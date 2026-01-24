# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Replicate, Shard

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
import torch.distributed as dist
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
import functools
import math

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        #########################################################
        # ISOPO hooks
        #########################################################

        self.isopo = self.config.get("use_proma_isopo", False)
        self.testing = self.config.get("isopo_testing", False)

        self.isopo_norm_neg_power = self.config.get("isopo_norm_neg_power", 0.0)
        self.isopo_overlap_neg_power = self.config.get("isopo_overlap_neg_power", 0.0)
        self.isopo_rel_overlap_neg_power = self.config.get("isopo_rel_overlap_neg_power", 0.0)

        self.isopo_rel_overlap_reg = self.config.get("isopo_rel_overlap_reg", 1.0)
        self.isopo_overlap_reg = self.config.get("isopo_overlap_reg", 1.0)
        self.isopo_norm_reg = self.config.get("isopo_norm_reg", 1.0)
        self.isopo_nat_reg = self.config.get("isopo_nat_reg", 1.0)

        self.override_pg_loss = self.config.get("override_pg_loss", False)
        self.isopo_keep_small_invariant = self.config.get("isopo_keep_small_invariant", True)
        self.isopo_nat = self.config.get("isopo_nat", False)
        self.proma_relative_bound = self.config.get("proma_relative_bound", 1.0)
        self.proma_shrinkage = self.config.get("proma_shrinkage", 1.0) # 1.0 means full proma, 0.0 means no proma
        self.quick_ntk = self.config.get("quick_ntk", False) # use fast Gram-Schmidt approximation instead of full NTK inverse

        self.bypass_isopo_scaling = self.config.get("bypass_isopo_scaling", False)

        if self.isopo:
            self.install_isopo_hooks()
            self.reset_isopo_cache()
            self.include_advantages_in_loss = False
        else:
            self.include_advantages_in_loss = True

        print(f"self.actor_optimizer: {self.actor_optimizer}")
        #import torch.optim as optim
        #is_sgd = isinstance(self.actor_optimizer, optim.SGD) or self.actor_optimizer.__class__.__name__.lower() == "sgd"

        self.done_tests = set()

    #########################################################

    def install_isopo_hooks(self):
        """Install ISOPO hooks that
        - store forward activations (act_in) per Linear layer
        - compute a per-layer scalar from act_in and backward grad_out in a full backward hook
        - use that scalar to scale the parameter gradients via param hooks

        This is lightweight and safe under FSDP: scaling is applied on sharded grads.
        """
        self.linear_modules = {n: sub for n, sub in self.actor_module.named_modules() if isinstance(sub, nn.Linear)}

        for i, (lname, lmod) in enumerate(self.linear_modules.items()):
            # storage on module
            lmod._isopo_act_in = None
            lmod._isopo_scale = 1.0

            # Forward hook to capture input activations
            def _fwd_hook(mod, inputs, output):
                in_tensor = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                try:
                    mod._isopo_act_in = in_tensor.detach().clone()
                except Exception:
                    mod._isopo_act_in = None

            # Full backward hook to compute a scalar using act_in and grad_out
            def _bwd_hook(mod, grad_input, grad_output, dump=False, lname=None):
                act_in = mod._isopo_act_in.clone()
                _g_out = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
                g_out = _g_out.clone()

                act_in = act_in.to(dtype=torch.float32)
                g_out = g_out.to(dtype=torch.float32)

                # Explicitly remove a leading singleton (e.g., [1, T, D] -> [T, D])
                if act_in.dim() >= 3 and act_in.size(0) == 1:
                    act_in = act_in[0]
                if g_out.dim() >= 3 and g_out.size(0) == 1:
                    g_out = g_out[0]

                # for overlap computation against a fixed set of samples
                perm = torch.randperm(act_in.shape[0], device=act_in.device)
                topk_idx = perm[:250]
                a0 = act_in[topk_idx]
                g0 = g_out[topk_idx]

                act_in_seqs = self.unflatten_attention_mask_list(act_in, self.attention_mask)
                g_out_seqs = self.unflatten_attention_mask_list(g_out, self.attention_mask)

                seq_grads = []
                for i, (act_in_seq, g_out_seq) in enumerate(zip(act_in_seqs, g_out_seqs)):
                    seq_grads.append(g_out_seq.T @ act_in_seq)

                def add_reg_to_square(x, reg_factor, name, keep_small_invariant=self.isopo_keep_small_invariant):
                    reg = reg_factor * self.batch_stats(f"{name}_squared_reg_{lname}", x.pow(2))
                    if reg_factor == 0.0:
                        keep_small_invariant = False

                    if keep_small_invariant:
                        return torch.sqrt(x.pow(2) / (reg + 1e-8) + 1.0)
                    else:
                        return torch.sqrt(x.pow(2) + reg)

                if self.isopo_nat:
                    ntk = torch.zeros((len(seq_grads), len(seq_grads)), dtype=torch.float32, device=seq_grads[0].device)

                    for i in range(len(seq_grads)):
                        for j in range(i, len(seq_grads)):

                            ntk[i, j] = torch.sum(seq_grads[i] * seq_grads[j])
                            ntk[j, i] = ntk[i, j]
                    D, U = torch.linalg.eigh(ntk)
                    reg = self.isopo_nat_reg * self.batch_stats(f"isopo_nat_reg_{lname}", torch.mean(D))
                    preconditioner = reg / (D + reg + 1e-8)
                    advantages_preconditioned = U @ (preconditioner * (U.T @ self.seq_advantages))

                    grad = torch.stack(seq_grads, dim=-1) @ advantages_preconditioned

                else:
                    grad = 0.0
                    for i, (seq_grad, advantage) in enumerate(zip(seq_grads, self.seq_advantages)):

                        if self.bypass_isopo_scaling:
                            grad += advantage * seq_grad
                            continue

                        overlap = torch.norm(torch.sum((g0 @ seq_grad) * a0, dim=1)) / torch.norm(torch.norm(g0, dim=1) * torch.norm(a0, dim=1) + 1e-12)
                        overlap_over_norm = overlap / (torch.norm(seq_grad) + 1e-12)

                        p,q,r = self.isopo_norm_neg_power, self.isopo_overlap_neg_power, self.isopo_rel_overlap_neg_power

                        assert not self.include_advantages_in_loss, "isopo to be used with separate_advantages"

                        norm_w_reg = add_reg_to_square(torch.norm(seq_grad), self.isopo_norm_reg, "norm")
                        overlap_w_reg = add_reg_to_square(overlap, self.isopo_overlap_reg, "overlap")
                        rel_overlap_w_reg = add_reg_to_square(overlap_over_norm, self.isopo_rel_overlap_reg, "rel_overlap")

                        scaling_factor = 1.0 / (norm_w_reg.pow(p) * overlap_w_reg.pow(q) * rel_overlap_w_reg.pow(r) + 1e-8)
                        scaling = scaling_factor * advantage

                        grad += scaling * seq_grad 

                if hasattr(mod, "suppo_grad"):
                    suppo_grad = mod.suppo_grad

                    # Calculate normalized sequence gradients
                    seq_grads_normed = [sg / (torch.norm(sg) + 1e-8) for sg in seq_grads]

                    if self.quick_ntk:
                        def project_to_complement(acc_grad):
                            for _ in range(2):
                                for sg in seq_grads_normed:
                                    acc_grad = acc_grad - torch.sum(acc_grad*sg) * sg
                            return acc_grad

                        projected_grad = suppo_grad - project_to_complement(suppo_grad)

                    else:
                        ntk = torch.zeros((len(seq_grads), len(seq_grads)), dtype=torch.float32, device=seq_grads[0].device)
                        for i in range(len(seq_grads_normed)):
                            for j in range(i, len(seq_grads_normed)):
                                ntk[i, j] = torch.sum(seq_grads_normed[i] * seq_grads_normed[j])
                                ntk[j, i] = ntk[i, j]

                        def project(acc_grad):
                            dot_products = torch.stack([torch.sum(acc_grad*sg) for sg in seq_grads_normed])
                            inv = torch.linalg.inv(ntk + 1e-2 * torch.eye(len(seq_grads_normed), device=ntk.device, dtype=ntk.dtype))
                            weights = inv @ dot_products
                            result = torch.zeros_like(seq_grads[0])

                            print(f"projection weights: {weights}")
                            for w, sg in zip(weights, seq_grads_normed):
                                result = result + w * sg
                            return result

                        projected_grad = project(suppo_grad)

                    abs_bound = torch.norm(grad) * self.proma_relative_bound
                    if torch.norm(projected_grad) > abs_bound:
                        projected_grad = projected_grad * abs_bound / (torch.norm(projected_grad) + 1e-8)

                    print(f"grad: {torch.norm(grad)}")
                    print(f"projected_grad: {torch.norm(projected_grad)}")

                    mod.suppo_grad = suppo_grad - self.proma_shrinkage * projected_grad + grad
                else:
                    mod.suppo_grad = grad


            if self.testing and i in [8,16,32,64,128]:
                lmod.register_forward_hook(_fwd_hook)
                lmod.register_full_backward_hook(functools.partial(_bwd_hook, dump=True, lname=lname))
            else:
                lmod.register_forward_hook(_fwd_hook)
                lmod.register_full_backward_hook(functools.partial(_bwd_hook, lname=lname))

    def batch_stats(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "done_batch_stats"):
            self.done_batch_stats = {}
        if not hasattr(self, "current_batch_stats"):
            self.current_batch_stats = {}

        if name not in self.current_batch_stats:
            self.current_batch_stats[name] = []

        self.current_batch_stats[name].append(value)
        
        if name in self.done_batch_stats:
            return self.done_batch_stats[name]
        else:
            print(f"no batch history for {name}, returning value {value}")
            return value

    def update_batch_stats(self, ema_decay: float = 0.9):
        if not hasattr(self, "current_batch_stats"):
            return
        for name, values in self.current_batch_stats.items():
            current_value = torch.mean(torch.stack(values))
            if name not in self.done_batch_stats:
                self.done_batch_stats[name] = current_value
            else:
                self.done_batch_stats[name] = self.done_batch_stats[name] * ema_decay + current_value * (1 - ema_decay)
            
            self.current_batch_stats[name].clear()

    def reset_isopo_cache(self):
        for lname, lmod in self.linear_modules.items():
            try:
                lmod.a_proj.clear()
                lmod.g_proj.clear()
            except Exception:
                lmod.a_proj = []
                lmod.g_proj = []

    def flatten_attention_mask(self, x: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Flatten non-padding response tokens to (N, ...).

        - Input `x` has shape `(B, R)` or `(B, R, H, ...)`.
        - `response_mask` has shape `(B, R)` with 1/True for valid response tokens.
        - Returns tensor with first two dims flattened to the valid rows only:
          `(N,)` if `x` is 2D, or `(N, H, ...)` if `x` has more dims, where `N = response_mask.sum()`.
        """
        if x.dim() == 1:
            x = x[:,None].expand_as(response_mask)
        
        assert x.dim() >= 2, "x must have at least 2 dims (B, R, ...)"
        assert x.shape[0] == response_mask.shape[0] and x.shape[1] == response_mask.shape[1], (
            f"x first two dims {x.shape[:2]} must match response_mask {response_mask.shape}"
        )
        # Ensure mask device matches x for boolean indexing
        sel = response_mask.to(device=x.device).bool()
        return x[sel]

    def dt_local_global_slices(self, g: DTensor) -> tuple[slice, ...]:
        """Return the global index slices that the local DTensor shard maps to.

        This uses the DTensor's device mesh coordinates and placements to compute,
        for each sharded dimension, the [start:end) range of the local shard in the
        full (global) tensor. Replicated dimensions are returned as slice(None).

        Notes:
        - No collectives are issued. This is cheap and safe to call during training.
        - Works with uneven splits (remainder distributed to lower ranks).
        - If the DTensor has Partial placements, the slice still denotes the logical
          region contributed by this rank, though values may be partially reduced.
        """
        assert isinstance(g, DTensor), "Expected a DTensor"

        global_shape = tuple(g.shape)
        mesh = g.device_mesh
        coord = mesh.get_coordinate()

        # Fallback: if no mesh coordinate (single-rank or not initialized), return full slices
        if coord is None:
            return tuple(slice(None) for _ in global_shape)

        slices = [slice(None)] * len(global_shape)
        for mesh_axis, placement in enumerate(g.placements):
            if isinstance(placement, Shard):
                dim = placement.dim
                S = global_shape[dim]
                world = mesh.size(mesh_axis)
                i = coord[mesh_axis]
                base = S // world
                rem = S % world
                start = i * base + min(i, rem)
                length = base + (1 if i < rem else 0)
                slices[dim] = slice(start, start + length)
            elif isinstance(placement, Replicate):
                # full dimension
                continue
            else:
                # Partial or other placements: keep slice(None) to denote full logical dim
                continue

        return tuple(slices)

    def dump_tensors(self, tensors, name="data"):
        import numpy as np
        from datetime import datetime
        id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"dump/{name}_{id}", exist_ok=True)
        for key, value in tensors.items():
            arr = value.detach().cpu().float().numpy()
            np.save(f"dump/{name}_{id}/{key}.npy", arr)

    #########################################################
    # seqwise isopo
    #########################################################

    def test_flatten_unflatten(self, attention_mask: torch.Tensor):
        if "test_flatten_unflatten" in self.done_tests:
            return
        self.done_tests.add("test_flatten_unflatten")

        x = torch.randn(attention_mask.shape[0], attention_mask.shape[1], 5, device=attention_mask.device) * attention_mask[:,:,None]

        flat_x = self.flatten_attention_mask(x, attention_mask)
        unflat_x = self.unflatten_attention_mask(flat_x, attention_mask)
        unflat_x_list = self.unflatten_attention_mask_list(flat_x, attention_mask)

        self.dump_tensors({
            f"attention_mask": attention_mask,
            f"x": x,
            f"flat_x": flat_x,
            f"unflat_x": unflat_x,
        }, name="test_flatten_unflatten")

        assert torch.allclose(x, unflat_x)
        for a, x, y in zip(attention_mask, x, unflat_x_list):
            assert torch.allclose(x[a.bool()], y)

        print("flatten and unflatten test passed")
        return unflat_x

    def unflatten_attention_mask(self, x_flat, attention_mask):
        B, R = attention_mask.shape[:2]
        m = attention_mask.view(B, R).to(x_flat.device).bool()
        out = x_flat.new_zeros((B, R) + x_flat.shape[1:])
        expected = int(m.sum().item())
        assert x_flat.shape[0] == expected, f"flat len {x_flat.shape[0]} != mask sum {expected}"
        out[m] = x_flat
        return out

    def unflatten_attention_mask_list(self, flat_x: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        res = []
        unflat_x = self.unflatten_attention_mask(flat_x, attention_mask)
        for row, a in zip(unflat_x, attention_mask):
            res.append(row[a.bool()])
        return res

    #########################################################

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    # Avoid in-place ops on views returned by custom Functions; use out-of-place scaling
                    logits_rmpad = logits_rmpad / temperature

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    # Avoid in-place ops on views returned by custom Functions; use out-of-place scaling
                    logits = logits / temperature
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for mcb_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    ################################################################################
                    if self.isopo:
                        self.attention_mask = attention_mask = model_inputs["attention_mask"]
                        self.seq_advantages = advantages[:, 0].to(attention_mask.device)
                        self.test_flatten_unflatten(attention_mask)
                    ################################################################################

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    if self.override_pg_loss:
                        if self.include_advantages_in_loss:
                            pg_loss = agg_loss(loss_mat=-log_prob*advantages, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        else:
                            pg_loss = agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    else:
                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages if self.include_advantages_in_loss else torch.ones_like(advantages),
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                    if self.testing:
                        break

                if self.isopo:
                    for lname, lmod in self.linear_modules.items():
                        for pname, p in lmod.named_parameters(recurse=False):
                            dtg = p.grad
                            if dtg is None:
                                continue
                            sl = self.dt_local_global_slices(dtg)

                            if len(sl) != 2:
                                with torch.no_grad():
                                    dtg.mul_(0.0)
                                print(f"WARN: rank {dist.get_rank()} {lname}.{pname} has {len(sl)} dimensions for shard mapping")
                                continue

                            # Work on local shard tensor; no collectives
                            g_local = dtg.to_local()
                            # Select matching per-row/col scalars and align dtype/device

                            grad_transformed = 0.0 * g_local.clone() + lmod.suppo_grad
                            lmod.suppo_grad = 0.0

                            with torch.no_grad():
                                g_local.copy_(grad_transformed)

                    self.reset_isopo_cache()
                    self.update_batch_stats()
                ################################################################################

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
