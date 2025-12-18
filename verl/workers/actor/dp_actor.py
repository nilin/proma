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
        # SEPPO hooks
        #########################################################

        self.seppo = self.config.get("use_seppo", False)
        self.testing = self.config.get("seppo_testing", False)
        self.seppo_static_fraction = self.config.get("seppo_static_fraction", 0.5)
        self.seppo_dim = self.config.get("seppo_dim", 32)
        self.projection_dim_margin = 8
        self.random_projection_dim = self.seppo_dim + self.projection_dim_margin
        self.seppo_ema_decay = self.config.get("seppo_ema_decay", 0.8)
        self.seppo_min_preconditioner = self.config.get("seppo_min_preconditioner", 0.2)
        self.seppo_linear_interpolation = self.config.get("seppo_linear_interpolation", False)
        self.seppo_squared = self.config.get("seppo_squared", False)
        self.seppo_len_lim = self.config.get("seppo_len_lim", 6000)
        self.seppo_skip_rank_1_threshold = self.config.get("seppo_skip_rank_1_threshold", 0.0)

        if self.seppo:
            self.install_seppo_hooks()
            self.reset_seppo_stats()

        print(f"self.actor_optimizer: {self.actor_optimizer}")


    #########################################################

    def install_seppo_hooks(self):
        """Install SEPPO hooks that
        - store forward activations (act_in) per Linear layer
        - compute a per-layer scalar from act_in and backward grad_out in a full backward hook
        - use that scalar to scale the parameter gradients via param hooks

        This is lightweight and safe under FSDP: scaling is applied on sharded grads.
        """
        self.linear_modules = {n: sub for n, sub in self.actor_module.named_modules() if isinstance(sub, nn.Linear)}

        for i, (lname, lmod) in enumerate(self.linear_modules.items()):
            # storage on module
            lmod._seppo_act_in = None
            lmod._seppo_scale = 1.0

            # Forward hook to capture input activations
            def _fwd_hook(mod, inputs, output):
                in_tensor = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                try:
                    mod._seppo_act_in = in_tensor.detach().clone()
                except Exception:
                    mod._seppo_act_in = None

            # Full backward hook to compute a scalar using act_in and grad_out
            def _bwd_hook(mod, grad_input, grad_output, dump=False, lname=None):
                act_in = mod._seppo_act_in.clone()
                _g_out = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
                g_out = _g_out.clone()

                # Explicitly remove a leading singleton (e.g., [1, T, D] -> [T, D])
                if act_in.dim() >= 3 and act_in.size(0) == 1:
                    act_in = act_in[0]
                if g_out.dim() >= 3 and g_out.size(0) == 1:
                    g_out = g_out[0]

                non0 = self.mcb_advantages != 0
                act_in[non0] = act_in[non0] / self.mcb_advantages[non0,None]
                g_out[non0] = g_out[non0] / self.mcb_advantages[non0,None]
                g_out = g_out / self.loss_scale_factor

                if act_in.shape[-1] > self.seppo_len_lim:
                    print(f"not registering seppo for {lname} because act_in.shape[-1] > {self.seppo_len_lim}")
                    return

                mod.a_proj.append(self.right_singular_rows(act_in, self.seppo_dim, skip_svd=True))
                mod.g_proj.append(self.right_singular_rows(g_out, self.seppo_dim, skip_svd=True))

                if dump:
                    print(f"dumping tensors for {lname}")
                    self.dump_tensors(**{
                        f"act_in_{lname}": act_in,
                        f"g_out_{lname}": g_out,
                        f"a_proj_{lname}": mod.a_proj,
                        f"g_proj_{lname}": mod.g_proj,
                        f"projection_block_{lname}": self.projection_block,
                        f"mcb_advantages": self.mcb_advantages,
                        })

            if self.testing and i in [8,16,32,64,128]:
                import functools
                lmod.register_forward_hook(_fwd_hook)
                lmod.register_full_backward_hook(functools.partial(_bwd_hook, dump=True, lname=lname))
            else:
                lmod.register_forward_hook(_fwd_hook)
                lmod.register_full_backward_hook(_bwd_hook)

    def right_singular_rows(self, A: torch.Tensor, k: int, iterations: int = 3, skip_svd: bool = False) -> torch.Tensor:
        A = A.to(dtype=torch.float32)
        n, d = A.shape
        k_ = k+self.projection_dim_margin
        N = torch.randn(d, k_, device=A.device, dtype=A.dtype)
        Y = A @ N
        for _ in range(iterations-1):
            N = A.T @ Y
            Y = A @ N
            
        Q, R = torch.linalg.qr(Y)

        if skip_svd:
            return Q.T @ A
        else:
            _, S, V = torch.linalg.svd(Q.T @ A, full_matrices=False)
            return S, V

    def reset_seppo_stats(self):
        for lname, lmod in self.linear_modules.items():
            try:
                lmod.a_proj.clear()
                lmod.g_proj.clear()
            except Exception:
                lmod.a_proj = []
                lmod.g_proj = []

    def flatten_response_window(self, x: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
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

    def get_random_projections(self, micro_batches):
        # Each element in micro_batches is a DataProto. Access tensors via .batch
        # rather than string-indexing the DataProto itself.
        mb_sizes = []

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_sizes.append(int(model_inputs["attention_mask"].sum()))

        n_samples = sum(mb_sizes)

        rand = torch.randn(n_samples, self.random_projection_dim, device=self.device_name)
        projection, S, _ = torch.linalg.svd(rand, full_matrices=False)

        return projection.split(mb_sizes)

    def get_ema(self, name, value, ema_decay):
        if not hasattr(self, "ema"):
            self.ema = {}
        if name not in self.ema:
            self.ema[name] = value
        else:
            self.ema[name] = self.ema[name] * ema_decay + value * (1 - ema_decay)
        return self.ema[name]

    def dump_tensors(self, **tensors):
        import pandas as pd
        os.makedirs("dump", exist_ok=True)
        for key, value in tensors.items():
            pd.DataFrame(value.detach().cpu().float().numpy()).to_parquet(f"dump/{key}.parquet")

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

                self.projections = self.get_random_projections(micro_batches)

                self.actor_optimizer.zero_grad()

                for mb_idx, micro_batch in enumerate(micro_batches):
                    self.projection_block = self.projections[mb_idx]
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

                    if self.seppo:
                        attention_mask = model_inputs["attention_mask"]
                        advantages_w_prompt = torch.zeros_like(attention_mask)
                        advantages_w_prompt[:, -advantages.shape[1]:] = advantages
                        self.mcb_advantages = self.flatten_response_window(advantages_w_prompt, attention_mask)
                        self.loss_scale_factor = loss_scale_factor

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

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
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

                ################################################################################

                if self.seppo:
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

                            def precondition(grad, proj, mode="right", actual_mode=None):

                                if grad.shape[-1] > self.seppo_len_lim:
                                    print(f"skipping {actual_mode} seppo for {lname} with dim {grad.shape} because grad.shape[-1] > {self.seppo_len_lim}")
                                    return grad

                                if mode == "left":
                                    return precondition(grad.T, proj, mode="right", actual_mode="left").T

                                if actual_mode is None:
                                    actual_mode = mode

                                S, V = self.right_singular_rows(proj, self.seppo_dim)

                                if self.seppo_linear_interpolation:
                                    preconditioner = torch.ones_like(S)
                                    preconditioner[:self.seppo_dim] = torch.linspace(self.seppo_min_preconditioner, 1.0, self.seppo_dim) 

                                else:
                                    if self.seppo_squared:
                                        preconditioner = 1.0 / (S.pow(2) + 1e-8)
                                    else:
                                        preconditioner = 1.0 / (S + 1e-8)

                                    scaling = 1.0 / preconditioner[self.seppo_dim-1]
                                    scaling = torch.clamp(scaling, min=self.seppo_min_preconditioner / preconditioner[0])
                                    scaling = self.get_ema(f"seppo_scaling_{lname}_{actual_mode}", scaling, self.seppo_ema_decay)
                                    preconditioner = preconditioner * scaling
                                    preconditioner = torch.clamp(preconditioner, 0.0, 1.0)

                                    n_precondition = (preconditioner < 1.0).sum().item()
                                    print(f"{lname}.{pname} preconditioner #<1.0: {n_precondition}")
                                    print(f"first10: {[float(f'{x.item():.2g}') for x in preconditioner.flatten()[:10]]}\n")

                                diag = preconditioner - 1.0

                                grad_transformed = grad + ((grad @ V.T) * diag) @ V

                                if self.seppo_skip_rank_1 and torch.norm(grad_transformed) < torch.norm(grad)*self.seppo_skip_rank_1_threshold:
                                    print(f"{lname}.{pname} skipping seppo because {torch.norm(grad_transformed)/torch.norm(grad):.2f} < {self.seppo_skip_rank_1_threshold:.2f}")
                                    return grad
                                else:
                                    return grad_transformed


                            grad_transformed = g_local.clone()

                            if len(lmod.a_proj) > 0:
                                g_proj = torch.cat(lmod.g_proj, dim=0)
                                a_proj = torch.cat(lmod.a_proj, dim=0)
                                lmod.a_proj.clear()
                                lmod.g_proj.clear()
                                grad_transformed = precondition(grad_transformed, g_proj[:,sl[0]], mode="left")
                                grad_transformed = precondition(grad_transformed, a_proj[:,sl[1]], mode="right")
                            else:
                                print(f"no a_proj for {lname}, skipping seppo")

                            with torch.no_grad():
                                g_local.copy_(grad_transformed)

                    self.reset_seppo_stats()

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
