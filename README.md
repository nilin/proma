# ISOPO: Proximal policy gradients without pi-old

**Nilin Abrahamsen**


This repo contains a demonstration of the Projected Microbatch Accumulation (PROMA), [arxiv:2601.10498](https://arxiv.org/abs/2601.10498). 

This demonstration is a fork of [VeRL](https://github.com/volcengine/verl).

Projected Microbatch Accumulation (PROMA) is a proximal policy update method for large language model fine-tuning. PROMA accumulates policy gradients across microbatches by projecting out sequence-wise gradient components before microbatch aggregation. The projection is applied layer-wise during the backward pass, enabling efficient implementation without additional forward or backward passes. Empirically, PROMA enforces tighter control of local KL divergence than GRPO, resulting in more stable policy learning. Unlike PPO and GRPO, PROMA achieves proximal updates without inducing entropy collapse and does not rely on a reference policy or likelihood-ratio clipping.


The implementation of PROMA (and [ISOPO](https://arxiv.org/abs/2512.23353)) are in https://github.com/nilin/isopo/blob/main/verl/workers/actor/dp_actor.py

- [setup example](setup.sh)
- [run script example](run/run.sh)
