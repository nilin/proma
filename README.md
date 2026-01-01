# ISOPO: Proximal policy gradients without pi-old

**Nilin Abrahamsen**


This repo contains a demonstration of the ISOPO isometric policy gradient (https://arxiv.org/pdf/2512.23353). It is a fork of [VeRL](https://github.com/volcengine/verl).


Isometric Policy Optimization (ISOPO) is an efficient method to approximate the natural policy gradient in a single gradient step. In comparison, existing proximal policy methods such as PPO, GRPO, GSPO, or CISPO use multiple gradient steps with variants of importance ratio clipping to approximate a natural gradient step relative to a reference policy. In its simplest form, ISOPO normalizes the log-probability gradient of each sequence in the Fisher metric before contracting with the advantages. Another variant of ISOPO transforms the microbatch advantages based on the neural tangent kernel in each layer. ISOPO applies this transformation layer-wise in a single backward pass and can be implemented with negligible computational overhead compared to vanilla REINFORCE.


<img width="948" height="388" alt="image" src="isopo/rectangle.png" />

The implementation of ISOPO is in https://github.com/nilin/isopo/blob/main/verl/workers/actor/dp_actor.py

[setup](setup.sh)
