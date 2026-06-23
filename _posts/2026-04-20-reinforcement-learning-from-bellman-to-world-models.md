---
layout: post
title: "Reinforcement Learning: From Bellman Equations to World Models"
date: 2026-04-20 10:00:00+0800
description: A technical overview of reinforcement learning from dynamic programming and Q-learning to deep RL, policy optimization, and modern world models.
tags: [reinforcement-learning, deep-rl, world-models]
categories: [research]
related_posts: false
_styles: |
  .post .post-header { margin-bottom: 3.5rem; }
  .post .post-title {
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.06;
    margin-bottom: 1.35rem;
  }
  .post .post-content {
    font-size: 1.22rem;
    font-weight: 300;
    letter-spacing: 0.002em;
    line-height: 1.75;
  }
  .post .post-content h2 {
    font-size: clamp(2.15rem, 3.4vw, 3rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.12;
    margin: 5.75rem 0 2.15rem;
  }
  .post .post-content h3 {
    font-size: clamp(1.6rem, 2.45vw, 2.1rem);
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.14;
    margin: 4.5rem 0 1.6rem;
  }
  .post .post-content p { margin-bottom: 1.35rem; }
  .post .post-content strong { font-weight: 700; }
  .post .post-content em { font-weight: 300; }
  .post .post-content a {
    text-decoration: underline;
    text-decoration-thickness: 1px;
    text-underline-offset: 0.16em;
  }
  .post .post-content li { margin-bottom: 0.7rem; }
  @media (max-width: 576px) {
    .post .post-header { margin-bottom: 2.4rem; }
    .post .post-content { font-size: 1.08rem; }
    .post .post-content h2 { margin-top: 4.25rem; }
    .post .post-content h3 { margin-top: 3.5rem; }
  }
toc:
  sidebar: right
  collapse: expanded
---

Reinforcement learning (RL) studies how an agent should act through interaction so as to maximize long-term return. The field is mathematically elegant, historically deep, and still evolving rapidly. What I find most helpful is not to memorize RL as a flat list of algorithms, but to read it as a sequence of answers to a few recurring questions:

- How should we formalize delayed consequences?
- How can we estimate long-term value from partial experience?
- How do we improve a policy without destabilizing learning?
- When is it worth learning a model of the environment?

This post follows that thread from Bellman equations and temporal-difference learning to policy gradients, deep RL, offline RL, and modern world models.

## The Markov decision process view

The standard formalism of RL is the **Markov Decision Process (MDP)**, defined by states $s$, actions $a$, a transition kernel $P(s' \mid s,a)$, a reward function $r(s,a)$, and a discount factor $\gamma \in [0,1)$. A policy $\pi(a \mid s)$ maps states to action distributions.

For a given policy, the state-value function is

$$
V^\pi(s)
=
\mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \,\middle|\, s_0=s\right],
$$

and the action-value function is

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \,\middle|\, s_0=s, a_0=a\right].
$$

These definitions are not merely bookkeeping devices. They encode the central RL difficulty: actions matter because they influence future state distributions, not just immediate rewards.

**Read an MDP as a causal loop, not a static dataset.** The agent chooses an action, the environment changes state, and the resulting reward is only one part of the evidence available for evaluating that decision.

- **State and transition.** *What can happen next.* $P(s' \mid s,a)$ captures the part of the world the agent must predict or reason through.
- **Policy.** *How the agent intervenes.* $\pi(a \mid s)$ changes the future state distribution, which is why supervised prediction of immediate reward is not enough.
- **Return.** *What is optimized.* Discounting makes the infinite-horizon objective well defined and expresses how much delayed outcomes should influence a current decision.

## Bellman equations as the backbone of RL

The key recursion is the **Bellman equation**, introduced in [Bellman (1957)](https://press.princeton.edu/books/hardcover/9780691651873/dynamic-programming):

$$
V^\pi(s)
=
\mathbb{E}_{a \sim \pi,\, s' \sim P}
\left[r(s,a) + \gamma V^\pi(s')\right].
$$

Similarly,

$$
Q^\pi(s,a)
=
\mathbb{E}_{s' \sim P,\, a' \sim \pi}
\left[r(s,a) + \gamma Q^\pi(s',a')\right].
$$

For optimal control, the Bellman optimality equation becomes

$$
Q^*(s,a)
=
\mathbb{E}_{s' \sim P}
\left[r(s,a) + \gamma \max_{a'} Q^*(s',a')\right].
$$

These recursions are the structural core of the field. Dynamic programming, temporal-difference learning, policy iteration, actor-critic methods, and many modern algorithms can all be understood as approximate ways of solving Bellman-style fixed-point equations under data and computation constraints.

## Dynamic programming and the model-based ideal

If the environment dynamics $P$ and reward function are known, we can solve an MDP using **dynamic programming**. Classical value iteration repeatedly applies the Bellman optimality operator, while policy iteration alternates between policy evaluation and policy improvement.

This setting is idealized but useful. It clarifies what makes RL hard in practice:

- the transition model is often unknown,
- the state space is too large for tabular enumeration,
- and exact backups become impossible under function approximation.

Thus, most of RL is about making Bellman reasoning work approximately from sampled experience.

## Monte Carlo and temporal-difference learning

Two classical routes to policy evaluation are **Monte Carlo** estimation and **temporal-difference (TD)** learning.

Monte Carlo methods wait until the end of a trajectory and regress value estimates toward full sampled returns. They are unbiased but can have high variance.

TD methods bootstrap from current estimates instead. The simplest TD(0) update for state values is

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha \left(r_t + \gamma V(s_{t+1}) - V(s_t)\right).
$$

This target uses one-step lookahead and therefore has lower variance, but it introduces bias because the target depends on an estimate rather than a realized full return.

The bias-variance tradeoff introduced here never disappears from RL. It reappears later in eligibility traces, actor-critic learning, advantage estimation, and modern large-scale policy optimization.

## Q-learning and off-policy control

[**Q-learning**](https://link.springer.com/article/10.1007/BF00992698) is one of the most important algorithms in RL because it provides a simple model-free route to optimal control. Its update is

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\Big(
r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)
\Big).
$$

Several properties made Q-learning historically central.

First, it is **off-policy**: the behavior policy used to collect data can differ from the greedy policy implied by the learned $Q$ function. Second, it uses bootstrapping, which makes learning sample-efficient in many discrete settings. Third, it avoids explicit model estimation.

At the same time, Q-learning also exposes fundamental challenges:

- maximization can amplify estimation error,
- exploration is often difficult,
- and combining off-policy bootstrapping with nonlinear function approximation can become unstable.

These challenges strongly shaped the next decades of RL research.

## The deadly triad

A classic warning in RL is the **deadly triad**:

- function approximation,
- bootstrapping,
- and off-policy learning.

Individually, each ingredient is useful. Together, they can cause divergence or severe instability. Much of deep RL can be read as an engineering and algorithmic effort to tame the deadly triad with replay, target networks, regularization, trust regions, conservative objectives, and better data usage.

Keeping this triad in mind is extremely helpful when reading new RL papers. Many methods differ in surface form, but they are often solving the same stability problem.

## Policy gradients: optimize the policy directly

Value-based methods are not always the most natural choice, especially in continuous action spaces. Policy-based methods instead parameterize the policy itself and optimize expected return directly.

The **policy gradient theorem** of [Sutton et al. (1999)](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) states that

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, Q^{\pi_\theta}(s_t,a_t)
\right].
$$

This result is important because it avoids differentiating the state distribution explicitly. In practice, one replaces $Q^{\pi_\theta}$ with sampled returns or lower-variance surrogates.

The simplest algorithm, REINFORCE, uses full-return Monte Carlo estimates:

$$
\nabla_\theta J(\theta)
\approx
\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t.
$$

This estimator is unbiased but often has high variance. That motivates baselines and critic networks.

## Advantage functions and actor-critic methods

The **advantage function**

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

measures how much better or worse an action is than the policy's average behavior at a state. Replacing raw returns with advantages often reduces gradient variance substantially.

This leads naturally to **actor-critic** methods:

- the **actor** updates the policy,
- the **critic** estimates value information used for the gradient.

Conceptually, actor-critic is one of the most durable ideas in RL because it combines:

- the flexibility of direct policy parameterization,
- the sample reuse benefits of value estimation,
- and the possibility of continuous control.

Many modern RL algorithms, from A2C/A3C to PPO and SAC, fit naturally into this actor-critic template.

## Deep Q-Networks and the birth of deep RL

The deep RL era was catalyzed by [**DQN**](https://www.nature.com/articles/nature14236), which applied Q-learning to high-dimensional observations such as Atari frames. This was historically significant because it showed that convolutional neural networks could learn control policies directly from pixels with a single algorithmic template.

Two stabilizing ideas were crucial:

**Experience replay** stores transitions in a replay buffer and samples them approximately independently for training. This breaks the strong temporal correlation of online trajectories, lets rare but useful transitions be reused, and makes neural-network optimization resemble ordinary supervised learning more closely. Its cost is that the data distribution becomes stale, so later methods often tune replay priorities, buffer age, and off-policy corrections carefully.

**Target networks** address a different instability: bootstrapping against a rapidly moving function approximator. DQN freezes a separate target network for several updates, turning a moving fixed-point problem into a sequence of better-behaved regression problems. Replay and targets are complementary; without either one, the combination of nonlinear approximation, bootstrapping, and off-policy data is markedly less stable.

<figure style="margin: 1.75rem auto; max-width: 760px;">
  <img src="{{ '/assets/img/blog/dqn-seaquest-result.png' | relative_url }}" alt="Seaquest Atari frame used by DQN" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">An Atari Seaquest observation, representative of the raw visual input used in DQN. Reproduced from <a href="https://arxiv.org/abs/1312.5602">Mnih et al. (2013)</a>.</figcaption>
</figure>

These ideas became part of the standard RL toolbox. Many later off-policy methods can be viewed as elaborations of replay-buffer learning with more careful targets and critics.

## Improvements to value-based deep RL

After DQN, a long sequence of refinements followed:

- **Double DQN** reduces overestimation bias by decoupling action selection from target evaluation.
- **Dueling networks** separate state value and action advantage estimation.
- **Prioritized replay** samples more informative transitions more often.
- **Distributional RL** models the full return distribution instead of just its expectation.

This period is instructive because it shows how deep RL progressed: not by one giant theoretical leap, but by repeatedly addressing specific failure modes in approximation, target bias, and data efficiency.

## Trust regions and stable policy improvement

Policy gradient methods can also be unstable, especially when each update changes the policy too aggressively. [**Trust Region Policy Optimization (TRPO)**](https://proceedings.mlr.press/v37/schulman15.html) formalized the principle that policy updates should stay within a local trust region in distribution space.

The basic idea is to maximize a surrogate objective while constraining the KL divergence between the new and old policies. This improved stability, but the optimization procedure is relatively cumbersome.

[**Proximal Policy Optimization (PPO)**](https://arxiv.org/abs/1707.06347) simplified the idea with the clipped objective

$$
\mathcal{L}^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min\left(
r_t(\theta)\hat{A}_t,\,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right],
$$

where

$$
r_t(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}.
$$

PPO became widely used because it hits a strong practical balance:

- simple to implement,
- relatively robust,
- broadly effective across control, robotics, and language-model fine-tuning settings.

## Generalized advantage estimation

One of the practical tools that made actor-critic methods work well is [**Generalized Advantage Estimation (GAE)**](https://arxiv.org/abs/1506.02438). It constructs advantage estimates by exponentially weighting multi-step TD residuals, trading off bias and variance smoothly.

The reason GAE matters is that it turned what could have been a fragile estimation problem into a tunable engineering component. In large-scale policy optimization, the quality of the advantage estimator is often nearly as important as the optimizer itself.

## Continuous control: DDPG, TD3, and SAC

Many real control problems have continuous action spaces, where maximizing over actions inside a Bellman backup is not trivial. This motivated specialized actor-critic methods.

**DDPG** combines deterministic policy gradients with replay-buffer training and target networks. It was influential because it showed that deep off-policy actor-critic learning could handle continuous actions, but it is often brittle: actor errors can exploit optimistic critic errors and quickly destabilize learning.

**TD3** improves this failure mode with clipped double critics, delayed policy updates, and target-policy smoothing. The common theme is conservative value estimation: when the critic is uncertain, avoiding systematic overestimation is often more important than pushing the actor toward the nominally best action.

**Soft Actor-Critic (SAC)** introduced a maximum-entropy objective

$$
J(\pi)
=
\sum_t
\mathbb{E}_{(s_t,a_t)\sim \pi}
\left[
r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))
\right].
$$

The entropy term encourages stochastic exploration and often produces more stable training. SAC is one of the most practically important RL algorithms because it combines strong sample efficiency with impressive robustness in continuous control.

## Model-based reinforcement learning

Model-free RL avoids explicit environment modeling, but it can be expensive in sample complexity. This motivates **model-based RL**, where the agent learns or uses a dynamics model to support planning, imagination, or better policy improvement.

There are several ways a model can be used:

- simulate rollouts for planning,
- improve value estimation,
- generate synthetic training data,
- or learn compact latent dynamics for control in representation space.

Historically, model-based RL has swung between optimism and disappointment. Simple learned models often accumulate prediction error over long horizons, causing compounding bias in planning. Recent world-model methods became much more successful by modeling latent structure rather than raw observations directly.

## World models and latent imagination

Modern **world models** typically learn:

- a representation model mapping observations to latent states,
- a transition model in latent space,
- and reward or value heads on top of those latent dynamics.

This makes imagined rollouts cheaper and often more stable than pixel-space prediction. The resulting framework is particularly appealing when real interaction is expensive, as in robotics, autonomous experimentation, or safety-constrained decision making.

Methods in the **Dreamer** family are especially important milestones. They show that latent imagination can be used not just for toy planning, but as a scalable control strategy across diverse domains. [**DreamerV3**](https://arxiv.org/abs/2301.04104) is notable for reducing per-domain retuning and improving robustness, suggesting that world-model RL is becoming more generally usable.

<figure style="margin: 1.75rem auto; max-width: 760px;">
  <img src="{{ '/assets/img/blog/dreamerv3-world-model-learning.png' | relative_url }}" alt="DreamerV3 world model learning architecture" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">DreamerV3 world-model learning: observations are encoded into discrete latents and predicted through recurrent dynamics. Reproduced from <a href="https://arxiv.org/abs/2301.04104">Hafner et al. (2023)</a>.</figcaption>
</figure>

<figure style="margin: 1.75rem auto; max-width: 900px;">
  <img src="{{ '/assets/img/blog/dreamerv3-cross-domain-results.png' | relative_url }}" alt="DreamerV3 results across RL benchmark domains" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">DreamerV3 results across diverse domains, showing one configuration used across benchmark families. Reproduced from <a href="https://arxiv.org/abs/2301.04104">Hafner et al. (2023)</a>.</figcaption>
</figure>

## Offline RL

Another major modern direction is **offline RL**, where the policy must be learned from a fixed dataset without further interaction. This setting is attractive when exploration is unsafe, expensive, or impossible.

Offline RL introduces a distinctive problem: the learned policy may choose actions not well supported by the dataset, leading to severe extrapolation error in value estimation. This motivates conservative critics, behavior regularization, and sequence-model-based approaches.

The offline setting is highly relevant for real applications because many domains already have logged historical data, but do not permit unconstrained online trial and error.

## RL as sequence modeling

Recent work has also reframed RL as a sequence modeling problem. [**Decision Transformer**](https://arxiv.org/abs/2106.01345) is a clean example: instead of explicitly learning Bellman backups, it models trajectories autoregressively conditioned on a target return.

This is conceptually striking because it shows that some RL problems can be attacked with tools from language modeling and supervised sequence prediction. It does not replace all of RL, but it broadens the design space. In data-rich settings, especially offline ones, the boundary between control and sequence modeling becomes much less rigid.

## Exploration, credit assignment, and long horizons

Despite decades of progress, several core difficulties remain central.

**Exploration** is difficult because reward signals can be sparse, delayed, or deceptive. Count-based bonuses, intrinsic motivation, entropy regularization, and curiosity objectives all add an incentive for visiting informative states, but they must be balanced against safety and task reward; novelty alone is not useful if it repeatedly drives the agent into irrelevant or unsafe regions.

**Credit assignment** becomes hard when a reward arrives long after the decision that enabled it. Eligibility traces, advantage estimation, temporal abstractions, and learned models all supply shortcuts for moving learning signal backward through time, yet long-horizon credit remains an open problem whenever relevant events are rare or partially observed.

**Distribution shift** links off-policy, offline, and model-based RL. A critic may evaluate actions that the dataset never supported, or a planner may optimize trajectories a learned model predicts poorly. Conservative objectives, uncertainty estimates, and explicit behavior regularization are therefore not optional add-ons; they are defenses against acting outside the evidence available to the learner.

## A practical taxonomy

I find it useful to organize RL methods by what they estimate most directly.

**Value-based methods** learn $V$ or $Q$ and derive a policy indirectly, typically by maximizing action value. They are natural for discrete control and can reuse data efficiently, but bootstrapping, off-policy data, and function approximation can interact unstably.

**Policy-based methods** parameterize the policy directly and optimize expected return. They are well suited to continuous or stochastic control, but variance reduction, trust regions, and exploration are central because policy gradients are estimated from noisy trajectories.

**Actor-critic methods** combine direct policy optimization with a learned value estimator. This is the dominant practical template because the critic reduces variance while the actor can represent rich continuous-action policies; most differences among modern algorithms concern how conservatively the critic is trained and how far the actor may move per update.

**Model-based methods** learn or exploit dynamics to improve planning and sample efficiency. They are especially attractive for data-limited, high-cost environments, but they must control compounding model error by planning in suitable latent spaces, keeping imagined rollouts short, or estimating uncertainty.

## Why RL still matters

Reinforcement learning remains important because it addresses a setting that ordinary supervised learning does not fully capture: **decision making under delayed consequences and interaction constraints**.

This is especially relevant for trustworthy AI and scientific discovery. Many systems in those domains must:

- optimize long-horizon objectives,
- balance exploration against risk,
- satisfy constraints under uncertainty,
- and learn from interaction rather than static labels alone.

Those are fundamentally RL-shaped problems, even when the eventual solution uses ideas from sequence modeling, control theory, or world models.

## Outlook

The field is moving toward greater integration. Deep RL now borrows heavily from supervised learning, generative modeling, and large-scale sequence modeling. World models and offline RL are making the data interface more realistic. Policy optimization remains essential, but increasingly it is embedded inside richer systems for representation learning, planning, and uncertainty management.

If there is one durable lesson across the history of RL, it is that the Bellman view never really disappears. Even when we use transformers, latent dynamics, or conservative offline objectives, we are still grappling with the same central issue Bellman formalized long ago: how should present decisions be evaluated through their effect on the future?

## References

1. Richard Bellman. *Dynamic Programming*. Princeton University Press, 1957. https://press.princeton.edu/books/hardcover/9780691651873/dynamic-programming
2. Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*. Second Edition, 2018. http://incompleteideas.net/book/the-book-2nd.html
3. Christopher J. C. H. Watkins and Peter Dayan. *Q-learning*. Machine Learning, 1992. https://link.springer.com/article/10.1007/BF00992698
4. Richard S. Sutton, David McAllester, Satinder Singh, and Yishay Mansour. *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS 1999. https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html
5. Volodymyr Mnih et al. *Human-level Control through Deep Reinforcement Learning*. Nature, 2015. https://www.nature.com/articles/nature14236
6. John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, and Pieter Abbeel. *Trust Region Policy Optimization*. ICML 2015. https://proceedings.mlr.press/v37/schulman15.html
7. Timothy P. Lillicrap et al. *Continuous Control with Deep Reinforcement Learning*. 2015. https://arxiv.org/abs/1509.02971
8. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. *Proximal Policy Optimization Algorithms*. 2017. https://arxiv.org/abs/1707.06347
9. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018. https://proceedings.mlr.press/v80/haarnoja18b.html
10. Hado van Hasselt, Arthur Guez, and David Silver. *Deep Reinforcement Learning with Double Q-learning*. AAAI 2016. https://arxiv.org/abs/1509.06461
11. Lili Chen et al. *Decision Transformer: Reinforcement Learning via Sequence Modeling*. NeurIPS 2021. https://arxiv.org/abs/2106.01345
12. Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. *Mastering Diverse Domains through World Models*. 2023. https://arxiv.org/abs/2301.04104
