---
layout: post
title: "Diffusion Models: From Langevin Dynamics to Discrete Diffusion"
date: 2025-06-12 10:00:00+0800
description: A technical overview of score-based generative modeling, DDPM, DDIM, guided diffusion, and discrete diffusion models.
tags: [diffusion-models, generative-ai, score-based-modeling]
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

Diffusion models have become one of the most influential paradigms in modern generative modeling. Their rise is not just an empirical story about image synthesis quality; it is also a mathematical story about how denoising, score estimation, stochastic dynamics, and probability transport fit together. What makes the area especially rich is that several lines of work that once looked separate now form a reasonably coherent picture: Langevin sampling, score matching, denoising diffusion probabilistic models, non-Markovian samplers, guidance methods, and more recent discrete-state variants.

This post gives a technical overview of that picture. I focus on the parts that are most useful for reading papers and implementing models in practice: the score-based starting point, the DDPM formulation, DDIM acceleration, guided diffusion, and discrete diffusion models.

## Why diffusion models are conceptually different

Classical likelihood-based generative models usually start by parameterizing a density directly or by constructing an invertible map from a simple latent distribution to data. Diffusion models take a different route. Instead of learning the data distribution in one shot, they define a sequence of slightly corrupted distributions and train a model to invert that corruption progressively.

This viewpoint has three important consequences.

First, learning is localized. The model only needs to answer a comparatively easier question: given a noisy sample, in which direction should we move to make it slightly less noisy? Second, the target distributions become smoother at high noise levels, which stabilizes training. Third, the framework naturally supports conditional control because reverse dynamics can be perturbed by gradients or conditional denoisers.

At a high level, diffusion models solve generation by learning many small inverse steps rather than one globally difficult transformation.

## Score functions and Langevin dynamics

The cleanest theoretical entry point is the **score function**

$$
\nabla_x \log p(x),
$$

which tells us in which direction probability density increases most rapidly.

If we knew the score exactly, then we could sample from the target distribution using **Langevin dynamics**. This score-based route to generative modeling was made practical by [Song and Ermon (2019)](https://arxiv.org/abs/1907.05600). In its overdamped form, the update is

$$
x_{k+1} = x_k + \frac{\epsilon}{2}\nabla_x \log p(x_k) + \sqrt{\epsilon}\, z_k,
\qquad z_k \sim \mathcal{N}(0, I).
$$

The first term drifts the sample toward regions of high probability and the second term injects Gaussian noise so that the chain explores rather than collapses. Under suitable regularity conditions and sufficiently small step size, this Markov chain converges to the target density.

In practice, we do not know $\nabla_x \log p(x)$, so we learn a neural approximation $s_\theta(x)$. This immediately gives the score-based sampling rule

$$
x_{k+1} = x_k + \frac{\epsilon}{2} s_\theta(x_k) + \sqrt{\epsilon}\, z_k.
$$

The difficulty is that the score of the raw data distribution can be hard to estimate accurately, especially when the data lie near a low-dimensional manifold inside a high-dimensional ambient space. This motivates a central diffusion idea: estimate scores not just for the original distribution, but for a family of progressively smoothed distributions.

## Noise perturbation and annealed score estimation

Let $p_\sigma(x)$ denote the data distribution convolved with Gaussian noise of scale $\sigma$. For large $\sigma$, the distribution is smooth and easy to model; for small $\sigma$, it becomes sharp and data-like. Noise-conditioned score networks learn

$$
s_\theta(x,\sigma) \approx \nabla_x \log p_\sigma(x)
$$

for many noise scales $\sigma_1 > \sigma_2 > \dots > \sigma_L$.

Sampling then proceeds through **annealed Langevin dynamics**:

1. Start from strong noise.
2. Sample at a high noise level using the corresponding score estimate.
3. Gradually reduce the noise level while continuing to refine the sample.

This already contains the qualitative logic of diffusion: generation is a trajectory through a family of noise scales. DDPM later made this intuition particularly practical by choosing a simple forward corruption process and a tractable training objective.

## Forward diffusion in DDPM

The standard **Denoising Diffusion Probabilistic Model (DDPM)**, introduced by [Ho, Jain, and Abbeel (2020)](https://arxiv.org/abs/2006.11239), defines a forward noising chain

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I\right),
\qquad t = 1,\dots,T,
$$

where $\beta_t \in (0,1)$ is a variance schedule.

Intuitively, each step keeps most of the previous signal but injects a small amount of Gaussian noise. After many steps, the original data sample $x_0$ is destroyed and $x_T$ becomes approximately standard Gaussian.

Defining $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, one obtains the extremely useful closed form

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

Equivalently,

$$
q(x_t \mid x_0)
=
\mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I\right).
$$

This formula matters because it lets us sample $x_t$ from $x_0$ directly, without simulating the entire chain from step $1$ to $t$. During training, one may therefore choose a random time index $t$, synthesize the corresponding noisy sample in one step, and optimize a denoising objective.

**The key asymmetry is worth keeping in view.** The forward chain is deliberately simple and completely specified; the reverse chain is the learning problem. This is why DDPM can use a variational objective without ever needing the intractable marginal likelihood in closed form.

- **Forward process.** *Known corruption.* Starting from $x_0$, the fixed Markov chain $q(x_t \mid x_{t-1})$ gradually removes information until $x_T$ is close to Gaussian noise. The schedule is a modeling choice, but no neural network is needed to evaluate or sample this chain.
- **Reverse process.** *Learned denoising.* The true conditional $q(x_{t-1} \mid x_t)$ depends on the unknown data distribution. DDPM replaces it with $p_\theta(x_{t-1} \mid x_t)$, whose mean is predicted from a noisy input and a timestep embedding.
- **Training signal.** *A local surrogate for a global density model.* Because $q(x_t \mid x_0)$ is analytically tractable, training can draw one random time, create one noisy observation, and ask the network to recover its noise or clean signal.

<figure style="margin: 1.75rem auto; max-width: 900px;">
  <img src="{{ '/assets/img/blog/ddpm-forward-reverse-chain.png' | relative_url }}" alt="DDPM forward and reverse diffusion chains" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">The fixed forward chain and learned reverse chain in DDPM. Adapted from <a href="https://arxiv.org/abs/2006.11239">Ho, Jain, and Abbeel (2020)</a> with explanatory annotations from <a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">Weng (2021)</a>.</figcaption>
</figure>

## Reverse diffusion and why it is learnable

If we could sample from the true reverse conditionals $q(x_{t-1}\mid x_t)$, then we could start from $x_T \sim \mathcal{N}(0,I)$ and reconstruct data by repeatedly removing noise. The core modeling problem is therefore the reverse transition

$$
p_\theta(x_{t-1}\mid x_t).
$$

For sufficiently small $\beta_t$, the reverse process is approximately Gaussian. More specifically, conditioning on the original clean sample gives a tractable posterior:

$$
q(x_{t-1} \mid x_t, x_0)
=
\mathcal{N}\!\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right),
$$

with

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

and

$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
+
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0.
$$

This expression shows why denoising is enough. If we can estimate either the clean sample $x_0$, the added noise $\epsilon$, or an equivalent score quantity from $x_t$, then we can parameterize the reverse mean and sample backward.

## Training objectives: noise, score, and variational bounds

The original DDPM paper derives a variational lower bound on the data likelihood, but in practice the most widely used objective is the simplified noise-prediction loss

$$
\mathcal{L}_{\text{simple}}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|\epsilon - \epsilon_\theta(x_t,t)\right\|_2^2
\right].
$$

This objective can look deceptively simple. Its effectiveness comes from a structural fact: predicting the Gaussian perturbation is equivalent to learning how to denoise samples across many signal-to-noise regimes.

There are several equivalent parameterizations.

- **Noise prediction** predicts $\epsilon$ directly.
- **Clean-data prediction** predicts $x_0$.
- **Score prediction** predicts a scaled score field.
- **Velocity prediction** uses a linear combination of $x_0$ and $\epsilon$, which is now common in modern latent diffusion and video systems.

These parameterizations are not identical numerically even if they are closely related analytically. They change gradient scaling across timesteps, interact differently with solver design, and can meaningfully affect optimization stability.

<figure style="margin: 1.75rem auto; max-width: 760px;">
  <img src="{{ '/assets/img/blog/ddpm-progressive-decoding.jpg' | relative_url }}" alt="DDPM progressive generation on CIFAR-10" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">Progressive generation in DDPM: large-scale image structure appears before fine details. Reproduced from <a href="https://arxiv.org/abs/2006.11239">Ho, Jain, and Abbeel (2020)</a>.</figcaption>
</figure>

## Model architecture

The diffusion objective specifies *what* the network should predict, but the backbone determines *which spatial and semantic information* it can combine at each noise level. The two dominant choices are **U-Net** and **Transformer** backbones.

**U-Net.** The original [U-Net](https://arxiv.org/abs/1505.04597) has a downsampling stack, an upsampling stack, and skip connections between matching resolutions. In diffusion, the down path aggregates global context while the up path restores local detail; skip connections preserve spatial information that would otherwise be lost. Timestep embeddings, residual blocks, and attention layers are added to tell the network how much noise remains and which condition it should follow.

- **Downsampling.** *Build global context.* Repeated convolution or residual blocks reduce resolution and increase channel capacity, allowing the network to reason about object identity and long-range structure.
- **Upsampling.** *Recover local detail.* Features are expanded back to the target resolution, while skip connections inject high-frequency information from the corresponding encoder stage.

<figure style="margin: 1.75rem auto; max-width: 900px;">
  <img src="{{ '/assets/img/blog/unet-architecture.png' | relative_url }}" alt="Original U-Net architecture" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">The classic U-Net encoder-decoder with skip connections. Reproduced from <a href="https://arxiv.org/abs/1505.04597">Ronneberger, Fischer, and Brox (2015)</a>.</figcaption>
</figure>

**Diffusion Transformers (DiT).** [DiT](https://arxiv.org/abs/2212.09748) replaces convolutional hierarchy with tokenized latent patches and Transformer blocks. This is attractive when scaling model size and attention capacity matters more than a fixed convolutional inductive bias. The key tradeoff is that attention can model broad interactions naturally, while the latent-space encoder and decoder must shoulder much of the local image structure.

<figure style="margin: 1.75rem auto; max-width: 680px;">
  <img src="{{ '/assets/img/blog/dit-architecture.png' | relative_url }}" alt="Diffusion Transformer architecture" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">A Diffusion Transformer block conditioned on timestep and class information. Reproduced from <a href="https://arxiv.org/abs/2212.09748">Peebles and Xie (2023)</a>.</figcaption>
</figure>

## The role of the noise schedule

The schedule $\{\beta_t\}$ or, equivalently, the log signal-to-noise ratio profile, is not a detail. It determines which corruption levels the model sees and how probability mass moves across time.

If early timesteps destroy the signal too quickly, the reverse process becomes unnecessarily hard. If the schedule is too conservative, training may waste capacity on nearly trivial denoising steps. This is why later work increasingly reasons in terms of continuous-time noise schedules, log-SNR parameterizations, or solver-friendly path design.

From a practical standpoint, one of the most important lessons is that diffusion performance depends not just on model architecture, but on the joint design of:

- noise schedule,
- target parameterization,
- timestep sampling strategy,
- and sampling solver.

## Connection to score-based SDEs

The discrete DDPM chain and the score-based modeling line can be unified in the continuous-time SDE framework of [Song et al. (2021)](https://arxiv.org/abs/2011.13456). Instead of indexing the state by discrete steps $t=1,\dots,T$, we consider a continuous process

$$
d x = f(x,t)\,dt + g(t)\,d w_t,
$$

where $w_t$ is standard Brownian motion.

The reverse-time dynamics then depend on the score:

$$
d x =
\left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d \bar{w}_t.
$$

This formulation unifies several models:

- variance-preserving SDEs recover DDPM-like behavior,
- variance-exploding SDEs resemble annealed score models,
- probability-flow ODEs replace stochastic reverse-time evolution with deterministic trajectories sharing the same marginals.

The SDE view is conceptually valuable because it decouples training from a specific discrete sampler and clarifies why better numerical solvers can improve generation without retraining the denoiser.

## DDIM and the separation of training from sampling

One of the most important developments after DDPM was **Denoising Diffusion Implicit Models (DDIM)** by [Song, Meng, and Ermon (2021)](https://arxiv.org/abs/2010.02502). The key insight is that the DDPM objective does not force us to sample with the original ancestral Markov chain. We can define a non-Markovian family of reverse processes that preserve the same training objective but admit deterministic trajectories.

In practice, DDIM uses the denoiser to estimate $x_0$ and then updates the sample using a chosen timestep subsequence. In its deterministic form, the update can be interpreted as moving along a probability-flow-like path instead of repeatedly injecting fresh noise.

This had two major effects on the field.

First, it showed that sampling speed can be improved dramatically without retraining. Second, it made clear that **the diffusion model is a learned time-dependent vector field or denoiser, while the sampler is a separate numerical design choice**. Much of the later progress on fast diffusion sampling follows directly from this separation.

## Guided diffusion for conditional generation

Diffusion became far more useful once the reverse trajectory could be steered toward desired semantics. Guidance methods provide this control.

### Classifier guidance

Suppose we want to sample conditionally on label or text information $y$. By Bayes' rule,

$$
\nabla_{x_t}\log p(x_t \mid y)
=
\nabla_{x_t}\log p(x_t)
+
\nabla_{x_t}\log p(y \mid x_t).
$$

The second term can be estimated by a noisy classifier, as in [Dhariwal and Nichol (2021)](https://arxiv.org/abs/2105.05233). During sampling, one adds the classifier gradient to the reverse update, often with a scale factor $w$ controlling the strength of conditioning.

This increases fidelity to the condition, but it also introduces a tradeoff: strong guidance typically improves alignment and sharpness while reducing diversity and sometimes causing oversaturation or other artifacts.

### Classifier-free guidance

Classifier guidance requires an external classifier. **Classifier-free guidance (CFG)** from [Ho and Salimans (2022)](https://arxiv.org/abs/2207.12598) avoids that by training the denoiser with conditioning dropped at random. At inference time, one evaluates both conditional and unconditional predictions and combines them:

$$
s_{\text{guided}}
=
s_{\text{uncond}}
+
w\left(s_{\text{cond}} - s_{\text{uncond}}\right).
$$

This is one of the most influential practical ideas in diffusion modeling. It is simple, effective, and easy to integrate with text, layout, or other side information. Most modern image and video diffusion systems rely on some form of CFG or a closely related variant.

<figure style="margin: 1.75rem auto; max-width: 760px;">
  <img src="{{ '/assets/img/blog/score-sde-conditional-generation.jpg' | relative_url }}" alt="Conditional image generation with score-based SDEs" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">Conditional score-based generation for inpainting and colorization. Reproduced from <a href="https://arxiv.org/abs/2011.13456">Song et al. (2021)</a>.</figcaption>
</figure>

## Discrete diffusion models

Gaussian noise is natural for images, audio waveforms, and continuous latent spaces, but many important domains are inherently discrete: text tokens, molecular symbols, amino-acid sequences, graph labels, and other categorical states. In such settings, continuous perturbation followed by rounding is often statistically awkward.

This motivates **discrete diffusion**.

Instead of adding Gaussian noise, we define a Markov chain over categorical variables. For a one-hot representation, a forward step can be written as

$$
q(x_t \mid x_{t-1}) = \mathrm{Cat}(Q_t x_{t-1}),
$$

where $Q_t$ is a transition matrix.

Different choices of $Q_t$ produce different notions of corruption:

- uniform random replacement,
- masking into a special absorbing state,
- nearest-neighbor or structure-aware transitions,
- task-specific kernels informed by chemistry, syntax, or graph topology.

The design space is large because discrete corruption has no canonical Gaussian analogue. This is both a challenge and an opportunity. Good transition kernels can encode prior structure that would be difficult to express in a purely continuous formulation.

### D3PM and structured transition kernels

The **D3PM** family from [Austin et al. (2021)](https://arxiv.org/abs/2107.03006) generalized multinomial diffusion by allowing structured discrete transitions rather than only uniform corruption. That flexibility made it easier to adapt diffusion ideas to domains where token relationships matter.

From a modeling standpoint, D3PM is important because it emphasizes that the forward corruption kernel is part of the inductive bias. For language, molecules, or biological sequences, not every token substitution should be treated as equally plausible.

<figure style="margin: 1.75rem auto; max-width: 900px;">
  <img src="{{ '/assets/img/blog/d3pm-discrete-process.png' | relative_url }}" alt="D3PM discrete forward and reverse processes" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">Different discrete transition kernels induce meaningfully different corruption paths and reverse problems. Reproduced from <a href="https://arxiv.org/abs/2107.03006">Austin et al. (2021)</a>.</figcaption>
</figure>

### Ratio-based and entropy-based discrete methods

Recent work such as **SEDD** by [Lou, Meng, and Ermon (2024)](https://arxiv.org/abs/2310.16834) revisits how to parameterize the reverse process in discrete spaces. Instead of forcing the model into an image-style denoising template, these methods derive objectives more directly aligned with discrete probability ratios or entropy-based characterizations.

This line of work suggests an important broader lesson: diffusion should be understood as a framework for learning reverse stochastic transport, not merely as “Gaussian noise plus a U-Net.”

## A unifying view across variants

Despite the diversity of formulations, the core structure of diffusion models is remarkably consistent:

1. Choose a tractable corruption or interpolation process.
2. Learn a local reverse operator, usually a denoiser or score field.
3. Integrate those local predictions into a global sampling trajectory.

The main axes of variation are:

- **state space**: continuous or discrete,
- **time**: discrete chain or continuous SDE/ODE,
- **prediction target**: noise, score, clean sample, velocity, or probability ratio,
- **conditioning**: unconditional, classifier-guided, classifier-free guided, or multimodal,
- **sampler**: ancestral, implicit, deterministic, solver-based, or distilled.

Once viewed this way, many “new” diffusion papers become easier to parse. They usually innovate along one or two of these axes rather than reinventing the whole framework.

## What has mattered most historically

If I compress the historical arc into a few central ideas, these are the ones I would keep.

**Denoising became a scalable route to generative modeling.** DDPM showed that high-quality generation can emerge from repeated local denoising even when every latent has the same dimensionality as the data. This avoids a brittle global inversion problem: the network learns local corrections across a controlled range of signal-to-noise ratios, while the Markov structure supplies the global composition rule.

**Score estimation provided the theoretical backbone.** The score-based perspective explains why a denoiser contains usable geometric information about the data distribution and why reverse stochastic dynamics are principled rather than heuristic. It also tells us what can be changed safely: a new sampler or noise path should remain compatible with the score field that the training objective estimates.

**Sampling is a numerical-analysis problem as much as a modeling problem.** DDIM, higher-order solvers, consistency methods, and distillation all reinforce the same point: once the reverse field is learned, quality and efficiency depend strongly on how we integrate that field. In practice, one should separate failures caused by a weak denoiser from failures caused by too few steps, an unsuitable noise schedule, or an unstable solver.

**Discrete structure is becoming increasingly important.** As diffusion expands from images to scientific design, biology, language, and multimodal reasoning, respecting the native state space becomes essential. A transition kernel that respects masking, chemistry, syntax, or graph structure can be more valuable than simply importing a continuous-image recipe and rounding its output.

## Outlook

Diffusion models are no longer just a method for image synthesis. They now form a broader language for generative modeling that connects stochastic processes, conditional control, and probability transport. For research problems involving structured uncertainty, iterative refinement, or native discrete constraints, that language is especially useful.

The practical frontier is still moving quickly: better samplers, better parameterizations, latent-space diffusion, consistency-style acceleration, diffusion transformers, and task-specific discrete-state models are all active directions. But the central mathematical intuition has remained stable: learn how to reverse a tractable corruption process, and generation becomes the accumulation of many small, well-behaved local steps.

## References

1. Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015. https://arxiv.org/abs/1503.03585
2. Yang Song and Stefano Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 2019. https://arxiv.org/abs/1907.05600
3. Jonathan Ho, Ajay Jain, and Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. https://arxiv.org/abs/2006.11239
4. Jiaming Song, Chenlin Meng, and Stefano Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021. https://arxiv.org/abs/2010.02502
5. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. https://arxiv.org/abs/2011.13456
6. Prafulla Dhariwal and Alexander Nichol. *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021. https://arxiv.org/abs/2105.05233
7. Jonathan Ho and Tim Salimans. *Classifier-Free Diffusion Guidance*. 2022. https://arxiv.org/abs/2207.12598
8. Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS 2021. https://arxiv.org/abs/2107.03006
9. Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, Baining Guo, and Daxin Jiang. *Efficient Diffusion Training via Min-SNR Weighting Strategy*. ICCV 2023. https://arxiv.org/abs/2303.09556
10. Aaron Lou, Chenlin Meng, and Stefano Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*. ICML 2024. https://arxiv.org/abs/2310.16834
11. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. https://arxiv.org/abs/1505.04597
12. William Peebles and Saining Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. https://arxiv.org/abs/2212.09748
