---
layout: post
title: "Flow-based Models: From Coupling Layers to Flow Matching"
date: 2025-12-07 10:00:00+0800
description: A technical review of flow-based generative models, from NICE and RealNVP to continuous flows, flow matching, and rectified flow.
tags: [flow-based-models, normalizing-flows, generative-ai]
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

Flow-based generative models occupy a special place in modern machine learning. They offer an unusually clean mathematical story: if we can learn an invertible transformation from a simple base distribution to the data distribution, then we obtain exact likelihood, exact latent-variable inference, and direct ancestral sampling in one framework. This is a very different philosophy from adversarial generation or diffusion-based denoising. Instead of learning to classify realism or reverse a corruption process, flows learn transport maps whose density transformation can be computed explicitly.

The classical form of this idea is the **normalizing flow**, but the area has expanded far beyond stacks of coupling layers. Continuous normalizing flows, invertible residual networks, stochastic interpolants, flow matching, and rectified flow have all pushed the field from “tractable Jacobians” toward a broader language of probability transport. This post follows that arc, from early discrete invertible architectures to more recent path-based formulations.

## Change of variables as the foundation

Let $z \sim p_Z(z)$ be a simple latent variable, often standard Gaussian, and let $x = f(z)$ for an invertible differentiable map $f$. Then the density of $x$ is given by the change-of-variables formula

$$
p_X(x) = p_Z(f^{-1}(x)) \left|\det J_{f^{-1}}(x)\right|,
$$

or equivalently

$$
\log p_X(x)
=
\log p_Z(f^{-1}(x))
+
\log \left|\det J_{f^{-1}}(x)\right|.
$$

This equation is the core of flow-based modeling. It immediately provides three capabilities:

- **Sampling.** *Prior to data.* Draw $z \sim p_Z$ and push it through $f$; this direction is attractive when the forward map is cheap.
- **Likelihood evaluation.** *Data to prior.* Invert $x$ and add the Jacobian correction. This is the defining advantage over generators that only support sampling.
- **Latent inference.** *An exact representation.* Apply $f^{-1}$ to map an observation into a latent coordinate system rather than relying on a separately trained encoder.

The architectural problem is therefore very precise: we need invertible maps that are expressive enough for complex data distributions while keeping the log-determinant of the Jacobian computationally tractable.

## Why invertibility is both a strength and a constraint

Invertibility gives flows mathematical transparency, but it also imposes a strong inductive bias. A generic deep network is not invertible, and even when a transformation is invertible in principle, computing the determinant of its Jacobian may be prohibitively expensive in high dimensions.

This created the central tension in early flow research:

- If we enforce too much structure, optimization and likelihood are easy but expressivity is weak.
- If we allow very flexible transformations, likelihood computation becomes intractable.

Much of the history of flows can be read as a sequence of compromises around this tension.

## NICE: coupling layers and triangular Jacobians

The breakthrough that made normalizing flows practical was the **coupling layer** design in [**NICE**](https://arxiv.org/abs/1410.8516). Split the input into two parts, $x = (x_a, x_b)$, and transform only one part conditioned on the other:

$$
y_a = x_a,
\qquad
y_b = x_b + m(x_a).
$$

This simple form has major advantages.

First, the inverse is trivial:

$$
x_a = y_a,
\qquad
x_b = y_b - m(y_a).
$$

Second, the Jacobian is triangular, so its determinant is easy to compute. For additive coupling, the determinant is exactly one, making the log-determinant term disappear.

NICE established an enduring design principle: if each layer only transforms part of the state while leaving the rest unchanged, then exact invertibility becomes manageable. However, additive coupling also limits local rescaling and can therefore be less expressive than one would like for complex image densities.

## RealNVP: affine coupling and multi-scale design

[**RealNVP**](https://arxiv.org/abs/1605.08803) extended NICE by introducing **affine coupling**:

$$
y_a = x_a,
\qquad
y_b = x_b \odot \exp(s(x_a)) + t(x_a).
$$

Now the Jacobian is still triangular, but no longer volume-preserving. Its log-determinant becomes

$$
\log |\det J|
=
\sum_i s_i(x_a).
$$

This was an important step because the model can now scale as well as shift features. RealNVP also introduced a multi-scale architecture that progressively factors out subsets of variables, improving computational efficiency and stabilizing training on images.

RealNVP became one of the first flow models that felt like a serious large-scale generative system rather than a conceptual proof of principle. It demonstrated that exact likelihood and usable image synthesis could coexist.

<figure style="margin: 1.75rem auto; max-width: 760px;">
  <img src="{{ '/assets/img/blog/realnvp-attribute-transfer.png' | relative_url }}" alt="RealNVP attribute transfer examples" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">Latent-space attribute transfer with RealNVP, illustrating the semantic structure available through an invertible representation. Reproduced from <a href="https://arxiv.org/abs/1605.08803">Dinh, Sohl-Dickstein, and Bengio (2017)</a>.</figcaption>
</figure>

## Autoregressive flows and directional tradeoffs

Another important branch of early development is the autoregressive family. Here, each output dimension depends on previous dimensions in an autoregressive order, which again yields a triangular Jacobian.

Two canonical examples are:

- [**MAF**](https://arxiv.org/abs/1705.07057): efficient for density estimation, slower for sampling.
- [**IAF**](https://arxiv.org/abs/1606.04934): efficient for sampling, slower for exact density evaluation.

This asymmetry is conceptually useful. It shows that “normalizing flow” is not one architecture, but a design space in which we may prioritize different computational directions.

If our main goal is maximum-likelihood density estimation, MAF is attractive. If our goal is fast ancestral generation or variational inference, IAF may be preferable. The broader lesson is that tractability is directional: invertibility alone does not guarantee that all operations are equally cheap.

## Glow and the refinement of image flows

[**Glow**](https://arxiv.org/abs/1807.03039) refined the coupling-layer recipe with three influential ingredients:

- **ActNorm**, a learnable affine normalization initialized from data,
- **invertible $1\times1$ convolutions**, replacing fixed channel permutations,
- and a carefully engineered hierarchical architecture.

The invertible convolution is especially elegant. A fixed permutation mixes channels only in a very restricted way, whereas a learnable linear transform can rotate feature channels before the next coupling block, greatly improving expressivity while preserving tractable determinants.

Glow mattered historically because it showed that carefully engineered flow architectures could scale to visually impressive image generation, not merely likelihood benchmarks.

<figure style="margin: 1.75rem auto; max-width: 900px;">
  <img src="{{ '/assets/img/blog/glow-latent-interpolation.png' | relative_url }}" alt="Glow latent interpolation" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">Smooth latent interpolation in Glow. Reproduced from <a href="https://arxiv.org/abs/1807.03039">Kingma and Dhariwal (2018)</a>.</figcaption>
</figure>

## Likelihood quality versus sample quality

One recurring theme in generative modeling is that strong likelihood does not automatically imply perceptually superior samples. Flows made this tension very visible. On one hand, they optimize exact log-likelihood rather than a variational bound or adversarial proxy. On the other hand, the invertibility constraint can make it hard to concentrate modeling capacity on perceptually salient directions in the way that GANs or later diffusion models often do.

This observation is not an argument against flows. Rather, it clarifies what they optimize well:

- precise density modeling,
- invertible latent semantics,
- principled uncertainty under exact change of variables.

When those properties matter, flows remain unusually attractive.

## Continuous normalizing flows

The next conceptual shift was to replace a finite stack of invertible layers with a continuous-time trajectory. Instead of writing

$$
x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z),
$$

we define a time-dependent state $z_t$ satisfying the ODE

$$
\frac{d z_t}{dt} = v_\theta(z_t,t).
$$

This leads to **Continuous Normalizing Flows (CNFs)**, built on the continuous-depth view of [Neural ODEs](https://arxiv.org/abs/1806.07366). The density evolves according to the instantaneous change-of-variables formula

$$
\frac{d}{dt}\log p(z_t)
=
-\operatorname{tr}\left(\frac{\partial v_\theta}{\partial z_t}\right).
$$

Integrating this from initial to final time yields the exact log-density correction. This perspective is powerful because it turns depth into time and lets us use adaptive ODE solvers. It also places generative modeling close to dynamical systems and optimal transport.

## Neural ODEs and FFJORD

[Neural ODEs](https://arxiv.org/abs/1806.07366) provided the general continuous-depth framework, while [**FFJORD**](https://arxiv.org/abs/1810.01367) showed how to make CNFs practical for high-dimensional density estimation using stochastic trace estimation.

The essential computational challenge in CNFs is the trace term

$$
\operatorname{tr}\left(\frac{\partial v_\theta}{\partial z_t}\right),
$$

which is expensive to compute directly. Hutchinson-style estimators reduce this cost by replacing exact traces with unbiased stochastic estimates. This makes large-scale training feasible, though not always cheap.

CNFs offer elegant geometry and adaptive trajectories, but they trade architectural constraints for numerical ones. The model is now simple to describe functionally, yet training and sampling depend on ODE integration accuracy, stiffness, and solver cost.

## Residual flows and Lipschitz-constrained invertibility

Another direction asked whether standard residual networks could be made invertible with mild constraints. Consider a residual block

$$
f(x) = x + g(x).
$$

If $g$ is sufficiently contractive, then $f$ is invertible. This gave rise to [**invertible residual networks**](https://arxiv.org/abs/1811.00995) and residual flows, where spectral or Lipschitz constraints ensure bijectivity.

This perspective is valuable because it narrows the conceptual gap between conventional deep learning and invertible generative modeling. Instead of building special-purpose coupling architectures, one modifies familiar residual blocks to preserve invertibility. The resulting models connect flow design to stability theory, fixed-point inversion, and operator norms.

## The transport viewpoint becomes central

Up to this point, much of flow research was driven by exact likelihood and tractable Jacobians. More recent work shifts the emphasis. The main object is no longer always an explicitly invertible block architecture; it is often a **probability path** from a simple source distribution to the target data distribution.

This shift matters because once we think in terms of paths rather than discrete layer stacks, we can connect flows to:

- optimal transport,
- Schrödinger bridge methods,
- diffusion probability paths,
- and deterministic samplers for score models.

In other words, the modern frontier is increasingly about **how to choose and learn transport trajectories**, not just how to multiply Jacobian determinants efficiently.

## Flow matching

[**Flow Matching**](https://arxiv.org/abs/2210.02747) is one of the clearest expressions of this modern view. Rather than learning densities through repeated Jacobian corrections, it directly learns a time-dependent vector field whose trajectories realize a desired probability path.

Suppose we define a conditional interpolation between a source sample $x_0$ and a target sample $x_1$, producing intermediate states $x_t$. Then we can derive a target velocity field $u_t(x_t \mid x_0, x_1)$ and train a model by regression:

$$
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{t,x_0,x_1,x_t}
\left[
\left\|v_\theta(x_t,t) - u_t(x_t \mid x_0,x_1)\right\|_2^2
\right].
$$

This looks deceptively straightforward, but it is conceptually important. Generation becomes the task of learning a transport ODE that matches a prescribed probability path, often without the need for stochastic reverse diffusion or likelihood-oriented layer design.

Flow matching is attractive because:

- it supports deterministic generation,
- it is compatible with a wide family of probability paths,
- and it often leads to simpler and faster samplers than diffusion-style ancestral chains.

<figure style="margin: 1.75rem auto; max-width: 760px;">
  <img src="{{ '/assets/img/blog/flow-matching-ot-path.png' | relative_url }}" alt="Flow Matching optimal-transport probability path" style="display: block; width: 100%; height: auto;" loading="lazy">
  <figcaption style="margin-top: 0.65rem; text-align: center; font-size: 0.9rem;">An optimal-transport probability path from a Gaussian source to a checkerboard target. Reproduced from <a href="https://arxiv.org/abs/2210.02747">Lipman et al. (2023)</a>.</figcaption>
</figure>

## Relation to diffusion and probability paths

One reason flow matching has become influential is that it can reuse probability paths inspired by diffusion. If a path gradually transforms a simple prior into data, we may train a vector field to follow that path directly instead of learning a stochastic reverse denoiser.

This creates a useful bridge between flows and diffusion:

- diffusion emphasizes reverse stochastic denoising,
- flow matching emphasizes deterministic velocity learning along probability paths.

These are not disjoint camps. In many modern papers, they are two parameterizations of a deeper transport problem.

## Rectified flow and straight trajectories

[**Rectified Flow**](https://arxiv.org/abs/2209.03003) pushes the transport viewpoint further by emphasizing **straightness** of trajectories. The intuition is simple: if sample paths are unnecessarily curved, then the ODE solver needs more steps and numerical error accumulates more easily. By learning straighter transport fields, one can often generate high-quality samples with fewer evaluations.

This is appealing for both conceptual and practical reasons.

Conceptually, it aligns generative modeling with efficient transport geometry. Practically, it speaks directly to the question that matters most in deployment: how many model evaluations are needed to produce a strong sample?

Rectified flow also influenced the broader idea that probability-path design is a first-class object. We do not merely fit a vector field to whatever path is convenient; we choose paths that are easier to learn and easier to integrate.

## Stochastic interpolants and unification

Related work on [**stochastic interpolants**](https://arxiv.org/abs/2303.08797) offers an even broader framework for connecting diffusions, ODE flows, and other transport-based generators. The central message is that many apparently different generative models can be interpreted as learning dynamics over interpolating distributions between source and target.

This is one of the most interesting conceptual developments in recent years. It suggests that the real unifying object is not “flow” or “diffusion” as a label, but rather:

- a family of intermediate distributions,
- a rule for moving probability mass across that family,
- and an objective that matches the corresponding local transport statistics.

## What changed from early flows to recent flows

The historical development can be summarized as a sequence of broadening viewpoints.

**From invertible layers to dynamical systems.** Early flows focused on finite-dimensional blocks whose Jacobians were deliberately tractable. CNFs retain the change-of-variables principle but move the design problem to a time-dependent vector field, trading discrete architectural constraints for ODE integration and trace estimation.

**From exact Jacobians to learned velocities.** Classical flows place the log-determinant at center stage because likelihood is the primary objective. Flow matching instead regresses a velocity field along a chosen probability path; this often removes the need for special coupling structure, but it makes the chosen path and numerical solver central modeling choices.

**From density estimation to transport design.** The modern question is not only how to represent a density, but how to move probability mass from prior to data along paths that are stable, efficient, and expressive. This explains why optimal transport, diffusion paths, rectification, and stochastic interpolants now appear in the same technical conversation.

## Practical tradeoffs

For implementation and paper reading, I find the following tradeoffs most useful.

**Exact likelihood versus computational simplicity.** Coupling-based flows provide exact likelihood in a transparent way, but they demand architectural constraints such as triangular Jacobians. Path-based methods can be simpler to train for generation, although they may emphasize transport quality rather than likelihood evaluation; the right choice depends on whether density estimation or sampling is the primary deliverable.

**Invertibility versus flexibility.** Hard invertibility guarantees are mathematically elegant but restrict layer design and memory tradeoffs. Continuous-time and path-based approaches relax this pressure by shifting complexity into vector fields and numerical solvers, which in turn introduces sensitivity to stiffness, tolerances, and solver choice.

**Solver cost versus path quality.** Once generation is expressed as ODE transport, numerical integration becomes central. A well-designed, nearly straight path can reduce the number of function evaluations dramatically, whereas a poor path can erase the apparent advantage of an otherwise expressive network.

## Why flow-based ideas still matter

Even though diffusion models currently dominate many generative benchmarks, flow-based ideas remain deeply relevant.

First, they provide one of the clearest mathematical languages for **invertible transport** and exact density transformation. Second, they connect generative modeling to ODEs, optimal transport, and continuous dynamics in a way that is often more explicit than denoising-based formulations. Third, recent flow-matching-style methods are actively influencing the design of fast generative samplers and hybrid diffusion-flow systems.

For scientific machine learning, this matters a great deal. Many problems are really transport problems in disguise: moving between distributions over structures, trajectories, fields, or designs while preserving geometric constraints. The flow perspective is often the most natural lens for those tasks.

## Outlook

The most interesting recent trend is that the boundary between flows and other generative paradigms has become softer. What used to be a narrowly defined architecture family is now part of a broader theory of probability paths and transport fields. This is healthy for the field. It means we can keep the conceptual precision of flow-based modeling while borrowing the strongest ideas from diffusion, optimal transport, and modern numerical methods.

If one wants a compact takeaway, it is this: the original flow literature taught us how to make density transformation exact, while the newer literature teaches us how to make probability transport efficient. Both ideas are still central.

## References

1. Laurent Dinh, David Krueger, and Yoshua Bengio. *NICE: Non-linear Independent Components Estimation*. 2014. https://arxiv.org/abs/1410.8516
2. Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. *Density Estimation using Real NVP*. ICLR 2017. https://arxiv.org/abs/1605.08803
3. Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. *Improving Variational Inference with Inverse Autoregressive Flow*. NeurIPS 2016. https://arxiv.org/abs/1606.04934
4. George Papamakarios, Theo Pavlakou, and Iain Murray. *Masked Autoregressive Flow for Density Estimation*. NeurIPS 2017. https://arxiv.org/abs/1705.07057
5. Diederik P. Kingma and Prafulla Dhariwal. *Glow: Generative Flow with Invertible 1x1 Convolutions*. NeurIPS 2018. https://arxiv.org/abs/1807.03039
6. Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. *Neural Ordinary Differential Equations*. NeurIPS 2018. https://arxiv.org/abs/1806.07366
7. Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*. ICLR 2019. https://arxiv.org/abs/1810.01367
8. Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, and Jörn-Henrik Jacobsen. *Invertible Residual Networks*. ICML 2019. https://arxiv.org/abs/1811.00995
9. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. *Flow Matching for Generative Modeling*. ICLR 2023. https://arxiv.org/abs/2210.02747
10. Xingchao Liu, Chengyue Gong, and Qiang Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. https://arxiv.org/abs/2209.03003
11. Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*. 2023. https://arxiv.org/abs/2303.08797
