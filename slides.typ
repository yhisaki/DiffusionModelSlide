#import "@preview/typslides:1.3.2": *
#import "@preview/ctheorems:1.1.3": *

#let theorem = thmbox(
  "theorem",
  "Theorem",
  fill: rgb("#e8f4fd"),
  stroke: rgb("#0074d9") + 1pt,
  padding: (top: 1em, bottom: 1em),
)

#let definition = thmbox(
  "definition",
  "Definition",
  fill: rgb("#eefce8"),
  stroke: rgb("#2d9c02") + 1pt,
)

#show: thmrules

#show: typslides.with(
  ratio: "16-9",
  theme: rgb("1f6f78"),
  font: ("Noto Sans CJK JP", "DejaVu Sans"),
  font-size: 18pt,
  link-style: "underline",
  show-progress: true,
)

#front-slide(
  title: "Score-Based Diffusion Model",
  subtitle: [An introduction],
  authors: "Yukinari Hisaki",
)

#table-of-contents()

#slide(title: "Introduction")[

  - The purpose of this talk is to understand the Diffusion Model, which is part of the Diffusion Planner algorithm.
  - Diffusion Model is more difficult than previous talk about Transformer. Diffusion Model is based on the difficult mathematical theory.
  - However, the outcome of this theory is a simple algorithm.
  - A rough understanding of the theory is sufficient; you don't need to worry about the details.

]

#title-slide[
  Motivation and background
]

#slide(title: "Motivation: From Single-step to Iterative Refinement")[
  #set text(size: 17pt)
  The central goal is to estimate an unknown data distribution $q(x)$ from samples $x tilde q(x)$.

  #v(0.8em)

  - *The Standard Approach (e.g., VAEs):*
    Introduce a latent variable $z tilde cal(N)(0, I)$ and map it to data via a *single* probabilistic decoding step $p_theta(x|z)$.
    $ z arrow.long^(p_theta (x|z)) x $

  #v(0.5em)

  - *The Limitation:*
    Learning the complex data distribution $q(x)$ via a *"single shot"* $p_theta (x|z)$ is challenging. The gap between the simple prior and the complex data manifold is too large.

  #v(0.5em)

  - *The Solution (Diffusion Models):*
    Decompose the generation into *multiple gradual transformations*. We refine the sample step-by-step.
    $x_T arrow.long x_(T-1) arrow.long ... arrow.long x_0$
    $ x_T tilde cal(N)( dot | 0, I), x_0 tilde q(x_0) $
]

#slide(title: "Representative Formulations")[
  #set text(size: 15pt)

  The idea of "gradual transformation" can be mathematically formulated in three main ways:

  #v(0.5em)

  1. *Denoising Diffusion Probabilistic Models (DDPM) @ddpm*
    - Formulated as discrete Markov chains.
    - Optimized via Variational Lower Bound (ELBO).
    - _Key idea:_ Learn to reverse the noise addition step-by-step.

  #v(0.2em)

  2. *Score-based Generative Models @score-based-generative-models*
    - Formulated via Stochastic Differential Equations (SDEs).
    - _Key idea:_ Learn the *score function* $nabla_x log p_t (x)$ (direction to higher density).

  #v(0.2em)

  3. *Flow Matching*
    - A more general framework based on Ordinary Differential Equations (ODEs).
    - _Key idea:_ Directly regress a *vector field* $v_t (x)$ to transport probability paths.

  #v(0.2em)
  *Focus of this talk:*
  We will focus on *2. Score Matching*.
]

#title-slide[
  Score-Based Generative Modeling with SDEs
]

#slide(title: "Background: Score-Based Modeling")[

  #set text(size: 16pt)

  Before introducing diffusion, let's define the "Score".

  #definition("Score")[
    For a probability density $q(x)$, the score is defined as the gradient of the log-density with respect to data $x$:
    $
      s_t (x) := nabla_x log p_t (x) = (nabla p_t (x)) / (p_t (x)).
    $
  ]

  - *Intuition: A Vector Field*
    - The score is a vector field pointing in the direction of *higher data density*.
    - At the modes (peaks) of the distribution, the score is zero ($nabla log q(x) = 0$).

  #v(0.2em)

  - *The Goal (Score Matching)*
    We want to train a neural network $s_theta(x)$ to approximate this true score:
    $ cal(L)_("SM")(theta) = bb(E)_(x tilde q) [ || s_theta (x) - nabla_x log q(x) ||^2_2 ] $
  // (Fisher Divergence)
]

#slide(title: "Solution: Denoising Score Matching")[

  #set text(size: 18pt)
  *Problem:* The true score $nabla_x log q(x)$ is intractable.

  #set text(size: 16pt)
  #theorem("Equivalence of Explicit and Denoising Score Matching")[
    Let $q(x)$ be the data distribution and $q(tilde(x)|x)$ be a perturbation kernel.
    Define the smoothed marginal distribution as $q(tilde(x)) = integral q(tilde(x)|x)q(x) dif x$.

    Optimizing the explicit score matching objective for the smoothed marginal,
    $
      cal(L)_("ESM")(theta) = bb(E)_(q(tilde(x))) [ || s_theta(tilde(x)) - nabla_tilde(x) log q(tilde(x)) ||_2^2 ],
    $
    is equivalent to optimizing the Denoising Score Matching objective up to a constant $C$ independent of $theta$:
    $
      cal(L)_("DSM")(theta) = bb(E)_(q(x)q(tilde(x)|x)) [ || s_theta (tilde(x)) - nabla_tilde(x) log q(tilde(x)|x) ||^2_2 ].
    $
    $ => cal(L)_("ESM")(theta) = cal(L)_("DSM")(theta) + C $
  ]

  #pagebreak()

  #set text(size: 18pt)

  *The Core Intuition:*
  Calculating the true score $nabla_tilde(x) log q(tilde(x))$ is computationally intractable. This theorem bypasses the problem by matching the score of $q(tilde(x)|x)$ instead.  Since we explicitly design this perturbation kernel, its score is fully known. We effectively replace an impossible calculation with a tractable, human-defined one.

  *Choice of Conditional Distribution:*
  Typically, we choose a *Gaussian with small variance* for $q(tilde(x)|x)$ (e.g., $cal(N)(tilde(x); x, sigma^2 I)$).
  This makes the conditional score analytically tractable:
  $ nabla_tilde(x) log q(tilde(x)|x) = - (tilde(x) - x) / sigma^2 $
  and allows $q(tilde(x))$ to approximate the data distribution $q(x)$ closely.
]

#slide(title: "Sampling via Langevin Dynamics")[

  Once we have the score function $s_theta(x) approx nabla_x log q(x)$, how do we generate samples?

  #v(0.5em)

  - *Langevin Dynamics:*
    We initialize $x_0$ arbitrarily and iteratively update it using the score:
    $
      x_(t+1) arrow.l x_t + underbrace(epsilon_t s_theta (x_t), "Gradient Ascent") + underbrace(sqrt(2 epsilon_t) z_t, "Random Noise")
    $
    where $z_t tilde cal(N)(0, I)$ and $epsilon$ is the step size.

  #v(0.5em)

  - *Mechanism:*
    1. *Gradient Term:* Pushes $x$ towards high-density regions (modes).
    2. *Noise Term:* Prevents collapsing to a single mode, allowing exploration of the full distribution.

  #v(0.5em)

  $arrow.double$ As $t arrow infinity$ and $epsilon_t arrow 0$, $x_t$ converges to a sample from $q(x)$.
]

#slide(title: "Challenges: Why Simple Score Matching Fails")[

  #set text(size: 15pt)

  Theory is sound, but training on raw data $x tilde q(x)$ often fails in practice.

  #v(0.5em)

  1. *The Manifold Hypothesis (Blindness)*
    - High-dimensional data concentrates on a low-dimensional manifold.
    - Outside this manifold, $q(x) approx 0$.
    - The score $nabla_x log q(x)$ is *undefined* or numerically unstable in the ambient space.

  #v(0.5em)

  2. *Inaccurate Estimation in Low-Density Regions*
    - The loss $bb(E)_(x tilde q)$ only minimizes error *where data exists*.
    - In low-density regions (most of the space), the model $s_theta(x)$ is untrained.
    - *Result:* When sampling starts from random noise (low density), the gradients are inaccurate, leading to garbage generation.

  #v(0.5em)

  3. *Slow Mixing of Langevin Dynamics*
    - If the data distribution has disjoint modes (isolated "islands" of data), the score in between is near zero.
    - The sampler gets stuck in one mode and cannot cross the gap to others.
]

#slide(title: "Enter Stochastic Differential Equations (SDEs)")[

  #set text(size: 16pt)

  How do we populate low-density regions and fix the undefined score?

  *Solution:* Continuously perturb the data into pure noise over time $t in [0, T]$.
  #theorem("Forward SDE and its marginal evolution")[
    Assume the *forward* diffusion is defined by the Itô SDE:
    $
      dif x = f(x, t) dif t + g(t) dif w, quad t in [0, T],
    $
    where $w$ is a standard Wiener process, $f(x, t)$ is the *drift* coefficient, and $g(t)$ is the scalar *diffusion* coefficient.

    The marginal density $p_t (x)$ exists and evolves according to the *Fokker--Planck PDE*:
    $
      partial_t p_t (x)
      = - nabla dot.c (f(x,t) p_t (x))
      + 1/2 g(t)^2 Delta p_t (x).
    $
  ]

  #v(0.5em)

  - *What this Theorem Means:*
    Simply put, if we take a data point sampled from $x_0 tilde p_0(x)$ and move it according to the SDE, this PDE describes exactly what its probability distribution $x_t tilde p_t (x)$ looks like at any time $t$.

  - *Designing the SDE:*
    We design the drift $f(x,t)$ and diffusion $g(t)$ specifically so that as $t arrow T$, the distribution $p_t (x)$ smoothly converges to a simple, tractable noise distribution. For example, setting $f(x,t) = -1/2 beta(t) x$ and $g(t) = sqrt(beta(t))$ ensures $p_T (x) approx cal(N)(0, I)$.

  - *The Continuous Perturbation Process:*
    - *At $t=0$: * $x_0 tilde p_0(x)$ (The complex, low-dimensional real data distribution).
    - *At $t in (0, T)$: * Noise gradually fills the empty space. The smoothed score $nabla_x log p_t (x)$ becomes well-defined everywhere.
    - *At $t=T$: * $x_T tilde p_T (x) approx cal(N)(0, I)$ (An unstructured, tractable prior).

  #v(0.5em)

  $arrow.double$ *Key Idea:* By designing $f(x,t)$ and $g(t)$, we construct a process that systematically destroys data into noise. If we can *reverse* this SDE, we can generate data from noise!
]

#slide(title: "Generating Data: The Probability Flow ODE")[

  #set text(size: 16pt)

  The forward SDE transforms data $p_0 (x)$ into pure noise $p_T (x) approx cal(N)(0, I)$. To generate data, we reverse this process from $t=T$ down to $t=0$.
  Remarkably, we can perfectly retrace the exact marginal densities $p_t (x)$ using a deterministic equation.

  #theorem("Probability Flow ODE (Song et al., 2020)")[
    For the forward SDE $dif x = f(x, t) dif t + g(t) dif w$, there exists a corresponding deterministic *Ordinary Differential Equation (ODE)*. This ODE shares the *exact same marginal probability densities* $p_t (x)$ at all times $t in [0, T]$:
    $
      dif x = [ f(x, t) - 1/2 g(t)^2 nabla_x log p_t (x) ] dif t.
    $
  ]

  #v(0.5em)

  - *The Missing Piece:* To simulate this ODE backwards from $t=T$ to $t=0$, $f(x,t)$ and $g(t)$ are already known (we designed them). The *only* unknown variable is the score function $nabla_x log p_t(x)$.
    $arrow.double$ *Our trained score model $s_theta (x, t)$ directly plugs in here!*
]

#slide(title: "Interim Summary: The Generative Blueprint")[

  #set text(size: 16pt)

  Let's pause and summarize the theoretical framework we have established.

  #v(0.5em)

  - *The Key Takeaway:*
    Given a forward SDE $dif x = f(x, t) dif t + g(t) dif w$, there exists a corresponding Probability Flow ODE:
    $ dif x = [ f(x, t) - 1/2 g(t)^2 nabla_x log p_t (x) ] dif t $
    The mathematical blueprint to generate data is entirely determined by the forward process and its score.

  #pagebreak()

  - *What remains to be done?*
    To turn this continuous-time theory into a practical generative model, we must complete two concrete tasks:

    #v(0.5em)

    1. *Design a Specific Forward SDE:*
      We need to explicitly define the drift $f(x, t)$ and diffusion $g(t)$ coefficients so that the complex data $p_0(x)$ reliably converges to a tractable noise prior $p_T (x) approx cal(N)(0, I)$.

    #v(0.5em)

    2. *Learn the Score Function:*
      Since the Reverse ODE requires $nabla_x log p_t (x)$, we need a way to train a time-dependent neural network $s_theta(x, t)$ to approximate this score for *all* continuous times $t in [0, T]$.

  #v(0.8em)

  $arrow.double$ Let's first address *Task 1* by looking at a standard choice: the *Variance Preserving (VP) SDE*.
]

#slide(title: "The Variance Preserving (VP) SDE")[

  #set text(size: 16pt)

  A standard choice for the forward process is the *Variance Preserving (VP) SDE*, which is the continuous-time limit of DDPM.

  #v(0.5em)

  - *The SDE Formulation:*
    We define the drift and diffusion using a time-dependent noise schedule $beta(t) > 0$:
    $
      dif x = underbrace(-1/2 beta(t) x, "Drift: Shrinks signal") dif t + underbrace(sqrt(beta(t)), "Diffusion: Injects noise") dif w
    $

  #v(0.5em)

  - *Why "Variance Preserving"?*
    If the initial data $x_0$ has unit variance, the variance of $x_t$ remains exactly $1$ for all $t$. The drift smoothly scales down the data toward the origin, while the diffusion perfectly compensates by adding Gaussian noise.

  #pagebreak()

  #set text(size: 14pt)


  - *The Tractable Transition Kernel:*
    Because this SDE is linear, the transition probability from $x_0$ to $x_t$ is always a tractable Gaussian. Let $alpha_t = exp(- integral_0^t beta(s) dif s)$:
    $
      p_(t|0)(x_t | x_0) = cal(N)(x_t; sqrt(alpha_t) x_0, (1 - alpha_t) I)
    $


  - *Practical Implementation (Linear Schedule):*
    In practice, we set $t in [0, 1]$ and define $beta(t)$ as a linear function:
    $ beta(t) = beta_"min" + t (beta_"max" - beta_"min") $
    Standard hyperparameters are $beta_"min" = 0.1$ and $beta_"max" = 20$.

    This gives a simple, closed-form solution for code implementation:
    $ integral_0^t beta(s) dif s = beta_"min" t + 1/2 (beta_"max" - beta_"min") t^2 $
    $ => alpha_t = exp(- beta_"min" t - 1/2 (beta_"max" - beta_"min") t^2) $

  #v(0.5em)

  $arrow.double$ At $t=1$, $alpha_1 approx exp(-10) approx 4.5 times 10^(-5) approx 0$.
  The signal is completely destroyed, ensuring $p_1(x) approx cal(N)(0, I)$.

  #pagebreak()

  - *Visualizing the Evolution ($x_0 = 1$):*
    How do these equations transform our data? Let's track a single 1D data point starting exactly at $x_0 = 1$ over time $t in [0, 1]$.

  #v(0.5em)

  #align(center)[
    #image("assets/vp_sde_plot.png", width: 50%)
  ]

  #v(0.5em)

  - *Drift (The Mean):*
    The mean $mu(t) = sqrt(alpha_t) x_0$ smoothly decays from $1$ down to $0$. The original signal is systematically erased.
  - *Diffusion (The Variance):*
    The $plus.minus 2 sigma(t)$ interval, where $sigma(t) = sqrt(1 - alpha_t)$, steadily expands until the distribution perfectly matches standard Gaussian noise $cal(N)(0, 1)$ at $t=1$.

]

#slide(title: "Continuous-Time Score Matching")[

  #set text(size: 16pt)

  We now have a specific SDE and its transition kernel. The final task is to train a neural network $s_theta(x, t)$ to approximate the score $nabla_x log p_t(x)$ across all continuous times $t in [0, 1]$.

  #v(0.5em)

  - *Extending Denoising Score Matching:*
    Just like in the discrete case, matching the intractable marginal score is mathematically equivalent to matching the *known* conditional score $nabla_(x_t) log p_(t|0)(x_t | x_0)$.

  #v(0.5em)

  - *The Conditional Score of the VP SDE:*
    Since the transition kernel is $p_(t|0)(x_t | x_0) = cal(N)(x_t; sqrt(alpha_t) x_0, (1 - alpha_t) I)$, the score is simply the gradient of the Gaussian log-density:
    $
      nabla_(x_t) log p_(t|0)(x_t | x_0) = - (x_t - sqrt(alpha_t) x_0) / (1 - alpha_t)
    $
    *Notice:* This is a deterministic, closed-form target!

  #pagebreak()

  #set text(size: 15pt)

  - *The Continuous-Time Objective:*
    We train our score network $s_theta(x_t, t)$ by minimizing the expected distance to this conditional score. We sample time $t$ uniformly, original data $x_0$, and perturbed data $x_t$.
    
    $
      cal(L)(theta) = bb(E)_(t tilde cal(U)(0, 1)) bb(E)_(x_0 tilde p_0) bb(E)_(x_t tilde p_(t|0)) [ lambda(t) || s_theta(x_t, t) - underbrace((- (x_t - sqrt(alpha_t) x_0) / (1 - alpha_t)), nabla_(x_t) log p_(t|0)(x_t | x_0)) ||^2_2 ]
    $

  #v(0.5em)

  - *The Weighting Function $lambda(t)$:*
    The score magnitude grows extremely large as $t arrow 0$ (because the variance $(1 - alpha_t) arrow 0$). To prevent gradient explosion and balance the loss across all times, we typically choose a positive weighting function proportional to the variance, such as $lambda(t) = 1 - alpha_t$.

  #v(0.5em)

  $arrow.double$ By optimizing this single objective, $s_theta (x, t)$ learns the continuous vector field. We can then plug it directly into the Reverse ODE to generate data!
]
// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)

