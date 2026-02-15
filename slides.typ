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

#show cite: set text(fill: black)

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

#title-slide[
  Why Diffusion?
]

#slide(title: "Generative Models for Planning")[
  - *The Goal:* How should we model complex human driving behavior using machine learning?

  #v(0.5em)

  - *The Naive Approach (Regression):*
    The simplest method is to learn a deterministic function $tau = f_theta(c)$ that maps the context $c$ to a single future trajectory $tau$.
    - Input $c$: Ego state, other vehicles, map data, etc.
    - Output $tau$: Sequence of states ${x_t, x_(t+1), ..., x_T}$.

  #v(0.5em)

  - *The Limitation:*
    *However*, human behavior is inherently *multimodal* and uncertain.

    - In ambiguous scenarios, there are multiple valid maneuvers.

  #v(0.5em)

  $arrow.double$ We need a *Generative Model* that learns the full distribution $p(tau | c)$, not just a single path.
]
#slide(title: "Motivation: Iterative Refinement")[
  #set text(size: 17pt)
  The central goal is to estimate an unknown data distribution $q(x)$ from samples $x tilde q(x)$.

  #v(0.8em)

  - *The Standard Approach (e.g., VAEs):*
    Introduce a latent variable $z tilde cal(N)(0, I)$ and map it to data via a *single* probabilistic decoding step $p_theta (x|z)$.
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

#slide(title: "Three Formulations and Today's Focus")[
  #set text(size: 16pt)

  The idea of "Diffusion Model" can be mathematically formulated in three main ways:

  #v(0.5em)

  1. *Denoising Diffusion Probabilistic Models (DDPM) @ddpm @nonequilibrium-diffusion*
    - Formulated as discrete Markov chains.
    - Optimized via Variational Lower Bound (ELBO).
    - _Key idea:_ Learn to reverse the noise addition step-by-step.

  #v(0.2em)

  2. *Score-based Generative Models @score-based-generative-models*
    - Formulated via Stochastic Differential Equations (SDEs).
    - _Key idea:_ Learn the *score function* $nabla_x log p_t (x)$ (direction to higher density).

  #v(0.2em)

  3. *Flow Matching @flow-matching*
    - A more general framework based on Ordinary Differential Equations (ODEs).
    - _Key idea:_ Directly regress a *vector field* $v_t (x)$ to transport probability paths.

  #v(0.2em)
  *Focus of this talk:*
  We will focus on *2. Score Matching*.
]

#title-slide[
  Score Matching Basics
]

#slide(title: "Motivation: Learning Unnormalized Distributions")[

  #set text(size: 14pt)

  *The Goal:*
  Given a dataset of samples ${x_1, ..., x_N}$ from an unknown distribution $q (x)$, we want to learn a model $p_theta (x)$ that approximates it.

  #v(0.5em)

  - *The Modeling Challenge:*
    Flexible probability models are often defined up to a normalization constant (Energy-Based Models):
    $ p_theta (x) = (exp(-E_theta (x))) / Z_theta, quad Z_theta = integral exp(-E_theta (x)) dif x $

  #v(0.5em)

  - *The Intractable Partition Function:*
    To train via Maximum Likelihood Estimation (MLE), we need to maximize $log p_theta (x)$. However, calculating $Z_theta$ (the integral over high-dimensional space) is computationally impossible.
    $ log p_theta (x) = -E_theta (x) - underbrace(log Z_theta, "Intractable!") $

  #v(0.5em)

  - *The Solution (Differentiation):*
    If we take the gradient with respect to input $x$, the constant $Z_theta$ vanishes!
    $ nabla_x log p_theta (x) = - nabla_x E_theta (x) - underbrace(nabla_x log Z_theta, "0") = - nabla_x E_theta (x) $
]

#slide(title: [The Score Function: Definition and Intuition])[

  #set text(size: 16pt)

  By modeling the gradient of the log-density instead of the density itself, we bypass the normalization problem entirely.

  #definition("The Score Function")[
    For a probability density $p(x)$, the score is defined as:
    $ s(x) := nabla_x log p(x). $
  ]

  - *Intuition: A Vector Field*
    - The score is a vector field pointing in the direction of *steepest ascent* (higher data density).
    - Crucially, it creates a valid training target without ever needing to compute $Z_theta$.

  #v(0.5em)

  - *The Objective (Score Matching):*
    We train a neural network $s_theta(x)$ to match the data score:
    $ cal(L)_("SM")(theta) = bb(E)_(x tilde q ) [ || s_theta (x) - nabla_x log q (x) ||^2_2 ] $
]

#slide(title: [Denoising Score Matching: Tractable Target @score-matching])[

  #set text(size: 18pt)
  *Problem:* The true score $nabla_x log q(x)$ is intractable.

  #set text(size: 16pt)
  #theorem("Equivalence of Explicit and Denoising Score Matching")[
    Let $q(x)$ be the data distribution and $q(tilde(x)|x)$ be a perturbation kernel.
    Define the smoothed marginal distribution as $q(tilde(x)) = integral q(tilde(x)|x)q(x) dif x$.

    Optimizing the explicit score matching objective for the smoothed marginal,
    $
      cal(L)_("ESM")(theta) = bb(E)_(q(tilde(x))) [ || s_theta (tilde(x)) - nabla_tilde(x) log q(tilde(x)) ||_2^2 ],
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

#slide(title: [Sampling: Langevin Dynamics from the Score @ncsn])[

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

#slide(title: "Why Naive Score Matching Fails")[

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

#title-slide[
  Diffusion as an SDE
]

#slide(title: "Forward Diffusion: SDE and Marginal Evolution")[

  #set text(size: 16pt)

  How do we populate low-density regions and fix the undefined score?

  *Solution:* Continuously perturb the data into pure noise over time $t in [0, T]$.

  #block(fill: luma(240), inset: 1em, radius: 5pt)[
    *Intuition: From "Cliffs" to "Gentle Hills"*
    Raw data is like isolated "islands" in a vast space; outside them, the gradient (direction) is unknown.
    The SDE adds noise, acting like ink spreading in water. This *smooths* the distribution, spreading probability mass into empty regions.
    Result: The score becomes defined everywhere, creating a "slope" that guides us back to the data.
  ]

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
    It bridges the *microscopic* stochastic path (SDE) and the *macroscopic* density evolution (PDE). Crucially, the diffusion term $1/2 g(t)^2 Delta p_t$ acts as a smoothing operator (like heat flow). This spreads probability mass into empty regions, ensuring $p_t(x) > 0$ everywhere so the score $nabla log p_t(x)$ becomes well-defined.

  - *Designing the SDE:*
    We design the drift $f(x,t)$ and diffusion $g(t)$ specifically so that as $t arrow T$, the distribution $p_t (x)$ smoothly converges to a simple, tractable noise distribution. For example, setting $f(x,t) = -1/2 beta(t) x$ and $g(t) = sqrt(beta(t))$ ensures $p_T (x) approx cal(N)(0, I)$.

  - *The Continuous Perturbation Process:*
    - *At $t=0$: * $x_0 tilde p_0(x)$ (The complex, low-dimensional real data distribution).
    - *At $t in (0, T)$: * Noise gradually fills the empty space. The smoothed score $nabla_x log p_t (x)$ becomes well-defined everywhere.
    - *At $t=T$: * $x_T tilde p_T (x) approx cal(N)(0, I)$ (An unstructured, tractable prior).

  #v(0.5em)

  $arrow.double$ *Key Idea:* By designing $f(x,t)$ and $g(t)$, we construct a process that systematically destroys data into noise. If we can *reverse* this SDE, we can generate data from noise!
]

#title-slide[
  Generation via Probability Flow ODE
]

#slide(title: "Generation: Probability Flow ODE")[

  #set text(size: 16pt)

  The forward SDE transforms data $p_0 (x)$ into pure noise $p_T (x) approx cal(N)(0, I)$. To generate data, we reverse this process from $t=T$ down to $t=0$.
  Remarkably, we can perfectly retrace the exact marginal densities $p_t (x)$ using a deterministic equation.

  #theorem([Probability Flow ODE @score-based-generative-models])[
    For the forward SDE $dif x = f(x, t) dif t + g(t) dif w$, there exists a corresponding deterministic *Ordinary Differential Equation (ODE)*. This ODE shares the *exact same marginal probability densities* $p_t (x)$ at all times $t in [0, T]$:
    $
      dif x = [ f(x, t) - 1/2 g(t)^2 nabla_x log p_t (x) ] dif t.
    $
  ]

  #v(0.5em)

  - *The Missing Piece:* To simulate this ODE backwards from $t=T$ to $t=0$, $f(x,t)$ and $g(t)$ are already known (we designed them). The *only* unknown variable is the score function $nabla_x log p_t (x)$.
    $arrow.double$ *Our trained score model $s_theta (x, t)$ directly plugs in here!*
]

#slide(title: "Roadmap: Design the SDE, Learn the Score")[

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
      Since the probability flow ODE requires $nabla_x log p_t (x)$, we need a way to train a time-dependent neural network $s_theta (x, t)$ to approximate this score for *all* continuous times $t in [0, T]$.

  #v(0.8em)

  $arrow.double$ Let's first address *Task 1* by looking at a standard choice: the *Variance Preserving (VP) SDE*.
]

#title-slide[
  Specify the Forward Process: VP SDE
]

#slide(title: "VP SDE: A Standard Forward Process")[

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

  - *The Tractable Transition Kernel:*
    Because this SDE is linear, the transition probability from $x_0$ to $x_t$ is always a tractable Gaussian. Let $alpha_t = exp(- integral_0^t beta(s) dif s)$:
    $
      p_(t|0)(x_t | x_0) = cal(N)(x_t; sqrt(alpha_t) x_0, (1 - alpha_t) I)
    $



  #pagebreak()

  #set text(size: 14pt)

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

  - *The Corresponding Probability Flow ODE:*
    By substituting our specific drift $f(x, t)$ and diffusion $g(t)$ into the general Probability Flow ODE formula, we get the following equation:
    $ dif x = (-1/2 beta(t) x -1/2 beta(t) nabla_x log p_t (x)) dif t $

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

#title-slide[
  Learn the Time-dependent Score
]

#slide(title: "Training: Continuous-Time Score Matching")[

  #set text(size: 16pt)

  We now have a specific SDE and its transition kernel. The final task is to train a neural network $s_theta(x, t)$ to approximate the score $nabla_x log p_t (x)$ across all continuous times $t in [0, 1]$.

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

#title-slide[
  Controllable Generation: Guiding the SDE
]

#slide(title: "Conditional Generation via Bayes' Rule")[

  #set text(size: 16pt)

  We now know how to generate unconditional data $x tilde p_0(x)$. What if we want to generate data conditioned on a signal $y$ (e.g., a text prompt or a class label)?

  #v(0.5em)

  - *The Conditional Score:*
    To generate conditional samples, we simply replace our unconditional score $nabla_x log p_t (x)$ with the conditional score $nabla_x log p_t (x | y)$ inside the Reverse ODE/SDE.

  #v(0.5em)

  - *Applying Bayes' Theorem:*
    By taking the gradient of the log-posterior $p_t (x | y) prop p_t (x) p_t (y | x)$, we get a beautiful decomposition:

    $
      nabla_x log p_t (x | y) = underbrace(nabla_x log p_t (x), "Unconditional Score" \ s_theta (x, t)) + underbrace(nabla_x log p_t (y | x), "Guidance Term" \ ("Likelihood"))
    $

  #v(0.5em)

  - *The Mechanism:*
    1. *Unconditional Score:* Ensures the generated sample looks like real, high-quality data.
    2. *Guidance Term:* Acts as a secondary vector field, actively pushing the trajectory toward regions of the data manifold that satisfy the condition $y$.
]

#slide(title: "Application: Guidance in Diffusion Planner")[

  #set text(size: 16pt)

  In *Diffusion Planner*, we want to generate valid trajectories that satisfy specific physical or operational constraints (e.g., avoiding obstacles, target speeds).

  #v(0.5em)

  - *Rethinking the Guidance Term:*
    Instead of training a separate neural network classifier to predict $p_t (y | x)$, we can mathematically define these rules using *human-designed cost functions* $cal(E)(x)$.

  #v(0.5em)

  - *Energy-Based Formulation:*
    We formulate the condition $y$ ("satisfies all constraints") as a Boltzmann distribution. The probability of a trajectory being valid decreases exponentially as the cost increases:
    $ p_t (y | x_t) prop exp(- lambda cal(E)(x_t)) $
    where $lambda > 0$ controls the guidance strength, and $cal(E)(x_t)$ is the total cost.

  #pagebreak()

  - *Analytic Guidance Field:*
    By plugging this into our Bayes' decomposition, the intractable log-probability derivative beautifully simplifies into an *analytic gradient*:
    $ nabla_(x_t) log p_t (y | x_t) = - lambda nabla_(x_t) cal(E)(x_t) $
    *(e.g., $cal(E)(x_t) = cal(E)_"obstacle" (x_t) + cal(E)_"velocity" (x_t)$)*

  #v(0.5em)

  $arrow.double$ *The Synergy:* The unconditional score $s_theta (x_t, t)$ ensures the trajectory follows realistic dynamics (learned from data), while $-lambda nabla_(x_t) cal(E)(x_t)$ acts as a "repulsive force" pushing the path away from obstacles during inference!
]

#title-slide[
  Advanced Capabilities Enabled by the ODE Formulation
]

#slide(title: "Application 1: Latent Space Interpolation")[

  #set text(size: 16pt)

  Because the Probability Flow ODE is completely *deterministic*, it defines a bijective (1-to-1) mapping between the complex data space $x_0 tilde p_0$ and the simple latent space $x_T tilde p_T$.

  #v(0.5em)

  - *How to Interpolate:*
    1. *Encode:* Take two real images $x_0^((A))$ and $x_0^((B))$. Simulate the ODE *forward* ($t=0 arrow T$) to find their exact latent representations $x_T^((A))$ and $x_T^((B))$.
    2. *Interpolate:* Blend the latents in the spherical prior space using a parameter $lambda in [0, 1]$ (e.g., $z_lambda = "Slerp"(x_T^((A)), x_T^((B)), lambda)$).
    3. *Decode:* Simulate the ODE *backward* ($t=T arrow 0$) starting from $z_lambda$ to generate the intermediate image $x_0^((lambda))$.

  #v(0.5em)

  - *Result:*
    Unlike pixel-space blending, ODE-based interpolation smoothly morphs the semantic features (e.g., age, hair color) while staying perfectly on the data manifold.

  #align(center)[
    #image("assets/celeba_interp.png", width: 65%)
  ]
]

#slide(title: "Application 2: Exact Log-Likelihood Computation")[

  #set text(size: 16pt)

  The Reverse ODE allows us to compute the *exact* probability density $p_0(x)$ of any data point.

  #v(0.5em)

  - *Instantaneous Change of Variables:*
    Let $f_("ODE")(x, t)$ be the vector field of our Probability Flow ODE. The change in the log-probability over time is given by the negative divergence of this vector field:
    $
      (partial log p_t (x)) / (partial t) = - nabla_x dot.c f_("ODE")(x, t)
    $

  #v(0.5em)

  - *Computing the Likelihood:*
    To find the exact log-likelihood of a real image $x_0$, we integrate this divergence along the ODE trajectory from $t=0$ to $t=T$:
    $
      log p_0(x_0) = log p_T (x_T) + integral_0^T nabla_(x_t) dot.c f_("ODE")(x_t, t) dif t
    $
    where $p_T (x_T)$ is just the simple Gaussian prior $cal(N)(0, I)$.

  #v(0.5em)

  - *Why this matters:* Exact likelihoods allow for principled model evaluation, rigorous comparison against other generative models (like Autoregressive models or Normalizing Flows), and out-of-distribution (anomaly) detection.
]

#title-slide[
  Accelerating Generation: DPM-Solver++
]

#slide(title: "The Bottleneck of Standard ODE Solvers")[

  #set text(size: 16pt)

  While the Probability Flow ODE is elegant, solving it with standard numerical methods (like Euler or Runge-Kutta) is notoriously slow, often requiring 200 to 1000 steps.

  #v(0.5em)

  - *Why so slow?*
    The ODE contains a "stiff" linear drift term. If we take large step sizes, the truncation error explodes, destroying the image.
    $
      dif x = underbrace(-1/2 beta(t) x, "Linear Drift (Stiff)") dif t - 1/2 beta(t) underbrace(epsilon_theta (x, t), "Neural Network") dif t
    $

  #v(0.5em)

  - *The Semi-Analytic Solution (DPM-Solver):*
    Instead of treating the entire ODE as a black box, Lu et al. (2022) proposed using an *Exponential Integrator*.
    By applying the variation of constants formula, DPM-Solver solves the linear drift part $f(x,t)$ *exactly analytically*. It only relies on numerical approximation for the non-linear neural network part $epsilon_theta(x, t)$.

  #v(0.5em)

  $arrow.double$ This reduces the required steps from 1000 down to around 50. But can we go even faster?
]

#slide(title: "DPM-Solver++: The Data Prediction Upgrade")[

  #set text(size: 15pt)

  DPM-Solver is fast, but it suffers from instability at very small $t$ (near the end of generation), especially when using large Classifier-Free Guidance scales. *DPM-Solver++* fixes this via a clever reparameterization.

  #v(0.5em)

  - *Noise vs. Data Prediction:*
    Instead of approximating the noise trajectory $epsilon_theta(x, t)$, DPM-Solver++ rewrites the exact ODE solution to integrate over the *data prediction* model $x_theta(x, t) approx x_0$:
    $ x_theta(x_t, t) = (x_t - sqrt(1 - alpha_t) epsilon_theta(x_t, t)) / sqrt(alpha_t) $

  #v(0.5em)

  - *Why predict the clean image $x_0$?*
    The predicted noise $epsilon$ oscillates sharply as the generation progresses. However, the model's prediction of the final clean image $x_theta$ becomes remarkably *stable and smooth* as $t$ decreases.
    Integrating a mathematically *smoother function* allows for Taylor expansions with much larger step sizes and lower truncation error!

  #v(0.5em)

  - *The Impact:*
    By applying a multi-step update rule on this smooth $x_theta$ trajectory, DPM-Solver++ can generate extremely high-quality images in just *10 to 20 steps*, effectively making real-time diffusion a reality.
]

// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)
