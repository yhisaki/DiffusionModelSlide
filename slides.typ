#import "@preview/typslides:1.3.2": *

#show: typslides.with(
  ratio: "16-9",
  theme: rgb("1f6f78"),
  font: ("Noto Sans CJK JP", "DejaVu Sans"),
  font-size: 18pt,
  link-style: "underline",
  show-progress: true,
)

#front-slide(
  title: "Diffusion Model",
  subtitle: [An introduction],
  authors: "Yukinari Hisaki",
)

#table-of-contents()

#slide(title: "Introduction")[

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
    $ z arrow.long^(p_theta(x|z)) x $

  #v(0.5em)

  - *The Limitation:*
    Modeling the complex data distribution $q(x)$ via a *"single shot"* transformation $p_theta(x|z)$ is challenging. The gap between the simple prior and the complex data manifold is too large.

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

  2. *Score-based Generative Models (Score Matching) @score-based-generative-models*
    - Formulated via Stochastic Differential Equations (SDEs).
    - _Key idea:_ Learn the *score function* $nabla_x log p_t(x)$ (direction to higher density).

  #v(0.2em)

  3. *Flow Matching*
    - A more general framework based on Ordinary Differential Equations (ODEs).
    - _Key idea:_ Directly regress a *vector field* $v_t(x)$ to transport probability paths.

  #v(0.2em)
  *Focus of this talk:*
  We will focus on *2. Score Matching* and *3. Flow Matching*, as they offer a unified continuous-time perspective.
]

#title-slide[
  Score Matching
]

#slide(title: "Background: Score-Based Modeling")[

  Before introducing diffusion, let's define the "Score".

  #v(0.5em)

  - *Definition: The Stein Score Function*
    For a probability density $q(x)$, the score is defined as the *gradient of the log-density* with respect to data $x$:
    $ s(x) eq.def nabla_x log q(x) $

  #v(0.2em)

  - *Intuition: A Vector Field*
    - The score is a vector field pointing in the direction of *higher data density*.
    - At the modes (peaks) of the distribution, the score is zero ($nabla log q(x) = 0$).

  #v(0.2em)

  - *The Goal (Score Matching)*
    We want to train a neural network $s_theta(x)$ to approximate this true score:
    $ cal(L)(theta) = bb(E)_(x tilde q) [ || s_theta (x) - nabla_x log q(x) ||^2_2 ] $
  // (Fisher Divergence)
]

#slide(title: "Solution: Denoising Score Matching")[

  #set text(size: 15pt)
  *Problem:* The true score $nabla_x log q(x)$ is unknown.

  #v(0.5em)

  *The Trick (#link("https://arxiv.org/abs/1103.5949")[Vincent, 2011]):*
  Instead of modeling raw data, we *perturb* data with Gaussian noise and learn to denoise it.

  1. Add noise: $tilde(x) = x + sigma z$, where $z tilde cal(N)(0, I)$.
  2. The conditional score $nabla_tilde(x) log q(tilde(x)|x)$ is known:

    $
      nabla_tilde(x) log q(tilde(x)|x) = nabla_tilde(x) (- frac(||tilde(x) - x||^2, 2sigma^2)) = frac(x - tilde(x), sigma^2)
    $

  #v(0.5em)

  *The Tractable Objective:*
  We match this *conditional* score instead:
  $
    cal(L)_(text(d))(theta) = bb(E)_(x, tilde(x)) [ || s_theta (tilde(x)) - underbrace(frac(x - tilde(x), sigma^2), "Target (Known)") ||^2_2 ]
  $

  #v(0.2em)

  *Key Result:*
  Minimizing this ensures $s_theta(tilde(x)) approx nabla_tilde(x) log q_sigma(tilde(x))$.
  // We learn the score of the "noisy" data distribution.
]

#slide(title: "Sampling via Langevin Dynamics")[

  Once we have the score function $s_theta(x) approx nabla_x log q(x)$, how do we generate samples?

  #v(0.5em)

  - *Langevin Dynamics (Noisy Gradient Ascent):*
    We initialize $x_0$ arbitrarily and iteratively update it using the score:
    $
      x_(t+1) arrow.l x_t + underbrace(frac(epsilon, 2) s_theta (x_t), "Gradient Ascent") + underbrace(sqrt(epsilon) z_t, "Random Noise")
    $
    where $z_t tilde cal(N)(0, I)$ and $epsilon$ is the step size.

  #v(0.5em)

  - *Mechanism:*
    1. *Gradient Term:* Pushes $x$ towards high-density regions (modes).
    2. *Noise Term:* Prevents collapsing to a single mode, allowing exploration of the full distribution.

  #v(0.5em)

  $arrow.double$ As $t arrow infinity$ and $epsilon arrow 0$, $x_t$ converges to a sample from $q(x)$.
]

// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)

