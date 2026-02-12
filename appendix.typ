#import "@preview/ctheorems:1.1.3": *

#set page(
  width: 180mm,
  height: 255mm,
  margin: (x: 20mm, y: 20mm),
)
#set text(size: 11pt, lang: "en")

#show: thmrules

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

#let proof = thmproof("proof", "Proof")

#let lemma = thmbox(
  "lemma",
  "Lemma",
  fill: rgb("#fff9e6"),
  stroke: rgb("#fbc02d") + 1pt,
  padding: (top: 1em, bottom: 1em),
)
// --- Content ---
= Score-based diffusion: Forward SDE and Reverse ODE

Let $(x_t)_(t in [0, T])$ be a diffusion process on $RR^d$ driven by a standard Wiener process $(w_t)$.

// Theorem 1
#theorem("Forward SDE and its marginal evolution")[
  Assume the *forward* diffusion is defined by the Itô SDE
  $
    dif x_t = f(x_t, t) dif t + g(t) dif w_t, quad t in [0, T],
  $
  where $w_t$ is a standard Wiener process in $RR^d$.
  The initial distribution is $x_0 tilde p_0$ (the data distribution), drift $f: RR^d times [0,T] -> RR^d$,
  and scalar diffusion coefficient $g: [0,T] -> (0, infinity)$.

  Then the marginal density $p_t (x)$ of $x_t$ exists and satisfies the
  *Fokker--Planck (forward Kolmogorov) PDE*:
  $
    partial_t p_t (x)
    = - nabla dot.c (f(x,t) p_t (x))
    + 1/2 g(t)^2 Delta p_t (x),
    quad forall x in RR^d.
  $
]

// Lemma: Itô's Lemma
#lemma("Itô's Lemma for test functions")[
  Let $x_t$ be the process defined above and $phi: RR^d -> RR$ be a smooth function ($C^2$).
  The stochastic differential of $phi(x_t)$ is given by:
  $
    dif phi(x_t)
    = lr([ nabla phi(x_t) dot.c f(x_t, t) + 1/2 g(t)^2 Delta phi(x_t) ]) dif t
    + g(t) nabla phi(x_t) dot.c dif w_t.
  $
] <lemma-ito>

#proof[
  Let $phi(x)$ be an arbitrary smooth test function with compact support ($phi in C_c^infinity (RR^d)$).

  We define the expectation $E(t) := EE[phi(x_t)]$.
  Integrating the SDE form given in @lemma-ito from $0$ to $t$ and taking the expectation:
  $
    EE[phi(x_t)] - EE[phi(x_0)]
    &= EE[ integral_0^t lr([ f(x_s, s) dot.c nabla phi(x_s) + 1/2 g(s)^2 Delta phi(x_s) ]) dif s ] \
    &quad + EE[ integral_0^t g(s) nabla phi(x_s) dot.c dif w_s ]. \
  $
  The second term (the stochastic integral) vanishes because the Itô integral with respect to a Wiener process is a martingale (and has zero expectation starting from 0).

  Differentiating the remaining term with respect to $t$:
  $
    (dif)/(dif t) EE[phi(x_t)]
    = EE[ f(x_t, t) dot.c nabla phi(x_t) + 1/2 g(t)^2 Delta phi(x_t) ].
  $

  Now, we rewrite the expectation using the probability density $p_t(x)$:
  $
    integral_(RR^d) phi(x) partial_t p_t(x) dif x
    = integral_(RR^d) lr([ f(x,t) dot.c nabla phi(x) + 1/2 g(t)^2 Delta phi(x) ]) p_t(x) dif x.
  $

  We apply integration by parts to move the derivatives from the test function $phi$ to the density $p_t$. Since $phi$ has compact support, the boundary terms vanish.

  1. For the drift term:
  $
    integral (f dot.c nabla phi) p_t dif x = - integral phi nabla dot.c (f p_t) dif x.
  $

  2. For the diffusion term (applying IBP twice):
  $
    integral (Delta phi) p_t dif x = integral phi Delta p_t dif x.
  $

  Substituting these back yields:
  $
    integral_(RR^d) phi(x) partial_t p_t(x) dif x
    = integral_(RR^d) phi(x) lr([ - nabla dot.c (f(x,t) p_t(x)) + 1/2 g(t)^2 Delta p_t(x) ]) dif x.
  $

  Rearranging terms to one side:
  $
    integral_(RR^d) phi(x) lr([ partial_t p_t(x) + nabla dot.c (f p_t) - 1/2 g(t)^2 Delta p_t ]) dif x = 0.
  $

  Since this equality holds for *any* smooth test function $phi$ with compact support, the fundamental lemma of calculus of variations implies that the term in the brackets must be zero almost everywhere. Thus, $p_t$ satisfies the Fokker--Planck equation.
]
#definition("Score of the marginal")[
  Define the *score* function as the gradient of the log-density:
  $
    s_t (x) := nabla_x log p_t (x) = (nabla p_t(x)) / p_t(x).
  $
]

// Theorem 2
#theorem("Reverse-time SDE")[
  Assume $p_t$ is smooth and strictly positive for $t in (0,T]$.
  Let $overline(w)_t$ be a standard Wiener process under time reversal.

  Then the *reverse-time* process that runs from $t=T$ down to $t=0$ is governed by the SDE:
  $
    dif x_t
    = lr([f(x_t,t) - g(t)^2 s_t (x_t)]) dif t + g(t) dif overline(w)_t,
  $
  integrated backward in time from $T -> 0$.
]

#proof[
  Consider the general form of the reverse drift $tilde(f)(x,t)$. For a diffusion process with scalar diffusion $g(t)$, the relationship between the forward drift $f$ and the reverse drift $tilde(f)$ is given by:
  $
    tilde(f)(x,t) = f(x,t) - 1/p_t(x) nabla dot.c (D(t) p_t(x)),
  $
  where $D(t) = g(t)^2 I$ is the diffusion tensor.
  Substituting $D(t)$ into the equation:
  $
    tilde(f)(x,t) & = f(x,t) - 1/p_t(x) nabla dot.c (g(t)^2 p_t(x)) \
                  & = f(x,t) - g(t)^2 (nabla p_t(x)) / p_t(x).
  $
  Using the definition of the score $s_t(x) = nabla log p_t(x) = (nabla p_t(x)) / p_t(x)$, we obtain:
  $
    tilde(f)(x,t) = f(x,t) - g(t)^2 s_t(x).
  $
  Thus, the reverse SDE has the modified drift $f - g^2 s_t$ and the same diffusion coefficient $g(t)$.
]

// Theorem 3
#theorem("Probability flow ODE")[
  Under the same assumptions, consider the deterministic ODE:
  $
    (dif x_t) / (dif t)
    = f(x_t,t) - 1/2 g(t)^2 s_t (x_t),
  $
  integrated backward in time from $T -> 0$.

  Then this *probability flow ODE* has the *same marginal densities* as the forward SDE.
]

#proof[
  We show that the probability density evolved by this ODE satisfies the same Fokker--Planck equation as the SDE.
  Recall the Fokker--Planck equation for the SDE from Theorem 1:
  $
    partial_t p_t = - nabla dot.c (f p_t) + 1/2 g(t)^2 Delta p_t.
  $
  We use the identity $Delta p_t = nabla dot.c (nabla p_t)$. By definition of the score, $nabla p_t = p_t s_t$.
  Therefore, we can rewrite the diffusion term as:
  $
    1/2 g(t)^2 Delta p_t
    = 1/2 g(t)^2 nabla dot.c (p_t s_t)
    = nabla dot.c (1/2 g(t)^2 s_t p_t).
  $
  Substituting this back into the Fokker--Planck equation:
  $
    partial_t p_t & = - nabla dot.c (f p_t) + nabla dot.c (1/2 g(t)^2 s_t p_t) \
                  & = - nabla dot.c lr([ (f - 1/2 g(t)^2 s_t) p_t ]).
  $
  This equation is exactly the *continuity equation* (Liouville equation) describing the evolution of densities under a deterministic flow field $v_t(x)$:
  $
    partial_t p_t + nabla dot.c (v_t p_t) = 0.
  $
  Comparing terms, we identify the velocity field:
  $
    v_t(x) = f(x,t) - 1/2 g(t)^2 s_t(x).
  $
  Since the ODE $(dif x_t) / (dif t) = v_t(x_t)$ implies the continuity equation for its marginals, and this equation is identical to the SDE's Fokker--Planck equation, the ODE and the SDE share the same marginal densities $p_t$.
]
