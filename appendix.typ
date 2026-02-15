#import "@preview/ctheorems:1.1.3": *

#set page(
  width: 180mm,
  height: 255mm,
  margin: (x: 20mm, y: 20mm),
)
#set text(size: 8pt, lang: "en")

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
  Let $x_t$ be the process defined above and $phi: RR^d -> RR$ be a smooth test function ($C^2$).
  The stochastic differential of $phi(x_t)$ is given by:
  $
    dif phi(x_t)
    = lr([ nabla phi(x_t) dot.c f(x_t, t) + 1/2 g(t)^2 Delta phi(x_t) ]) dif t
    + g(t) nabla phi(x_t) dot.c dif w_t.
  $
] <lemma-ito>

#proof[
  Let $phi(x)$ be a smooth test function with compact support (arbitrary).
  Applying @lemma-ito to the scalar field $phi(x_t)$:

  $
    dif phi(x_t) & = (nabla phi(x_t) dot.c f(x_t, t) + 1/2 g(t)^2 Delta phi(x_t)) dif t \
                 & + g(t) nabla phi(x_t) dot.c dif w_t
  $

  Taking the expectation $bb(E)$ on both sides with respect to the density $p_t (x)$, and noting that the expectation of the martingale term (the $dif w_t$ term) is zero:

  $
    frac(dif, dif t) bb(E)[phi(x_t)] = bb(E) [ nabla phi(x_t) dot.c f(x_t, t) + 1/2 g(t)^2 Delta phi(x_t) ]
  $

  Rewriting the expectation using the density integral $bb(E)[h(x)] = integral h(x) p_t (x) dif x$:

  $
    integral phi(x) partial_t p_t (x) dif x = integral [ nabla phi(x) dot.c f(x, t) + 1/2 g(t)^2 Delta phi(x) ] p_t (x) dif x
  $
  Now, we apply *Integration by Parts (IBP)* to transfer the spatial derivatives from the test function $phi(x)$ to the density $p_t (x)$.
  Since $phi(x)$ has compact support, it vanishes at infinity ($phi(x) -> 0$ as $|x| -> infinity$), so all boundary terms become zero.

  1. *Drift Term (1st order derivative):*
  Using the vector calculus identity $nabla dot.c (psi bold(A)) = nabla psi dot.c bold(A) + psi nabla dot.c bold(A)$, we expand the divergence of the product $phi(x) f(x,t) p_t (x)$:

  $
    integral (nabla phi(x) dot.c f(x,t)) p_t (x) dif x
    &= integral [ nabla dot.c (phi(x) f(x,t) p_t (x)) - phi(x) nabla dot.c (f(x,t) p_t (x)) ] dif x \
    &= underbrace(integral nabla dot.c (phi(x) f(x,t) p_t (x)) dif x, "Boundary term = 0 (Divergence Thm)")
    - integral phi(x) nabla dot.c (f(x,t) p_t (x)) dif x \
    &= - integral phi(x) nabla dot.c (f(x,t) p_t (x)) dif x
  $

  2. *Diffusion Term (2nd order derivative):*
  We perform IBP twice. First, note that $g(t)$ is independent of $x$, so we pull it out.
  We focus on the Laplacian term $integral (Delta phi(x)) p_t (x) dif x$.
  Recall that $Delta phi = nabla dot.c nabla phi$.

  *Step A (First IBP):* Move one $nabla$.
  $
    integral (nabla dot.c nabla phi(x)) p_t (x) dif x & = underbrace(integral nabla dot.c (p_t (x) nabla phi(x)) dif x, "= 0 (Boundary)")
                                                              - integral nabla phi(x) dot.c nabla p_t (x) dif x \
                                                            & = - integral nabla phi(x) dot.c nabla p_t (x) dif x
  $

  *Step B (Second IBP):* Move the remaining $nabla$.
  $
    - integral nabla phi(x) dot.c nabla p_t (x) dif x & = - [ underbrace(integral nabla dot.c (phi(x) nabla p_t (x)) dif x, "= 0 (Boundary)")
                                                                - integral phi(x) nabla dot.c nabla p_t (x) dif x ] \
                                                            & = integral phi(x) Delta p_t (x) dif x
  $

  Combining Step A and B with the coefficient $1/2 g(t)^2$:
  $
    integral 1/2 g(t)^2 Delta phi(x) p_t (x) dif x
    = integral phi(x) [ 1/2 g(t)^2 Delta p_t (x) ] dif x
  $

  Substituting these results back into the original expectation equation:

  $
    integral phi(x) partial_t p_t (x) dif x
    = integral phi(x) [ - nabla dot.c (f(x,t) p_t (x)) + 1/2 g(t)^2 Delta p_t (x) ] dif x
  $

  Rearranging to group all terms on one side:
  $
    integral phi(x) underbrace(( partial_t p_t (x) + nabla dot.c (f(x,t) p_t (x)) - 1/2 g(t)^2 Delta p_t (x) ), "Must be 0") dif x = 0
  $
]

#definition("Score of the marginal")[
  Define the *score* function as the gradient of the log-density:
  $
    s (x) := nabla_x log p (x).
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
    tilde(f)(x,t) = f(x,t) - 1/(p_t (x)) nabla dot.c (D(t) p_t (x)),
  $
  where $D(t) = g(t)^2 I$ is the diffusion tensor.
  Substituting $D(t)$ into the equation:
  $
    tilde(f)(x,t) & = f(x,t) - 1/(p_t (x)) nabla dot.c (g(t)^2 p_t (x)) \
                  & = f(x,t) - g(t)^2 (nabla p_t (x)) / (p_t (x)).
  $
  Using the definition of the score $s_t (x) = nabla log p_t (x) = (nabla p_t (x)) / (p_t (x))$, we obtain:
  $
    tilde(f)(x,t) = f(x,t) - g(t)^2 s_t (x).
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
  This equation is exactly the *continuity equation* (Liouville equation) describing the evolution of densities under a deterministic flow field $v_t (x)$:
  $
    partial_t p_t + nabla dot.c (v_t p_t) = 0.
  $
  Comparing terms, we identify the velocity field:
  $
    v_t (x) = f(x,t) - 1/2 g(t)^2 s_t (x).
  $
  Since the ODE $(dif x_t) / (dif t) = v_t (x_t)$ implies the continuity equation for its marginals, and this equation is identical to the SDE's Fokker--Planck equation, the ODE and the SDE share the same marginal densities $p_t$.
]
