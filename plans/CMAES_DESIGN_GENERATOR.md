# CMA-ES Design Generator: Theoretical Foundations and Implementation Plan

> Status, verified against the live sources on 2026-07-30. Implemented, with two qualifications. The generator classes specified in section 4b exist in `co_optimisation/runners/usd_generator.py` and drive every co-optimisation run. First, the hyperparameter recommendations of this document were superseded by [COPT_INVESTIGATION_PLAN.md](COPT_INVESTIGATION_PLAN.md) section 3, which reduced the initial step size and widened the parameter ranges after the original values were found to collapse the search against the box boundary. Second, this document contains one erratum, it states that the pycma library applies a penalty based boundary handler, whereas `../context/cmaes.md` section 4 establishes by direct reading that the default is a saturating transform, a distinction that matters because the two handlers bias the search differently near a bound. Read this document for the mathematics of the outer loop, which remains the fullest treatment in the workspace. See [README.md](README.md) for the full register.

## 1. Introduction

This document extends the documentation for the design-and-controller
co-optimisation pipeline introduced in
[../CO_OPTIMISATION.md](../CO_OPTIMISATION.md). The reference document establishes a
hybrid two-tier architecture in which a Proximal Policy Optimisation (PPO)
inner loop trains a locomotion policy on IsaacLab and RSL-RL while an outer
design generator periodically replaces the robot morphology used by the
parallel environments. The two tiers are interfaced through the abstract class
`DesignGeneratorBase` and its derivatives in
`co_optimisation.runners.usd_generator`, which produces a `Population` of
perturbed URDF/USD assets and a set of post-respawn `IdentifiedActuator`
overrides every outer-loop generation. The existing implementation,
`RandomDesignGenerator`, samples scale factors uniformly at random from a
bounded box; it is sufficient as a placeholder but does not learn from the
per-individual fitness signal exposed by `CoptOnPolicyRunner`.

Formally, design-controller co-optimisation is a bilevel program over a
continuous vector of design parameters $\boldsymbol{\theta}_d \in
\mathbb{R}^n$ (link lengths, link masses, actuator gains, joint limits) and a
control policy $\pi_{\boldsymbol{\theta}_c}$ that is near-optimal under the
morphology induced by $\boldsymbol{\theta}_d$:

$$
\boldsymbol{\theta}_d^{\star} \;=\; \arg\max_{\boldsymbol{\theta}_d \in \Theta_d} \;
F(\boldsymbol{\theta}_d), \qquad
F(\boldsymbol{\theta}_d) \;=\; \mathbb{E}_{\tau \sim \pi^{\star}_{\boldsymbol{\theta}_d}}\!\left[\sum_{t} \gamma^t r(s_t, a_t)\right],
$$

where $\pi^{\star}_{\boldsymbol{\theta}_d} = \arg\max_{\pi}
\mathbb{E}_{\tau \sim \pi}[\sum_t \gamma^t r(s_t, a_t) \mid
\boldsymbol{\theta}_d]$ is the optimal policy under morphology
$\boldsymbol{\theta}_d$. The outer objective $F(\boldsymbol{\theta}_d)$ is
non-convex, non-differentiable in $\boldsymbol{\theta}_d$ — because rigid-body
contact, mass-inertia recomputation, and PD-saturation are non-smooth — and
accessible only through expensive black-box evaluations: a single query
requires running an entire inner-loop training segment in a vectorised
simulator. Search efficiency therefore hinges on (i) faithfully exploring the
design space, (ii) exploiting the local curvature of the fitness landscape
near promising designs, and (iii) using as few evaluations as possible
because each evaluation invokes a costly simulation block. These three
desiderata motivate the use of *population-based stochastic optimisers* —
specifically, genetic and evolution-strategy algorithms — that maintain a set
of candidate designs, score them against the inner-loop's empirical return,
and bias subsequent samples towards regions of high estimated fitness.

The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) of Hansen and
Ostermeier [2001] is the de facto standard population-based optimiser for
continuous black-box problems of moderate dimension ($n \lesssim 100$). Its
defining feature is a *fully adaptive* multivariate Gaussian sampling
distribution whose covariance matrix is learned online from the ranks of past
evaluations; this gives the algorithm two empirically critical properties —
*invariance to monotone re-scalings of the objective* and *invariance to
affine reparameterisations of the search space* — which together explain its
robustness across heterogeneous problems such as robot co-design, where the
parameters have wildly different physical units (lengths, masses, gains) and
correlate strongly through the dynamics. Luck et al. [2020] are, to the best
of our knowledge, the first to deploy CMA-ES as the outer loop of a deep-RL
co-design pipeline, alternating CMA-ES generations over a continuous design
vector with SAC inner-loop training; Cheng, Han et al. [2024] (SERL) use a
related — though distinct — binary-encoded GA for the analogous bipedal-robot
problem on the *Wow Orin* platform. The pipeline in this repository follows
the SERL bi-level template but, motivated by the limitations of a discrete GA
catalogued in Section 2, will use CMA-ES as the outer optimiser. The
remainder of this document develops the theory and engineering necessary to
implement that replacement.

The document is organised as follows. Section 2 surveys the literature on
evolutionary robot co-design, introduces the canonical genetic-algorithm
framework, and positions CMA-ES as a continuous-space evolution strategy
(ES). Section 3 derives CMA-ES from first principles, gives the canonical
$(\mu/\mu_W, \lambda)$-CMA-ES formulation with active negative weights, and
documents the `cma` Python package (v4.4.4) that the implementation will use.
Section 4 dissects the existing `RandomDesignGenerator` and presents a
detailed implementation plan for `CMAESDesignGenerator`. Section 5 provides
API documentation for every class touched. Section 6 collects references.

## 2. Genetic Algorithms in Design Optimisation

### a. Literature Survey

Evolutionary computation has been applied to robot design since the
foundational work of Sims [1994] on co-evolving morphology and neural
controllers for simulated creatures. Sims' representation — a directed graph
whose nodes carry rigid-body part parameters and whose edges carry connection
placements and the sensor/effector/internal-neuron wiring of the controller —
established the now-standard pattern of representing a robot as a parameter
vector (or graph) and evaluating individuals through embodied simulation.
Lipson and Pollack [2000] closed the sim-to-real loop with GOLEM: bar-actuator
robots evolved in simulation, 3-D printed, and physically deployed. The biped
community adopted these ideas in the early 2000s: Paul and Bongard [2001]
demonstrated that for a five-link planar biped, coupled morphology +
controller evolution discovers stable gaits that controller-only evolution
(with a fixed body) fails to reach within the same compute budget; Endo,
Maeno and Kitano [2003] co-evolved link lengths/masses and central-pattern-
generator (CPG) walking parameters of a humanoid biped, penalising joint-
torque saturation directly inside the fitness — a precursor to modern motor-
limit constraints. Lipson [2014] and Cheney et al. [2018] subsequently
catalogued the three bottlenecks of automated co-design — high-fidelity
simulators, expressive actuation models, and component standardisation —
motivating the large-scale parallel simulators (Isaac Gym, MuJoCo MJX, Brax)
now standard in the field.

A persistent failure mode of naïve co-evolution, identified and named by
Cheney, Bongard, SunSpiral and Lipson [2018] as the absence of
*morphological innovation protection*, is the following: when a morphology
mutation occurs (longer legs, redistributed mass), the candidate's controller
has not yet adapted, so its immediate fitness drops below its parent's. A
standard $(\mu,\lambda)$-ES discards the variant before its controller can
re-adapt, trapping the search in a body + controller local optimum. Cheney
et al. fix this by maintaining an *age* objective alongside fitness, with age
reset whenever the body plan mutates; selection runs on the age-fitness
Pareto front, granting recently mutated bodies a few generations of shielded
time during which controller mutations can re-adapt. The same principle
re-emerges in modern bi-level deep-RL methods under a different name — the
*inner-loop training horizon per candidate* — which gives a fresh morphology
enough simulation budget for its policy to catch up before the fitness is
recorded.

The deep-RL-era methods that this work most directly inherits from are summarised
below.

- **Schaff et al. [2019]** (*Jointly Learning to Construct and Control
  Agents using Deep RL*) maintain a parametric Gaussian-mixture distribution
  $p_\phi(\boldsymbol{\omega})$ over design vectors and update it by a
  REINFORCE-style policy gradient on the design distribution; the inner loop
  is PPO with a single shared policy *conditioned* on $\boldsymbol{\omega}$.
  On Roboschool Hopper / Walker2D / Ant (8 to 25 design dimensions) their
  evolved designs outperform Bayesian-optimisation and random-sampling
  baselines.
- **Luck, Ben Amor and Calandra [2020]** (*Data-efficient Co-Adaptation of
  Morphology and Behaviour with Deep RL*, CoRL) name *PSO or CMA-ES* as the
  outer optimiser over a continuous design vector $\boldsymbol{\xi}$ and use
  a shared SAC critic as a *surrogate* for ranking new designs without
  re-simulating them. This is — to our knowledge — the first paper to
  explicitly use CMA-ES inside a deep-RL co-design loop, and is the most
  direct precedent for the design generator developed in Section 4.
- **Gupta, Savarese, Ganguli and Fei-Fei [2021]** introduce DERL (*Deep
  Evolutionary Reinforcement Learning*) and the UNIMAL articulated-tree
  design space; an asynchronous tournament EA over $\sim 4000$ morphologies
  is paired with a 5M-step PPO inner loop, exhibiting both the emergence of
  morphological intelligence under environmental complexity and a
  *morphological Baldwin effect* in which evolution selects bodies that are
  intrinsically easier to control.
- **Yuan et al. [2022]** (*Transform2Act*, ICLR) sidestep the bi-level
  structure entirely by folding the morphology search into the policy:
  each episode begins with a sequence of *transform actions* (add joint,
  remove joint, set parameter) that mutate a graph-conditioned GNN policy
  before control actions begin; experience is shared across designs via the
  GNN parameters.
- **Hejna, Pinto and Abbeel [2021]** (*Task-Agnostic Morphology Evolution*,
  ICLR) evolve morphologies under an information-theoretic fitness
  (state-space coverage, action causality) *without* per-morphology RL,
  and show that the discovered bodies match those of task-supervised
  co-evolution at downstream evaluation time.
- **Cheng, Han et al. [2024]** (SERL, IROS) is the most recent — and most
  directly comparable — published bipedal co-design pipeline. SERL is
  applied to the *Wow Orin* 9-DoF biped (10.5 kg, 0.88 m, Unitree A1
  motors, bionic-fishbone Bowden-cable ankle) and searches a two-dimensional
  design vector $(l_t, l_s) \in [0.20, 0.40]^2$ m (thigh length, shin
  length) under a uniform-mass-density consistency constraint. The inner
  loop is Rudin-style massively-parallel PPO (Isaac Gym) with a
  privileged-information encoder; each generation grants every individual
  10 PPO iterations after a 500-iteration shared warm-start. SERL's
  optimum — $(l_t, l_s) \approx (0.31, 0.36)$ m, with the shin counter-
  intuitively longer than the thigh — gives a cost-of-transport of 0.407
  at 2.1 m s$^{-1}$, ~45 % below Cassie and Unitree H1.

A clarification on SERL is essential before Section 4. Despite the surface
similarity between SERL's pipeline and the present one — bi-level outer
loop, RL inner loop, Isaac-Gym vectorised simulation, identical engineering
goal — **SERL's outer optimiser is a binary-encoded classical genetic
algorithm, not CMA-ES**. Table I of Cheng, Han et al. [2024] specifies
9-bit DNA per design parameter (giving SERL's 0.01 m length resolution),
population size 250, crossover rate 0.8, mutation rate 0.03, and
roulette-wheel fitness-proportionate selection. CMA-ES is not used or
cited. We highlight this because the implementation plan in Section 4
treats CMA-ES as an *improvement* over SERL's GA on three concrete axes:
(i) operating natively in continuous, bounded spaces without 9-bit
quantisation, (ii) adapting a *full covariance matrix* to handle the
anisotropic fitness landscape that is clearly visible in SERL's
Figs. 11–12, and (iii) eliminating SERL's manually tuned
$\mathrm{mutation\_rate} = 0.03$ and $\mathrm{crossover\_rate} = 0.8$
through cumulative step-size adaptation. The biped community has therefore
*already validated the bi-level template*; CMA-ES brings the outer loop in
line with current best practice in continuous black-box optimisation.

A summary comparison of the methods discussed above is given in Table 1.

| Method | Year | Morphology rep. | Outer loop | Inner loop | Fitness | Parallelisation |
|--------|------|-----------------|-----------|-----------|---------|-----------------|
| Cheney et al. | 2018 | Voxel grid (CPPN) | $(\mu{=}25,\lambda{=}25)$-ES + age-fitness Pareto | open-loop sinusoidal | locomotion dist. | CPU pop. |
| Schaff et al. | 2019 | Continuous lengths/radii | GMM $p_\phi(\omega)$, REINFORCE | PPO conditioned on $\omega$ | PPO return | Batched $\omega$ in PPO |
| Luck et al. | 2020 | Continuous design vector $\xi$ | **PSO or CMA-ES** with $Q_{\mathrm{Pop}}$ surrogate | SAC (individual + population nets) | $Q$-surrogate | Serial per design |
| DERL | 2021 | UNIMAL articulated tree | Async tournament (size 4), pop 576 | PPO, 5M steps per design | Mean RL return | 1152 CPUs, 288 parallel |
| Transform2Act | 2022 | Skeletal graph + transform actions | folded into policy | PPO on GNN policy | PPO return | Single run |
| SERL | 2024 | 2-D continuous $(l_t, l_s)$, 9-bit DNA | **Binary GA** (pop 250, cx 0.8, mut 0.03, roulette) | PPO (Rudin / Isaac Gym) | Cumulative PPO reward | 250 ind. on RTX 4090 |
| *This work* | 2026 | 19-D continuous scale-factor vector | **CMA-ES** (popsize $\sim$ 16–64, $\sigma_0{\approx}0.2$, $[0,1]^n$ bounds) | PPO (RSL-RL / IsaacLab) | Mean RL return per individual | $\sim$num_envs / num_individuals envs per design |

### b. The Genetic Algorithm Framework

A *genetic algorithm* (GA) maintains a population $\mathcal{P}_k =
\{\boldsymbol{x}^{(1)}_k, \ldots, \boldsymbol{x}^{(\lambda)}_k\}$ of $\lambda$
candidate solutions at generation $k$. Each individual $\boldsymbol{x}^{(i)}_k
\in \mathcal{X}$ encodes one robot design — for the present application, a
vector of scale factors over a base URDF. The algorithm iterates four
canonical operators until a termination criterion is met:

1. **Evaluation.** Each individual is scored by a fitness function
   $f: \mathcal{X} \to \mathbb{R}$. In a co-optimisation context the fitness
   is the mean episode return of the inner-loop policy trained under design
   $\boldsymbol{x}^{(i)}_k$. Evaluation is the dominant computational cost
   because each query is an entire RL training segment.
2. **Selection.** A subset of high-fitness individuals — the *parents* — is
   chosen. Selection may be deterministic (truncation: pick the top
   $\mu < \lambda$ individuals) or stochastic (fitness-proportionate /
   roulette wheel, as in SERL; rank-based; tournament). Truncation
   selection is the choice in CMA-ES.
3. **Variation.** New candidates are generated from the parents through
   *recombination* (mixing components of two or more parents) and
   *mutation* (additive perturbation). Real-valued GAs use Gaussian
   mutation; binary-encoded GAs (like SERL) use bit-flip mutation and
   single- or multi-point crossover.
4. **Replacement.** The new candidates form $\mathcal{P}_{k+1}$. In a
   *generational* GA the previous population is discarded entirely; in
   *steady-state* or *elitist* GAs some of the best previous individuals
   survive. Cheney et al.'s [2018] age-Pareto scheme is an extension of
   elitism that conditions survival on both fitness and morphological age.

The framework as stated above has two practical degrees of freedom that
profoundly affect performance: the *encoding* of $\mathcal{X}$ (discrete,
real-valued, tree, graph) and the *variation operators* (mutation
distribution, crossover topology). For continuous, bounded, low-to-moderate-
dimensional design spaces — the regime of co-design problems of interest
here — the right specialisation of the framework is the *Evolution Strategy*
(ES), in which (i) individuals are real-valued vectors, (ii) recombination
is replaced by a weighted *intermediate* recombination of the best $\mu$
parents to form a single distribution mean, and (iii) the mutation
distribution is itself an object of adaptation.

### c. From Evolution Strategies to CMA-ES

The earliest ES is Rechenberg's $(1+1)$-ES [Rechenberg 1973]: a single parent
$\boldsymbol{x}_k$ generates one offspring $\boldsymbol{x}' = \boldsymbol{x}_k
+ \sigma\,\boldsymbol{z}$ with $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0},
\mathbf{I})$, and the offspring replaces the parent iff
$f(\boldsymbol{x}') < f(\boldsymbol{x}_k)$. Convergence speed is governed
entirely by $\sigma$: too small and progress stalls, too large and most
candidates are rejected. Rechenberg's *one-fifth success rule* ties $\sigma$
to the empirical acceptance rate — increase $\sigma$ if more than one fifth
of candidates are accepted, decrease it otherwise — making the $(1+1)$-ES
the first instance of *adaptive step-size control* in continuous black-box
optimisation.

Schwefel [1981] generalised to a population of $\lambda$ offspring sampled
from a Gaussian centred at the weighted mean of the $\mu$ best parents, with
$\sigma$ itself encoded in the genome and *mutatively self-adapted* via a
log-normal perturbation per generation. Ostermeier, Gawelczyk and Hansen
[1994] removed the stochasticity from $\sigma$ adaptation by accumulating an
*evolution path*: a low-pass-filtered sum of past selection steps whose
length reports whether successive steps have been positively or negatively
correlated. This is *cumulative step-size adaptation* (CSA) and survives in
modern CMA-ES essentially unchanged.

Hansen and Ostermeier [2001] added the final ingredient — a *full covariance
matrix* $\mathbf{C}$ in addition to the scalar $\sigma$ — and adapted both
from the same evolution path via a *rank-one* update of $\mathbf{C}$. Hansen,
Müller and Koumoutsakos [2003] subsequently added a *rank-$\mu$* update from
the current generation's selection scatter, giving the algorithm a stable
high-rank covariance estimator. The two updates together are the
$(\mu/\mu_W, \lambda)$-CMA-ES that this document refers to as the
"canonical" algorithm. Subsequent refinements (Auger and Hansen [2005] IPOP
restart strategy; Jastrebski and Arnold [2006] / Hansen and Ros [2010]
active negative-weight update; Hansen [2009] BIPOP-CMA-ES) are documented in
Sections 3.b.5 and 3.b.9.

A pivotal modern result is the recasting of CMA-ES as an *approximate
natural-gradient ascent* on the log-likelihood of the Gaussian sampling
distribution under a rank-based utility transform of $f$ [Akimoto et al.
2010, 2012; Wierstra et al. 2014; Ollivier et al. 2017]. This information-
geometric view, summarised in §3.a.6, explains the algorithm's parameter-
free character — almost every hyperparameter is a closed-form function of
the dimension $n$ and selection mass $\mu_{\mathrm{eff}}$ — and motivates
the strong empirical performance of CMA-ES across the heterogeneous
parameter spaces encountered in robot co-design.

## 3. Covariance Matrix Adaptation Evolution Strategy

CMA-ES, introduced by Hansen and Ostermeier [2001] and refined in
Hansen [2016], is a stochastic, derivative-free optimiser for non-linear
non-convex problems in $\mathbb{R}^n$. It maintains a multivariate Gaussian
search distribution $\mathcal{N}(\boldsymbol{m}, \sigma^2 \boldsymbol{C})$
over the parameter space and adapts the mean $\boldsymbol{m} \in
\mathbb{R}^n$, the global step-size $\sigma > 0$, and the covariance matrix
$\boldsymbol{C} \in \mathbb{S}^n_{++}$ from the ranks of evaluated
offspring. The strategy is invariant under monotonic transformations of the
objective and under affine transformations of the search space, which
together explain its empirical robustness across heterogeneous problems.

This section builds the algorithm in three stages. §3.a develops the
mathematical preliminaries, derives the canonical update rules from
maximum-likelihood and natural-gradient principles, and establishes the
invariance properties. §3.b states the canonical CMA-ES algorithm in full,
including the rank-one update, the rank-$\mu$ update, the active negative-
weight extension, the evolution paths, and cumulative step-size adaptation.
§3.c describes the `cma` Python package (version 4.4.4) and the ask/tell
interface the design generator will hook into.

### a. Mathematical Background

#### a.1 Notational preliminaries

Throughout, $n$ is the search-space dimension, $k$ is the generation
index, $\lambda$ is the population size per generation, and $\mu$ is the
number of selected parents. Vectors are bold lowercase
($\boldsymbol{x}, \boldsymbol{m}, \boldsymbol{z}$); matrices are bold
uppercase ($\boldsymbol{C}, \boldsymbol{B}, \boldsymbol{D}$);
$\mathbb{S}^n_{++}$ denotes the cone of $n \times n$ symmetric
positive-definite matrices.

*Multivariate normal.* A random vector $\boldsymbol{x} \in \mathbb{R}^n$ is
$\mathcal{N}(\boldsymbol{m}, \boldsymbol{C})$ with density

$$
p(\boldsymbol{x}) \;=\; (2\pi)^{-n/2} \, |\boldsymbol{C}|^{-1/2} \, \exp\!\left[-\tfrac{1}{2} (\boldsymbol{x} - \boldsymbol{m})^{\top} \boldsymbol{C}^{-1} (\boldsymbol{x} - \boldsymbol{m})\right].
$$

*Eigendecomposition and SPD square root.* Because $\boldsymbol{C} \succ 0$
and symmetric, $\boldsymbol{C} = \boldsymbol{B} \boldsymbol{D}^2
\boldsymbol{B}^{\top}$, where $\boldsymbol{B}$ is orthogonal (eigenvectors)
and $\boldsymbol{D}$ is diagonal with positive square-roots of the
eigenvalues. Then $\boldsymbol{C}^{1/2} = \boldsymbol{B} \boldsymbol{D}
\boldsymbol{B}^{\top}$ and $\boldsymbol{C}^{-1/2} = \boldsymbol{B}
\boldsymbol{D}^{-1} \boldsymbol{B}^{\top}$. A sample
$\boldsymbol{x} = \boldsymbol{m} + \sigma \, \boldsymbol{B} \boldsymbol{D}
\, \boldsymbol{z}$ with $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0},
\boldsymbol{I})$ obeys $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{m},
\sigma^2 \boldsymbol{C})$. The eigendecomposition is the dominant
per-generation cost of CMA-ES: $\mathcal{O}(n^3)$ amortised across
$\mathcal{O}(n / 10)$ generations.

*Mahalanobis distance.* The Mahalanobis distance under $\boldsymbol{C}$ is

$$
d_{\boldsymbol{C}}(\boldsymbol{x}, \boldsymbol{m}) \;=\; \left\| \boldsymbol{C}^{-1/2} (\boldsymbol{x} - \boldsymbol{m}) \right\|.
$$

Multiplying by $\boldsymbol{C}^{-1/2}$ *whitens* $\boldsymbol{x}$ so that
the resulting vector has identity covariance; this whitening is precisely
what allows the conjugate evolution path $\boldsymbol{p}_\sigma$ to be
compared with the unit-isotropic reference distribution.

*Weighted recombination and effective selection mass.* Given non-negative
weights $w_1, \ldots, w_\mu$ summing to one, the *variance-effective
selection mass* is

$$
\mu_{\mathrm{eff}} \;=\; \frac{1}{\sum_{i=1}^{\mu} w_i^2}, \qquad 1 \le \mu_{\mathrm{eff}} \le \mu.
$$

It is the effective sample size of the weighted mean: for equal weights
$w_i = 1/\mu$ one has $\mu_{\mathrm{eff}} = \mu$; for the default
logarithmic weights $w_i \propto \ln((\mu+1)) - \ln i$ one has
$\mu_{\mathrm{eff}} \approx \lambda / 4$. The square root
$\sqrt{\mu_{\mathrm{eff}}}$ is the multiplicative variance-reduction factor
of the weighted mean relative to a single sample, and reappears as a
normaliser in the evolution-path updates.

*Expected norm of $\mathcal{N}(\mathbf{0}, \boldsymbol{I})$.* The norm of a
standard $n$-D normal follows a chi distribution with $n$ degrees of
freedom, with mean

$$
\mathbb{E} \|\mathcal{N}(\mathbf{0}, \boldsymbol{I})\| \;=\; \sqrt{2} \,\frac{\Gamma((n+1)/2)}{\Gamma(n/2)} \;\approx\; \sqrt{n}\left(1 - \frac{1}{4n} + \frac{1}{21 n^2}\right).
$$

This is the reference length that cumulative step-size adaptation drives
$\|\boldsymbol{p}_\sigma\|$ towards under the null hypothesis of random
selection.

#### a.2 From hill-climbing to the (1+1)-ES

Consider the continuous minimisation problem $\min_{\boldsymbol{x} \in
\mathbb{R}^n} f(\boldsymbol{x})$ with $f$ available only as a black box. A
naïve search starts from $\boldsymbol{x}_0$ and at each step proposes a
candidate $\boldsymbol{x}' = \boldsymbol{x}_k + \boldsymbol{z}$ with
$\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \boldsymbol{I})$; the
candidate replaces $\boldsymbol{x}_k$ iff $f(\boldsymbol{x}') <
f(\boldsymbol{x}_k)$. Rechenberg's analysis on the sphere model
$f(\boldsymbol{x}) = \|\boldsymbol{x}\|^2$ identifies a near-optimal
acceptance probability of $\approx 0.27$ that translates to the *one-fifth
success rule*: multiply $\sigma$ by $\exp(+ \alpha)$ when the success rate
over a moving window exceeds $1/5$, by $\exp(- \alpha / 4)$ otherwise. This
is the earliest instance of step-size adaptation and the conceptual
ancestor of CSA.

#### a.3 The $(\mu/\mu_W, \lambda)$-ES and weighted recombination

Increasing the population from one to $\lambda$ offspring sampled in
parallel turns the search into a *population-based ES*. At each generation
$k$, the $\lambda$ candidates

$$
\boldsymbol{x}^{(i)}_k \;=\; \boldsymbol{m}_k + \sigma_k \boldsymbol{B}_k \boldsymbol{D}_k \boldsymbol{z}^{(i)}_k, \quad \boldsymbol{z}^{(i)}_k \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})
$$

are drawn, ranked by $f$, and the $\mu$ best are recombined with positive
weights to form

$$
\boldsymbol{m}_{k+1} \;=\; \sum_{i=1}^{\mu} w_i \, \boldsymbol{x}^{(i:\lambda)}_k,
$$

where $i\!:\!\lambda$ denotes the $i$-th best of $\lambda$. The standard
weights are logarithmic:

$$
w_i \propto \ln(\mu + 1) - \ln i, \qquad \sum_{i=1}^{\mu} w_i = 1,
$$

normalised so that $\sum w_i = 1$. The use of *ranks* rather than raw
fitness values is what makes the resulting algorithm invariant under
monotone re-scalings of $f$; the use of *positive weights* rather than
top-$\mu$ truncation preserves more variance information from the best few
individuals.

#### a.4 Why a covariance matrix? The inverse-Hessian correspondence

If the level sets of $f$ are isotropic, $\boldsymbol{C} = \boldsymbol{I}$
is optimal — but on real engineering problems with heterogeneous variables
this is essentially never the case. Consider the local quadratic
approximation $f(\boldsymbol{x}) \approx \tfrac{1}{2} (\boldsymbol{x} -
\boldsymbol{x}^{\star})^{\top} \boldsymbol{H} (\boldsymbol{x} -
\boldsymbol{x}^{\star})$. The expected log-fitness gain of a sample
$\boldsymbol{m} + \boldsymbol{u}$ with $\boldsymbol{u} \sim
\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ is maximised, subject to a
trace constraint $\mathrm{tr}(\boldsymbol{\Sigma}) = \mathrm{const}$, by
$\boldsymbol{\Sigma} \propto \boldsymbol{H}^{-1}$. The level sets of the
squared Mahalanobis norm under $\boldsymbol{H}^{-1}$ then coincide with the
level sets of $f$ and the sampling distribution becomes locally
*sphere-equivalent* (isotropic in the natural coordinates of $f$).

CMA-ES does not have access to $\boldsymbol{H}$, but the *successful*
selection steps $(\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m}_k) /
\sigma_k$ are samples whose empirical second moment, accumulated across
generations, approximates $\boldsymbol{H}^{-1}$ up to a scalar. Hansen and
Auger have shown that on convex-quadratics $\boldsymbol{C}$ converges (up
to a global scale absorbed by $\sigma$) to $\boldsymbol{H}^{-1}$, so CMA-ES
implements a *variable-metric* search analogous to quasi-Newton BFGS but
without ever computing a gradient or Hessian [Hansen 2016 §5; Akimoto and
Hansen 2024].

#### a.5 Maximum-likelihood derivation of the mean and rank-$\mu$ updates

A cleaner derivation of the canonical updates treats CMA-ES as an
*iterated maximum-likelihood estimator* of the sampling distribution.
Given selected offspring $\{\boldsymbol{x}^{(i:\lambda)}_k\}_{i=1}^{\mu}$
with weights $w_i$, the weighted log-likelihood under $\mathcal{N}(
\boldsymbol{m}, \boldsymbol{C})$ is

$$
\mathcal{L}(\boldsymbol{m}, \boldsymbol{C}) \;=\; -\tfrac{1}{2} \sum_{i=1}^{\mu} w_i \left[ \ln |\boldsymbol{C}| + (\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m})^{\top} \boldsymbol{C}^{-1} (\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m}) \right].
$$

Setting $\nabla_{\boldsymbol{m}} \mathcal{L} = 0$ yields

$$
\widehat{\boldsymbol{m}}_{\mathrm{ML}} \;=\; \sum_{i=1}^{\mu} w_i \, \boldsymbol{x}^{(i:\lambda)}_k,
$$

which is precisely the mean update of §3.b.2 with learning rate $c_m = 1$.
Setting $\nabla_{\boldsymbol{C}} \mathcal{L} = 0$ yields the weighted
scatter matrix

$$
\widehat{\boldsymbol{C}}_{\mathrm{ML}} \;=\; \sum_{i=1}^{\mu} w_i \, \boldsymbol{y}^{(i:\lambda)}_k \, (\boldsymbol{y}^{(i:\lambda)}_k)^{\top}, \qquad \boldsymbol{y}^{(i:\lambda)}_k = (\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m}_k) / \sigma_k,
$$

the *rank-$\mu$ estimator* of $\boldsymbol{C}$. Equation (29) of Hansen
[2016] is the *exponentially smoothed* version with finite learning rate
$c_\mu$, anchoring towards the prior $\boldsymbol{C}_k$:

$$
\boldsymbol{C}^{\mathrm{rank}\text{-}\mu}_{k+1} \;=\; (1 - c_\mu) \boldsymbol{C}_k + c_\mu \sum_{i=1}^{\mu} w_i \, \boldsymbol{y}^{(i:\lambda)}_k \, (\boldsymbol{y}^{(i:\lambda)}_k)^{\top}.
$$

Two practical issues remain. First, $\mu$ is typically much smaller than
$n$, so $\widehat{\boldsymbol{C}}_{\mathrm{ML}}$ has rank at most $\mu$
and a single generation does not yield a usable full-rank covariance
estimate. Second, the maximum-likelihood estimator ignores the *direction*
of consecutive steps: if successive selection steps consistently point in
the same direction, that direction is more informative than any single
within-generation sample suggests. The next subsection addresses both
issues via the evolution path and rank-one update.

#### a.6 Natural-gradient and information-geometric view

Akimoto et al. [2010, 2012] and Ollivier et al. [2017] showed that
CMA-ES is an approximate *natural-gradient ascent* on the expected utility

$$
J(\boldsymbol{\theta}) \;=\; \mathbb{E}_{\boldsymbol{x} \sim p_{\boldsymbol{\theta}}}\!\left[ U(f(\boldsymbol{x})) \right]
$$

with $\boldsymbol{\theta} = (\boldsymbol{m}, \boldsymbol{C})$ and
$U$ a monotone rank-based *utility transform*. The Euclidean gradients of
$\log p_{\boldsymbol{\theta}}$ for a Gaussian are

$$
\nabla_{\boldsymbol{m}} \log p_{\boldsymbol{\theta}} \;=\; \boldsymbol{C}^{-1} (\boldsymbol{x} - \boldsymbol{m}), \qquad \nabla_{\boldsymbol{C}} \log p_{\boldsymbol{\theta}} \;=\; \tfrac{1}{2} \boldsymbol{C}^{-1} \left[ (\boldsymbol{x} - \boldsymbol{m})(\boldsymbol{x} - \boldsymbol{m})^{\top} - \boldsymbol{C} \right] \boldsymbol{C}^{-1}.
$$

Pre-conditioning these by $\boldsymbol{F}^{-1}$ — the inverse Fisher
information of the Gaussian family — yields the *natural* gradient. For
the mean,

$$
\widetilde{\nabla}_{\boldsymbol{m}} J \;\propto\; \sum_{i=1}^{\mu} w_i (\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m}_k),
$$

which is exactly the canonical mean update of §3.b.2. For the covariance,

$$
\widetilde{\nabla}_{\boldsymbol{C}} J \;\propto\; \sum_{i=1}^{\mu} w_i \left[ (\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m}_k)(\boldsymbol{x}^{(i:\lambda)}_k - \boldsymbol{m}_k)^{\top} - \sigma_k^2 \boldsymbol{C}_k \right],
$$

which is exactly the rank-$\mu$ update (modulo a smoothing learning rate
and the $\sigma_k^2$ factor absorbed into $\boldsymbol{y}^{(i:\lambda)}_k$).
The rank-one update is, in the MAP-IGO derivation of Akimoto and Hansen
[2024], a low-rank prior contribution arising from a concentrated prior on
$\boldsymbol{m}$ aligned with the evolution path. The natural-gradient
formalism explains the algorithm's *parameter-free* character: all
hyperparameters reduce to closed-form functions of $n$ and
$\mu_{\mathrm{eff}}$ via the Fisher metric, with no per-problem tuning.

#### a.7 Invariance properties

CMA-ES enjoys two invariances that are unmatched among continuous black-box
optimisers and that account for its empirical robustness across
heterogeneous problems.

- **Monotone-in-$f$ invariance.** Selection uses only the ranks of the
  offspring, so CMA-ES on $f$ and on $\phi \circ f$ for any strictly
  increasing $\phi$ produces an identical sequence of distributions. This
  matters in practice because the inner-loop fitness for a co-design
  problem (mean episode return) may be heavy-tailed, sign-changing, or
  reward-shaped without rendering the outer loop unstable.
- **Affine invariance in $\boldsymbol{x}$.** Under the change of variables
  $\boldsymbol{x} \mapsto \boldsymbol{A} \boldsymbol{x} + \boldsymbol{b}$
  with $\boldsymbol{A}$ non-singular, the algorithm produces statistically
  identical iterates *if* the initial distribution
  $(\boldsymbol{m}_0, \sigma_0 \boldsymbol{C}_0^{1/2})$ is transformed by
  the same $(\boldsymbol{A}, \boldsymbol{b})$. Equivalently, CMA-ES is
  *adaptively learning* a linear change of coordinates such that the
  rotated problem becomes sphere-like; the parameterisation choice
  (unit-cube encoding vs. physical-unit encoding) does not affect the
  trajectory in the limit.

These two invariances are the formal counterpart of the practitioner's
intuition that CMA-ES "self-tunes" to the conditioning of the problem.

### b. CMA-ES Formulation

This sub-section states the canonical $(\mu/\mu_W, \lambda)$-CMA-ES with
active negative weights and the default recommendations from Hansen [2016].
Equation numbers refer to that tutorial throughout. Let $g$ be the
generation index, $\boldsymbol{m}^{(g)} \in \mathbb{R}^n$ the distribution
mean, $\sigma^{(g)} > 0$ the step-size, $\boldsymbol{C}^{(g)} \in
\mathbb{S}^n_{++}$ the covariance matrix, and $\boldsymbol{p}^{(g)}_\sigma,
\boldsymbol{p}^{(g)}_c \in \mathbb{R}^n$ the two evolution paths.

#### b.1 Sampling — Hansen 2016 Eq. (5)

Each generation samples $\lambda$ offspring

$$
\boldsymbol{x}^{(i)}_{g+1} \;\sim\; \mathcal{N}\!\left(\boldsymbol{m}^{(g)}, \, (\sigma^{(g)})^2 \boldsymbol{C}^{(g)}\right), \quad i = 1, \ldots, \lambda,
$$

implemented as $\boldsymbol{x}^{(i)}_{g+1} = \boldsymbol{m}^{(g)} +
\sigma^{(g)} \boldsymbol{B}^{(g)} \boldsymbol{D}^{(g)}
\boldsymbol{z}^{(i)}_{g+1}$ with $\boldsymbol{z}^{(i)}_{g+1} \sim
\mathcal{N}(\mathbf{0}, \boldsymbol{I})$. Sorting the offspring by fitness
yields the order statistics $\boldsymbol{x}^{(1:\lambda)}_{g+1}, \ldots,
\boldsymbol{x}^{(\lambda:\lambda)}_{g+1}$.

#### b.2 Mean update — Eq. (9)

The new mean is the weighted recombination of the best $\mu$ offspring:

$$
\boldsymbol{m}^{(g+1)} \;=\; \boldsymbol{m}^{(g)} + c_m \sum_{i=1}^{\mu} w_i \, \left(\boldsymbol{x}^{(i:\lambda)}_{g+1} - \boldsymbol{m}^{(g)}\right),
$$

with learning rate $c_m = 1$ by default. A learning rate $c_m \le 1$
slows the mean update — useful for stabilising noisy optimisation, but
not required for the present application.

#### b.3 Evolution paths — Eq. (31) and Eq. (24)

Two low-pass-filtered sums of the selection step $(\boldsymbol{m}^{(g+1)}
- \boldsymbol{m}^{(g)}) / \sigma^{(g)}$ are maintained.

*The conjugate evolution path* $\boldsymbol{p}_\sigma$ is updated in
$\boldsymbol{C}^{-1/2}$-space — the *whitened* coordinate system — so that
under the null hypothesis of random selection it has unit isotropic
covariance:

$$
\boldsymbol{p}^{(g+1)}_\sigma \;=\; (1 - c_\sigma) \boldsymbol{p}^{(g)}_\sigma \; + \; \sqrt{c_\sigma (2 - c_\sigma) \mu_{\mathrm{eff}}} \; (\boldsymbol{C}^{(g)})^{-1/2} \, \frac{\boldsymbol{m}^{(g+1)} - \boldsymbol{m}^{(g)}}{\sigma^{(g)}}.
$$

Pre-multiplication by $(\boldsymbol{C}^{(g)})^{-1/2}$ whitens the
displacement so that $\|\boldsymbol{p}_\sigma\|$ can be compared with
$\mathbb{E}\|\mathcal{N}(\mathbf{0}, \boldsymbol{I})\|$ regardless of
$\boldsymbol{C}^{(g)}$'s anisotropy; this is what makes the CSA update of
§3.b.6 dimensionally meaningful. The normalising factor
$\sqrt{c_\sigma (2 - c_\sigma) \mu_{\mathrm{eff}}}$ is chosen so that, if
successive mean-shifts were independent of selection, the path's
stationary variance equals identity.

*The anisotropic evolution path* $\boldsymbol{p}_c$ accumulates the raw
(un-whitened) selection step, gated by a Heaviside indicator $h_\sigma$
that suppresses the update when $\|\boldsymbol{p}_\sigma\|$ is unusually
large:

$$
\boldsymbol{p}^{(g+1)}_c \;=\; (1 - c_c) \boldsymbol{p}^{(g)}_c \; + \; h_\sigma \sqrt{c_c (2 - c_c) \mu_{\mathrm{eff}}} \, \frac{\boldsymbol{m}^{(g+1)} - \boldsymbol{m}^{(g)}}{\sigma^{(g)}},
$$

with

$$
h_\sigma \;=\; \mathbf{1}\!\left[\frac{\|\boldsymbol{p}^{(g+1)}_\sigma\|}{\sqrt{1 - (1 - c_\sigma)^{2(g+1)}}} \;<\; \left(1.4 + \frac{2}{n+1}\right) \mathbb{E}\|\mathcal{N}(\mathbf{0}, \boldsymbol{I})\|\right].
$$

*Why the Heaviside?* When $\sigma$ is growing rapidly (large
$\|\boldsymbol{p}_\sigma\|$), several consecutive mean shifts add up to a
long $\boldsymbol{p}_c$ that mostly reflects step-size growth rather than
a genuine repeated direction. Multiplying $\boldsymbol{p}_c
\boldsymbol{p}_c^{\top}$ into $\boldsymbol{C}$ would then elongate
$\boldsymbol{C}$ along the search trajectory and cause runaway growth. The
Heaviside switch freezes $\boldsymbol{p}_c$ for that generation; the
divisor $\sqrt{1 - (1 - c_\sigma)^{2(g+1)}}$ corrects for the path's
initial-transient bias from $\boldsymbol{p}_\sigma^{(0)} = \mathbf{0}$.

#### b.4 Rank-one and rank-$\mu$ covariance updates — Eq. (28) and Eq. (29)

The covariance matrix is updated by combining a *rank-one* contribution
from $\boldsymbol{p}_c$ (capturing inter-generational correlation) and a
*rank-$\mu$* contribution from the current generation's selection scatter
(capturing within-generation curvature). Without the active negative-weight
extension of §3.b.5, the update reads

$$
\boldsymbol{C}^{(g+1)} \;=\; \underbrace{(1 - c_1 - c_\mu + c_s) \boldsymbol{C}^{(g)}}_{\text{decay}} \; + \; \underbrace{c_1 \boldsymbol{p}^{(g+1)}_c (\boldsymbol{p}^{(g+1)}_c)^{\top}}_{\text{rank-one}} \; + \; \underbrace{c_\mu \sum_{i=1}^{\mu} w_i \, \boldsymbol{y}^{(i:\lambda)}_{g+1} (\boldsymbol{y}^{(i:\lambda)}_{g+1})^{\top}}_{\text{rank-}\mu},
$$

where $\boldsymbol{y}^{(i:\lambda)}_{g+1} = (\boldsymbol{x}^{(i:\lambda)}_{g+1}
- \boldsymbol{m}^{(g)}) / \sigma^{(g)}$ and $c_s = (1 - h_\sigma^2) \, c_1
\, c_c (2 - c_c)$ is a small *variance compensation* that keeps the
expected trace of $\boldsymbol{C}$ invariant when $h_\sigma = 0$.

*Why the rank-one term captures inter-generational correlation.* The
exponentially-smoothed path $\boldsymbol{p}_c$ is the *signed* mean-shift
direction averaged over the last $\sim 1/c_c$ generations. If shifts are
temporally correlated (the algorithm is walking persistently in roughly
the same direction in $\boldsymbol{C}$-coordinates), $\boldsymbol{p}_c$
becomes long, and $\boldsymbol{p}_c \boldsymbol{p}_c^{\top}$ is a rank-one
SPD matrix *aligned with that persistent direction*. Adding it to
$\boldsymbol{C}$ stretches the sampling ellipsoid along the trajectory,
exactly the direction in which useful steps have been observed. Sign
cancellation between $+\boldsymbol{y}$ and $-\boldsymbol{y}$ steps is
the reason CMA-ES uses the *signed* mean-shift rather than per-individual
outer products: $\boldsymbol{p}_c \boldsymbol{p}_c^{\top}$ captures
coherent direction, not just magnitude.

*Why both updates are needed.* In one generation,
$\sum_{i=1}^{\mu} w_i \boldsymbol{y}_i \boldsymbol{y}_i^{\top}$ has rank
at most $\mu < n$. Any single-generation update therefore cannot encode
all $n(n+1)/2$ degrees of freedom of $\boldsymbol{C}$; the rank-one
component supplies the missing information by accumulating the mean-shift
direction across generations, providing a consistent estimator of the
leading axis of useful steps and breaking the within-generation rank
deficiency.

#### b.5 Active CMA-ES with negative weights — Hansen 2016 §4 Eq. (47)

The current default (`CMA_active=True` in pycma) extends the update with
*negative* weights on the worst $\lambda - \mu$ individuals, contracting
$\boldsymbol{C}$ in unfavourable directions. Let $w_1 \ge \cdots \ge w_\mu
> 0 \ge w_{\mu+1} \ge \cdots \ge w_\lambda$ with the positive weights
summing to $+1$ and the negative weights rescaled to preserve
positive-definiteness:

$$
w_i^{\circ} \;=\; w_i \cdot \min\!\left(1, \; \frac{n}{\|\boldsymbol{C}^{-1/2} \boldsymbol{y}^{(i:\lambda)}\|^2}\right) \quad \text{for } w_i < 0, \qquad w_i^{\circ} = w_i \text{ otherwise.}
$$

The combined update is

$$
\boldsymbol{C}^{(g+1)} \;=\; \left(1 + c_1 \delta(h_\sigma) - c_1 - c_\mu \sum_{i=1}^{\lambda} w_i\right) \boldsymbol{C}^{(g)} \; + \; c_1 \boldsymbol{p}^{(g+1)}_c (\boldsymbol{p}^{(g+1)}_c)^{\top} \; + \; c_\mu \sum_{i=1}^{\lambda} w_i^{\circ} \, \boldsymbol{y}^{(i:\lambda)}_{g+1} (\boldsymbol{y}^{(i:\lambda)}_{g+1})^{\top},
$$

with $\delta(h_\sigma) = (1 - h_\sigma) \, c_c (2 - c_c)$. Active CMA-ES
yields a 1.5×–2× speed-up on most ill-conditioned and large-$n$ problems
[Jastrebski and Arnold 2006; Hansen and Ros 2010] and is the recommended
default. Its only practical caveat is that the negative-weight rescaling
$w_i^{\circ}$ requires forming $\boldsymbol{C}^{-1/2}
\boldsymbol{y}^{(i:\lambda)}$, which adds a single triangular solve per
negative-weighted individual; the cost is negligible compared to the
inner-loop RL evaluation.

#### b.6 Cumulative step-size adaptation — Eq. (37)

The global step-size is updated by comparing the length of
$\boldsymbol{p}_\sigma$ with its expected length under random selection:

$$
\sigma^{(g+1)} \;=\; \sigma^{(g)} \, \exp\!\left( \frac{c_\sigma}{d_\sigma} \left[ \frac{\|\boldsymbol{p}^{(g+1)}_\sigma\|}{\mathbb{E}\|\mathcal{N}(\mathbf{0}, \boldsymbol{I})\|} - 1 \right] \right).
$$

*Intuition.* Under random selection, $\boldsymbol{p}_\sigma \sim
\mathcal{N}(\mathbf{0}, \boldsymbol{I})$ and $\mathbb{E}\|\boldsymbol{p}_\sigma\|
= \mathbb{E}\|\mathcal{N}(\mathbf{0}, \boldsymbol{I})\|$. Successive
selection steps that are *positively correlated* (the search keeps walking
in the same direction) lengthen the path beyond the expectation and
$\sigma$ grows; *anti-correlated* steps shorten it and $\sigma$ shrinks.
The damping $d_\sigma$ bounds the multiplicative changes per generation;
$c_\sigma$ sets the path's memory length to $\sim 1/c_\sigma$ generations.

#### b.7 Default hyperparameter recommendations — Hansen 2016 Table 1

For dimension $n$ and population size $\lambda$, the canonical defaults are
those in Table 2 below; pycma 4.4.4 uses these unless overridden. The
critical observation is that *every* hyperparameter except
$(\boldsymbol{m}_0, \sigma_0, \lambda)$ is a closed-form function of $n$
and $\mu_{\mathrm{eff}}$ — there is essentially nothing to tune.

| Quantity | Default value |
|----------|---------------|
| $\lambda$ | $4 + \lfloor 3 \ln n \rfloor$ |
| $\mu$ | $\lfloor \lambda / 2 \rfloor$ |
| $w'_i$ | $\ln((\lambda + 1) / 2) - \ln i$ for $i = 1, \ldots, \lambda$ |
| $w_i$ | $w'_i$ rescaled so $\sum_{i \le \mu} w_i = 1$ (negative weights for $i > \mu$ in active CMA) |
| $\mu_{\mathrm{eff}}$ | $1 / \sum_{i=1}^{\mu} w_i^2 \;\approx\; \lambda / 4$ |
| $c_m$ | $1$ |
| $c_\sigma$ | $(\mu_{\mathrm{eff}} + 2) / (n + \mu_{\mathrm{eff}} + 5)$ |
| $d_\sigma$ | $1 + 2 \max(0, \sqrt{(\mu_{\mathrm{eff}} - 1) / (n + 1)} - 1) + c_\sigma$ |
| $c_c$ | $(4 + \mu_{\mathrm{eff}} / n) / (n + 4 + 2 \mu_{\mathrm{eff}} / n)$ |
| $c_1$ | $2 / ((n + 1.3)^2 + \mu_{\mathrm{eff}})$ |
| $c_\mu$ | $\min(1 - c_1, \; 2 (\mu_{\mathrm{eff}} - 2 + 1 / \mu_{\mathrm{eff}}) / ((n + 2)^2 + \mu_{\mathrm{eff}}))$ |

The two parameters that *do* need user-supplied values in any application
are the initial mean $\boldsymbol{m}_0$ and the initial step-size
$\sigma_0$. The former should reflect the best prior guess of the optimum;
the latter should be approximately one quarter of the expected search
range per coordinate, so that $\pm 2\sigma_0$ covers the expected basin
of the optimum. For a unit-hypercube encoding $[0, 1]^n$ as recommended
for the design generator in §4, the conventional choice is
$\sigma_0 \in [0.2, 0.3]$. Population size $\lambda$ can be increased
above the default to improve global-search behaviour at the cost of more
evaluations per generation; for the present application it is set to
`num_individuals` (matched to the runner's environment capacity).

#### b.8 Algorithmic skeleton

```text
Initialise m_0, σ_0, C_0 = I, p^σ_0 = p^c_0 = 0, generation g = 0
while not terminate:
    # Sample
    for i in 1..λ:
        z_i ~ N(0, I)
        x_i = m_g + σ_g · B_g · D_g · z_i
        f_i = fitness(x_i)
    # Rank
    sort offspring by fitness ascending → x_{1:λ}, …, x_{λ:λ}
    # Mean update
    m_{g+1} = m_g + c_m · Σ_{i=1..μ} w_i · (x_{i:λ} − m_g)
    # Evolution paths
    p^σ_{g+1} = (1 − c_σ) p^σ_g + √(c_σ(2−c_σ)μ_eff) · C_g^{−1/2} · (m_{g+1} − m_g) / σ_g
    h_σ = 𝟙[‖p^σ_{g+1}‖ / √(1 − (1−c_σ)^{2(g+1)}) < (1.4 + 2/(n+1)) · E‖N(0,I)‖]
    p^c_{g+1} = (1 − c_c) p^c_g + h_σ · √(c_c(2−c_c)μ_eff) · (m_{g+1} − m_g) / σ_g
    # Covariance update (active CMA: positive + negative weights)
    C_{g+1} = (1 + c_1·δ(h_σ) − c_1 − c_μ·Σ w_i) · C_g
              + c_1 · p^c_{g+1} (p^c_{g+1})^T
              + c_μ · Σ_{i=1..λ} w_i^° · y_{i:λ} y_{i:λ}^T
    # Step-size update
    σ_{g+1} = σ_g · exp((c_σ / d_σ) · (‖p^σ_{g+1}‖ / E‖N(0,I)‖ − 1))
    g ← g + 1
```

#### b.9 Restart strategies — IPOP and BIPOP

For multimodal landscapes, a single CMA-ES run can converge to a local
optimum from which it cannot escape. The standard remedy is *restart with
an increased population size*:

- **IPOP-CMA-ES** [Auger and Hansen 2005] runs CMA-ES until a termination
  criterion (`tolfun`, `tolx`, stalled $\sigma$) fires, then restarts with
  $\lambda \leftarrow 2 \lambda$. The geometric increase covers a wide
  range of basin scales and won the CEC-2005 benchmark.
- **BIPOP-CMA-ES** [Hansen 2009] runs two interleaved regimes — large
  IPOP-style restarts and small random-budget restarts — and is the
  current best-performing variant on the BBOB testbed for weakly-
  structured multimodal functions.

For the present co-design application, restart is unlikely to be the
right control surface because each generation costs a full inner-loop
training segment. The implementation in §4 therefore omits restarts in
the initial release; a follow-up may add IPOP if convergence stagnation
is observed in practice.

### c. API Structure and Sample Usage

The `cma` Python package, version 4.4.4 (Hansen, Akimoto and Baudis
[2024]), implements the algorithm of §3.b through the
`CMAEvolutionStrategy` class with an *ask/tell* iteration interface. The
interface decouples candidate generation from fitness evaluation, which
is essential for the present application: the *generation* of designs
must happen synchronously inside `DesignGeneratorBase.generate_population`,
but the *evaluation* of fitness happens asynchronously over
`ea_update_interval` PPO iterations inside `CoptOnPolicyRunner.learn`.

#### c.1 Construction

```python
cma.CMAEvolutionStrategy(x0, sigma0, inopts=None)
```

- `x0`: initial mean (one-dimensional NumPy array of length $n$). For
  unit-cube encoded problems with no informative prior, `np.full(n, 0.5)`
  is the bias-free default.
- `sigma0`: initial step-size (positive scalar). Recommended
  $\approx 1/4$ of the search range per coordinate; for the
  unit-hypercube encoding used in §4, $\sigma_0 \in [0.2, 0.3]$.
- `inopts` / `options`: a dict or `cma.CMAOptions`; commonly used keys are
  documented in §c.3.

#### c.2 Ask / tell loop

The canonical usage pattern, on which §4.b is built, is:

```python
import cma
import numpy as np

n = 19
x0 = np.full(n, 0.5)
sigma0 = 0.2
opts = cma.CMAOptions()
opts.set({
    "popsize": 16,
    "bounds": [np.zeros(n), np.ones(n)],
    "seed": 42,
    "verb_disp": 0,
})

es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

while not es.stop():
    solutions = es.ask()              # list[np.ndarray] of length popsize
    costs = [objective(s) for s in solutions]
    es.tell(solutions, costs)         # update m, σ, C, evolution paths
```

`es.ask()` returns `popsize` candidate vectors; `es.tell(solutions, costs)`
consumes the same list of solutions together with corresponding
*minimisation* costs (smaller is better) and performs the mean, path,
covariance, and step-size updates of §3.b. State between `ask` and `tell`
is held entirely inside `es`; the caller is responsible only for evaluation.

The `tron1-rl-isaaclab-cozum/cmaes.py:489-547` reference uses this pattern
verbatim, with `popsize = 8` for a 13-dimensional 5-bar linkage co-design
problem and `multiprocessing.Pool.map` for fitness parallelisation. (One
quirk to be aware of: lines 475–476 of that file set `verb_disp` twice in
the same `opts.set` call; the second value silently overrides the first.
The design generator avoids this by passing a single, idempotent dict.)

#### c.3 Common options

The most relevant entries of `cma.CMAOptions()` are listed below. Values
listed as expressions are evaluated by pycma with `N` (the dimension) and
`popsize` in scope.

| Key | Default | Meaning |
|-----|---------|---------|
| `popsize` | `4 + floor(3 * log(N))` | $\lambda$ |
| `popsize_factor` | `1` | Multiplier on default popsize (used by IPOP) |
| `bounds` | `[None, None]` | Per-coordinate box constraints: `[lower, upper]` |
| `CMA_active` | `True` | Use the active negative-weight update of §3.b.5 |
| `CMA_diagonal` | `100 * N / popsize**0.5` | Initial iterations to run as sep-CMA-ES |
| `seed` | `time` | RNG seed; integer for reproducibility |
| `maxiter` | $\sim 100 + 150 (n+3)^2 / \sqrt{\lambda}$ | Hard cap on generations |
| `maxfevals` | `inf` | Function-evaluation cap |
| `tolfun` | `1e-11` | Stop on small $\Delta f$ over `tolfunhist` |
| `tolx` | `1e-11` | Stop on small $\Delta \boldsymbol{x}$ and $\Delta \sigma$ |
| `verb_disp` | `100` | Console-print interval; `0` to silence |
| `integer_variables` | `[]` | Indices treated as integers (rounded inside `ask`) |
| `verb_filenameprefix` | `outcmaes/` | Logger output directory / prefix |

For the design generator in §4, the relevant keys are `popsize`,
`bounds`, `seed`, and `verb_disp` (set to `0` because RSL-RL's
TensorBoard writer owns logging). The other defaults are left untouched.

#### c.4 State persistence and warm-starting

```python
data = es.pickle_dumps()                                # bytes
es2  = cma.CMAEvolutionStrategy.pickle_loads(data)
```

`save(filename)` / `load(filename)` are also supported. The serialised
state includes $\boldsymbol{m}_g$, $\sigma_g$, $\boldsymbol{C}_g$,
$\boldsymbol{p}^\sigma_g$, $\boldsymbol{p}^c_g$, the random state, and the
generation counter — everything needed to resume from a checkpoint. This
is the recommended persistence path for the design generator and is
covered in §4.b.11.

#### c.5 Solution injection

```python
es.inject([x_seed_1, x_seed_2])
```

External candidates are appended to the *next* `ask()` call (or replace
random samples beyond the popsize). pycma bounds the path contribution of
injected solutions so that a poor seed cannot blow up $\sigma$. The
mechanism is useful for seeding the first generation with the centroid of
prior best-known designs after a checkpoint resume; it is not required
for the initial implementation in §4.

#### c.6 Box constraints and bound transforms

`bounds = [lower, upper]` with the default `BoundPenalty` handler adds a
soft quadratic penalty to $f$ proportional to the unprojected distance
from the box; the internal Gaussian state operates on the *unconstrained*
search space. For tight box constraints — the situation in §4, where
*every* coordinate is bounded to $[0, 1]$ — a smoother alternative is
`cma.BoundDomainTransform`, a bijective $\tanh$-like map between
$\mathbb{R}^n$ and the box that preserves affine invariance away from the
boundary. Either handler is acceptable for the design generator; the
implementation in §4.b uses the default `bounds`-based handler for
simplicity, with the `BoundDomainTransform` upgrade noted as a follow-up.

#### c.7 Parallelisation patterns

For black-box objectives that fit one-per-process, the idiom in
`tron1-rl-isaaclab-cozum/cmaes.py` is canonical:

```python
pool = multiprocessing.Pool(processes=8)
while not es.stop():
    solutions = es.ask()
    costs = pool.map(get_cost, solutions)
    es.tell(solutions, costs)
```

For the design generator, parallelisation is performed by IsaacLab's
vectorised environments rather than by `multiprocessing.Pool`: each
individual is mapped to a contiguous subset of `num_envs` via the
round-robin assignment in `_assign_individuals_to_envs`. The `ask()` call
remains synchronous and cheap; the heavy fitness evaluation is
distributed across GPU envs.

## 4. CMA-ES Design Generator

This section connects the algorithm of §3 to the existing pipeline.
§4.a recapitulates the current implementation in `usd_generator.py` and
`copt_on_policy_runner.py`, focusing on the abstractions that the CMA-ES
generator must conform to. §4.b lays out the implementation plan: data
flow, encoding of the design vector, ask/tell hooks, refactoring strategy,
constraint handling, checkpointing, and validation milestones.

### a. The Existing Random Design Generator

The current implementation in
`tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py`
defines three primary classes — `Population`, `DesignGeneratorBase`,
`RandomPopulation`, `RandomDesignGenerator` — together with a utility
constants table (`DEFAULT_PARAM_RANGES`, `ACTUATOR_BASELINES`,
`JOINT_TO_ACTUATOR`, `ABAD_HIP_LINKS`, `IMU_LINK_NAME`) and a baseline-
extraction helper (`_actuator_baseline`). The runner-side contract lives
in
`tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`.

#### a.1 The `Population` abstraction

`Population` (`usd_generator.py:157`) is an abstract base class with two
methods:

- `get_usd_files() -> list[str]`: returns one filesystem path per
  individual; each path resolves to a USD asset converted from a
  perturbed URDF.
- `get_actuator_params() -> list[dict[str, dict]]`: returns one dict per
  individual; the outer dict is keyed by actuator group name
  (`"abad"`, `"hip"`, `"knee"`, `"ankle"`) and the inner dict carries
  scalar overrides for `IdentifiedActuator` tensor attributes
  (e.g. `effort_limit`, `friction_static`, `stiffness`, `damping`).

`RandomPopulation` (`usd_generator.py:214`) is the concrete
implementation: it stores the two lists in its constructor and returns
them on demand. The `Population` object is consumed by
`respawn_robots(env, population.get_usd_files())` and
`apply_actuator_params(env, population.get_actuator_params())` in
`co_optimisation.utils.respawn` — the runner does not interrogate the
population beyond these two getters.

#### a.2 The `DesignGeneratorBase` abstraction

`DesignGeneratorBase` (`usd_generator.py:182`) declares three methods:

- `generate_population(generation: int) -> Population`: produces a fresh
  population for the given outer-loop iteration.
- `update_with_fitness(population: Population, fitness: list[float]) -> None`:
  optionally updates internal state from per-individual fitness scores.
  The default implementation is a no-op (random search).
- `sample_batch() -> None`: a placeholder hook used by the runner before
  the first call to `generate_population`. The default implementation is
  a no-op.

The runner contract is enacted in
`CoptOnPolicyRunner._reload_morphology` (`copt_on_policy_runner.py:255`):

```text
if current_population is None:        # first generation
    design_generator.sample_batch()
else:                                 # subsequent generations
    design_generator.update_with_fitness(current_population, fitness)
current_population = design_generator.generate_population(generation)
respawn_robots(env, current_population.get_usd_files())
apply_actuator_params(env, current_population.get_actuator_params())
zero per-individual fitness accumulators
```

In other words, `sample_batch` is invoked exactly once at the start of
training and `update_with_fitness` once per generation thereafter,
*before* the call to `generate_population`. Any design generator that
maintains internal optimiser state (CMA-ES, Bayesian optimisation,
REINFORCE over the design parameters) must update that state inside
`update_with_fitness` so that the subsequent `generate_population` returns
candidates drawn from the updated distribution.

#### a.3 The `RandomDesignGenerator` implementation

`RandomDesignGenerator` (`usd_generator.py:230`) parameterises a design
by a vector of scalar scale factors over a base URDF. The scale-factor
vocabulary is defined in `DEFAULT_PARAM_RANGES` (`usd_generator.py:126`)
and covers nineteen parameters in five groups:

1. **Geometric scales** (4 keys): `thigh_length_scale`,
   `shank_length_scale`, `actuator_radius_scale`, `actuator_length_scale`.
2. **Link mass** (1 key): `link_mass_scale` — applied uniformly to all
   link masses and inertia diagonals except `limx_imu`.
3. **Per-joint limits** (2 keys): `joint_effort_scale`,
   `velocity_limit_scale`.
4. **Friction / saturation / armature** (4 keys): `friction_static_scale`,
   `friction_dynamic_scale`, `saturation_effort_scale`, `armature_scale`
   — applied post-respawn through `apply_actuator_params`.
5. **Per-group stiffness and damping** (8 keys): one stiffness and one
   damping scale per actuator group (`abad`, `hip`, `knee`, `ankle`),
   with ranges derived from the symmetric equivalents of the runtime
   domain-randomisation abs ranges (tightest half wins for asymmetric
   ranges).

For each individual, `_generate_individual(generation, idx)`
(`usd_generator.py:296`):

1. Seeds `numpy.random.default_rng` with `generation * 10000 + idx` to
   ensure determinism across reruns.
2. Samples scales uniformly from the per-parameter ranges
   (`_sample_scales`).
3. Parses the base URDF with `xml.etree.ElementTree` and applies
   geometric and mass scales to `<joint origin>`, `<link mass>`,
   `<link inertia>`, `<cylinder radius/length>`, and
   `<joint limit effort/velocity>` elements (skipping the `limx_imu`
   link), then writes the modified URDF under
   `{output_dir}/gen_{g:04d}/individual_{i:04d}.urdf`.
4. Converts the URDF to USD via
   `isaaclab.sim.converters.UrdfConverter` (with `merge_fixed_joints=
   False`, `fix_base=False`, `self_collision=False`,
   `collider_type="convex_hull"`, `force_usd_conversion=True`) and
   records the USD path.
5. Builds the actuator-overrides dict for each of the four actuator
   groups by multiplying each baseline scalar (read from the
   `TRON1_*_ACTUATOR_CFG` configs via `_actuator_baseline`) by the
   appropriate scale factor.

The output of `_generate_individual` is a `(usd_path, act_params)` tuple;
`generate_population` aggregates `num_individuals` such tuples into a
`RandomPopulation`. `update_with_fitness` and `sample_batch` are not
overridden — `RandomDesignGenerator` is a pure random-search baseline.

#### a.4 The runner-side contract: round-robin assignment and fitness aggregation

The assignment of environments to individuals is *round-robin*: env $i$ is
assigned individual $i \bmod N_\text{ind}$, computed once in
`_assign_individuals_to_envs` (`copt_on_policy_runner.py:299`) and stored
on `self._env_to_individual`. The same round-robin is used implicitly by
`respawn_robots`, which calls `spawn_multi_usd_file` with
`random_choice=False` so that env $i$ deterministically receives
`usd_paths[i \bmod N_\text{ind}]`. Per-individual fitness is the mean
episode return over all environments assigned to that individual
(`copt_on_policy_runner.py:288–297`); if an individual's environments
completed no episodes in the interval, its fitness defaults to $0.0$.

Several invariants of this contract bear on the CMA-ES implementation:

- `num_individuals` is passed *both* to the generator constructor and to
  the runner's `train_cfg["copt"]` dict; the runner does not assert
  these match, so the caller is responsible for keeping them in sync.
- The dimensionality of the design vector is fixed across generations:
  `param_ranges` is set at construction time and is not re-keyed.
- The population contract is minimal: only two getters are required, so a
  CMA-ES-specific population class is unnecessary and the implementation
  reuses `RandomPopulation`.
- `respawn_robots` runs `sim.stop`, prim deletion, `spawn_multi_usd_file`,
  and `sim.reset` in a single blocking call; per-generation respawn cost
  must therefore be amortised over a sufficiently long
  `ea_update_interval` (the default of 100 PPO iterations is typically
  more than enough; the current call site uses 1000).

### b. Implementation Plan

The CMA-ES design generator replaces the *uniform sampling step* of
`RandomDesignGenerator` with sampling from a CMA-ES search distribution
while preserving the rest of the URDF-mutation, USD-conversion, and
actuator-override pipeline. This subsection is organised around twelve
concrete topics: (1) requirements, (2) design vector encoding,
(3) CMA-ES lifecycle, (4) method mapping, (5) fitness sign and noise,
(6) refactoring `_apply_scales`, (7) data flow, (8) skeleton
implementation, (9) wiring into the runner, (10) constraint handling and
bound encoding, (11) logging / checkpointing / reproducibility,
(12) validation milestones, and a closing list of open questions.

#### b.1 Requirements

**R1. API conformance.** The CMA-ES generator must subclass
`DesignGeneratorBase` and return a `RandomPopulation` (or an identically-
typed subclass); no changes to `CoptOnPolicyRunner` or to the
`Population` / `RandomPopulation` interfaces are required.

**R2. Design vector encoding.** The design vector $\boldsymbol{x} \in
[0, 1]^n$ encodes one scale factor per key in `param_ranges`, normalised
into the unit hypercube. The dimensionality $n$ equals the number of keys
in the merged `DEFAULT_PARAM_RANGES`-plus-overrides dict (currently 19).
Unit-cube normalisation is essential: it removes the algorithm's
sensitivity to the heterogeneous physical scales of the underlying
parameters (lengths in $[0.85, 1.15]$, friction in $[0.70, 1.40]$, etc.)
and lets the algorithm's defaults — derived in §3.b.7 for sphere-like
problems — apply directly.

**R3. Deterministic mapping vector $\to$ URDF.** The mapping
$\boldsymbol{x} \mapsto (\text{usd path}, \text{actuator overrides})$
must be deterministic and reuse the existing URDF-mutation pipeline in
`_generate_individual`. This is achieved by factoring out a helper
`_apply_scales(scales: dict[str, float], generation: int, idx: int) ->
tuple[str, dict[str, dict]]` from `RandomDesignGenerator._generate_individual`
(the body from step ③ onwards) and calling it from both generators.

**R4. CMA-ES lifecycle inside the generator.** A single
`cma.CMAEvolutionStrategy` instance is constructed at generator
`__init__` time with `x0 = 0.5 * np.ones(n)`, `sigma0` from a constructor
argument (default `0.2`), `popsize = num_individuals`,
`bounds = [np.zeros(n), np.ones(n)]`, `seed` from a constructor
argument, and `verb_disp = 0`. The optimiser state is owned by
`self._es`; *no* internal `ask` is performed before `sample_batch` is
called.

**R5. Method mapping.**

- `sample_batch()`: pre-`ask` for the first generation. Stores the
  popsize candidate list in `self._pending_solutions`. Must be called
  exactly once, before the first `generate_population`.
- `generate_population(generation)`: consumes
  `self._pending_solutions`, denormalises each vector into a scales
  dict, calls `_apply_scales` to obtain `(usd_path, act_params)` per
  individual, stores `self._last_solutions = solutions` (so the next
  `update_with_fitness` can pass them back to CMA-ES), clears
  `self._pending_solutions`, and returns a `RandomPopulation`.
- `update_with_fitness(population, fitness)`: converts the list of mean
  episode returns into costs (`cost_i = -fitness_i` because CMA-ES
  minimises), calls `self._es.tell(self._last_solutions, costs)`, polls
  `self._es.stop()` for termination, and then immediately calls
  `self._es.ask()`, storing the result in `self._pending_solutions` for
  consumption by the next `generate_population`. The `tell → ask`
  sequence preserves the canonical CMA-ES loop while fitting the
  runner's interleaving order.

**R6. Fitness sign convention.** RSL-RL returns sums of per-step rewards;
higher is better. CMA-ES minimises. The generator negates the fitness in
`update_with_fitness` so the optimiser ascends the return surface. No
other sign conventions need to be considered because the runner already
returns a scalar mean per individual.

**R7. Termination and warm-start.** `es.stop()` returns a non-empty dict
when any CMA-ES termination criterion fires (`tolfun`, `tolx`,
`maxiter`, etc.). The generator polls `es.stop()` inside
`update_with_fitness`; if non-empty, it logs the reason and continues to
return the current incumbent as a degenerate population (all individuals
equal to `es.result.xfavorite`). For warm-starting, the constructor
optionally accepts `es_state_path` from which
`cma.CMAEvolutionStrategy.pickle_loads` reconstructs `self._es`.

**R8. Reproducibility.** The CMA-ES seed and the per-individual URDF-write
seed are independent: CMA-ES owns the *what to sample* randomness, while
`_apply_scales` is deterministic given the scales dict. Logging both
seeds in the run config is sufficient for full reproducibility.

**R9. Backwards compatibility.** Adding the CMA-ES generator must not
alter the behaviour of `RandomDesignGenerator` under default
configuration; only new constructor arguments and new top-level keys in
`train_cfg["copt"]` are introduced.

#### b.2 Design vector encoding and dimensionality

The design vector
$\boldsymbol{x} = (x_1, \ldots, x_n) \in [0, 1]^n$ encodes the scale
factors:

$$
x_i \;=\; \frac{s_i - \mathrm{lo}_i}{\mathrm{hi}_i - \mathrm{lo}_i}, \qquad s_i = \mathrm{lo}_i + x_i (\mathrm{hi}_i - \mathrm{lo}_i),
$$

with $(\mathrm{lo}_i, \mathrm{hi}_i)$ taken from `param_ranges[key_i]`.
The bijection $(s_i \leftrightarrow x_i)$ is implemented by
`_denormalise(x)`; the inverse (used only at the optional warm-start)
is `_normalise(scales)`.

The dimensionality $n$ is the cardinality of `param_ranges` at
construction time and is *frozen*. Adding or removing a key between
runs is supported (a different generator instance is constructed) but
checkpoint resume across different $n$ is not supported by pycma's
serialised state.

The ordering of the dimensions is the insertion order of
`param_ranges` (Python 3.7+ dict semantics), captured in
`self._param_keys`. Two implementation notes follow from this:

- Any reordering of `DEFAULT_PARAM_RANGES` between training runs
  invalidates pickled CMA-ES state. Resume across config changes is
  therefore brittle; the constructor should verify
  `len(self._param_keys) == self._es.N` and refuse to load an
  inconsistent checkpoint.
- The `param_ranges` overrides supplied at construction *merge into*
  `DEFAULT_PARAM_RANGES`, not replace it. Keys not present in
  `param_ranges` retain their default range.

#### b.3 CMA-ES lifecycle inside the generator

The generator owns one `cma.CMAEvolutionStrategy` instance for its
entire lifetime. Construction options are fixed at `__init__` time:

```python
opts = cma.CMAOptions()
opts.set({
    "popsize": num_individuals,
    "bounds": [np.zeros(n), np.ones(n)],
    "seed": int(seed),
    "verb_disp": 0,
    "verb_filenameprefix": str(Path(output_dir) / "cma_log") + "/",
})
self._es = cma.CMAEvolutionStrategy(np.full(n, 0.5), sigma0, opts)
```

`popsize` must equal `num_individuals` because the runner consumes
exactly `num_individuals` USDs per generation. The `bounds` handler is
the default `BoundPenalty` (clipping with soft quadratic penalty);
`BoundDomainTransform` is an upgrade documented in §b.10.

The two state pointers managed by the generator are:

- `self._pending_solutions: list[np.ndarray] | None`: candidates
  returned by the most recent `es.ask()`; consumed by the next
  `generate_population`.
- `self._last_solutions: list[np.ndarray] | None`: candidates that
  produced the population currently being evaluated; passed back to
  `es.tell` in `update_with_fitness`.

These pointers enforce the `sample_batch → generate_population →
update_with_fitness → generate_population → …` ordering: any deviation
raises an `AssertionError` with a clear diagnostic, surfacing runner-
generator contract violations as test failures.

#### b.4 Method mapping (sample_batch / generate_population / update_with_fitness)

The mapping of the three abstract methods to CMA-ES operations was
sketched under R5 above; this subsection is the canonical reference.

`sample_batch()` is the first-generation pre-`ask`. The runner calls
this once, before the first `generate_population`. Implementation:

```python
def sample_batch(self) -> None:
    assert self._pending_solutions is None, "sample_batch called twice"
    self._pending_solutions = self._es.ask()
```

`generate_population(generation)` consumes the pending batch into USDs
and actuator overrides:

```python
def generate_population(self, generation: int) -> Population:
    assert self._pending_solutions is not None, (
        "generate_population called before sample_batch or after a "
        "spent batch was consumed without a fresh ask"
    )
    solutions = self._pending_solutions
    usd_files: list[str] = []
    actuator_params: list[dict[str, dict]] = []
    for idx, x in enumerate(solutions):
        scales = self._denormalise(x)
        usd_path, act_params = self._apply_scales(scales, generation, idx)
        usd_files.append(usd_path)
        actuator_params.append(act_params)
    self._last_solutions = solutions
    self._pending_solutions = None
    return RandomPopulation(usd_files, actuator_params)
```

`update_with_fitness(population, fitness)` negates fitness (CMA-ES
minimises), calls `tell`, checks termination, and immediately calls
`ask` to pre-generate the next batch:

```python
def update_with_fitness(
    self, population: Population, fitness: list[float]
) -> None:
    assert self._last_solutions is not None, (
        "update_with_fitness called before generate_population"
    )
    costs = [-float(f) for f in fitness]
    self._es.tell(self._last_solutions, costs)
    if self._es.stop():
        # log but do not raise; runner will keep calling generate_population.
        print(f"[CMA-ES] terminated: {self._es.stop()}")
    self._pending_solutions = self._es.ask()
```

If termination has fired and the generator is still asked for more
populations, pycma's `ask()` continues to sample from the *current*
$\mathcal{N}(\boldsymbol{m}, \sigma^2 \boldsymbol{C})$ — i.e. the
incumbent distribution — which is the desired degenerate behaviour: the
runner keeps training around the best-known design until external
budget exhausts.

#### b.5 Fitness sign and noise handling

The runner reports `fitness[i] = mean episode return for individual i
since the last reset`. This signal has two properties that bear on the
update:

- **Sign:** higher is better. The generator negates it to match
  pycma's minimisation convention.
- **Noise:** the mean is taken over a small number of completed
  episodes per individual within one `ea_update_interval`. For a
  typical configuration (`num_envs = 4096`, `num_individuals = 64`,
  `ea_update_interval = 100` PPO iters, $\sim 100$ rollout steps per
  iter, episode length $\sim 1000$ steps) each individual sees
  $\sim (4096/64) \times 100 \times 100 / 1000 = 640$ episodes per
  generation — high enough for the fitness signal to be reasonably
  low-variance. With shorter `ea_update_interval` or fewer envs per
  individual, the fitness becomes noisy and a `cma.NoiseHandler`
  wrapper around `tell` should be considered. The initial
  implementation does *not* enable noise handling; if convergence
  stagnates in practice, this is the first recommended diagnostic.

Two edge cases deserve explicit handling:

- **Zero-episode individuals.** If `_individual_episode_counts[i] == 0`
  for some individual (no envs assigned to that individual completed an
  episode in the interval), the runner currently reports
  `fitness[i] = 0.0`. Under CMA-ES this is *uninformative*: a flat
  fitness of zero across individuals leaves the algorithm's selection
  step degenerate. The generator should log a warning when any
  per-individual count is zero, and the runner config should ensure
  `ea_update_interval` is long enough for every individual to complete
  several episodes.
- **Inf / NaN fitness.** A simulation that diverges may produce
  non-finite reward sums. pycma's `tell` will raise on `nan` costs.
  The generator should sanitise costs to a large finite penalty
  (e.g. `1e6`) before passing them to `tell`.

#### b.6 Refactoring `_apply_scales`

The current implementation of `RandomDesignGenerator._generate_individual`
inlines the URDF mutation, USD conversion, and actuator-override
construction (`usd_generator.py:296–385`). The CMA-ES generator reuses
all of this; cleanly sharing it requires a refactor that factors out
the post-sampling pipeline into a method whose only inputs are a
`scales` dict and identifying integers.

The refactor introduces a single new method on
`RandomDesignGenerator`:

```python
def _apply_scales(
    self,
    scales: dict[str, float],
    generation: int,
    idx: int,
) -> tuple[str, dict[str, dict]]:
    """Apply *scales* to the base URDF and return (usd_path, act_params).

    Body identical to the post-sampling portion of _generate_individual:
    parse URDF, apply geometric/mass/actuator-geometry/joint-limit
    mutations, write URDF, convert to USD, build actuator-overrides dict.
    """
```

The existing `_generate_individual` then simplifies to:

```python
def _generate_individual(
    self, generation: int, idx: int
) -> tuple[str, dict[str, dict]]:
    rng = np.random.default_rng(seed=generation * 10000 + idx)
    scales = self._sample_scales(rng)
    return self._apply_scales(scales, generation, idx)
```

`CMAESDesignGenerator` then inherits from `RandomDesignGenerator`
purely to reuse `_apply_scales` (and `_denormalise`, `_sample_scales`,
the static URDF helpers, and the `_convert_urdf_to_usd` helper). Two
design choices follow:

- **Inheritance vs composition.** Inheritance is preferred here
  because the shared surface is large (six methods plus a constants
  table) and because the only behavioural difference is the *source*
  of the scales dict. Composition would require `CMAESDesignGenerator`
  to hold a `RandomDesignGenerator` instance and forward six methods
  to it, which adds boilerplate without removing any coupling.
- **Override of `__init__`.** `CMAESDesignGenerator.__init__` calls
  `RandomDesignGenerator.__init__(base_urdf_path, num_individuals,
  param_ranges, output_dir)` then constructs `self._es`. The
  `num_individuals` argument is passed *twice* — once to the parent
  for the URDF mutation pipeline, once to `cma.CMAOptions["popsize"]`
  — which guarantees coherence.

The refactor is *behaviourally inert* for `RandomDesignGenerator`: the
existing `_generate_individual` body is moved verbatim into
`_apply_scales`, with the only call-site change being the routing of
`scales` through a parameter rather than a local variable. A unit
test that compares the USD paths and actuator overrides produced by
the refactored generator with those of the pre-refactor generator,
under the same `(generation, idx)` seed, is sufficient to confirm the
refactor is bug-free.

#### b.7 Data flow

```text
                       (generation g)
CoptOnPolicyRunner._reload_morphology
   │
   ├── if current_population is None:            # g = 0
   │     design_generator.sample_batch()
   │         └── self._pending = self._es.ask()       # initial λ candidates
   │
   ├── else:                                     # g > 0
   │     fitness_g = _compute_individual_fitness()
   │     design_generator.update_with_fitness(pop_{g−1}, fitness_g)
   │         ├── costs = [-f for f in fitness_g]      # sign flip
   │         ├── self._es.tell(self._last, costs)     # CMA-ES update
   │         ├── log es.stop() if non-empty
   │         └── self._pending = self._es.ask()       # pre-generate batch
   │
   ├── pop_g = design_generator.generate_population(g)
   │       ├── solutions = self._pending
   │       ├── for idx, x in enumerate(solutions):
   │       │     scales = self._denormalise(x)
   │       │     (usd, act) = self._apply_scales(scales, g, idx)
   │       │     append to lists
   │       ├── self._last = solutions
   │       ├── self._pending = None
   │       └── return RandomPopulation(usds, acts)
   │
   ├── respawn_robots(env, pop_g.get_usd_files())
   ├── apply_actuator_params(env, pop_g.get_actuator_params())
   └── reset _individual_fitness and _individual_episode_counts to zero
```

The `tell → ask → consume in next generate_population` ordering is the
canonical CMA-ES ask/tell pattern shifted by one half-iteration to fit
the runner's `update_with_fitness → generate_population` order. It
introduces no extra fitness evaluations: each call to `ask` produces
exactly `num_individuals` candidates, all of which are evaluated by the
runner over the next `ea_update_interval` PPO iterations.

#### b.8 Skeleton implementation

The full additions to
`tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py`
are sketched below. Imports, `_apply_scales` body, and helper methods
are omitted for brevity; they are the relevant lines of the existing
file, moved as described in §b.6.

```python
# co_optimisation/runners/usd_generator.py (additions)

from pathlib import Path
import cma
import numpy as np


class CMAESDesignGenerator(RandomDesignGenerator):
    """Generates robot designs by sampling from a CMA-ES search distribution.

    The design vector is the unit-hypercube encoding of the scale-factor
    dictionary used by :class:`RandomDesignGenerator`; CMA-ES adapts the
    multivariate Gaussian search distribution from the per-individual
    mean episode return reported by :class:`CoptOnPolicyRunner`.
    """

    def __init__(
        self,
        base_urdf_path: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
        sigma0: float = 0.2,
        seed: int = 0,
        es_state_path: str | None = None,
    ) -> None:
        super().__init__(
            base_urdf_path=base_urdf_path,
            num_individuals=num_individuals,
            param_ranges=param_ranges,
            output_dir=output_dir,
        )
        self._param_keys: list[str] = list(self.param_ranges.keys())
        self._n: int = len(self._param_keys)
        self._sigma0: float = float(sigma0)
        self._seed: int = int(seed)

        if es_state_path is not None:
            with open(es_state_path, "rb") as fh:
                self._es = cma.CMAEvolutionStrategy.pickle_loads(fh.read())
            assert self._es.N == self._n, (
                f"Checkpoint dimension mismatch: "
                f"pickled {self._es.N}, expected {self._n}"
            )
        else:
            opts = cma.CMAOptions()
            opts.set({
                "popsize": int(num_individuals),
                "bounds": [np.zeros(self._n), np.ones(self._n)],
                "seed": self._seed,
                "verb_disp": 0,
                "verb_filenameprefix": str(Path(output_dir) / "cma_log") + "/",
            })
            self._es = cma.CMAEvolutionStrategy(
                np.full(self._n, 0.5), self._sigma0, opts
            )

        self._pending_solutions: list[np.ndarray] | None = None
        self._last_solutions: list[np.ndarray] | None = None

    # --- Public API ---------------------------------------------------

    def sample_batch(self) -> None:
        assert self._pending_solutions is None, "sample_batch called twice"
        self._pending_solutions = self._es.ask()

    def update_with_fitness(
        self, population: Population, fitness: list[float]
    ) -> None:
        assert self._last_solutions is not None, (
            "update_with_fitness called before generate_population"
        )
        costs = [self._sanitise_cost(-float(f)) for f in fitness]
        self._es.tell(self._last_solutions, costs)
        if self._es.stop():
            print(f"[CMA-ES] terminated: {self._es.stop()}")
        self._pending_solutions = self._es.ask()

    def generate_population(self, generation: int) -> Population:
        assert self._pending_solutions is not None, (
            "generate_population called before sample_batch"
        )
        solutions = self._pending_solutions
        usd_files: list[str] = []
        actuator_params: list[dict[str, dict]] = []
        for idx, x in enumerate(solutions):
            scales = self._denormalise(x)
            usd_path, act_params = self._apply_scales(scales, generation, idx)
            usd_files.append(usd_path)
            actuator_params.append(act_params)
        self._last_solutions = solutions
        self._pending_solutions = None
        return RandomPopulation(usd_files, actuator_params)

    # --- Internal helpers --------------------------------------------

    def _denormalise(self, x: np.ndarray) -> dict[str, float]:
        scales: dict[str, float] = {}
        for i, key in enumerate(self._param_keys):
            lo, hi = self.param_ranges[key]
            xi = float(np.clip(x[i], 0.0, 1.0))
            scales[key] = float(lo + xi * (hi - lo))
        return scales

    @staticmethod
    def _sanitise_cost(c: float, fallback: float = 1e6) -> float:
        if not np.isfinite(c):
            return fallback
        return c

    def save_state(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self._es.pickle_dumps())
```

The `_apply_scales` method is inherited unchanged from
`RandomDesignGenerator` after the refactor in §b.6.

#### b.9 Wiring into the runner

`CoptOnPolicyRunner` already calls `design_generator.sample_batch()`
on the first generation and `design_generator.update_with_fitness(…)`
thereafter (`copt_on_policy_runner.py:255`). No runner changes are
required; only the choice of generator at the call site needs to
switch from `RandomDesignGenerator(…)` to `CMAESDesignGenerator(…)`.

The current call site in
`tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py:188-210` becomes:

```python
if args_cli.policy_type == "COPT":
    _base_urdf = os.path.join(...)
    _num_individuals = 64

    copt_cfg = {
        "ea_update_interval": 1000,
        "num_individuals": _num_individuals,
        "design_generator": args_cli.design_generator,   # NEW
        "cma_sigma0": 0.2,                               # NEW
        "cma_seed": int(args_cli.seed),                  # NEW
    }

    if copt_cfg["design_generator"] == "cmaes":
        design_generator = CMAESDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
            sigma0=copt_cfg["cma_sigma0"],
            seed=copt_cfg["cma_seed"],
        )
    else:
        design_generator = RandomDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
        )

    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["copt"] = copt_cfg
    runner = CoptOnPolicyRunner(
        env, design_generator, agent_cfg_dict,
        log_dir=log_dir, device=agent_cfg.device,
    )
```

The CLI flag `--design_generator {random,cmaes}` and `--seed` provide
selection and reproducibility; the other two new keys
(`cma_sigma0`, `cma_seed`) are surfaced through the `copt_cfg` dict
for completeness and for downstream log/CSV consumers.

#### b.10 Constraint handling and bound encoding

The unit-hypercube encoding of §b.2 turns every coordinate constraint
into the same simple form $0 \le x_i \le 1$. pycma offers three
handlers for this case, in order of increasing sophistication:

1. **`bounds = [lower, upper]`** with the default `BoundPenalty`.
   Soft quadratic penalty proportional to the unprojected distance from
   the box; internal Gaussian operates on $\mathbb{R}^n$. Cheap and
   robust but can cause samples to cluster against the boundary on
   tight problems.
2. **`bounds = [lower, upper]` with `BoundTransform`** in lieu of
   `BoundPenalty`. Reflective projection of infeasible samples back
   into the box; preserves the invariance properties of §3.a.7 better
   than the penalty handler.
3. **`cma.BoundDomainTransform([lower, upper])`** wrapper, with the
   internal Gaussian operating on the *transformed* unbounded space.
   The bijection is a $\tanh$-like saturating map; the strongest choice
   for tight box constraints with active boundaries.

The initial implementation uses option 1 (the default in §b.3). Option
3 is an optional upgrade noted for follow-up if samples concentrate at
$x_i \in \{0, 1\}$ in practice. A diagnostic test, run inside the unit
suite of §b.12.1, can compare the boundary-incidence rate across the
three handlers on the synthetic objective.

For physical-feasibility constraints beyond the box (e.g. total robot
mass below a threshold, motor torque within rated saturation), there
are three options:

- **Rejection.** Re-sample any individual that violates the constraint,
  capped at a maximum number of retries. Simple but distorts the
  CMA-ES sampling distribution; not invariance-preserving.
- **Penalty.** Add a constraint-violation penalty to the cost passed
  to `tell`. Easy to wire in but tuning the penalty weight is
  problem-specific.
- **`cma.constraints_handler.AugmentedLagrangian`.** Adapts a
  per-constraint Lagrange multiplier inside `tell`. Recommended when
  more than one inequality constraint is active simultaneously.

For the current design vocabulary, no constraint is needed beyond the
box: every parameter is bounded and the URDF mutation pipeline
maintains physical consistency by construction. Future additions
(e.g. structural mass constraints) should adopt the augmented-Lagrangian
handler.

#### b.11 Logging, checkpointing, and reproducibility

The generator emits four streams of diagnostic information, all routed
through paths that match RSL-RL's conventions:

- **pycma's internal data logger** writes `outcmaes*.dat` files to
  `{output_dir}/cma_log/` (set via `verb_filenameprefix` in §b.3).
  These six files (`outcmaesxmean.dat`, `outcmaessigma.dat`,
  `outcmaesaxlen.dat`, `outcmaesstddev.dat`, `outcmaesfit.dat`,
  `outcmaesxrecentbest.dat`) are the canonical CMA-ES diagnostic
  trace and can be post-processed with `cma.CMADataLogger(...).plot()`.
- **Per-generation TensorBoard scalars** added inside `update_with_fitness`:
  `copt/cma_sigma`, `copt/cma_axis_ratio`,
  `copt/cma_best_fitness`, `copt/cma_mean_fitness`,
  `copt/cma_stagnation`. The runner's `self.writer` is already
  initialised before `_reload_morphology` is first called, so the
  generator can take `writer` as an optional constructor argument or
  receive it via a setter.
- **A CSV of per-individual outcomes** (`{output_dir}/cma_log/individuals.csv`):
  one row per `(generation, idx)` with normalised solution,
  denormalised scales, fitness, USD path, and any termination flag.
- **Pickled CMA-ES state at every checkpoint** of the inner-loop
  runner: `{log_dir}/cma_state_{iteration}.pkl`. The save is wired
  into `CoptOnPolicyRunner.save` via a generator-side `save_state(path)`
  method (shown in §b.8). On resume, the constructor's `es_state_path`
  argument restores the optimiser; the runner's existing checkpoint
  logic restores the policy.

The two seeds owned by the generator — `cma_seed` and the per-individual
`generation * 10000 + idx` seed in `_generate_individual` (now unused
by the CMA-ES path) — are written to a `cma_config.json` file at
generator-construction time. This file is the single source of truth
for reproducibility audits.

#### b.12 Validation milestones

The implementation will be validated in three stages:

1. **Unit-level (no IsaacLab).** Drive `CMAESDesignGenerator`
   against a synthetic fitness function $f(\boldsymbol{x}) =
   -\|\boldsymbol{x} - \boldsymbol{x}^{\star}\|_2^2$ over the unit
   hypercube, with the URDF mutation pipeline replaced by an identity
   mock; verify (i) `sample_batch → generate_population →
   update_with_fitness → generate_population → …` lifecycle without
   `AssertionError`, (ii) the mean trajectory $\|\boldsymbol{m}_g -
   \boldsymbol{x}^{\star}\|$ decreases monotonically after a transient,
   (iii) reaching $\|\boldsymbol{m}_g - \boldsymbol{x}^{\star}\| < 0.05$
   within $\mathcal{O}(10^2)$ generations with `popsize = 8`, (iv) full
   determinism under a fixed seed across reruns. The synthetic test
   also exercises the boundary-clipping (clamp inputs outside $[0, 1]$
   and verify finite costs) and the `nan`/`inf` sanitisation.
2. **URDF-pipeline integration (no RL).** Drive
   `CMAESDesignGenerator` with the real URDF mutation pipeline but a
   synthetic fitness — e.g. fitness proportional to one scale factor.
   Verify (i) URDFs and actuator-overrides are produced under the
   `{output_dir}/gen_{g:04d}/individual_{i:04d}` layout without
   error, (ii) the USD files load successfully into a headless
   IsaacLab `Articulation`, (iii) the optimiser drives the target
   scale towards its bound.
3. **End-to-end integration.** Instantiate the generator inside
   `CoptOnPolicyRunner` on the existing TRON1 SF biped task with a
   short `ea_update_interval` (e.g. 10) and confirm (i) the
   `_reload_morphology` cycle completes without exception, (ii)
   `respawn_robots` and `apply_actuator_params` consume the new
   populations successfully across generations, (iii)
   `_compute_individual_fitness` returns finite values for every
   individual after the first interval. Run for $\sim 20$ generations
   and inspect the pycma diagnostic plots and TensorBoard scalars.
4. **Comparative training.** Run a full co-optimisation training under
   both `RandomDesignGenerator` and `CMAESDesignGenerator` with
   identical PPO config and compare per-generation best fitness,
   mean fitness, and final morphology. CMA-ES should produce a
   measurable upward trend in mean fitness across generations whereas
   random search should remain flat in expectation. A non-decreasing
   $\boldsymbol{m}_g \to \boldsymbol{x}^{\star}_{\text{best}}$
   contraction (decreasing $\sigma$) is the expected long-horizon
   signature of CMA-ES convergence.

#### b.13 Open questions for follow-up

- *Population-size / num_envs mismatch.* If `num_envs` is not an
  integer multiple of `popsize`, fitness aggregation is uneven across
  individuals; this is already handled by the round-robin assignment
  in `_assign_individuals_to_envs`, but the variance of the
  per-individual estimator differs across individuals. For CMA-ES,
  low-variance fitness estimates near the recombination boundary
  ($i \approx \mu$) matter most; a future refinement could weight
  environment assignments accordingly.
- *Non-stationary inner objective.* The policy is being trained
  concurrently with the morphology, so the fitness signal at generation
  $g$ is biased relative to the asymptotic return under design
  $\boldsymbol{x}_g$. This is the morphological-innovation-protection
  problem of Cheney et al. [2018] and is currently mitigated only by a
  generous `ea_update_interval`. The principled extensions are (a) a
  pre-EA shared PPO warm-start (SERL's 500-iter convention), (b)
  inflating $\sigma_0$ early in training to compensate for noisy
  fitness, and (c) wrapping `tell` in `cma.NoiseHandler`.
- *Constraint handling beyond the box.* The current scale-factor
  vocabulary is unconstrained beyond box bounds, but future inclusion
  of structural constraints (total mass, peak motor torque) would
  motivate the augmented-Lagrangian path of §b.10.
- *Active vs vanilla CMA-ES.* The default of `CMA_active = True` is
  retained; if the negative-weight update interacts pathologically
  with the noisy RL fitness, the implementation should expose a
  constructor flag to disable it. No evidence of pathology is expected
  on this problem, but the diagnostic is cheap.
- *Restart strategies.* IPOP / BIPOP are *not* enabled in the initial
  implementation because each generation costs a full inner-loop
  training segment. If multimodality of the fitness landscape becomes
  evident (e.g. distinct gait regimes correspond to disjoint design
  basins), an IPOP wrapper around the entire `CoptOnPolicyRunner.learn`
  loop is the recommended path.

## 5. Key Classes and Interfaces

Below is the API documentation for the classes introduced or extended
by this document. Conventions follow [../CO_OPTIMISATION.md §4.8](../CO_OPTIMISATION.md):
role, constructor arguments, key variables, key methods.

#### `Population`
**File**: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py` (line 157)
- **Role**: Abstract base class for a generation's design population.
  Holds a fixed set of robot designs (individuals), each represented by
  a USD file path and an actuator-parameter override dict.
- **Key Methods**:
  - `get_usd_files() -> list[str]`: Returns one USD file path per
    individual, in the order in which environments are assigned
    (round-robin).
  - `get_actuator_params() -> list[dict[str, dict]]`: Returns one
    actuator-overrides dict per individual; each is keyed by actuator
    group name (`"abad"`, `"hip"`, `"knee"`, `"ankle"`), and each value
    is a dict of scalar overrides for `IdentifiedActuator` tensor
    attributes (e.g. `effort_limit`, `friction_static`, `stiffness`,
    `damping`).

#### `RandomPopulation`
**File**: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py` (line 214)
- **Role**: Concrete `Population` backed by pre-computed lists of USD
  paths and actuator-override dicts. Reused unchanged by
  `CMAESDesignGenerator`.
- **Constructor Args**:
  - `usd_files` (`list[str]`): One USD file path per individual.
  - `actuator_params` (`list[dict[str, dict]]`): One actuator-overrides
    dict per individual.
- **Key Methods**: `get_usd_files`, `get_actuator_params` (return the
  constructor lists as-is).

#### `DesignGeneratorBase`
**File**: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py` (line 182)
- **Role**: Abstract base class for any design generator. Encapsulates
  the outer-loop optimisation algorithm — random search, CMA-ES,
  Bayesian optimisation, etc. — and produces a fresh `Population` every
  outer-loop generation.
- **Key Methods**:
  - `generate_population(generation: int) -> Population`: Produces a
    new population for the given outer-loop iteration. Must return
    exactly `num_individuals` designs.
  - `update_with_fitness(population: Population, fitness: list[float]) -> None`:
    Optionally updates internal optimiser state from per-individual
    mean episode return. Default implementation is a no-op (random
    search).
  - `sample_batch() -> None`: Pre-generation hook called once at the
    start of training, before the first `generate_population`. Default
    implementation is a no-op; CMA-ES uses it to pre-generate the
    first batch of candidates via `es.ask()`.

#### `RandomDesignGenerator`
**File**: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py` (line 230)
- **Role**: Reference generator that samples scale factors uniformly at
  random from per-parameter ranges and applies them to the base URDF.
  Serves as the random-search baseline and as the parent class of
  `CMAESDesignGenerator` (for inheritance-based reuse of the URDF
  mutation pipeline; see §4.b.6).
- **Constructor Args**:
  - `base_urdf_path` (`str`): Absolute path to the template URDF.
  - `num_individuals` (`int`): Population size per generation.
  - `param_ranges` (`dict[str, tuple[float, float]] | None`): Optional
    overrides over `DEFAULT_PARAM_RANGES`; only the supplied keys are
    overridden.
  - `output_dir` (`str`, default `"/tmp/copt_usds"`): Directory under
    which generated URDFs and USDs are written, organised as
    `{output_dir}/gen_{g:04d}/individual_{i:04d}.{urdf,usd}`.
- **Key Methods**:
  - `generate_population(generation)`: Iterates `num_individuals`
    times, calls `_generate_individual` per individual, and aggregates
    the results into a `RandomPopulation`.
  - `_generate_individual(generation, idx)`: After the refactor in
    §4.b.6, seeds a generation-local `numpy.random.Generator`, samples
    scales, and delegates the rest of the pipeline to `_apply_scales`.
  - `_apply_scales(scales, generation, idx)` *(new)*: URDF mutation,
    USD conversion, and actuator-override construction — the shared
    pipeline between `RandomDesignGenerator` and `CMAESDesignGenerator`.
  - `_sample_scales(rng)`: Uniform sampling of one float per
    `param_ranges` key.
  - `_scale_joint_z_origin(root, joint_names, scale)`: Static helper;
    multiplies the Z component of `<origin xyz>` of named joints by
    `scale`.
  - `_convert_urdf_to_usd(urdf_path, usd_out_dir, idx)`: Static helper;
    runs `isaaclab.sim.converters.UrdfConverter` with
    `merge_fixed_joints=False`, `fix_base=False`,
    `self_collision=False`, `collider_type="convex_hull"`,
    `force_usd_conversion=True`.

#### `CMAESDesignGenerator` *(planned, see §4.b.8)*
**File**: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/usd_generator.py` (to be added)
- **Role**: Generator that samples designs from a
  `cma.CMAEvolutionStrategy` search distribution over the unit
  hypercube of scale factors. Adapts the multivariate Gaussian search
  distribution from per-individual mean episode return using the
  rank-one, rank-$\mu$, and active negative-weight updates of §3.b.
- **Constructor Args**:
  - `base_urdf_path` (`str`): Absolute path to the template URDF.
  - `num_individuals` (`int`): Population size per generation; sets
    `cma`'s `popsize` option.
  - `param_ranges` (`dict[str, tuple[float, float]] | None`): Optional
    overrides over `DEFAULT_PARAM_RANGES`. The dimensionality $n$ of
    the CMA-ES search space equals `len(self.param_ranges)` and is
    fixed for the lifetime of the generator.
  - `output_dir` (`str`): Directory for generated URDFs/USDs and for
    the pycma diagnostic log directory `{output_dir}/cma_log/`.
  - `sigma0` (`float`, default `0.2`): Initial CMA-ES step-size over
    the normalised unit hypercube. The default corresponds to
    $\sigma_0 \approx 1/4$ of the unit range, which is the standard
    recommendation.
  - `seed` (`int`, default `0`): RNG seed passed to `cma.CMAOptions`
    for reproducibility.
  - `es_state_path` (`str | None`): Optional path to a pickled
    `CMAEvolutionStrategy` for warm-starting after a checkpoint
    resume; the constructor asserts dimensionality consistency.
- **Key Variables**:
  - `_es` (`cma.CMAEvolutionStrategy`): The optimiser state — mean
    $\boldsymbol{m}$, step-size $\sigma$, covariance matrix
    $\boldsymbol{C}$, evolution paths $\boldsymbol{p}_\sigma$,
    $\boldsymbol{p}_c$ — wholly owned by the generator.
  - `_param_keys` (`list[str]`): Ordered list of parameter keys;
    defines the bijection between vector indices and `param_ranges`
    entries.
  - `_n` (`int`): Search-space dimensionality, equal to
    `len(_param_keys)`.
  - `_sigma0`, `_seed`: cached constructor arguments for diagnostic
    logging.
  - `_pending_solutions` (`list[np.ndarray] | None`): Candidates
    returned by the most recent `es.ask()`; consumed by the next
    `generate_population`.
  - `_last_solutions` (`list[np.ndarray] | None`): Candidates that
    produced the population currently being evaluated; passed back to
    `es.tell` in `update_with_fitness`.
- **Key Methods**:
  - `sample_batch()`: Calls `self._es.ask()` and stores the result in
    `_pending_solutions`. Asserts that the batch has not already been
    sampled.
  - `update_with_fitness(population, fitness)`: Negates `fitness`
    (CMA-ES minimises), sanitises non-finite costs, calls
    `self._es.tell(self._last_solutions, costs)`, polls
    `self._es.stop()` for termination, and immediately calls
    `self._es.ask()` to pre-generate the next batch.
  - `generate_population(generation)`: Denormalises each pending
    solution into a per-parameter scales dict, calls `_apply_scales`
    (inherited) to produce `(usd_path, act_params)` per individual,
    stores `_last_solutions`, and returns a `RandomPopulation`.
  - `_denormalise(x)`: Maps a unit-hypercube vector $\boldsymbol{x}
    \in [0, 1]^n$ to a scales dict using the per-parameter
    `(lo, hi)` ranges; clips $\boldsymbol{x}$ to $[0, 1]$ before
    scaling for robustness against numerical bound violations.
  - `_sanitise_cost(c, fallback=1e6)`: Replaces `nan`/`inf` costs
    with a large finite penalty so that `tell` does not raise.
  - `save_state(path)`: Pickles `self._es` to `path` for checkpoint
    persistence.

#### `CoptOnPolicyRunner`
**File**: `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/copt_on_policy_runner.py` (line 29)
- **Role**: PPO runner with an outer evolutionary loop over robot
  morphology. Every `ea_update_interval` PPO iterations it evaluates
  per-individual fitness from the accumulated episode return, advances
  the design generator, hot-swaps robot articulations via
  `respawn_robots`, patches `IdentifiedActuator` tensor attributes via
  `apply_actuator_params`, and resets fitness accumulators.
- **Constructor Args**:
  - `env` (`VecEnv`): The `RslRlVecEnvWrapper`-wrapped
    `ManagerBasedRLEnv`.
  - `design_generator` (`DesignGeneratorBase`): The outer-loop design
    optimiser. Passing `CMAESDesignGenerator` here is the *only*
    change required to switch from random search to CMA-ES.
  - `train_cfg` (`dict`): RSL-RL training config plus an optional
    `"copt"` sub-dict with keys `"ea_update_interval"` (int, default
    100), `"num_individuals"` (int, default 16), and — after §4.b.9
    — optional keys `"design_generator"`, `"cma_sigma0"`, `"cma_seed"`.
  - `log_dir` (`str | None`): Logging directory.
  - `device` (`str`): Compute device.
- **Key Variables**:
  - `_design_generator` (`DesignGeneratorBase`): The injected
    generator.
  - `_ea_update_interval` (`int`): PPO iterations between outer-loop
    generations.
  - `_num_individuals` (`int`): Population size; must equal the
    generator's `num_individuals`.
  - `generation` (`int`): Current outer-loop generation, incremented
    in `_reload_morphology`.
  - `current_population` (`Population | None`): Population currently
    spawned in the simulation; `None` before the first generation.
  - `_individual_fitness` (`torch.Tensor`, shape `(num_individuals,)`):
    Sum of completed-episode returns per individual since the last
    generation reset.
  - `_individual_episode_counts` (`torch.Tensor`, shape
    `(num_individuals,)`): Number of completed episodes per individual
    since the last generation reset.
  - `_env_to_individual` (`list[int]`): Round-robin map env index →
    individual index, computed at construction time.
- **Key Methods**:
  - `learn(num_learning_iterations, init_at_random_ep_len=False)`:
    Main training loop. Calls `_reload_morphology` once before the
    rollout loop, then again every `ea_update_interval` iterations.
  - `_reload_morphology()`: One outer-loop cycle: if
    `current_population is None` calls `design_generator.sample_batch()`,
    else calls `update_with_fitness(current_population, fitness)`;
    calls `generate_population`; respawns the robots; applies
    actuator-param overrides; zeros the fitness accumulators.
  - `_compute_individual_fitness()`: Returns the per-individual mean
    episode return (`fitness[i] / count[i]`), with `0.0` for
    individuals whose envs produced no completed episodes.
  - `_assign_individuals_to_envs()`: Computes the round-robin
    assignment `env_idx → env_idx % num_individuals`.

#### `cma.CMAEvolutionStrategy`
**File**: `cma` package, version 4.4.4 (`cma.evolution_strategy.CMAEvolutionStrategy`)
- **Role**: Reference implementation of the $(\mu/\mu_W, \lambda)$-CMA-ES
  with cumulative step-size adaptation, rank-one and rank-$\mu$
  covariance updates, the active negative-weight extension, and box-
  constraint handling. Exposes an ask/tell iteration interface that
  decouples candidate generation from fitness evaluation.
- **Constructor Args**:
  - `x0` (`np.ndarray | list`): Initial mean $\boldsymbol{m}_0$.
  - `sigma0` (`float`): Initial step-size $\sigma_0$.
  - `inopts` (`cma.CMAOptions | dict | None`): Options dict; commonly
    used keys are `popsize`, `bounds`, `seed`, `maxiter`, `tolx`,
    `tolfun`, `verb_disp`, `integer_variables`, `CMA_active`.
- **Key Variables**:
  - `mean` (`np.ndarray`): Current distribution mean $\boldsymbol{m}_g$.
  - `sigma` (`float`): Current step-size $\sigma_g$.
  - `C` (`np.ndarray`): Current covariance matrix $\boldsymbol{C}_g$.
  - `popsize` (`int`): $\lambda$.
  - `countiter` (`int`): Current generation index.
- **Key Methods**:
  - `ask(number=None) -> list[np.ndarray]`: Samples and returns
    `popsize` candidate vectors from $\mathcal{N}(\boldsymbol{m}_g,
    \sigma_g^2 \boldsymbol{C}_g)$, applying box-constraint repair if
    `bounds` is set.
  - `tell(solutions, function_values) -> None`: Consumes the same
    list returned by the most recent `ask` together with corresponding
    *minimisation* costs; performs the mean, evolution-path,
    covariance, and step-size updates.
  - `stop() -> dict`: Returns a (possibly empty) dict of termination
    criteria that have fired; non-empty indicates convergence or
    budget exhaustion.
  - `disp() -> None`: Console diagnostic print.
  - `result -> CMAEvolutionStrategyResult`: Namedtuple with `xbest`,
    `fbest`, `xfavorite`, `stds`, `evals_best`, `evaluations`,
    `iterations`, `stop`.
  - `pickle_dumps()` / `pickle_loads(blob)`: Serialise and restore
    optimiser state for checkpoint/resume.
  - `inject(solutions) -> None`: Inject externally proposed candidates
    into the next `ask` call.

#### `cma.CMAOptions`
**File**: `cma` package, version 4.4.4 (`cma.evolution_strategy.CMAOptions`)
- **Role**: Configuration container for `CMAEvolutionStrategy`. Behaves
  like a dict with validation and documentation; values may be strings
  that are evaluated with `N` and `popsize` in scope.
- **Key Fields** (most relevant to the design generator):
  - `popsize` (`int`): Population size $\lambda$; defaults to
    $4 + \lfloor 3 \ln n \rfloor$. The design generator sets this to
    `num_individuals`.
  - `bounds` (`[lower, upper]`): Box constraints per coordinate; the
    design generator sets this to $[0, 1]^n$ to enforce the unit
    hypercube encoding.
  - `seed` (`int`): RNG seed; threaded through from the runner's
    `"copt"` config.
  - `CMA_active` (`bool`, default `True`): Use the active negative-
    weight covariance update of §3.b.5. Recommended to leave on; a
    constructor flag is exposed for disabling if pathology is observed.
  - `maxiter` (`int`): Hard cap on generations; unused in the design
    generator (the runner controls termination), but available for
    unit tests.
  - `tolx`, `tolfun` (`float`): Convergence thresholds; default values
    are typically left unchanged.
  - `verb_disp` (`int`): Console-print interval; the design generator
    sets this to `0` to defer logging to RSL-RL's `TensorBoard` writer.
  - `verb_filenameprefix` (`str`): Logger output directory / prefix;
    set to `{output_dir}/cma_log/` so the diagnostic `.dat` files
    land alongside the per-generation URDFs.

---

## 6. References

- Akimoto, Y., Nagata, Y., Ono, I., and Kobayashi, S. (2010).
  *Bidirectional Relation between CMA Evolution Strategies and Natural
  Evolution Strategies.* PPSN XI, LNCS 6238.
  doi:10.1007/978-3-642-15844-5_16.
- Akimoto, Y., Auger, A., and Hansen, N. (2012).
  *Theoretical Foundation for CMA-ES from Information Geometry
  Perspective.* Algorithmica 64(4), 698–716.
  doi:10.1007/s00453-011-9564-8.
- Akimoto, Y. and Hansen, N. (2024).
  *Natural-Gradient Interpretation of Rank-One Update in CMA-ES.* PPSN
  2024. arXiv:2406.16506.
- Auger, A. and Hansen, N. (2005).
  *A Restart CMA Evolution Strategy with Increasing Population Size.*
  CEC 2005.
- Bongard, J. (2011). *Morphological change in machines accelerates the
  evolution of robust behavior.* PNAS 108(4):1234–1239.
  doi:10.1073/pnas.1015390108.
- Cheney, N., Bongard, J., SunSpiral, V. and Lipson, H. (2018).
  *Scalable co-optimization of morphology and control in embodied
  machines.* J. R. Soc. Interface 15:20170937. arXiv:1706.06133.
  doi:10.1098/rsif.2017.0937.
- Cheng, Y., Han, C., Min, Y., Ye, L., Liu, H. and Liu, H. (2024).
  *Structural Optimization of Lightweight Bipedal Robot via SERL.*
  IROS 2024. arXiv:2408.15632.
- Endo, K., Maeno, T. and Kitano, H. (2002).
  *Co-evolution of morphology and walking pattern of biped humanoid
  robot using evolutionary computation.* IROS 2002.
- Gupta, A., Savarese, S., Ganguli, S. and Fei-Fei, L. (2021).
  *Embodied intelligence via learning and evolution.* Nature
  Communications 12:5721. arXiv:2102.02202.
  doi:10.1038/s41467-021-25874-z.
- Hansen, N. (2009). *Benchmarking a BI-Population CMA-ES on the
  BBOB-2009 Function Testbed.* GECCO 2009 Workshops.
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial.*
  arXiv:1604.00772.
- Hansen, N., Akimoto, Y. and Baudis, P. (2024). *CMA-ES/pycma on
  GitHub.* Zenodo (DOI: 10.5281/zenodo.2559634); package `cma`
  version 4.4.4. https://github.com/CMA-ES/pycma.
- Hansen, N., Müller, S. D. and Koumoutsakos, P. (2003).
  *Reducing the Time Complexity of the Derandomized Evolution Strategy
  with Covariance Matrix Adaptation (CMA-ES).* Evolutionary
  Computation 11(1), 1–18.
- Hansen, N. and Ostermeier, A. (2001). *Completely Derandomized
  Self-Adaptation in Evolution Strategies.* Evolutionary Computation
  9(2), 159–195. doi:10.1162/106365601750190398.
- Hansen, N. and Ros, R. (2010). *Benchmarking a Weighted Negative
  Covariance Matrix Update on the BBOB-2010 Noiseless Testbed.* GECCO
  2010 Workshops.
- Hejna, D., Abbeel, P. and Pinto, L. (2021).
  *Task-Agnostic Morphology Evolution.* ICLR 2021. arXiv:2102.13100.
- Jastrebski, G. A. and Arnold, D. V. (2006).
  *Improving Evolution Strategies through Active Covariance Matrix
  Adaptation.* CEC 2006.
- Lipson, H. (2014). *Challenges and opportunities for design,
  simulation, and fabrication of soft robots.* Soft Robotics 1(1):21–27.
  doi:10.1089/soro.2013.0007.
- Lipson, H. and Pollack, J. B. (2000). *Automatic design and
  manufacture of robotic lifeforms.* Nature 406:974–978.
  doi:10.1038/35023115.
- Luck, K. S., Ben Amor, H. and Calandra, R. (2020).
  *Data-efficient Co-Adaptation of Morphology and Behaviour with Deep
  Reinforcement Learning.* CoRL 2020. arXiv:1911.06832.
- Ollivier, Y., Arnold, L., Auger, A. and Hansen, N. (2017).
  *Information-Geometric Optimization Algorithms.* JMLR 18, 1–65.
- Ostermeier, A., Gawelczyk, A. and Hansen, N. (1994). *A Derandomized
  Approach to Self-Adaptation of Evolution Strategies.* Evolutionary
  Computation 2(4), 369–380.
- Paul, C. and Bongard, J. C. (2001). *The Road Less Travelled:
  Morphology in the Optimization of Biped Robot Locomotion.* IROS 2001.
  doi:10.1109/IROS.2001.973363.
- Rechenberg, I. (1973). *Evolutionsstrategie: Optimierung technischer
  Systeme nach Prinzipien der biologischen Evolution.* Frommann-
  Holzboog.
- Rudin, N., Hoeller, D., Reist, P. and Hutter, M. (2021). *Learning to
  Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2021.
- Schaff, C., Yunis, D., Chakrabarti, A. and Walter, M. R. (2019).
  *Jointly Learning to Construct and Control Agents using Deep
  Reinforcement Learning.* ICRA 2019. arXiv:1801.01432.
- Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models.*
  Wiley.
- Sims, K. (1994). *Evolving Virtual Creatures.* SIGGRAPH '94, 15–22.
  doi:10.1145/192161.192167.
- Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and
  Schmidhuber, J. (2014). *Natural Evolution Strategies.* JMLR 15,
  949–980.
- Yuan, Y., Song, Y., Luo, Z., Sun, W. and Kitani, K. (2022).
  *Transform2Act: Learning a Transform-and-Control Policy for Efficient
  Agent Design.* ICLR 2022 Oral. arXiv:2110.03659.
