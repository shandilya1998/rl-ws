# Bandit Methods for Robot Design Optimisation

Written 2026-10-01, revised the same day at the user's request to carry full derivations, a definition for every symbol, the complete mechanics of every surveyed method, and a further investigation of how the bandit problem may be made part of the proximal policy optimisation formulation. The document is a literature survey and idea generation report. It asks whether the theory of multi-armed bandits, and in particular bandits whose decision rule is a neural network, can serve as the design search of this workspace's co-optimisation pipeline, where a design comprises continuous link lengths, a continuous nominal pose and a discrete choice of actuator per joint group. It is written to be read on its own, so that a reader who has not opened the surveyed papers can follow every argument from first principles, and it is long for that reason rather than from want of economy.

Every claim drawn from the literature carries a numbered marker resolved in the bibliography of section 14, and every claim about the workspace carries a `file:line` citation. Currency, the codebase citations were read against the tree as it stood on the date above, and every literature entry was verified by retrieval of its arXiv, Crossref, PMLR, NeurIPS or publisher record in the same session. The mathematical statements attributed to a source were read from that source's own text, and where a derivation is given in this document rather than quoted, it is marked as the standard argument of the textbook it follows, so that a reader can tell a reproduced proof from a reported result.

Codebase paths are abbreviated as follows. `usd_generator.py` and `copt_on_policy_runner.py` stand for the files of those names under `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/runners/`, `respawn.py` for `tron1-rl-isaaclab-cozum/co_optimisation/co_optimisation/utils/respawn.py`, `train.py` for `tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py`, `limx_base_env_cfg.py` for `tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`, `limx_rsl_rl_ppo_cfg.py` for `tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`, and `joint_actions.py`, `observations.py` and `rewards.py` for the Isaac Lab files of those names under `IsaacLab/source/isaaclab/isaaclab/envs/mdp/`, the first of them in its `actions/` subdirectory. The co-optimisation audit is cited as `audit` and lives at `tron1-rl-isaaclab-cozum/context/co_optimisation_audit.md`.

## 0. How to read this document, and the notation it uses

The report advances from the general to the particular. Section 1 states the engineering problem and the proposal. Section 2 assembles the probability and calculus on which every later argument rests, so that no later derivation needs to pause for a definition. Sections 3 to 6 build the theory of bandits from the finite case outward, through continuous, hybrid and neural decision spaces. Sections 7 and 8 survey the use of that theory in robot design and in the co-optimisation of design and control. Section 9 answers the further question of how the bandit may be absorbed into proximal policy optimisation itself. Section 10 surveys adjacent domains, section 11 derives the modifications the workspace's setting requires, section 12 distils candidate contributions, section 13 records the limits of the survey, and section 14 is the bibliography. A reader interested only in the proposal may read sections 1, 6.8, 8, 9.7, 11 and 12, returning to the earlier sections for the mathematics they cite.

Symbols are reused with one meaning throughout wherever possible, and the table below fixes the most frequent. Where a section must reuse a letter for a different object, as when the literature it reports does so, the section redefines the letter locally and says that it has done so.

| Symbol | Meaning |
|---|---|
| $K$ | number of arms of a finite bandit |
| $a$, $A_t$ | an arm, and the arm chosen at round $t$ |
| $X_t$ | the reward observed at round $t$ |
| $\mu_a$, $\mu^\star$ | the mean reward of arm $a$, and the largest mean |
| $\Delta_a = \mu^\star - \mu_a$ | the gap of arm $a$ |
| $T_a(t)$ or $N_a(t)$ | the number of times arm $a$ was chosen in the first $t$ rounds |
| $\hat\mu_a(t)$ | the empirical mean reward of arm $a$ after round $t$ |
| $n$ or $T$ | the horizon, the total number of rounds |
| $R_n$ | the cumulative regret over $n$ rounds |
| $\mathcal{X}$ | a set of arms that may be infinite, such as a box of design parameters |
| $f$ or $\mu(\cdot)$ | the unknown mean reward as a function on $\mathcal{X}$ |
| $k(\cdot,\cdot)$ | a kernel, or covariance function, of a Gaussian process |
| $\sigma^2$ | the variance of observation noise |
| $\mu_t(x)$, $\sigma_t(x)$ | the posterior mean and standard deviation of $f(x)$ after $t$ observations |
| $\gamma_T$ | the maximum information gain after $T$ observations |
| $\phi$, $p_\phi$ | the parameters of a search distribution, and the distribution |
| $\theta$ | the parameters of a neural network or a control policy |
| $\pi_\theta$ | a stochastic policy |
| $x = (z, c)$ | a design, with continuous part $z$ and categorical part $c$ |
| $G$, $M$ | the number of actuator groups, and the number of catalogue entries per group |
| $B$ | the batch, or population, size of one generation |
| $F_k(x)$ | the merit of design $x$ under the policy of generation $k$ |
| $\mathbb{E}$, $\Pr$ | expectation and probability |
| $\mathbb{1}[\cdot]$ | the indicator, one when its argument holds and zero otherwise |
| $\mathrm{KL}(P, Q)$ or $D(P,Q)$ | the relative entropy of $P$ with respect to $Q$ |

## 1. Introduction

### 1.1 The co-design problem in its general form

A legged robot is a body and a controller. The body is fixed by design variables, here the lengths of the thigh and the shank, the pose about which the joints are commanded, and the actuator that drives each joint group, and the controller is a policy that maps sensed state to joint commands. Write $x$ for the vector of design variables and $\pi$ for the policy. The performance of the pair is an expected return $J(\pi, x)$, the long run sum of rewards that the policy collects when it drives the body $x$ on the task. The co-design problem is to choose both,

$$\max_{x \in \mathcal{X}} \; \max_{\pi \in \Pi} \; J(\pi, x),$$

where $\mathcal{X}$ is the set of admissible designs and $\Pi$ the set of representable policies. Written this way the problem is bilevel, an outer search over bodies wrapped around an inner search over controllers, and the inner search is itself a full reinforcement learning problem that may take hours of simulation for one body [1]. The difficulty of the field follows directly from this nesting. Every evaluation of a design is expensive, the evaluation is noisy because the inner learner is stochastic, and the value assigned to a design depends on how well the inner learner happened to adapt to it.

A second formulation removes the nesting by training one controller across many bodies at once. If the controller observes the design, so that it is a design-conditioned policy $\pi(a \mid s, x)$, then a single policy can serve every design, and the outer search can be carried out concurrently with training,

$$\max_{p} \; \max_{\pi} \; \mathbb{E}_{x \sim p}\big[J(\pi, x)\big],$$

where $p$ is a probability distribution over designs that the outer search moves toward the better region as training proceeds [2]. This concurrent form is the one this workspace implements.

### 1.2 The co-optimisation loop as it stands

The pipeline trains one control policy across a population of robot designs at once. A design-conditioned actor-critic is optimised by proximal policy optimisation, while every 480 policy iterations an outer search proposes a fresh population of 256 designs (`train.py:222`, `train.py:225`), the first 12000 iterations being spent on randomly drawn designs before the search begins (`train.py:226`). Each design is assigned to a fixed residue class of environments (`copt_on_policy_runner.py:746`), so that with 4096 environments in the solefoot configuration (`limx_base_env_cfg.py:1237`) each design is driven by sixteen of them. The fitness of a design is the mean return of the episodes its environments complete during the interval (`copt_on_policy_runner.py:372`, `copt_on_policy_runner.py:733`), and the search receives the negated fitness as a cost (`usd_generator.py:619`). The critic observes the masses and inertias of the design it evaluates (`limx_base_env_cfg.py:289`, `limx_base_env_cfg.py:290`) and, in the co-optimisation task, its link lengths (`limx_base_env_cfg.py:1276`), as section 5.2 of `copt.md` first established.

The time scales are worth fixing because later sections depend on them. Each policy iteration collects 25 control steps per environment (`limx_rsl_rl_ppo_cfg.py:98`), the control period is 0.02 seconds and an episode lasts 20 seconds, that is 1000 control steps (`limx_base_env_cfg.py:1250`, `limx_base_env_cfg.py:1254`). One generation of 480 iterations therefore gives each environment 12000 control steps, enough for at most twelve full episodes, so a design's fitness rests on at most 192 full-length episodes across its sixteen environments, and on more but shorter ones where episodes end early. The co-optimisation runner trains for 45000 iterations (`limx_rsl_rl_ppo_cfg.py:146`), so after the random phase there are about $(45000 - 12000)/480 \approx 68$ search generations, and about $68 \times 256 \approx 17{,}400$ design evaluations in a run.

Two searches are available. `CMAESDesignGenerator` adapts a Gaussian over two link length scales bounded to the interval from 0.75 to 1.25 (`usd_generator.py:216`), and `CatCMAESDesignGenerator` (`usd_generator.py:729`) searches those two scales jointly with one categorical actuator choice per joint group, through the mixed-variable strategy CatCMAwM [3] constructed at `usd_generator.py:829`. Each scaled length is rounded to the centimetre (`usd_generator.py:416`), which the audit (section D8) shows leaves 17 by 19, or 323, length designs for the biped and 12 by 12, or 144, for the quadruped. The biped has four actuator groups and the quadruped three (`usd_generator.py:179`, `usd_generator.py:186`), drawing on catalogues of fifteen and fourteen entries, so the biped's design space holds $323 \times 15^4 = 16{,}351{,}875$ discrete designs and the quadruped's $144 \times 14^3 = 395{,}136$. A run evaluates about one design in a thousand of the biped's space even if no design were ever repeated. The audit (section 4.1) has further established that the nominal pose ought to be a design variable, since nothing in a velocity tracking task fixes the height at which a robot stands, and that the pose, unlike the lengths, would not be rounded.

### 1.3 The proposal under examination

The proposal is to replace or augment the outer search with a method in the spirit of the multi-armed bandit, in which a neural network, trained over a large number of trials, predicts the link lengths, the nominal pose and the actuator of a good design. A bandit is the simplest formal model of learning by trial, a sequence of choices among options of unknown worth in which each choice both earns a reward and reveals information. The design search has exactly that character. A design must be built and trained upon before its merit is known, every generation spent on a poor design is a generation not spent on a good one, and the number of designs vastly exceeds the number that can ever be tried, so the search must infer the merit of untried designs from the tried ones. A follow-up question asks whether proximal policy optimisation itself can be modified so that the design choice becomes part of its formulation rather than a separate outer loop, and whether prior work has done so in this or a similar domain. Section 9 answers it.

### 1.4 Plan of the report

Section 2 assembles the probability, concentration, information and gradient identities that every later derivation uses. Section 3 develops the finite-armed bandit in full, its regret, its lower bounds and its four families of algorithm, answering the fourth request. Sections 4 and 5 treat bandits over continuous and over hybrid decision spaces, answering the second and third. Section 6 treats neural network bandits and their learning, answering the eighth. Sections 7 and 8 survey the use of bandits in robot design and in the co-optimisation of design and control, answering the first and fifth. Section 9 treats the integration of the bandit into proximal policy optimisation, answering the follow-up. Section 10 surveys adjacent domains, answering the seventh, and section 11 derives the modifications a bandit method requires here, answering the sixth. Section 12 distils the survey into candidate contributions and section 13 records what the retrieved literature does not contain.

## 2. Mathematical background

This section collects the tools on which the rest of the report depends. Each is stated, each symbol is defined, and each is derived where the derivation is short enough to be instructive. A reader comfortable with probability may skip to section 3 and return here when a later argument cites a numbered tool.

### 2.1 Random variables, expectation and conditioning

A random variable $X$ is a quantity whose value is determined by chance, described by its distribution, the probabilities with which it takes each value. Its expectation $\mathbb{E}[X]$ is its probability-weighted average, $\sum_x x \Pr(X = x)$ for a discrete variable and $\int x\, p(x)\, dx$ for one with density $p$, and its variance is $\mathbb{V}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$. Expectation is linear, $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ for constants $a, b$, whether or not $X$ and $Y$ are independent. For two independent variables the variance of the sum is the sum of the variances, and the variance of the mean of $n$ independent copies of $X$ is $\mathbb{V}[X]/n$, which is the reason averages concentrate.

The conditional expectation $\mathbb{E}[X \mid Y]$ is the average of $X$ computed with $Y$ held at its observed value, and is therefore itself a random variable, a function of $Y$. Two of its properties are used constantly. The tower rule states $\mathbb{E}\big[\mathbb{E}[X \mid Y]\big] = \mathbb{E}[X]$, that averaging a conditional average over the conditioning variable recovers the plain average. And any function of the conditioning variable may be taken outside, $\mathbb{E}[g(Y) X \mid Y] = g(Y)\,\mathbb{E}[X \mid Y]$. In bandit analysis the conditioning variable is the history $\mathcal{H}_{t-1} = (A_1, X_1, \dots, A_{t-1}, X_{t-1})$ of choices and rewards before round $t$, and $\mathbb{E}_{t-1}[\cdot]$ abbreviates $\mathbb{E}[\cdot \mid \mathcal{H}_{t-1}]$.

### 2.2 Concentration, from Markov to Hoeffding

Every confidence bound in this report rests on the fact that an average of independent observations is unlikely to stray far from its mean, and the precise rate at which that likelihood falls is what fixes the width of the bound. The argument is built in four steps, following the standard treatment [4].

The first step is Markov's inequality. For a non-negative random variable $Y$ and any $c > 0$, $P(Y \ge c) \le \mathbb{E}[Y]/c$. It follows from $\mathbb{E}[Y] \ge \mathbb{E}[Y \mathbb{1}[Y \ge c]] \ge c P(Y \ge c)$, the first inequality because $Y$ is non-negative and the second because $Y \ge c$ wherever the indicator is one.

The second step defines a class of well-behaved variables. A random variable $X$ with mean zero is $\sigma$-subgaussian if its moment generating function satisfies

$$\mathbb{E}[\exp(\lambda X)] \le \exp\!\big(\lambda^2\sigma^2/2\big) \quad \text{for all } \lambda \in \mathbb{R},$$

where $\lambda$ is a free real parameter and $\sigma > 0$ plays the role of a standard deviation [4]. A Gaussian with variance $\sigma^2$ satisfies this with equality, and any variable confined to an interval of length $b - a$ is $(b-a)/2$-subgaussian, which is Hoeffding's lemma. Rewards in $[0,1]$ are therefore $\tfrac12$-subgaussian about their means.

The third step is the Cramér-Chernoff method, which converts the definition into a tail bound. For $\varepsilon \ge 0$ and any $\lambda > 0$, applying Markov's inequality to the non-negative variable $\exp(\lambda X)$ gives

$$P(X \ge \varepsilon) = P\big(e^{\lambda X} \ge e^{\lambda\varepsilon}\big) \le \mathbb{E}[e^{\lambda X}] e^{-\lambda\varepsilon} \le \exp\!\Big(\frac{\lambda^2\sigma^2}{2} - \lambda\varepsilon\Big),$$

and choosing $\lambda = \varepsilon/\sigma^2$, which minimises the exponent, yields $P(X \ge \varepsilon) \le \exp(-\varepsilon^2/(2\sigma^2))$ [4]. The tail of a subgaussian variable thus decays as fast as a Gaussian tail.

The fourth step passes to averages. If $X_1, \dots, X_n$ are independent and each $X_i - \mu$ is $\sigma$-subgaussian, then the sum of independent subgaussian variables is subgaussian with the variances added, so the centred mean $\hat\mu - \mu = \frac1n\sum_i (X_i - \mu)$ is $\sigma/\sqrt n$-subgaussian, and

$$P(\hat\mu \ge \mu + \varepsilon) \le \exp\!\Big(-\frac{n\varepsilon^2}{2\sigma^2}\Big), \qquad P(\hat\mu \le \mu - \varepsilon) \le \exp\!\Big(-\frac{n\varepsilon^2}{2\sigma^2}\Big),$$

where $\hat\mu$ is the empirical mean of $n$ observations, $\mu$ their common mean and $\varepsilon$ the deviation of interest [4]. Read in the other direction, setting the right-hand side to a small probability $\delta$ and solving for $\varepsilon$ shows that with probability at least $1 - \delta$ the true mean lies below $\hat\mu + \sqrt{2\sigma^2\log(1/\delta)/n}$. That expression is the confidence width of every upper confidence bound in section 3, and its two features, a square root of a logarithm of the inverse failure probability and a division by the square root of the number of samples, recur throughout.

### 2.3 Relative entropy

Lower bounds ask how hard it is to tell two reward distributions apart, and the measure of that difficulty is the relative entropy. For distributions $P$ and $Q$ with densities $p$ and $q$, the relative entropy, or Kullback-Leibler divergence, is

$$\mathrm{KL}(P, Q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx,$$

the expected log likelihood ratio of a sample drawn from $P$. It is non-negative, zero exactly when $P = Q$, and not symmetric. For two Gaussians of equal variance, $\mathrm{KL}(\mathcal{N}(\mu_1,\sigma^2), \mathcal{N}(\mu_2,\sigma^2)) = (\mu_1 - \mu_2)^2/(2\sigma^2)$, so distinguishing means that differ by $\Delta$ costs on the order of $2\sigma^2/\Delta^2$ samples. Two further facts are used in section 3. The divergence between the joint laws of two bandit interactions decomposes into a sum over arms of the expected number of pulls times the per-arm divergence [4], and the Bretagnolle-Huber inequality states that for any event $E$, $P(E) + Q(E^c) \ge \tfrac12\exp(-\mathrm{KL}(P,Q))$, where $E^c$ is the complement of $E$, so that two laws with small divergence cannot both assign small probability to complementary events [4].

In the continuous setting the relative entropy also measures distance between policies and between search distributions, and its second-order expansion defines the Fisher information. For a parametric family $p_\phi$ and a small change $\delta\phi$ of its parameters,

$$\mathrm{KL}(p_{\phi + \delta\phi}, p_\phi) \approx \tfrac12\,\delta\phi^\top F(\phi)\,\delta\phi, \qquad F(\phi) = \mathbb{E}_{x \sim p_\phi}\big[\nabla_\phi \log p_\phi(x)\,\nabla_\phi \log p_\phi(x)^\top\big],$$

where $F(\phi)$ is the Fisher information matrix of the family at $\phi$ [5]. The Fisher matrix is the local metric under which a change in parameters is measured by the change it produces in the distribution, which is what sections 2.6 and 4.9 exploit.

### 2.4 Bayes' rule, conjugacy and Gaussian conditioning

Bayesian bandits maintain a belief about unknown quantities and revise it as data arrive. If $\theta$ is an unknown parameter with prior density $p(\theta)$ and data $\mathcal{D}$ have likelihood $p(\mathcal{D} \mid \theta)$, Bayes' rule gives the posterior

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{\int p(\mathcal{D} \mid \theta')\,p(\theta')\,d\theta'}.$$

A prior is conjugate to a likelihood when the posterior lies in the same family as the prior, which makes updating a matter of adjusting a few numbers. Two conjugate pairs appear in section 3. For a Bernoulli arm with success probability $\theta$ and a $\mathrm{Beta}(\alpha, \beta)$ prior, whose density is proportional to $\theta^{\alpha-1}(1-\theta)^{\beta-1}$, observing $S$ successes and $F$ failures multiplies the prior by $\theta^S(1-\theta)^F$, so the posterior is $\mathrm{Beta}(\alpha + S, \beta + F)$. For a Gaussian arm with unknown mean $\theta$, known noise variance $\sigma^2$ and prior $\mathcal{N}(m_0, \tau^2)$, completing the square in the exponent of the product of $n$ Gaussian likelihoods and the prior gives a Gaussian posterior with precision, the inverse variance, equal to the prior precision plus $n$ times the noise precision, $s_n^{-2} = \tau^{-2} + n\sigma^{-2}$, and mean $m_n = s_n^2(m_0\tau^{-2} + \sigma^{-2}\sum_i X_i)$, a precision-weighted average of the prior mean and the data.

Gaussian processes, in section 4.5, rest on one further identity. If two jointly Gaussian vectors $u$ and $v$ have zero mean and covariance blocks $\Sigma_{uu}$, $\Sigma_{uv}$ and $\Sigma_{vv}$, then the conditional distribution of $u$ given an observed $v$ is Gaussian with

$$\mathbb{E}[u \mid v] = \Sigma_{uv}\Sigma_{vv}^{-1} v, \qquad \mathrm{Cov}[u \mid v] = \Sigma_{uu} - \Sigma_{uv}\Sigma_{vv}^{-1}\Sigma_{vu}.$$

The identity follows by verifying that $u - \Sigma_{uv}\Sigma_{vv}^{-1}v$ is uncorrelated with $v$, hence independent of it because the two are jointly Gaussian, so that conditioning on $v$ leaves its distribution unchanged [6]. The second term of the covariance, the reduction in uncertainty, does not depend on the observed value of $v$, only on which variables were observed, a fact that makes batch selection in section 11 tractable.

### 2.5 The gradient of an expectation

Many methods in this report improve a probability distribution by following the gradient of the expected reward it earns. Let $p_\phi(x)$ be a distribution over decisions $x$ with parameters $\phi$, and let $F(x)$ be a reward that cannot be differentiated, because it is produced by a simulator, a trained policy or a physical experiment. The objective $J(\phi) = \mathbb{E}_{x \sim p_\phi}[F(x)] = \int F(x)\,p_\phi(x)\,dx$ can nevertheless be differentiated, because $\phi$ enters only through the density. Using $\nabla_\phi p_\phi = p_\phi \nabla_\phi \log p_\phi$,

$$\nabla_\phi J(\phi) = \int F(x)\,\nabla_\phi p_\phi(x)\,dx = \int F(x)\,p_\phi(x)\,\nabla_\phi\log p_\phi(x)\,dx = \mathbb{E}_{x\sim p_\phi}\big[F(x)\,\nabla_\phi \log p_\phi(x)\big].$$

This is the likelihood ratio, or score function, identity, and the term $\nabla_\phi\log p_\phi(x)$ is called the score [7]. The identity turns a gradient into an expectation that can be estimated by sampling, $\nabla_\phi J \approx \frac1N\sum_{i=1}^N F(x_i)\nabla_\phi\log p_\phi(x_i)$ for samples $x_i \sim p_\phi$, without ever differentiating $F$.

The estimator is unbiased but its variance can be large, and a baseline reduces it. Because the score has mean zero, $\mathbb{E}[\nabla_\phi \log p_\phi(x)] = \int \nabla_\phi p_\phi(x)\,dx = \nabla_\phi \int p_\phi(x)\,dx = \nabla_\phi 1 = 0$, any constant $b$ may be subtracted from the reward without changing the expectation,

$$\nabla_\phi J(\phi) = \mathbb{E}_{x\sim p_\phi}\big[(F(x) - b)\,\nabla_\phi \log p_\phi(x)\big],$$

and a $b$ near the typical reward, such as the batch average, shrinks the terms being averaged and hence their variance [7, 8]. In reinforcement learning the decision $x$ is a whole trajectory and the baseline becomes a learned value function, which is the origin of the advantage in section 9.

Two scores are needed repeatedly. For a Gaussian $p(x) = \mathcal{N}(x; \mu, \sigma^2)$ with mean $\mu$ and standard deviation $\sigma$, differentiating $\log p = -\log\sigma - (x-\mu)^2/(2\sigma^2) + \text{const}$ gives $\partial_\mu \log p = (x - \mu)/\sigma^2$ and $\partial_\sigma \log p = ((x-\mu)^2 - \sigma^2)/\sigma^3$ [9]. For a categorical distribution over $M$ outcomes with logits $\ell \in \mathbb{R}^M$ and probabilities $\mathrm{softmax}(\ell)_j = e^{\ell_j}/\sum_{j'} e^{\ell_{j'}}$, the score of outcome $c$ is $\nabla_\ell \log\mathrm{softmax}(\ell)_c = e_c - \mathrm{softmax}(\ell)$, where $e_c$ is the indicator vector of $c$, because $\log\mathrm{softmax}(\ell)_c = \ell_c - \log\sum_{j'} e^{\ell_{j'}}$ and the gradient of the log-sum-exp is the softmax.

### 2.6 The natural gradient

A plain gradient step treats every parameter direction alike, yet a fixed step in one parameter may change the distribution far more than the same step in another. The natural gradient corrects this by asking for the steepest ascent per unit change of distribution rather than per unit change of parameter. Maximising the linearised objective $J(\phi + \delta\phi) \approx J(\phi) + \delta\phi^\top \nabla_\phi J$ subject to the constraint $\mathrm{KL}(p_{\phi + \delta\phi}, p_\phi) = \varepsilon$, and replacing the divergence by its second-order expansion of section 2.3, the Lagrangian condition is $F(\phi)\,\delta\phi = \beta\,\nabla_\phi J$ for a multiplier $\beta > 0$, so the optimal direction is

$$\tilde\nabla_\phi J = F(\phi)^{-1}\nabla_\phi J(\phi),$$

the natural gradient [5]. Its practical virtue is invariance. The update it produces does not depend on how the distribution is parameterised, so a search that follows it behaves the same whether a Gaussian is described by its standard deviation or by the logarithm of it. Natural evolution strategies, CMA-ES, CatCMA, trust region policy optimisation and, approximately, proximal policy optimisation are all instances of this one idea, which is why they recur together in sections 4.9, 5.5 and 9.

## 3. The stochastic multi-armed bandit

### 3.1 The interaction protocol and the regret

The problem takes its name from a gambler facing a row of slot machines, and its formal study begins with Robbins [10] and, earlier still, with Thompson's rule for choosing between two unknown probabilities [11]. Three monographs give the modern account [4, 12, 13], and this section follows the first of them except where stated.

There are $K$ arms. Arm $a \in \{1, \dots, K\}$ carries a reward distribution $P_a$, unknown to the learner, with mean $\mu_a$. At each round $t = 1, \dots, n$ the learner chooses an arm $A_t$ using only the history $\mathcal{H}_{t-1}$ of earlier choices and rewards, and possibly an independent source of randomness, and then observes a reward $X_t$ drawn from $P_{A_t}$ independently of everything else given $A_t$. A rule mapping histories to choices is a policy. Write $\mu^\star = \max_a \mu_a$ for the best mean, $\Delta_a = \mu^\star - \mu_a \ge 0$ for the gap of arm $a$, and $T_a(t) = \sum_{s=1}^t \mathbb{1}[A_s = a]$ for the number of times arm $a$ has been chosen by the end of round $t$. The cumulative regret of a policy over $n$ rounds is

$$R_n = n\mu^\star - \mathbb{E}\Big[\sum_{t=1}^n X_t\Big],$$

the expected reward forgone against an oracle that knows the best arm and plays it always [4].

### 3.2 The regret decomposition

The regret can be rewritten as a weighted count of mistakes, and this rewriting is the starting point of almost every analysis. Since exactly one arm is chosen in each round, $\sum_a \mathbb{1}[A_t = a] = 1$, so

$$R_n = \sum_{t=1}^n \mathbb{E}[\mu^\star - X_t] = \sum_{a=1}^K \sum_{t=1}^n \mathbb{E}\big[(\mu^\star - X_t)\mathbb{1}[A_t = a]\big].$$

Conditioning on $A_t$ and using that the reward has mean $\mu_{A_t}$ given the choice, $\mathbb{E}[(\mu^\star - X_t)\mathbb{1}[A_t = a] \mid A_t] = \mathbb{1}[A_t = a](\mu^\star - \mu_a) = \mathbb{1}[A_t = a]\Delta_a$, and taking expectations by the tower rule and summing over $t$,

$$R_n = \sum_{a=1}^K \Delta_a\, \mathbb{E}[T_a(n)].$$

This is the regret decomposition lemma [4]. It says that regret is governed entirely by how often each inferior arm is tried, weighted by how inferior it is. An arm with a large gap must be abandoned quickly, an arm with a small gap may be tried often without much loss, and an arm never tried cannot be judged at all. This is the exploration and exploitation trade-off in its plainest form.

### 3.3 Why naive strategies fail

The greedy policy, which tries each arm once and then always plays the arm with the highest empirical mean, can suffer regret growing linearly in $n$. If the best arm happens to return a poor first reward, its empirical mean may sit below that of an inferior arm forever, since the greedy policy never samples it again to correct the estimate. Some deliberate exploration is therefore necessary.

The simplest remedy is explore-then-commit. It plays each arm $m$ times in turn, then commits for the remaining $n - mK$ rounds to the arm with the highest empirical mean after exploration. Its analysis shows the trade-off quantitatively [4]. Each suboptimal arm $i$ is played $m$ times in exploration and then, if chosen at commitment, $n - mK$ more times, so $\mathbb{E}[T_i(n)] \le m + (n - mK)\Pr(\hat\mu_i \ge \hat\mu_1)$, where arm 1 is optimal and the empirical means are taken after $m$ samples each. The difference $(\hat\mu_i - \mu_i) - (\hat\mu_1 - \mu_1)$ is a difference of two independent means of $m$ samples, so for 1-subgaussian rewards it is $\sqrt{2/m}$-subgaussian, and the concentration bound of section 2.2 gives $\Pr(\hat\mu_i \ge \hat\mu_1) \le \exp(-m\Delta_i^2/4)$. Substituting into the decomposition,

$$R_n \le m\sum_{i=1}^K \Delta_i + (n - mK)\sum_{i=1}^K \Delta_i \exp\!\Big(-\frac{m\Delta_i^2}{4}\Big),$$

where $m$ is the number of exploratory pulls per arm [4]. The first term grows with $m$ and the second shrinks, so $m$ must be tuned, and the best $m$ depends on the unknown gaps. With two arms and a gap $\Delta$, the optimum lies near $m \approx (4/\Delta^2)\log(n\Delta^2/4)$, which needs the gap to be known, whereas a choice of $m$ that depends only on $n$ guarantees regret of order $n^{2/3}$ whatever the gap, short of the $\sqrt n$ that adaptive policies achieve [4]. The ε-greedy policy, which with probability $\varepsilon$ plays a uniformly random arm and otherwise the empirical best, suffers the same disease in another form, since a fixed $\varepsilon$ wastes a constant fraction of rounds forever. The algorithms that follow avoid tuning to the gaps by letting the data decide, round by round, how much each arm still deserves to be explored.

### 3.4 Lower bounds, the limit on what any policy can achieve

Before constructing good policies it is worth knowing how good a policy can possibly be. Two kinds of lower bound exist, one for the worst case over problems and one for each fixed problem.

The minimax bound states that for $K > 1$ arms and $n \ge K - 1$ rounds, for every policy there is a Gaussian bandit with unit variance and means in $[0,1]$ on which its regret is at least $\frac{1}{27}\sqrt{(K-1)n}$ [4]. The proof illustrates a technique used repeatedly. Fix a policy and the bandit $\nu$ with means $(\Delta, 0, \dots, 0)$, so arm 1 is best. Some arm $i \ne 1$ must be pulled at most $n/(K-1)$ times in expectation, because the expected counts sum to $n$. Construct a second bandit $\nu'$ identical except that arm $i$ has mean $2\Delta$, so that $i$ is now best. By the divergence decomposition, the relative entropy between the laws of the interaction under $\nu$ and $\nu'$ equals $\mathbb{E}_\nu[T_i(n)]\cdot(2\Delta)^2/2 \le 2n\Delta^2/(K-1)$, since the two bandits differ only on arm $i$. If the policy plays arm 1 fewer than $n/2$ times it suffers regret at least $n\Delta/2$ under $\nu$, and if it plays arm 1 more than $n/2$ times it suffers at least $n\Delta/2$ under $\nu'$. The Bretagnolle-Huber inequality then bounds the sum of the two probabilities below by $\frac12\exp(-2n\Delta^2/(K-1))$, so the sum of the two regrets is at least $\frac{n\Delta}{4}\exp(-2n\Delta^2/(K-1))$, and choosing $\Delta = \sqrt{(K-1)/(4n)}$ gives the stated order. No policy can do better than $\sqrt{Kn}$ against every problem, because a problem can always be found on which a rarely sampled arm is in truth the best.

The instance-dependent bound, due to Lai and Robbins, fixes the problem and asks how regret grows with $n$ [14]. A policy is consistent over a class of bandits if its regret grows more slowly than every power of $n$ on every bandit of the class, that is $R_n/n^p \to 0$ for every $p > 0$, which excludes policies that ignore data. For a consistent policy, every suboptimal arm $i$ satisfies

$$\liminf_{n\to\infty} \frac{\mathbb{E}[T_i(n)]}{\log n} \ge \frac{1}{d_i}, \qquad d_i = \inf\big\{\mathrm{KL}(P_i, P') : P' \text{ in the class, with mean } > \mu^\star\big\},$$

where $d_i$ is the smallest relative entropy between arm $i$'s distribution and any distribution of the class that would make arm $i$ the best [14, 4]. Multiplying by the gaps and summing through the decomposition, $\liminf_n R_n/\log n \ge \sum_{i:\Delta_i>0}\Delta_i/d_i$. The proof uses the same two-bandit argument. Replace arm $i$'s distribution by the nearest distribution $P'$ that makes it optimal, observe that the divergence between the two interactions is $\mathbb{E}[T_i(n)]\,\mathrm{KL}(P_i, P')$, and note that consistency forces the policy to play each bandit's own optimal arm almost always, arm $i$ rarely in the original and often in the altered one, which by Bretagnolle-Huber is impossible unless $\mathbb{E}[T_i(n)]$ is at least $\log n/\mathrm{KL}(P_i, P')$ asymptotically [4]. For Gaussian arms of variance $\sigma^2$, $d_i = \Delta_i^2/(2\sigma^2)$, so a good policy must pull each suboptimal arm about $2\sigma^2\log n/\Delta_i^2$ times, the number of samples needed to resolve a gap of $\Delta_i$ to confidence $1/n$. Every algorithm below is judged against these two benchmarks, logarithmic growth on each fixed problem and square root growth in the worst case.

### 3.5 Optimism, the upper confidence bound

The first family of good policies acts as though each arm were as good as the data plausibly allow, a principle called optimism in the face of uncertainty. After $T_a(t-1)$ pulls of arm $a$ with empirical mean $\hat\mu_a(t-1)$, the concentration bound of section 2.2 says that the true mean lies below $\hat\mu_a(t-1) + \sqrt{2\log(1/\delta)/T_a(t-1)}$ with probability at least $1-\delta$ for 1-subgaussian rewards. The UCB policy plays the arm whose upper bound is largest,

$$A_t = \arg\max_a \; \mathrm{UCB}_a(t-1,\delta), \qquad \mathrm{UCB}_a(t-1, \delta) = \hat\mu_a(t-1) + \sqrt{\frac{2\log(1/\delta)}{T_a(t-1)}},$$

where $\delta$ is a confidence level and every arm is played once before the rule applies [4]. The bound is large either because the estimate is high or because the arm has been sampled little, and the second reason expires of its own accord as the arm is sampled.

The analysis is worth giving in full, since every optimistic method in this report, including GP-UCB and NeuralUCB, repeats its structure [4]. Suppose arm 1 is optimal and fix a suboptimal arm $i$. Define a good event $G_i$ under which two things hold. First, the upper bound of arm 1 never falls below $\mu_1$ at any round. Second, after some number $u_i$ of samples, the upper bound of arm $i$ computed from its first $u_i$ samples lies below $\mu_1$. On $G_i$, arm $i$ is played at most $u_i$ times, because to be played a $(u_i+1)$-th time its bound, computed from exactly $u_i$ samples, would have to exceed the bound of arm 1, yet the former is below $\mu_1$ and the latter above it. Since $T_i(n) \le n$ always, $\mathbb{E}[T_i(n)] \le u_i + n\Pr(G_i^c)$. It remains to bound the probability that $G_i$ fails. The first part can fail only if for some sample count $s \le n$ the mean of arm 1's first $s$ samples undershoots $\mu_1$ by more than the width, which by a union bound over $s$ has probability at most $n\delta$. For the second part, choose $u_i$ large enough that $\Delta_i - \sqrt{2\log(1/\delta)/u_i} \ge c\Delta_i$ for a constant $c \in (0,1)$, so that failure requires arm $i$'s mean to overshoot $\mu_i$ by at least $c\Delta_i$, which has probability at most $\exp(-u_i c^2\Delta_i^2/2)$. The smallest such $u_i$ is $\lceil 2\log(1/\delta)/((1-c)^2\Delta_i^2)\rceil$. With $\delta = 1/n^2$ and $c = \tfrac12$ these give $\mathbb{E}[T_i(n)] \le 3 + 16\log n/\Delta_i^2$, and substituting into the decomposition,

$$R_n \le 3\sum_{i=1}^K\Delta_i + \sum_{i:\Delta_i > 0}\frac{16\log n}{\Delta_i}.$$

This matches the Lai-Robbins order, logarithmic in $n$ and inversely proportional to the gaps [4]. Splitting the arms into those with gaps below and above a threshold $\Delta$ and optimising $\Delta$ also yields the gap-free bound $R_n \le 8\sqrt{nK\log n} + 3\sum_i\Delta_i$, within a logarithmic factor of the minimax lower bound [4]. Auer, Cesa-Bianchi and Fischer proved the original finite-time result for the variant UCB1, whose index is $\hat\mu_a + \sqrt{2\ln t/T_a}$ with the round number $t$ in place of a fixed confidence, for rewards in $[0,1]$, with regret at most $8\sum_{a:\Delta_a>0}\ln n/\Delta_a + (1 + \pi^2/3)\sum_a\Delta_a$ [15].

### 3.6 Posterior sampling, Thompson's rule

The second family is Bayesian. A prior is placed over the unknown means, the posterior is updated after each reward by Bayes' rule, and at each round one sample $\tilde\theta \sim p(\theta \mid \mathcal{H}_{t-1})$ is drawn from the posterior, after which the arm best under that sample is played,

$$A_t = \arg\max_a \mu_a(\tilde\theta),$$

where $\mu_a(\theta)$ is the mean of arm $a$ under parameter $\theta$ [11]. Each arm is thereby chosen with exactly the posterior probability that it is the best, a property called probability matching. For Bernoulli arms with uniform priors the posterior of arm $a$ after $S_a$ successes and $F_a$ failures is $\mathrm{Beta}(1 + S_a, 1 + F_a)$, and for Gaussian arms the posterior is the Gaussian of section 2.4, so each round costs one random draw per arm. An arm whose posterior is wide is sometimes sampled high and therefore sometimes chosen, which is how the rule explores, and the width shrinks as the arm is sampled, which is how exploration ends.

Its Bayesian regret, the regret averaged over problems drawn from the prior, admits an analysis that explains why posterior sampling and optimism share their guarantees [16, 4]. Let $A^\star$ be the optimal arm, a random variable because the means are drawn from the prior, and let $U_t(a)$ be any upper confidence bound computed from the history. Probability matching means that given the history, $A_t$ and $A^\star$ have the same distribution, so $\mathbb{E}_{t-1}[U_t(A_t)] = \mathbb{E}_{t-1}[U_t(A^\star)]$. Adding and subtracting this quantity,

$$\mathbb{E}_{t-1}[\mu_{A^\star} - \mu_{A_t}] = \mathbb{E}_{t-1}[\mu_{A^\star} - U_t(A^\star)] + \mathbb{E}_{t-1}[U_t(A_t) - \mu_{A_t}].$$

The first term is non-positive whenever the upper bounds hold, and the second is the width of the bound at the chosen arm, which is exactly the quantity summed in the UCB analysis. Summing over rounds, the Bayesian regret of posterior sampling is bounded by the regret bound of any upper confidence bound policy whose bounds are valid, although posterior sampling never computes those bounds [16]. For $K$ arms with 1-subgaussian rewards this gives a Bayesian regret of order $\sqrt{Kn\log n}$ [4], and Agrawal and Goyal proved frequentist logarithmic regret for Bernoulli arms [17]. Posterior sampling has one property of special value in this workspace. Because each choice is a random draw, a batch of $B$ independent draws yields $B$ distinct and sensible choices without any coordination between them, which section 11 uses to choose 256 designs at once.

### 3.7 Preference learning, the gradient bandit

The third family learns a probability distribution over arms directly, and it is the family to which a neural network that outputs designs belongs. Sutton and Barto present it as the gradient bandit algorithm [8]. Each arm carries a numerical preference $H(a)$, arms are drawn from the softmax distribution

$$\pi(a) = \frac{e^{H(a)}}{\sum_{b=1}^K e^{H(b)}},$$

and the aim is to raise the expected reward $J(H) = \sum_a \pi(a)\mu_a$. The derivative of the softmax is $\partial\pi(b)/\partial H(a) = \pi(b)(\mathbb{1}[a=b] - \pi(a))$, and because these derivatives sum to zero over $b$, any constant baseline $\bar R$ may be subtracted from the means without changing the gradient, as in section 2.5,

$$\frac{\partial J}{\partial H(a)} = \sum_b (\mu_b - \bar R)\,\pi(b)\big(\mathbb{1}[a = b] - \pi(a)\big) = \mathbb{E}_{A\sim\pi}\big[(\mu_A - \bar R)(\mathbb{1}[a = A] - \pi(a))\big].$$

Replacing the expectation by the single observed reward $R_t$ of the arm $A_t$ actually drawn gives the stochastic gradient ascent update

$$H_{t+1}(a) = H_t(a) + \alpha\,(R_t - \bar R_t)\big(\mathbb{1}[a = A_t] - \pi_t(a)\big),$$

where $\alpha$ is a step size and $\bar R_t$ the running mean reward [8]. If the observed reward beats the baseline, the chosen arm's preference rises and every other arm's falls in proportion to its probability, and the reverse if the reward falls short. The rule is the REINFORCE estimator of section 2.5 applied to a softmax [7], and it generalises to any parameterised distribution over decisions, discrete or continuous, which is how a neural network learns to predict good designs.

Its convergence has been settled only recently. With the exact gradient, softmax policy gradient converges to the optimal arm at a rate of order $1/t$, with constants that depend on the initialisation through the smallest probability the optimal arm ever has, and adding an entropy regulariser makes the convergence linear, of order $e^{-ct}$, toward the regularised optimum [18]. With sampled rewards, the stochastic gradient bandit converges to the globally optimal arm almost surely at rate $1/t$ even with a constant step size, because the noise of its updates shrinks as progress slows and because the update itself prevents any arm's probability from decaying faster than order $1/t$, so every arm is sampled infinitely often [19]. The practical weakness remains plain. The rule carries no explicit measure of uncertainty and explores only through the randomness of $\pi$, so where its probabilities have sharpened early on a mediocre arm, as they may under a poor initialisation, recovery can take a very long time, which is why an entropy term or a floor on the probabilities is routinely imposed.

### 3.8 Adversarial rewards and exponential weights

The fourth family drops the assumption that each arm's reward distribution is fixed and allows the rewards to be chosen by an adversary, subject only to lying in $[0,1]$. Regret is then measured against the best single arm in hindsight. The key device is importance weighting. If arm $a$ is drawn with probability $P_t(a)$ and the reward $X_t$ is observed only for the drawn arm, the estimate

$$\hat X_t(a) = 1 - \frac{\mathbb{1}[A_t = a](1 - X_t)}{P_t(a)}$$

has conditional expectation equal to the reward arm $a$ would have earned, because the indicator has expectation $P_t(a)$ which cancels the denominator, so every arm's reward is estimated without bias even though only one is seen [4]. Exp3, the exponential-weight algorithm for exploration and exploitation, keeps the running sums $\hat S_{t-1}(a) = \sum_{s<t}\hat X_s(a)$ and plays

$$P_t(a) = \frac{\exp(\eta\hat S_{t-1}(a))}{\sum_b\exp(\eta\hat S_{t-1}(b))},$$

where $\eta > 0$ is a learning rate [4, 20]. With $\eta = \sqrt{\log K/(nK)}$ its regret against the best fixed arm is at most $2\sqrt{nK\log K}$ [4].

The proof is short and shows where the learning rate comes from. Let $W_t = \sum_a\exp(\eta\hat S_t(a))$, with $W_0 = K$. For any arm $i$, $\exp(\eta\hat S_n(i)) \le W_n = K\prod_{t=1}^n W_t/W_{t-1}$. Each ratio equals $\sum_a P_t(a)\exp(\eta\hat X_t(a))$, and because $\hat X_t(a) \le 1$ the inequalities $e^x \le 1 + x + x^2$ for $x \le 1$ and $1 + x \le e^x$ give $W_t/W_{t-1} \le \exp(\eta\sum_a P_t(a)\hat X_t(a) + \eta^2\sum_a P_t(a)\hat X_t(a)^2)$. Taking logarithms and dividing by $\eta$,

$$\hat S_n(i) - \sum_{t=1}^n\sum_a P_t(a)\hat X_t(a) \le \frac{\log K}{\eta} + \eta\sum_{t=1}^n\sum_a P_t(a)\hat X_t(a)^2.$$

The left side has expectation equal to the regret against arm $i$, by unbiasedness and the tower rule, and the expectation of the inner sum on the right is at most $K$ per round, so the regret is at most $\log K/\eta + \eta nK$, which the stated $\eta$ minimises [4]. The original algorithm of Auer and colleagues mixes a uniform term into the distribution, $P_t(a) = (1-\gamma)w_t(a)/\sum_b w_t(b) + \gamma/K$ with $\gamma \in (0,1]$, which keeps every probability above $\gamma/K$ [20], a floor whose value is the subject of section 11.8.

Two variants matter later. When the identity of the best arm changes over time, regret against a single fixed arm is the wrong criterion, and Exp3.S tracks a shifting best arm by mixing a little of every arm's weight into every other arm's at each round, the fixed share device [20, 21]. When several distinct arms must be chosen at once, Exp3.M draws a set of arms whose marginal inclusion probabilities match the desired probabilities exactly, through a procedure called dependent rounding [22]. EXP3 matters in this workspace because the value of a design drifts as the policy trains upon it, and an algorithm that assumes nothing about the stationarity of rewards is robust to that drift, which is why several of the hybrid methods of section 5 use it for their discrete variables.

### 3.9 Pure exploration and the identification of the best arm

Cumulative regret charges every round, which suits a learner that must earn while it learns. A designer instead wants the best design at the end and is indifferent to the merit of the designs tried along the way. The matching criterion is the simple regret $r_n = \mathbb{E}[\Delta_{A_{n+1}}]$ of the arm $A_{n+1}$ recommended after $n$ rounds of experimentation [4]. Bubeck, Munos and Stoltz proved that the two criteria conflict, a lower bound on the simple regret rising as the cumulative regret falls, so that a policy tuned to one is poor at the other [23]. The intuition is that a policy with small cumulative regret pulls inferior arms only logarithmically often, too seldom to be sure they are inferior.

The simplest pure exploration policy samples arms in turn for all $n$ rounds and recommends the empirical best. For 1-subgaussian rewards its simple regret is at most $\min_{\Delta\ge0}\big(\Delta + \sum_{i:\Delta_i>\Delta}\Delta_i\exp(-\lfloor n/K\rfloor\Delta_i^2/4)\big)$, by the same concentration argument as explore-then-commit [4]. Uniform sampling wastes effort on arms that are obviously poor, and two refinements avoid that waste. In the fixed-budget setting, sequential halving divides the budget into $L = \lceil\log_2 K\rceil$ phases, samples every surviving arm $\lfloor n/(L|A_\ell|)\rfloor$ times in phase $\ell$, where $A_\ell$ is the set of survivors, and eliminates the worse half after each phase [24]. Its probability of recommending a suboptimal arm is at most

$$3\log_2 K\,\exp\!\Big(-\frac{n}{16\,H_2\log_2 K}\Big), \qquad H_2 = \max_{i:\Delta_i>0}\frac{i}{\Delta_i^2},$$

for 1-subgaussian rewards with arms sorted by decreasing mean, where $H_2$ measures the hardness of the problem through the gaps of the arms nearest the best [4, 24]. In the fixed-confidence setting, the learner chooses when to stop and must recommend the best arm with probability at least $1 - \delta$. Any such sound procedure needs, in expectation, at least $c^\star(\nu)\log(1/(4\delta))$ samples, where $c^\star(\nu)^{-1} = \sup_\alpha \inf_{\nu'}\sum_a\alpha_a\mathrm{KL}(\nu_a, \nu'_a)$, the supremum running over sampling proportions $\alpha$ and the infimum over alternative bandits $\nu'$ with a different best arm [4], and Garivier and Kaufmann gave a procedure that attains the bound asymptotically by tracking the optimal proportions [25].

Jamieson and Talwalkar carried successive halving to hyperparameter search, where an arm is a configuration being trained and pulling it means granting it more training, so that its observed loss settles toward a limit rather than being drawn independently each time [26]. The co-optimisation loop sits between the two criteria. The final design is what is wanted, which is a simple regret objective, but the shared policy trains on whatever is sampled, so sampling poor designs costs the controller training data, which has a cumulative flavour. Section 11 returns to this tension.

### 3.10 Contextual and linear bandits

In a contextual bandit the learner observes a context $c_t$ before choosing, and the reward depends on both, so what is learned is a policy mapping contexts to arms [4]. A contextual bandit is reinforcement learning with a horizon of one step, a fact section 9 exploits. When contexts are many, arms cannot be treated independently, and the reward must be modelled. The linear model assumes $\mathbb{E}[X_t \mid c_t, A_t] = \langle\theta^\star, \varphi(c_t, A_t)\rangle$, where $\varphi(c, a) \in \mathbb{R}^d$ is a known feature vector and $\theta^\star \in \mathbb{R}^d$ an unknown parameter. Writing $\varphi_s$ for the feature of the choice made at round $s$, the regularised least squares estimate minimises $\sum_s(X_s - \langle\theta, \varphi_s\rangle)^2 + \lambda\|\theta\|^2$, and setting its gradient to zero gives

$$\hat\theta_t = V_t^{-1}\sum_{s=1}^t \varphi_s X_s, \qquad V_t = \lambda I + \sum_{s=1}^t\varphi_s\varphi_s^\top,$$

where $\lambda > 0$ is the regularisation weight and $V_t$ the design matrix [4]. The estimation error is controlled in the norm the matrix defines. With probability at least $1-\delta$, simultaneously for all $t$,

$$\|\hat\theta_{t-1} - \theta^\star\|_{V_{t-1}} \le \sqrt{\beta_t}, \qquad \sqrt{\beta_t} = \sqrt\lambda\, m_2 + \sqrt{2\log(1/\delta) + d\log\!\Big(\frac{d\lambda + tL^2}{d\lambda}\Big)},$$

where $\|v\|_V = \sqrt{v^\top V v}$, $m_2$ bounds $\|\theta^\star\|$ and $L$ bounds $\|\varphi\|$ [4, 27]. The set of parameters within this distance of $\hat\theta$ is an ellipsoid, and the optimistic rule chooses the arm whose best plausible reward over the ellipsoid is largest, which by the Cauchy-Schwarz inequality is

$$A_t = \arg\max_a\; \langle\hat\theta_{t-1}, \varphi(c_t, a)\rangle + \sqrt{\beta_t}\,\|\varphi(c_t, a)\|_{V_{t-1}^{-1}}.$$

This is LinUCB [28], and its confidence set was sharpened by Abbasi-Yadkori, Pál and Szepesvári [27]. The bonus $\|\varphi\|_{V^{-1}}$ is the standard deviation of the predicted reward under the posterior that a Gaussian prior on $\theta$ would give, so a feature direction that has been sampled often carries little bonus.

The regret analysis introduces a lemma that every kernel and neural bandit in sections 4 and 6 reuses. On the event that the ellipsoid contains $\theta^\star$, the per-round regret is at most $2\sqrt{\beta_t}\|\varphi_t\|_{V_{t-1}^{-1}}$, by the same argument as for UCB. The elliptical potential lemma then bounds the sum of the squared widths,

$$\sum_{t=1}^n \min\big(1, \|\varphi_t\|^2_{V_{t-1}^{-1}}\big) \le 2\log\frac{\det V_n}{\det V_0} \le 2d\log\!\Big(\frac{\mathrm{trace}\,V_0 + nL^2}{d\det(V_0)^{1/d}}\Big),$$

which holds because each new feature multiplies the determinant of the design matrix by $1 + \|\varphi_t\|^2_{V_{t-1}^{-1}}$, so the widths sum to the logarithm of the growth of the determinant [4]. Combining the two through the Cauchy-Schwarz inequality gives regret of order $d\sqrt{n}\log n$ [4]. The quantity $\log\det V_n/\det V_0$ measures how much the data have taught about $\theta^\star$, and section 4.6 shows that its kernel analogue is the information gain. Replacing the fixed features $\varphi$ with the features of a neural network is the step that produces the neural bandits of section 6.

## 4. Bandits over continuous decision spaces

### 4.1 Smoothness as the price of infinitely many arms

When the arms form a continuum $\mathcal{X} \subset \mathbb{R}^d$, as link lengths and joint angles do, a learner can never pull every arm even once, and nothing can be learned about an untried arm unless the mean reward $\mu(x)$ varies smoothly, so that a trial at one point informs its neighbours. The standard assumption places the arms in a metric space $(\mathcal{X}, D)$, a set with a distance function $D$ that is non-negative, symmetric, zero only between identical points and obeys the triangle inequality, and requires the mean reward to be Lipschitz with respect to it,

$$|\mu(x) - \mu(y)| \le D(x, y) \quad \text{for all } x, y \in \mathcal{X},$$

so that arms that are close in the metric have close means [29, 12]. With $D(x,y) = L\|x - y\|$ this is the familiar Lipschitz condition with constant $L$. Four strategies then exist, to discretise the space, to partition it adaptively, to model the reward function as a whole, or to move a search distribution without any model of the reward, and the sections below take them in turn.

### 4.2 Discretisation

The simplest strategy fixes a finite set $S \subset \mathcal{X}$ of arms and runs a finite-armed algorithm on it [12]. The regret then splits into two parts, the regret against the best arm of $S$ and the discretisation error $\mathrm{DE}(S) = \sup_{x\in\mathcal{X}}\mu(x) - \max_{x\in S}\mu(x)$ paid in every round,

$$\mathbb{E}[R(T)] \le c\sqrt{|S|\,T\log T} + T\cdot\mathrm{DE}(S),$$

where $c$ is the constant of the finite-armed algorithm's worst-case bound [12]. On the interval $[0,1]$ with a uniform grid of spacing $\epsilon$, every point lies within $\epsilon$ of the grid, so $\mathrm{DE} \le L\epsilon$ and $|S| \approx 1/\epsilon$. Balancing the two terms by choosing $\epsilon = (TL^2/\log T)^{-1/3}$ gives

$$\mathbb{E}[R(T)] \le L^{1/3}\,T^{2/3}\,(1 + c)(\log T)^{1/3},$$

and this rate cannot be improved in the worst case, because a reward function that is flat everywhere except for a narrow bump of height $\epsilon$ and width $\epsilon/L$ around an unknown point forces any algorithm to search about $1/\epsilon$ cells, each needing about $1/\epsilon^2$ samples to resolve the bump, which reproduces the $T^{2/3}$ order [12, 30]. Kleinberg proved that this one-dimensional rate is tight to within a sub-logarithmic factor, improving on the first treatment of the continuum-armed problem [30, 31].

In $d$ dimensions a grid of spacing $\epsilon$ holds about $\epsilon^{-d}$ points, and the same balancing gives regret of order $T^{(d+1)/(d+2)}$ up to logarithmic factors [12]. The exponent approaches one as $d$ grows, which is the curse of dimensionality for discretisation. The general statement replaces $d$ by the covering dimension of the metric space, the smallest $d$ such that the space can be covered by about $\epsilon^{-d}$ sets of diameter $\epsilon$ for every $\epsilon$ [12]. The centimetre rounding of `usd_generator.py:416` is in effect a fixed discretisation of this kind, harmless with two length scales and ruinous were the design vector to grow.

### 4.3 Adaptive discretisation, the zooming algorithm

A fixed grid spends the same resolution on regions where the reward is poor as on regions where it is good. The zooming algorithm of Kleinberg, Slivkins and Upfal refines the grid only where the reward is high [29]. It maintains a set of active arms. For an active arm $x$ that has been played $n_t(x)$ times before round $t$ with average reward $\bar\mu_t(x)$, its confidence radius and confidence ball are

$$r_t(x) = \sqrt{\frac{2\log T}{n_t(x) + 1}}, \qquad B_t(x) = \{y \in \mathcal{X} : D(x, y) \le r_t(x)\},$$

where the radius is the width within which the true mean lies with high probability and the ball is the region of arms that $x$ cannot yet be distinguished from [12]. Two rules define the algorithm. The activation rule maintains the invariant that every arm of the space is covered by the ball of some active arm, activating any arm that becomes uncovered as the balls shrink. The selection rule plays the active arm with the largest index,

$$\mathrm{index}_t(x) = \bar\mu_t(x) + 2r_t(x),$$

which is an upper bound not only on $\mu(x)$ but, through the Lipschitz condition, on the mean of every arm in the ball of $x$ [12]. Because the balls shrink only around arms that are played, and arms are played only where the index is high, the algorithm activates many arms in good regions and few in poor ones, which is the sense in which it zooms.

The analysis rests on one lemma. On the event that every confidence radius is valid, the gap of any arm satisfies $\Delta(x) \le 3r_t(x)$ whenever $x$ is played [12]. The proof is a two-line chain. The best arm $x^\star$ lies in the ball of some active arm $y$, so $\mathrm{index}(x) \ge \mathrm{index}(y) \ge \mu(y) + r_t(y) \ge \mu(x^\star)$, the last step by the Lipschitz condition, while $\mathrm{index}(x) \le \mu(x) + 3r_t(x)$ by validity of the radius, and the two together give $\mu^\star - \mu(x) \le 3r_t(x)$. Two consequences follow. An arm with gap $\Delta(x)$ is played at most $O(\log T)/\Delta(x)^2$ times, and two active arms are separated by more than one third of the smaller of their gaps, because a new arm is activated only outside the ball of every existing one. Grouping arms by gap into the shells $X_r = \{x : r \le \Delta(x) < 2r\}$, the separation property bounds the number of active arms in a shell by the number of sets of diameter $r/3$ needed to cover it, $N_{r/3}(X_r)$, and the regret becomes a sum over shells. The zooming dimension is the smallest $d$ with $N_{r/3}(X_r) \le c\,r^{-d}$ for all $r$, and the regret is

$$\mathbb{E}[R(T)] \le O\big(T^{(d+1)/(d+2)}(c\log T)^{1/(d+2)}\big),$$

the same form as uniform discretisation but with the zooming dimension, a property of the problem's near-optimal region, in place of the covering dimension of the whole space [12]. Where the near-optimal designs occupy a small region, as when a reward has a single sharp peak, the zooming dimension is much smaller than the ambient dimension and the algorithm is correspondingly faster.

### 4.4 Hierarchical optimistic optimisation and tree search

HOO, hierarchical optimistic optimisation, implements adaptive refinement with a tree [32]. The space is covered by a binary tree of nested regions $P_{h,i}$, where $h$ is the depth and $i$ the index within the depth, each region being the union of its two children, and the regions at depth $h$ have diameter at most $\nu_1\rho^h$ for constants $\nu_1 > 0$ and $\rho \in (0,1)$. For each node that has been visited $T_{h,i}(n)$ times with average reward $\hat\mu_{h,i}(n)$, HOO forms the optimistic value

$$U_{h,i}(n) = \hat\mu_{h,i}(n) + \sqrt{\frac{2\ln n}{T_{h,i}(n)}} + \nu_1\rho^h,$$

whose second term accounts for the noise in the average and whose third term accounts for the largest variation of the mean over the region, and then tightens it recursively by

$$B_{h,i}(n) = \min\Big\{U_{h,i}(n),\; \max\{B_{h+1,2i-1}(n),\, B_{h+1,2i}(n)\}\Big\},$$

since the best mean in a region is bounded both by the region's own optimistic value and by the larger of its children's bounds [32]. Each round descends from the root along the child with the larger $B$ value, adds a new leaf, pulls an arm in it and updates the statistics on the path. Its regret is governed by the near-optimality dimension, the growth rate as $\varepsilon \to 0$ of the number of balls of radius $\varepsilon$ needed to pack the set of $c\varepsilon$-optimal arms, and is of order $n^{(d'+1)/(d'+2)}(\ln n)^{1/(d'+2)}$ for any $d'$ above that dimension [32]. When the reward has finitely many maxima around which it behaves smoothly with a known degree of smoothness, the near-optimality dimension is zero and the regret is of order $\sqrt n$ up to a logarithmic factor whatever the dimension of the space [32]. The same optimistic descent applied to a tree of sequential decisions, rather than of nested regions, is Monte Carlo tree search in its UCT form, which runs a UCB index at every node to decide which child to expand [33].

### 4.5 Gaussian processes as priors over reward functions

The third strategy places a probability distribution over the whole reward function and lets every observation update the belief about every arm at once. A Gaussian process $f \sim \mathcal{GP}(m, k)$ is a distribution over functions on $\mathcal{X}$ such that for any finite set of points $x_1, \dots, x_t$ the vector $(f(x_1), \dots, f(x_t))$ is jointly Gaussian with mean $(m(x_1), \dots, m(x_t))$ and covariance matrix $[k(x_i, x_j)]_{ij}$, where $m$ is the mean function and $k$ the covariance function, or kernel [6]. The kernel encodes the belief about how the reward varies. It must be positive semi-definite, meaning that every covariance matrix it produces has non-negative eigenvalues, and two choices dominate practice. The squared exponential kernel

$$k_{\mathrm{SE}}(x, x') = s^2\exp\!\Big(-\frac{\|x - x'\|^2}{2\ell^2}\Big)$$

has signal variance $s^2$ and length scale $\ell$, the distance over which the reward decorrelates, and produces infinitely smooth functions. The Matérn family, indexed by a smoothness parameter $\nu$, produces rougher functions whose smoothness grows with $\nu$, approaches the squared exponential as $\nu \to \infty$, and with $\nu = 5/2$ is a common default, which CASMOPOLITAN for one adopts for its continuous variables [34, 6, 35]. Sums and products of positive semi-definite kernels are again positive semi-definite, which is how kernels for mixed spaces are built in section 5.

Given observations $y_i = f(x_i) + \varepsilon_i$ with independent noise $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ and a zero prior mean, the vector of observations $y_t$ and the unknown value $f(x)$ are jointly Gaussian, and the conditioning identity of section 2.4 gives the posterior at any point $x$,

$$\mu_t(x) = k_t(x)^\top(K_t + \sigma^2 I)^{-1}y_t, \qquad \sigma_t^2(x) = k(x, x) - k_t(x)^\top(K_t + \sigma^2 I)^{-1}k_t(x),$$

where $K_t = [k(x_i, x_j)]_{i,j\le t}$ is the kernel matrix of the observed points, $k_t(x) = [k(x_1, x), \dots, k(x_t, x)]^\top$ the vector of kernel values between $x$ and them, and $I$ the identity [34, 6]. The mean interpolates the data smoothly, the variance is small near observed points and returns to the prior variance far from them, and, as section 2.4 noted, the variance depends only on where observations were made and not on what was observed. The kernel's own parameters, the length scales, signal variance and noise variance, are fitted by maximising the log marginal likelihood of the data,

$$\log p(y_t \mid X_t) = -\tfrac12 y_t^\top(K_t + \sigma^2 I)^{-1}y_t - \tfrac12\log\det(K_t + \sigma^2 I) - \tfrac t2\log 2\pi,$$

whose first term rewards fit to the data and whose second penalises complexity [6]. The cost of exact inference is cubic in the number of observations, because of the matrix inverse, which is the limitation that the neural surrogates of section 6 address.

### 4.6 GP-UCB and its analysis

GP-UCB is the upper confidence bound policy with the Gaussian process posterior in place of empirical means [34],

$$x_t = \arg\max_{x\in\mathcal{X}}\; \mu_{t-1}(x) + \beta_t^{1/2}\sigma_{t-1}(x),$$

where $\beta_t$ is a confidence parameter that grows slowly with $t$. Its analysis by Srinivas, Krause, Kakade and Seeger introduced the quantity that governs every kernel and neural bandit since, the maximum information gain

$$\gamma_T = \max_{A\subset\mathcal{X},\,|A| = T} I(y_A; f_A) = \max_{A\subset\mathcal{X},\,|A| = T}\tfrac12\log\det\big(I + \sigma^{-2}K_A\big),$$

the largest reduction in uncertainty about $f$ that $T$ noisy observations can achieve, where $f_A$ is the vector of function values at the points of $A$, $y_A$ the observations there and $I(\cdot\,;\cdot)$ the mutual information [34]. The analysis proceeds in four lemmas, each short [34].

The first lemma establishes validity of the confidence bounds. For a finite set $\mathcal{X}$ of $|\mathcal{X}|$ points and $f$ drawn from the prior, conditioned on the history $f(x)$ is Gaussian with mean $\mu_{t-1}(x)$ and standard deviation $\sigma_{t-1}(x)$. A standard normal $r$ satisfies $\Pr(r > c) \le \tfrac12 e^{-c^2/2}$, so $\Pr(|f(x) - \mu_{t-1}(x)| > \beta_t^{1/2}\sigma_{t-1}(x)) \le e^{-\beta_t/2}$. A union bound over the points of $\mathcal{X}$ and over rounds, with $\beta_t = 2\log(|\mathcal{X}|t^2\pi^2/(6\delta))$ so that the failure probabilities $\delta\cdot 6/(\pi^2t^2)$ sum to $\delta$, shows that with probability at least $1-\delta$ every bound holds at every round [34].

The second lemma bounds the regret of one round. If the bounds hold, then $\mu_{t-1}(x_t) + \beta_t^{1/2}\sigma_{t-1}(x_t) \ge \mu_{t-1}(x^\star) + \beta_t^{1/2}\sigma_{t-1}(x^\star) \ge f(x^\star)$ by the choice of $x_t$ and validity at the optimum $x^\star$, and $f(x_t) \ge \mu_{t-1}(x_t) - \beta_t^{1/2}\sigma_{t-1}(x_t)$ by validity at $x_t$, so the instantaneous regret satisfies $r_t = f(x^\star) - f(x_t) \le 2\beta_t^{1/2}\sigma_{t-1}(x_t)$ [34].

The third lemma expresses the information gained by the chosen points through their predictive variances. Since the entropy of the observations decomposes by the chain rule and each new observation, given the past, is Gaussian with variance $\sigma^2 + \sigma_{t-1}^2(x_t)$, the mutual information is

$$I(y_T; f_T) = \tfrac12\sum_{t=1}^T\log\big(1 + \sigma^{-2}\sigma_{t-1}^2(x_t)\big),$$

which is the kernel analogue of the log-determinant growth in the elliptical potential lemma of section 3.10 [34].

The fourth lemma joins the others. Because $s^2 \le C_2\log(1 + s^2)$ for $s^2$ in $[0, \sigma^{-2}]$ with $C_2 = \sigma^{-2}/\log(1 + \sigma^{-2})$, the squared regrets satisfy $\sum_t r_t^2 \le 4\beta_T\sum_t\sigma_{t-1}^2(x_t) \le C_1\beta_T\,I(y_T; f_T) \le C_1\beta_T\gamma_T$, where $C_1 = 8/\log(1+\sigma^{-2})$. The Cauchy-Schwarz inequality $R_T^2 \le T\sum_t r_t^2$ then gives the theorem,

$$\Pr\Big(R_T \le \sqrt{C_1 T\beta_T\gamma_T}\ \text{for all } T \ge 1\Big) \ge 1 - \delta,$$

for a finite set of arms [34]. The same paper extends the result to compact convex subsets of $[0, r]^d$ with a $\beta_t$ that adds a term of order $2d\log(t^2 d)$, under a condition that the sample paths have derivatives that are bounded with high probability, and to an agnostic setting in which $f$ is any function of bounded norm in the reproducing kernel Hilbert space of $k$ and the noise is bounded, with $\beta_t = 2B + 300\gamma_t\log^3(t/\delta)$, where $B$ bounds the squared norm [34]. The regret is sublinear whenever $\gamma_T$ grows more slowly than $T$, and the information gain was bounded for the common kernels as

$$\gamma_T = O(d\log T)\ \text{(linear kernel)},\qquad \gamma_T = O\big((\log T)^{d+1}\big)\ \text{(squared exponential)},\qquad \gamma_T = O\big(T^{d(d+1)/(2\nu + d(d+1))}\log T\big)\ \text{(Matérn, } \nu > 1\text{)},$$

where $d$ is the dimension of the space and $\nu$ the Matérn smoothness [34]. The reproducing kernel Hilbert space of a kernel is, informally, the space of functions that can be written as weighted sums of kernel functions centred at points of the space, and its norm measures how rough a function is relative to the kernel's notion of smoothness [6].

The finite-set version applies to the centimetre lattice of section 1.2 without modification, with $|\mathcal{X}| = 323$ or $144$ length designs. Chowdhury and Gopalan later tightened the agnostic analysis with a self-normalised concentration inequality for kernel regression, obtaining the improved rule IGP-UCB with $\beta_t^{1/2} = B + R\sqrt{2(\gamma_{t-1} + 1 + \ln(1/\delta))}$, where $B$ now bounds the norm itself and $R$ is the subgaussian constant of the noise, with regret of order $\sqrt T(B\sqrt{\gamma_T} + \gamma_T)$ [36]. They also analysed GP-TS, which draws a whole function from the posterior with its covariance inflated by a factor $v_t^2$ of the same form and maximises the draw, with regret of order $\sqrt{(\gamma_T + \ln(2/\delta))\,d\ln(BdT)}\,(\sqrt{T\gamma_T} + B\sqrt{T\ln(2/\delta)})$ [36]. The Thompson variant is the natural one for batches, since independent draws give distinct maximisers.

### 4.7 Bayesian optimisation and its acquisition functions

Bayesian optimisation is the applied practice of the same idea, a Gaussian process surrogate combined with an acquisition function that scores candidate points, whose maximiser is evaluated next [37, 38]. It has been applied to the tuning of learning algorithms, notably by Snoek, Larochelle and Adams [39], and its upper confidence bound acquisition is GP-UCB itself. Two other acquisitions recur in the robot design literature. The probability of improvement scores a point by $\Pr(f(x) > f^+) = \Phi((\mu(x) - f^+)/\sigma(x))$, where $f^+$ is the best value observed so far and $\Phi$ the standard normal distribution function. The expected improvement scores it by $\mathbb{E}[\max(f(x) - f^+, 0)]$, which for a Gaussian posterior has the closed form

$$\mathrm{EI}(x) = (\mu(x) - f^+)\,\Phi(z) + \sigma(x)\,\phi(z), \qquad z = \frac{\mu(x) - f^+}{\sigma(x)},$$

where $\phi$ is the standard normal density [38]. The closed form follows by writing $f(x) = \mu + \sigma Z$ with $Z$ standard normal and integrating $(\mu + \sigma Z - f^+)$ over the region $Z > -z$, the first term coming from the linear part and the second from $\int_{-z}^\infty u\,\phi(u)\,du = \phi(z)$. Expected improvement rewards points whose mean is high and points whose uncertainty is high, in a proportion fixed by the data rather than by a tuned $\beta$.

Where a context $c$ is present, Krause and Ong place a kernel over context and action pairs, typically the product $k((c, x), (c', x')) = k_c(c, c')\,k_x(x, x')$, and play the optimistic action for the observed context, with regret governed by the information gain of the product kernel [40]. In this workspace a context could be a terrain level or a commanded speed band, so that one surrogate serves designs for many tasks, which section 11.10 takes up.

### 4.8 Zeroth-order gradients

The fourth strategy keeps no model of the reward and instead moves a search distribution along an estimate of the gradient of its expected reward, which is section 2.5 transposed to a continuum. The idea first appears in bandit convex optimisation. Flaxman, Kalai and McMahan showed that a single evaluation at a randomly perturbed point gives an unbiased gradient of a smoothed objective [41]. Define the smoothed function $\hat f(x) = \mathbb{E}_{v}[f(x + \delta v)]$, with $v$ uniform in the unit ball of $\mathbb{R}^d$ and $\delta > 0$ a smoothing radius. Then

$$\mathbb{E}_{u}\big[f(x + \delta u)\,u\big] = \frac{\delta}{d}\,\nabla\hat f(x),$$

with $u$ uniform on the unit sphere, because by Stokes' theorem the gradient of the integral of $f$ over a ball equals the integral of $f$ times the outward normal over the bounding sphere, and the normalising volumes differ by the factor $d/\delta$ [41]. Running gradient descent with the estimate $(d/\delta)f(x + \delta u)u$ attains regret of order $T^{3/4}$ on convex costs, using one function value per round [41].

Evolution strategies make the same move with a Gaussian [42]. For a search distribution $\mathcal{N}(\theta, \sigma^2 I)$ with mean $\theta$ and fixed standard deviation $\sigma$, the expected objective is $\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[F(\theta + \sigma\epsilon)]$, a Gaussian-blurred version of $F$, and the score function identity gives

$$\nabla_\theta\,\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[F(\theta + \sigma\epsilon)] = \frac1\sigma\,\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\big[F(\theta + \sigma\epsilon)\,\epsilon\big],$$

estimated from a population of $n$ perturbations as $\frac{1}{n\sigma}\sum_i F_i\epsilon_i$, where $F_i$ is the return of the $i$-th perturbed parameter vector [42]. Two refinements reduce the variance of the estimate. Antithetic sampling evaluates each perturbation in pairs $\pm\epsilon$, and fitness shaping replaces the raw returns by a rank transformation, which removes the influence of outliers and makes the update invariant to monotone changes of the objective [42].

### 4.9 Natural evolution strategies, information-geometric optimisation and CMA-ES

Natural evolution strategies follow the natural gradient of section 2.6 rather than the plain gradient, updating the parameters of the search distribution by $\theta \leftarrow \theta + \eta\,F^{-1}\nabla_\theta J$, where $F$ is the Fisher information of the search distribution and is estimated from the same scores used for the gradient [5]. The information-geometric optimisation framework of Ollivier, Arnold, Auger and Hansen gives the general recipe from which CMA-ES and its relatives follow [43]. It replaces the raw objective by a rank-based transform. For a sample $x$ it computes the probability $q_\theta(x)$ that a fresh sample from the current distribution would be better, and assigns the weight $w(q_\theta(x))$ for a non-increasing selection function $w$, such as $w(q) = \mathbb{1}[q \le q_0]$, which keeps the best fraction $q_0$. With $N$ samples ranked by objective value, the sample of rank $i$ receives the weight $\hat w_i = \frac1N w\big((i - \tfrac12)/N\big)$, and the parameters are updated by

$$\theta^{t+\delta t} = \theta^t + \delta t\; F^{-1}(\theta^t)\sum_{i=1}^N \hat w_i\,\frac{\partial\log p_\theta(x_{i:N})}{\partial\theta}\Big|_{\theta = \theta^t},$$

where $x_{i:N}$ is the sample of rank $i$ and $\delta t$ a step size [43]. The update is invariant to any strictly increasing transformation of the objective, because only ranks enter it, and invariant to reparameterisation of the distribution, because the natural gradient is. Applied to independent Bernoulli distributions on binary strings, where the inverse Fisher information is diagonal with entries $\theta_i(1-\theta_i)$, the update simplifies to $\theta_i \leftarrow \theta_i + \delta t\sum_j w_j([x_{j:N}]_i - \theta_i)$, which moves each probability toward the frequency of a one among the selected samples and recovers the population-based incremental learning algorithm [43].

Applied to a Gaussian, the same recipe yields the core of CMA-ES, whose full form adds two cumulation mechanisms [44]. Each generation samples $\lambda$ points

$$x_k = m + \sigma\,y_k, \qquad y_k = BDz_k \sim \mathcal{N}(0, C), \qquad z_k \sim \mathcal{N}(0, I),$$

where $m$ is the distribution mean, $\sigma$ the global step size, $C = BD^2B^\top$ the covariance matrix with eigenvectors in the columns of $B$ and square roots of eigenvalues on the diagonal of $D$. The points are ranked by objective, and with positive weights $w_1 \ge \dots \ge w_\mu$ summing to one for the best $\mu$ points the mean moves to their weighted average, $m \leftarrow m + c_m\sigma\langle y\rangle_w$ with $\langle y\rangle_w = \sum_{i=1}^\mu w_i y_{i:\lambda}$. The step size is controlled by an evolution path $p_\sigma$, an exponentially smoothed sum of successive normalised steps,

$$p_\sigma \leftarrow (1 - c_\sigma)p_\sigma + \sqrt{c_\sigma(2 - c_\sigma)\mu_{\mathrm{eff}}}\;C^{-1/2}\langle y\rangle_w, \qquad \sigma \leftarrow \sigma\exp\!\Big(\frac{c_\sigma}{d_\sigma}\Big(\frac{\|p_\sigma\|}{\mathbb{E}\|\mathcal{N}(0, I)\|} - 1\Big)\Big),$$

where $\mu_{\mathrm{eff}} = (\sum_i w_i^2)^{-1}$ is the variance-effective selection mass, $c_\sigma$ a learning rate and $d_\sigma$ a damping constant [44]. The path is longer than a random walk would be when successive steps point the same way, in which case the step size grows, and shorter when they cancel, in which case it shrinks. The covariance is adapted by a rank-one term from a second evolution path $p_c$ and a rank-$\mu$ term from the selected steps,

$$C \leftarrow (1 - c_1 - c_\mu)C + c_1\,p_cp_c^\top + c_\mu\sum_{i}w_i\,y_{i:\lambda}y_{i:\lambda}^\top,$$

in its simplest form with positive weights, where $c_1$ and $c_\mu$ are learning rates [44]. The rank-$\mu$ term is the natural gradient step of the information-geometric framework for the covariance, which explains why the method learns the shape of the objective's level sets [43, 44].

The incumbent search is therefore already a bandit algorithm of this fourth kind, a search distribution moved by a rank-based natural gradient estimated from one generation of 256 samples. What it lacks is a model. CMA-ES moves its mean and covariance by the ranks of the current generation alone (`cmaes.md:115`, `cmaes.md:182`), so the evidence of earlier generations survives only through the distribution parameters, and an evaluation is never revisited once its generation has passed. CatCMA inherits that machinery for mixed spaces, as section 5.5 shows [45].

## 5. Bandits over hybrid continuous and discrete spaces

### 5.1 The structure of a hybrid design space

Write a design as $x = (z, c)$, where $z \in [0,1]^{d_z}$ collects the continuous variables after normalisation to the unit box, here two length scales and the nominal pose, and $c = (c_1, \dots, c_G)$ collects the categorical ones, with $c_g \in \{1, \dots, M\}$ the actuator chosen for joint group $g$ from a catalogue of $M$ entries. A categorical variable differs from an integer one in having no order, actuator 7 is not between actuators 6 and 8 in any meaningful sense, so methods that exploit order or distance on the integers do not apply. The discrete block alone is combinatorial, $M^G$ being $15^4 = 50{,}625$ for the biped and $14^3 = 2{,}744$ for the quadruped, and the two blocks interact, since the torque an actuator delivers bounds the crouch that a leg of given length can hold and the mass it adds changes which lengths are viable. A hybrid method must therefore share evidence across categories, so that a trial with one actuator informs the others, while still permitting their interaction. The literature offers five families of answer, taken in turn below.

### 5.2 Relaxation and rounding, and why it fails

The naive course encodes each categorical variable as continuous coordinates, either one coordinate per category through a one-hot code or a single coordinate through an integer code, and rounds before evaluation. Garrido-Merchán and Hernández-Lobato showed that a Gaussian process unaware of the rounding models variation that cannot occur, believing for instance that two continuous points which round to the same category may have different values, and so wastes evaluations and misplaces its uncertainty [46]. Their remedy applies the rounding inside the kernel,

$$k'(x_i, x_j) = k\big(T(x_i), T(x_j)\big),$$

where $T$ rounds every integer coordinate to the nearest integer and, for each categorical variable, sets the largest of its one-hot coordinates to one and the rest to zero [46]. The model then sees the objective as constant on every region that rounds to the same configuration, its uncertainty on such a region vanishes after a single evaluation there, and the acquisition function stops proposing points it has in effect already tried [46].

The evolutionary counterpart of the same failure is the collapse of the step size below the width of a rounding plateau. When CMA-ES searches a continuous relaxation of an integer variable, the ranks of the samples carry no information about the variable once all samples round to the same integer, the step size keeps shrinking, and the search never leaves that integer. Hamano and colleagues repaired this by lower-bounding the marginal probability of leaving the current integer [47]. They sample $x_i = m + \sigma y_i$ as usual, evaluate the affinely transformed points $v_i = m + \sigma A y_i$ with a diagonal matrix $A$ initialised to the identity, and after each update enlarge the entries of $A$, or shift the mean, just enough that the probability of generating a value on the far side of the nearest rounding threshold stays at least a margin $\alpha$, with default $\alpha = 1/(N\lambda)$ for $N$ variables and population $\lambda$ [47]. The thresholds are the midpoints between consecutive admissible values. An integer code imposes on an actuator catalogue an order it does not possess, which is one reason the workspace moved to a categorical search, as the cluster 29 entry of `literature.md` records.

### 5.3 Kernels for categorical and mixed inputs

The second course gives the categorical variables a kernel of their own and combines it with a continuous kernel, so that a single Gaussian process covers the whole space. Three constructions are surveyed here, each a different answer to how similarity between two actuator choices should be measured.

The overlap kernel counts agreeing categories. For categorical vectors $h, h'$ with $c$ components it is $k_h(h, h') = \frac{\sigma}{c}\sum_{i=1}^c\mathbb{1}[h_i = h'_i]$, where $\sigma$ is a variance parameter, so two designs with the same actuator in three of four groups are more similar than designs agreeing in one [48]. CASMOPOLITAN exponentiates it and gives each categorical dimension its own length scale,

$$k_h(h, h') = \exp\!\Big(\frac{1}{d_h}\sum_{i=1}^{d_h}\ell_i\,\delta(h_i, h'_i)\Big),$$

where $d_h$ is the number of categorical variables, $\ell_i$ the length scale of variable $i$ and $\delta$ the Kronecker delta, so the model can learn that the objective is sensitive to the knee actuator and indifferent to the abduction actuator [35]. CoCaBO combines a categorical kernel $k_h$ and a continuous kernel $k_x$ as

$$k(z, z') = (1 - \lambda)\big(k_h(h, h') + k_x(x, x')\big) + \lambda\,k_h(h, h')\,k_x(x, x'),$$

with a mixing weight $\lambda \in [0,1]$ learned from the data, where $z = (h, x)$ stacks the categorical part $h$ and the continuous part $x$ [48]. The sum term lets a good length setting found with one actuator inform every other actuator, because two designs with the same lengths are correlated through $k_x$ whatever their actuators. The product term is large only when both parts agree, so it lets the best length depend on the actuator. The combined kernel is positive semi-definite because sums, products and positive multiples of positive semi-definite kernels are, and CASMOPOLITAN adopts the same mixture with its exponentiated categorical kernel and a Matérn 5/2 continuous kernel [35]. CASMOPOLITAN adds one further mechanism for high-dimensional categorical spaces, trust regions. It restricts the search to a Hamming ball of radius $L_h$ around the best categorical configuration so far and a box of side $L_x$ around the best continuous point, optimises the acquisition by local search inside them, shrinks them after repeated failures and expands or recentres them after successes, and restarts when they become too small, while retaining a convergence guarantee in the mixed space that its authors note CoCaBO lacks [35].

HyBO measures similarity by diffusion on a graph [49]. The discrete space is represented as a graph whose nodes are configurations and whose edges join configurations differing in exactly one variable, and the diffusion kernel is the matrix exponential $k = \exp(-\beta L(G))$ of the negative graph Laplacian $L(G)$, scaled by a length-scale-like parameter $\beta$, which expresses the similarity of two configurations as the amount of heat that would diffuse from one to the other along the graph in time $\beta$. For a single binary variable the closed form is proportional to $1 + e^{-2\beta}$ when the values agree and $1 - e^{-2\beta}$ when they differ. HyBO then forms additive kernels of every order over the base kernels of the individual dimensions,

$$K_p = \theta_p^2\sum_{1\le i_1 < \dots < i_p \le m+n}\;\prod_{d=1}^p k_{i_d}(x_{i_d}, x'_{i_d}),$$

where $m$ and $n$ are the numbers of discrete and continuous variables, $k_{i}$ is the diffusion kernel for a discrete dimension or a squared exponential for a continuous one, and $\theta_p$ weights the interactions of order $p$, so that the data decide whether pairwise or higher interactions between actuator and length matter [49].

MiVaBO replaces the Gaussian process by a Bayesian linear model on hand-built features [50]. It writes $f(x) = \sum_{j\in\{d,c,m\}}w_j^\top\varphi_j(x_j)$, with discrete features $\varphi_d$ forming a second-order polynomial in the binary encoding of the categorical variables, continuous features $\varphi_c$ forming random Fourier features that approximate a squared exponential Gaussian process, and mixed features $\varphi_m$ formed as all pairwise products of discrete and continuous features, for a total of $M_d + M_c + M_dM_c$ features. It places a Gaussian prior on the weights, draws weights from the posterior for Thompson sampling, and optimises the drawn function by alternating between the two blocks, the discrete block as a binary integer quadratic programme handed to an off-the-shelf solver and the continuous block by a quasi-Newton method, each with the other held fixed, which also lets it respect linear and quadratic constraints on the discrete variables [50].

### 5.4 Hierarchical bandits

The third course treats each categorical configuration as an arm and the continuous variables as a Bayesian optimisation beneath the arm. Three variants differ in how the levels share information.

CoCaBO selects the categorical configuration with EXP3, using the observed function value as the arm's reward on the grounds that EXP3 makes few assumptions about the reward distribution, and then maximises the GP-UCB acquisition over the continuous variables with the categories fixed, under the mixed kernel of section 5.3 so that observations under every category inform the model [48]. For batches it draws several categorical configurations with EXP3.M and allocates continuous points to each in proportion to how often it was drawn, filling them in sequence with the kriging believer heuristic, which treats each pending point's predicted mean as if it had been observed [48].

Nguyen and colleagues treat the case in which each category has its own continuous function, which fits an actuator choice that changes the meaning of the continuous variables [51]. Each category $c$ is an arm whose mean is $f^\star_c = \max_x f_c(x)$, the best value attainable with that category, and each category has its own Gaussian process. To choose a point, the algorithm draws one function $\tilde f_c$ from every category's posterior, maximises each draw to obtain $\tilde x^\star_c$ and the value $\tilde f^\star_c = \tilde f_c(\tilde x^\star_c)$, plays the category with the largest $\tilde f^\star_c$, and evaluates it at $\tilde x^\star_c$. Thompson sampling thereby selects both levels at once, choosing each category with the posterior probability that its best attainable value is the largest, and a batch is formed by repeating the procedure with independent draws [51]. The method has sublinear regret [51].

Parker-Holder and colleagues built the same hierarchy for the hyperparameters of reinforcement learning tuned during a single training run, the setting closest to this workspace's [22]. Categories are chosen by TV.EXP3.M, a time-varying version of EXP3 with multiple plays that selects a batch of $B$ distinct categories out of $C$ through dependent rounding, so that each category $c$ is included with probability exactly $p_c$, and that renormalises its weights to keep any single category from dominating. With exploration rate $\gamma = \min\{1, \sqrt{C\ln(C/B)/((e-1)BT)}\}$ and a fixed-share rate of $1/T$, its expected regret against a sequence of best categories that changes at most $V$ times satisfies

$$\mathbb{E}[R_{TB}] \le (1 + e + V)\sqrt{\frac{(e-1)CT}{B}\ln\frac{CT}{B}},$$

where $T$ is the number of rounds and $B$ the batch size [22]. Continuous variables are then chosen by a time-varying GP-UCB, either with one Gaussian process per category or with a single process under the mixed kernel of section 5.3 [22]. Of every method surveyed this last is the closest in setting to the present one, and section 8.6 examines its continuous component in full.

### 5.5 Joint search distributions, CatCMA

The fourth course places one probability distribution over the whole hybrid space and moves it by natural gradient, extending section 4.9. CatCMA uses the product of a multivariate Gaussian over the continuous variables and an independent categorical distribution over each categorical variable [45],

$$p(x, c \mid m, C, q) = \mathcal{N}(x; m, C)\prod_{n=1}^{N_{\mathrm{ca}}}\prod_{k=1}^{K_n}q_{n,k}^{c_{n,k}},$$

where $x \in \mathbb{R}^{N_{\mathrm{co}}}$ is the continuous part, $c_n$ is a one-hot vector over the $K_n$ categories of the $n$-th of $N_{\mathrm{ca}}$ categorical variables, and $q_{n,k}$ is the probability of category $k$ for variable $n$. Because the joint distribution is a product, its Fisher information is block diagonal, with the familiar Gaussian blocks and, for each categorical variable parameterised by its first $K_n - 1$ probabilities, an inverse Fisher block equal to the multinomial covariance $\mathrm{diag}(q_n) - q_nq_n^\top$. Multiplying the score of each block by its inverse Fisher block gives the information-geometric updates, with $\lambda$ samples ranked by objective and weights $w_i$ on the ranked samples,

$$m \leftarrow m + c_m\sum_{i=1}^\lambda w_i(x_{i:\lambda} - m), \qquad C \leftarrow C + c_\mu\sum_{i=1}^\lambda w_i\big((x_{i:\lambda} - m)(x_{i:\lambda} - m)^\top - C\big), \qquad q_n \leftarrow q_n + \eta_n\sum_{i=1}^\lambda w_i(c_{i:\lambda,n} - q_n),$$

where $c_m$, $c_\mu$ and $\eta_n$ are learning rates and the subscript $i{:}\lambda$ denotes the sample of rank $i$ [45]. The categorical update moves each probability vector toward the weighted frequency of each category among the best samples, exactly as the Bernoulli case of section 4.9 moved a bit probability, and CatCMA adds to the Gaussian part the step-size and rank-one mechanisms of CMA-ES and an adaptive learning rate for the categorical part [45].

A margin prevents the categorical distribution from fixing prematurely. After each update every probability is raised to at least $q^{\min}_n$ and the excess is removed proportionally from the others,

$$q_{n,k} \leftarrow \max\{q_{n,k}, q^{\min}_n\}, \qquad q_{n,k} \leftarrow q_{n,k} + \frac{1 - \sum_{k'}q_{n,k'}}{\sum_{k'}(q_{n,k'} - q^{\min}_n)}\,(q_{n,k} - q^{\min}_n),$$

which keeps every category reachable while preserving the sum to one [45]. Too large a margin keeps producing samples with non-optimal categories, which disturbs the convergence of the Gaussian part since only the best half of the samples carry weight, and too small a margin lets a category be lost. The authors derive the setting

$$q^{\min}_n = \frac{1 - (1 - \xi)^{1/N_{\mathrm{ca}}}}{K_n - 1}, \qquad \xi = 0.27,$$

which guarantees, once the distribution has converged on the optimal categories, that with probability at least $0.95$ fewer than $\lambda - \lfloor\lambda/2\rfloor$ of the $\lambda \ge 6$ samples contain a non-optimal category, so that the weighted half is uncontaminated [45]. CatCMA with margin extends the scheme to integer variables alongside categorical and continuous ones [3].

This is the workspace's incumbent hybrid search. `CatCMAESDesignGenerator` gives the continuous block the unit-box encoding of the length scales and one categorical variable per actuator group whose categories are the catalogue entries, and it starts each group's categorical distribution with probability 0.5 on the donor robot's actuator and the remainder spread evenly over the others (`usd_generator.py:841`), so that generation zero is centred on the donor.

### 5.6 Parameterised actions in reinforcement learning

Reinforcement learning meets the same structure in parameterised action spaces, where an agent chooses a discrete action together with continuous parameters that belong to it [52]. The action set is $\mathcal{A} = \{(k, x_k) : k \in \{1, \dots, K\},\ x_k \in \mathcal{X}_k\}$, where $\mathcal{X}_k$ is the parameter space of the $k$-th discrete action [53]. Three methods illustrate the options. P-DQN learns a value $Q(s, k, x_k)$ for each discrete action and its parameters, together with a deterministic network $x_k(s)$ that approximates the maximising parameters, and acts by $\arg\max_k Q(s, k, x_k(s))$, so the Bellman update maximises over $k$ and, through the network, over the parameters [53]. H-PPO uses one discrete actor producing a softmax over $k$ and one continuous actor producing a Gaussian over the parameters, with a single critic, and updates each actor separately with the clipped objective of proximal policy optimisation using the same advantage [54]. HyAR learns a compact latent action space, an embedding table for the discrete actions and a conditional variational autoencoder that embeds the continuous parameters given the discrete embedding and the state, trains a conventional continuous-action agent in that latent space, and decodes its actions back into the hybrid space [55].

With a horizon of one step these are hybrid bandits, and the choice of an actuator per joint group with continuous geometry and pose beside it is exactly a parameterised action. A design policy of this kind factorises its likelihood as

$$\log\pi_\phi(c, z) = \sum_{g=1}^{G}\log\mathrm{softmax}(\ell_{\phi,g})_{c_g} + \log\mathcal{N}\big(z;\ m_\phi(c),\ \mathrm{diag}\,\sigma_\phi^2(c)\big),$$

where $\ell_{\phi,g} \in \mathbb{R}^M$ are the logits of group $g$, and the continuous head's mean $m_\phi(c)$ and standard deviations $\sigma_\phi(c)$ may be conditioned on the chosen actuators. The scores of section 2.5 then train both heads at once with any policy gradient estimator, which section 9.7 develops for proximal policy optimisation.

### 5.7 Summary of the hybrid methods

| Method | Categorical block | Continuous block | Batch | Drifting rewards | Model of the reward |
|---|---|---|---|---|---|
| Relaxation with rounding in the kernel [46] | rounded coordinate | Gaussian process | possible | no | Gaussian process |
| CoCaBO [48] | EXP3 | GP-UCB, mixed kernel | yes | partly, through EXP3 | Gaussian process |
| CASMOPOLITAN [35] | trust region local search | trust region, mixed kernel | yes | no | Gaussian process |
| HyBO [49] | diffusion kernel | squared exponential, additive | possible | no | Gaussian process |
| MiVaBO [50] | quadratic features | random Fourier features | possible | no | Bayesian linear model |
| Bandit-BO [51] | Thompson sampling | Gaussian process per category | yes | no | Gaussian process per arm |
| PB2-Mix [22] | time-varying EXP3 | time-varying GP-UCB | yes | yes | Gaussian process with forgetting |
| CatCMA with margin [45, 3] | categorical distribution | Gaussian distribution | yes | implicitly, through ranks | none |
| Parameterised action policy [54, 55] | softmax head | Gaussian head | yes | through retraining | optional critic |

## 6. Neural network bandits and their learning

### 6.1 Three roles for a network

A neural network can enter a bandit in three ways, and the proposal must choose among them before anything else. In the first role the network is a reward model $f_\theta(x) \approx \mu(x)$, a regression from designs to their expected merit, equipped with some measure of its own uncertainty, from which designs are chosen by an optimistic or a posterior sampling rule. The literature calls this a neural bandit, and sections 6.2 to 6.5 treat it. In the second role the network is a policy $\pi_\phi(x \mid c)$ that outputs a probability distribution over designs, possibly given a context $c$, trained by the score function estimator of section 2.5, which is what the phrase "a network that predicts the link lengths, the pose and the actuator" most naturally describes, and section 6.6 treats it. In the third role the network is an amortised optimiser, a map from a description of a problem to that problem's solution, learned across many problems so that each new one is solved in a single forward pass rather than by a fresh search [56]. The three are complementary, and the strongest constructions in section 12 combine the first two.

### 6.2 Neural reward models with a Bayesian last layer

The simplest neural bandit trains a network by regression and treats its last hidden layer as a fixed feature map, reducing the problem to the linear bandit of section 3.10. Let $\varphi_\theta(x) \in \mathbb{R}^p$ be the activations of the last hidden layer for design $x$, and model the reward as $y = w^\top\varphi_\theta(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ and a Gaussian prior $w \sim \mathcal{N}(0, \lambda^{-1}I)$ on the last layer's weights. Stacking the features of the $t$ observed designs as the rows of $\Phi \in \mathbb{R}^{t\times p}$ and their rewards in $y$, the posterior over $w$ follows from section 2.4,

$$w \mid \mathcal{D} \sim \mathcal{N}(\bar w, \Sigma), \qquad \Sigma = \big(\lambda I + \sigma^{-2}\Phi^\top\Phi\big)^{-1}, \qquad \bar w = \sigma^{-2}\Sigma\Phi^\top y,$$

and the predictive distribution at a new design has mean $\bar w^\top\varphi_\theta(x)$ and variance $\varphi_\theta(x)^\top\Sigma\varphi_\theta(x) + \sigma^2$. Thompson sampling draws $\tilde w$ from the posterior and plays the design maximising $\tilde w^\top\varphi_\theta(x)$, and an optimistic rule adds a multiple of the predictive standard deviation to the mean. The representation $\theta$ is trained by ordinary regression on all data and refreshed periodically, while the cheap linear posterior is updated after every observation.

Riquelme, Tucker and Snoek compared many ways of approximating a neural network's posterior inside Thompson sampling on contextual bandit problems [57]. They found that methods successful in supervised learning often failed in the sequential setting because their uncertainty estimates converged too slowly, while this last-layer construction, which they named NeuralLinear, was easy to tune, robust to its hyperparameters, and on their hardest exploration problem the best of all, almost an order of magnitude better than the plain network and its bootstrapped, dropout and parameter noise variants [57]. They attributed its success to the separation of representation learning, done by the network, from uncertainty estimation, done exactly in the linear layer [57]. Snoek and colleagues had used the same device, which they called adaptive basis function regression, to make Bayesian optimisation scale linearly rather than cubically in the number of observations, and so to evaluate thousands of hyperparameter configurations in parallel [58].

### 6.3 Gradient features and the neural tangent kernel

A bolder construction uses as features the gradient of the network's output with respect to all of its parameters, $g(x;\theta) = \nabla_\theta f(x;\theta) \in \mathbb{R}^p$, where $p$ is the total number of parameters. The justification is the neural tangent kernel [59]. Expanding the network to first order about its initial parameters $\theta_0$,

$$f(x;\theta) \approx f(x;\theta_0) + g(x;\theta_0)^\top(\theta - \theta_0),$$

shows that a network whose parameters move little during training behaves as a linear model in the fixed features $g(x;\theta_0)$, that is a kernel method with kernel $\Theta(x, x') = g(x;\theta_0)^\top g(x';\theta_0)$. Jacot, Gabriel and Hongler proved that this is exactly what happens as the width of every layer grows without bound. Under gradient descent the network function follows the kernel gradient of the loss with respect to $\Theta$, and in the infinite-width limit $\Theta$ converges to a deterministic kernel that stays constant throughout training, so that training a wide network by least squares is equivalent to solving a linear differential equation in function space, which converges fastest along the leading eigenfunctions of the kernel [59]. They also recall that at initialisation an infinitely wide network is a Gaussian process [59], a fact section 6.5 exploits.

NeuralUCB turns this into a bandit algorithm [60]. With a fully connected network $f(x;\theta)$ of width $m$ and depth $L$, it maintains the matrix $Z_t = Z_{t-1} + g(x_t;\theta_{t-1})g(x_t;\theta_{t-1})^\top/m$, with $Z_0 = \lambda I$, plays

$$x_t = \arg\max_x\; f(x;\theta_{t-1}) + \gamma_{t-1}\sqrt{g(x;\theta_{t-1})^\top Z_{t-1}^{-1}g(x;\theta_{t-1})/m},$$

and after each observation retrains the network by gradient descent on

$$\mathcal{L}(\theta) = \sum_{i=1}^t\tfrac12\big(f(x_i;\theta) - r_i\big)^2 + \tfrac{m\lambda}{2}\|\theta - \theta_0\|_2^2,$$

where $r_i$ is the observed reward, $\lambda$ a regularisation weight and the penalty keeps the parameters near initialisation where the linearisation holds [60]. The exploration bonus is the LinUCB bonus of section 3.10 with the gradient as feature, and $\gamma_t$ plays the role of the confidence radius. The regret analysis assumes the kernel matrix $H$ of the limiting neural tangent kernel on all candidate inputs is non-singular and defines the effective dimension

$$\tilde d = \frac{\log\det(I + H/\lambda)}{\log(1 + TK/\lambda)},$$

where $T$ is the horizon and $K$ the number of candidate arms per round, a quantity that measures how quickly the eigenvalues of $H$ decay and is the neural analogue of the information gain [60]. For a sufficiently wide network and suitable step size, the regret is of order $\tilde d\sqrt T$ up to logarithmic factors, without any parametric assumption on the reward beyond boundedness [60]. Neural Thompson sampling uses the same quantities to define a posterior. For each candidate it draws a reward $\tilde r \sim \mathcal{N}(f(x;\theta), \nu^2\sigma_t^2(x))$ with $\sigma_t^2(x) = \lambda\,g(x;\theta)^\top U_{t-1}^{-1}g(x;\theta)/m$, where $U_t$ accumulates the gradient outer products as $Z_t$ does and $\nu$ scales the exploration, plays the candidate with the largest draw, and enjoys regret of order $\sqrt T$ [61].

The $p\times p$ matrix of a network with $p$ parameters is the practical obstacle, since $p$ is easily a million. Three remedies exist. Diagonal approximations keep only the diagonal of $Z_t$. Neural-LinUCB confines the exploration to the last layer, playing $\arg\max_k\langle\varphi(x_k; w_{t-1}), \theta_{t-1}\rangle + \alpha_t\|\varphi(x_k; w_{t-1})\|_{A_{t-1}^{-1}}$ with $A_t = \lambda I + \sum_i\varphi_i\varphi_i^\top$ and $\theta_t = A_t^{-1}b_t$ as in LinUCB, and retraining the hidden layers $w$ only every $H$ rounds on the loss $\sum_i(\theta_i^\top\varphi(x_i; w) - r_i)^2$, the scheme its authors call deep representation and shallow exploration [62]. EE-Net abandons confidence widths altogether. It trains an exploitation network $f_1$ on the rewards and a second, exploration network $f_2$ that takes the normalised gradient of $f_1$ as input and is trained to predict the residual $r - f_1(x)$, the potential gain of an arm over the current estimate, and decides with a third network or a weighted sum $w_1f_1 + w_2f_2$ [63].

### 6.4 Ensembles and other approximate posteriors

A cheaper and widely used alternative trains several networks and reads their disagreement as uncertainty. Lakshminarayanan, Pritzel and Blundell train $M$ networks from different random initialisations, each outputting a mean $\mu_m(x)$ and a variance $\sigma_m^2(x)$ and trained on the Gaussian negative log likelihood $\tfrac12\log\sigma_m^2(x) + (y - \mu_m(x))^2/(2\sigma_m^2(x))$, and combine them as a uniform mixture approximated by a single Gaussian with

$$\mu_\star(x) = \frac1M\sum_m\mu_m(x), \qquad \sigma_\star^2(x) = \frac1M\sum_m\big(\sigma_m^2(x) + \mu_m^2(x)\big) - \mu_\star^2(x),$$

where the first part of the variance is the average noise each member predicts and the second the spread of the members' means, which is the epistemic part [64]. Thompson sampling with an ensemble amounts to picking one member at random and acting greedily on it, the construction Osband and colleagues introduced for deep exploration with $K$ heads trained on bootstrap resamples of the data, each transition being shown to each head according to a random mask [65]. Epistemic neural networks reduce the cost of ensembles by attaching a small auxiliary network, the epinet, to an ordinary base network, $f(x, z) = \mu_\zeta(x) + \sigma_\eta(\mathrm{sg}[\varphi_\zeta(x)], z)$, where $z$ is a random epistemic index drawn from a standard Gaussian, $\varphi_\zeta(x)$ are features of the base network passed through a stop-gradient $\mathrm{sg}$, and $\sigma_\eta$ is the sum of a trainable network and a fixed random prior network, so that varying $z$ traces out the model's uncertainty at the cost of one base forward pass, and the authors report it outperforming ensembles of hundreds of members [66].

Ensembles suit a reward model built upon an existing network, such as the critic of this pipeline, because they need neither per-candidate gradient features nor a matrix over all parameters, and section 8.3 shows that a published design method already trains its critic as an ensemble of heads without exploiting their disagreement.

### 6.5 Neural bandits over continuous and batched domains

The methods above are stated for a finite set of candidates per round, and the maximisation over a continuous design space must then be done by sampling candidates. Sample-Then-Optimize Batch Neural Thompson Sampling removes that restriction [67]. It rests on an observation from the infinite-width theory. If a wide network is initialised at random parameters $\theta_0$, perturbed by a random linear term $\langle\nabla_\theta f(x;\theta_0), \theta'_0\rangle$ with a second random draw $\theta'_0$ whose last layer is zeroed, and trained to convergence from $\theta_0$ on the loss

$$\mathcal{L}_t(\theta) = \sum_{\tau,j}\big(y_\tau^j - f_t^i(x_\tau^j;\theta)\big)^2 + \beta_t^2\sigma^2\|\theta - \theta_0\|_2^2,$$

where the sum runs over all observed points, then the trained function is a sample from the posterior of the Gaussian process whose kernel is the neural tangent kernel, with variance inflated by $\beta_t^2$ [67]. Maximising the trained network over a continuous domain is therefore a Thompson step, repeating the procedure with $B$ independent random draws gives a batch of $B$ queries, and no $p\times p$ matrix is ever formed. The authors prove regret bounds for the batch setting and report the method on automated machine learning and reinforcement learning tasks [67].

For large finite action sets, SquareCB reduces the bandit to online regression with any model [68]. Given the model's predicted losses $\hat y_a$ for every arm and the predicted best $b = \arg\min_a\hat y_a$, it plays each $a \ne b$ with probability

$$p_a = \frac{1}{K + \gamma(\hat y_a - \hat y_b)},$$

and the predicted best with the remaining probability, where $K$ is the number of arms and $\gamma$ a learning rate set from the horizon and the regression model's error [68]. This inverse gap weighting explores an arm in inverse proportion to how much worse the model believes it to be, so arms the model is unsure about, whose predicted gaps are small, are tried often, and its regret is at most $4\sqrt{KT\,\mathrm{Reg}_{\mathrm{Sq}}(T)}$ plus lower order terms, where $\mathrm{Reg}_{\mathrm{Sq}}(T)$ is the regression model's own cumulative square loss regret [68]. Its appeal for this workspace is that any network trained by regression becomes a bandit without any uncertainty estimate. Majzoubi and colleagues extended the reduction to continuous actions by smoothing each chosen action over a band of width $h$ and organising the action space as a tree of discretisations, giving a computationally tractable algorithm that composes with most supervised learning representations [69].

### 6.6 Policy networks trained by the score function estimator

The second role trains a generator. With $J(\phi) = \mathbb{E}_{x\sim\pi_\phi}[F(x)]$ and the estimator of section 2.5, a batch of $B$ designs $x_1, \dots, x_B$ drawn from the generator and their fitnesses $F_1, \dots, F_B$ give the update

$$\phi \leftarrow \phi + \alpha\,\frac1B\sum_{i=1}^B (F_i - b)\,\nabla_\phi\log\pi_\phi(x_i),$$

where $\alpha$ is a step size and $b$ a baseline. A Gaussian head contributes the scores $(x - \mu)/\sigma^2$ and $((x-\mu)^2 - \sigma^2)/\sigma^3$, the forms Ha used to learn a robot's body alongside its policy [9], and a categorical head contributes $e_c - \mathrm{softmax}(\ell)$, which is the gradient bandit update of section 3.7 exactly. The baseline may be the batch mean of the fitness or a learned value, and an entropy bonus, $\beta_H\,\mathcal{H}(\pi_\phi)$ added to the objective with $\mathcal{H}$ the entropy, resists the collapse noted in section 3.7.

The most celebrated generator of this kind is the controller of neural architecture search [70]. A recurrent network emits an architecture as a sequence of $T$ discrete choices $a_1, \dots, a_T$, each conditioned on the previous ones, and is trained by the estimator

$$\nabla_{\theta_c}J(\theta_c) \approx \frac1m\sum_{k=1}^m\sum_{t=1}^T\nabla_{\theta_c}\log P(a_t \mid a_{t-1:1};\theta_c)\,(R_k - b),$$

where $m$ architectures are sampled per batch, $R_k$ is the validation accuracy of the $k$-th after training and $b$ is an exponential moving average of past accuracies [70]. Its successor shares parameters across all candidate architectures, so that evaluating a candidate requires no training from scratch [71], which section 10.2 shows to be the exact analogue of a design-conditioned policy. When the quantity of interest is the best design rather than the average, the expectation is the wrong objective, and section 9.6 presents the risk-seeking alternative.

### 6.7 How a neural reward model is trained in practice

The surveyed algorithms share a training procedure whose details matter more in practice than their regret bounds. The data set is every design ever evaluated with its observed fitness, never only the latest generation, which is the main advantage over a search distribution. The targets should be standardised to zero mean and unit variance, or rank-transformed as in section 4.8, so that the network's scale does not drift as fitness improves. The network is retrained periodically rather than after every observation, Neural-LinUCB retraining its representation every $H$ rounds while updating its last layer continuously [62], and NeuralUCB regularising toward the initial parameters so that retraining does not discard what earlier rounds taught [60]. Where the reward drifts, the loss should weight recent observations more heavily, which section 11.3 derives. And the acquisition must be maximised over the design space, which for a mixed space is itself a mixed optimisation, handled by enumeration where the categorical block is small, by alternation between the blocks as MiVaBO does [50], or by gradient ascent on the continuous block from several starting points for each categorical configuration considered.

### 6.8 What a network adds over a search distribution

One consequence of the foregoing deserves to be stated plainly, since it bears directly on the novelty of the proposal. A generator network with no input outputs a constant distribution, so a network that predicts a design without being told anything is merely an elaborate parameterisation of the Gaussian and categorical family that CatCMA already maintains with natural gradients [45]. A network earns its place in one of three ways. As a reward model it can generalise across the whole design space from every evaluation ever made, where the incumbent retains only its distribution parameters (section 4.9). As a conditional generator it can map a context, such as a terrain, a commanded speed band or a payload, to a design, which is a contextual bandit and an amortised optimisation [40, 56]. And given a random latent input it can represent a multimodal distribution over hybrid designs, which a single Gaussian cannot, the need Schaff and colleagues met with an eight-component Gaussian mixture [2]. A proposal that claims none of the three has not yet identified what its network is for.

## 7. Bandits in robot design optimisation

The surveyed robot design literature uses bandit methods in three ways, to choose designs with a Gaussian process surrogate, to search trees of discrete design decisions, and to allocate training budget among candidate designs. A fourth question, how a discrete catalogue of components is searched, is answered in the retrieved literature by other means, and section 7.4 records how.

### 7.1 Bayesian optimisation of morphology

Gaussian process bandits, under the name of Bayesian optimisation, are the bandit method most often met in the retrieved robot design literature.

Liao and colleagues designed a simulated six-legged microrobot whose morphology can only be changed by fabricating a new batch of robots, an expensive and slow process, while its controller can be retuned cheaply [72]. Their method, hierarchical process-constrained batch Bayesian optimisation, nests two Gaussian process searches. For each morphology $\theta^m_k$ in a batch of $K$, the controller parameters $\theta^c$ are optimised by contextual Bayesian optimisation with GP-UCB, the morphology serving as an observed context so that one shared controller model generalises across morphologies, and the best reward found for that morphology is recorded. A second Gaussian process maps morphologies to their best rewards, and the next batch of $K$ morphologies is chosen from it by GP-UCB, each choice after the first being made as though the earlier choices of the batch had already returned the model's predicted reward, the hallucination device that section 11.2 analyses [72]. Choosing a whole batch per fabrication cycle reduced the number of production cycles by 360 per cent against standard Bayesian optimisation, from a hypothetical 21 months of manufacturing to 4 [72].

Bjelonic and colleagues co-designed the parallel elastic actuators of a quadruped's knees in two phases [73]. They first trained one design-conditioned locomotion policy by proximal policy optimisation, the policy observing the design parameters together with privileged information about terrain and contacts, with design parameters and task parameters sampled at random throughout training. They then froze the policy and optimised the design by Bayesian optimisation with the HEBO algorithm, which they report won the NeurIPS 2020 black-box optimisation challenge, minimising the expectation over tasks of a cost of torque,

$$\mathrm{CoTr} = \frac{\int\tau^2\,dt}{mg\,\Delta s},$$

where $\tau$ is the joint torque, $m$ the total mass, $g$ gravitational acceleration and $\Delta s$ the distance travelled, a proxy for Joule heating in the motors normalised so that designs and speeds can be compared, estimated by Monte Carlo rollouts of the frozen policy in simulation [73]. This is the published construction closest to this workspace's own, a design-conditioned policy trained across a distribution of designs with a Gaussian process search over the design, differing in that the search runs after training rather than during it.

Wang, Fang, Hanna and Xiong co-designed passive-wheel roller supports for a quadruped that skates, with an outer Bayesian optimisation over the wheel yaw installation angle, fitting a Gaussian process to the performance of evaluated designs and maximising an acquisition function, and an inner reinforcement learning loop that trains a separate policy for each candidate design [74]. Chen and colleagues co-optimised the morphology and gait of a small legged robot as a bilevel problem, with central pattern generators and deep reinforcement learning at the lower level and Bayesian optimisation over morphology at the upper level, avoiding the training of a policy from scratch for each candidate [75].

Two works apply Bayesian optimisation to combinatorial robot structures by changing the space it searches. GLSO trains a graph variational autoencoder on robot structures generated from a graph grammar, with a property predictor that encourages physically similar robots to lie close together in the latent space, and then runs Bayesian optimisation with expected improvement in that continuous latent space, turning a combinatorial search into a continuous one [76]. The Evolution Gym benchmark for soft robots includes Bayesian optimisation among its three design optimisers, in the form of batch Bayesian optimisation with a Gaussian process surrogate and batch Thompson sampling, with the batch equal to the population of the evolutionary baselines and a separate controller trained by proximal policy optimisation for each design [77].

One frequently cited legged co-design work does not use a bandit at all, and is recorded here to forestall the assumption that it does. Belmonte-Baeza and colleagues trained a locomotion policy by meta reinforcement learning with model-agnostic meta-learning, sampling design parameters and terrains at random, so that the policy adapts quickly to any design, and then optimised a quadruped's kinematics and actuator parameters with a genetic algorithm that uses the adapted policy as its evaluator [78].

### 7.2 Bandit-based tree search over discrete designs

Where designs are assembled from discrete parts, the search becomes a tree of decisions and the bandit appears as Monte Carlo tree search. RoboGrammar expresses robot structures as derivations of a graph grammar, so that each design is a sequence of rule applications, and compares its own graph heuristic search, which learns a function predicting the best performance reachable from a partial design, against Monte Carlo tree search as a baseline [79]. The baseline is UCB applied at every node of the derivation tree [33]. LA-MCTS learns online a recursive partition of a continuous search space into regions of high and low value, navigates the partition by Monte Carlo tree search with upper confidence bounds, and runs an existing optimiser such as Bayesian optimisation within the region selected [80].

### 7.3 Bandit allocation of the training budget

A third use spends the bandit on the budget rather than on the choice. ECoDe applies the successive halving brackets of Hyperband, described in section 10.1, to co-design, with a universal policy network that receives the design parameters as part of its state and so transfers controllers between neighbouring designs [81]. With $M$ the maximum number of configurations, $\eta$ the elimination rate, $F_{\max} = \lfloor\log_\eta M\rfloor$ and a budget $B = (F_{\max} + 1)M$ per filter, filter $F$ samples $n = \lceil (B/M)\,\eta^F/(F+1)\rceil$ designs with $p = M\eta^{-F}$ resource units each, and in round $i$ of the filter evaluates $n_i = \lfloor n\eta^{-i}\rfloor$ survivors with $p_i = p\eta^i$ units before keeping the best $\lfloor n_i/\eta\rfloor$ [81]. Its authors observed a bias that bears directly on this workspace. A shared policy creates a preference for designs that have received more training, since even a bad design trained on many samples can produce a seemingly better policy than a good design trained on few, and ECoDe reduces the bias by running Hyperband's filters in reverse, from the high-fidelity filters with few designs to the low-fidelity filters with many, so that the shared policy is already competent before cheap evaluations begin [81].

Schneider and colleagues co-designed the mounting of a manipulator on a mobile base, six continuous parameters, with BOHB, a combination of Bayesian optimisation and Hyperband described in section 10.1, training a separate reinforcement learning agent for each design with between 300 thousand and one million steps, and more than doubled task success for one arm, from 31.8 to 72.7 per cent, while showing that a manipulability heuristic correlated only weakly with task performance [82]. Radulov and colleagues use BOHB for the same reason in co-designing a laboratory dispensing tool with its policy, over a space that mixes continuous depth and width with a discrete rim topology, and warm-start each candidate's policy from the cached policy of the most similar previously trained design under a geometric similarity metric [83].

### 7.4 The treatment of actuator selection

Actuator choice is treated in the retrieved robot design literature by enumeration, by rounding, or by evolutionary search, rather than as a bandit. Huang and colleagues co-design geared actuators for a jumping leg with surrogate models of the motor and a hierarchical mixed-variable strategy that combines discrete enumeration with continuous search over dimensions and over real-valued indices that are rounded to admissible discrete values before each evaluation [84]. Belmonte-Baeza and colleagues optimise actuator parameters with a genetic algorithm [78]. This workspace treats the catalogue as a categorical distribution under CatCMA (section 5.5). No retrieved work in robot co-design treats an actuator catalogue as the arms of a bandit, the hybrid bandits of section 5 having been developed for hyperparameter search instead.

## 8. Bandits in the co-optimisation of design and control

### 8.1 Bilevel and single-level formulations

Wang and colleagues' survey of embodied co-design formalises the two views of section 1.1 and organises more than a hundred studies around them [1]. The bilevel view writes

$$\omega^\star = \arg\max_{\omega\in\Omega}R(\pi^\star, \omega) \quad\text{subject to}\quad \pi^\star = \arg\max_{\pi\in\Pi}R(\pi, \omega),$$

where $\omega$ is the morphology, $\Omega$ the morphology space, $\pi$ the controller and $R$ the task performance, so that the upper level searches bodies and the lower level trains a controller for each, the same structure as the search for a neural architecture and its weights [1]. The single-level view writes co-design as one Markov decision process with a design stage followed by a control stage, a nine-tuple $\langle S^D, S^C, A^D, A^C, F^D, F^C, R^D, R^C, \gamma\rangle$ of design and control state spaces, action spaces, transition functions and rewards with a discount $\gamma$, in which a design policy $\pi^d(a^d \mid s^d)$ builds the morphology $\omega = s^d_{T_d}$ over $T_d$ design steps and a control policy $\pi^c(a^c \mid s^c, \omega)$ then acts, with the single objective

$$R = \mathbb{E}_{\pi^d,\pi^c}\Big[\sum_{t=0}^{T_d}\gamma^t r^d_t + \sum_{t=T_d+1}^{\infty}\gamma^t r^c_t\Big],$$

where $r^d_t$ and $r^c_t$ are the rewards of the two stages [1]. The survey classifies methods as bilevel, single-level, generative and open-ended [1]. In bandit terms, the bilevel view treats each design as an arm whose pull is a full training run, and the single-level view treats the design as the first action of an episode, which section 9 shows is a one-step bandit embedded in the reinforcement learning problem.

### 8.2 Nested and concurrent loops

Two architectures recur within the bilevel view. In the nested form a controller is trained afresh for each design, the design search is an outer bandit, and each pull is a full training run, as in Liao, Evolution Gym, the skating quadruped and Schneider [72, 77, 74, 82]. In the concurrent form one design-conditioned controller is trained across a distribution of designs, and the distribution is moved toward better designs during training, as in Schaff, Ha and this workspace [2, 9]. Bjelonic and Belmonte-Baeza occupy a middle position, training a conditioned or meta-learned controller first and searching afterwards [73, 78]. The concurrent form is far cheaper per pull but couples the arms through the shared controller, which section 8.5 shows to be its defining difficulty.

### 8.3 The concurrent loop is a gradient bandit

Schaff and colleagues gave the concurrent form its canonical algorithm [2]. They maintain a design distribution $p_\phi(\omega)$, a Gaussian mixture of eight components with diagonal covariances whose mixing probabilities are kept uniform, append the design parameters $\omega$ to the controller's observation so that the policy $\pi_\theta(a \mid s, \omega)$ can tailor its behaviour to each design, and alternate two updates. Each iteration samples $n$ designs $\omega_i \sim p_\phi$, runs the policy on each for a fixed number of steps, and updates $\theta$ by proximal policy optimisation on the pooled trajectories. After a warm-up of one hundred million steps in which only the policy trains, each iteration also computes the average episode return $R_i$ of each sampled design and moves the design distribution by the score function estimator,

$$\nabla_\phi \approx \frac1n\sum_{i=1}^n\nabla_\phi\log p_\phi(\omega_i)\,R_i,$$

and every hundred million steps it evaluates one hundred designs from each mixture component and removes the half of the components with the lowest average reward [2]. Ha treats the design parameters as further entries of the policy, samples them from a factored Gaussian with learnable means and standard deviations, and updates designs and policy weights together with the same population-based REINFORCE, using a population of 192 agents each evaluated over 16 episodes [9]. Both are gradient bandits in the sense of section 3.7, and the incumbent CMA-ES and CatCMA searches are their natural gradient relatives (section 4.9). A neural network that predicts designs, trained over many trials, is therefore not a departure from the published concurrent loop but its generalisation, and its novelty must be sought in what section 6.8 identified.

### 8.4 The critic as a reward model

The concurrent loop already contains a neural reward model, the design-conditioned value function. Luck, Ben Amor and Calandra used a morphology-conditioned Q function, learned by soft actor-critic across all designs tried so far, to score candidate designs without building them, solving

$$\max_\xi\;\frac1n\sum_{s\in\mathcal{S}_{\mathrm{batch}}}Q\big(s, \pi(s, \xi), \xi\big)$$

over designs $\xi$ by particle swarm optimisation with about 700 particles for 250 iterations, where $\mathcal{S}_{\mathrm{batch}}$ is a batch of initial states and $\pi(s, \xi)$ the design-conditioned policy's action, and they alternated these exploitative designs with randomly drawn ones for exploration [85]. Bohlinger and Peters train one embodiment-aware policy and critic across many robots, up to fifty, including a Unitree Go2 quadruped and a humanoid, then freeze the critic and optimise new designs by following its gradient [86]. Their critic has $K$ heads whose mean $\bar V(s, \Phi(f)) = \frac1K\sum_k V_k(s, \Phi(f))$ serves as the objective, where $f$ is the normalised design vector and $\Phi$ maps it to the embodiment parameters the critic observes, averaged over a bank of $M$ states from rollouts of the final policy. They found that unconstrained maximisation of the critic drove designs toward the boundary of the design space, where the critic predicted high returns but real performance collapsed, and therefore optimise the penalised objective

$$\hat J_\lambda(f) = \frac1M\sum_{m=1}^M\bar V\big(s_m, \Phi(f)\big) - \lambda\,\frac{\|f - f_{\mathrm{ref}}\|_2^2}{d_{\mathrm{design}}},$$

with a soft trust region of weight $\lambda$ around a reference design $f_{\mathrm{ref}}$ normalised by the design dimension $d_{\mathrm{design}}$, by Adam gradient ascent with each parameter's update clipped to a maximum and the design clipped to the valid box, on design spaces of more than 1100 continuous parameters, comparing favourably with CMA-ES, Bayesian optimisation and other black-box searches using the same objective [86].

This workspace's critic observes the design (section 1.2), so a design-conditioned estimate of return is available at every generation at no cost in simulation. The two published uses differ in how they treat the critic's unreliability far from the designs it has seen. Luck and colleagues counter it with random exploratory designs [85], and Bohlinger and Peters with a distance penalty, although their critic is an ensemble of heads whose disagreement could have measured the unreliability directly [86]. Neither uses the critic's uncertainty to decide where to explore, and the bandit view supplies exactly that missing element.

### 8.5 Rewards that depend on past allocation

In the concurrent loop the reward of an arm is not a fixed property of the design. Let $\pi_k$ be the policy at generation $k$, trained on the designs drawn in generations $1$ to $k$, and define the merit of design $x$ at generation $k$ as

$$F_k(x) = \mathbb{E}\big[\text{return} \mid \pi_k, x\big].$$

Then $F_k$ changes with $k$, and it rises fastest for the designs the policy has trained upon most, so a design sampled often is made to look better by the sampling itself. The co-design literature knows this as fragile co-adaptation and premature convergence of the morphology. Cheney and colleagues identified it as the central obstacle to co-optimising morphology and control and proposed giving newly changed bodies more time for their controllers to adapt before they must compete [87, 88]. Mertan and Cheney then showed concrete high-performing regions of a morphology space that co-optimisation never reaches but that are easily found when morphologies are optimised under a fixed controller [88]. Their explanation is directly a bandit one. They propose viewing the fitness landscape as a surface over body plans that carries a range of possible values per body, the realised value depending on how much controller optimisation the body has received. Bodies found early accrue more optimisation and climb toward their maximum, a first-mover advantage, while promising bodies found later cannot compete until a controller is optimised for them and seldom survive long enough to receive one [88]. They connect this view explicitly to surrogate fitness models, to the multi-armed bandit, and to the weight-sharing supernets of neural architecture search [88]. ECoDe's authors observed the same bias in a universal policy network (section 7.3) [81].

In bandit terms the arms are non-stationary, their drift is endogenous, and an arm's value depends on its own pull history, so an algorithm built on stationary arms will under-rate every design it has neglected. Section 11.3 develops a model of this drift and the corrections it suggests.

### 8.6 Population-based bandits as the nearest algorithmic precedent

Population based training runs a population of agents in parallel, each with its own weights and hyperparameters [89]. When a member is ready, after a minimum number of training steps, an exploit step replaces it if it ranks in the bottom 20 per cent of the population with a copy of the weights and hyperparameters of a member drawn uniformly from the top 20 per cent, and an explore step perturbs the copied hyperparameters by a factor of 0.8 or 1.2 or resamples them from their prior [89]. The population thus learns a schedule of hyperparameters during a single training run, and populations of 20 to 40 members suffice for consistent gains [89].

PB2 replaced the random explore step with a time-varying Gaussian process bandit [90]. Its derivation begins from a model of how the objective changes with time. Bogunovic, Scarlett and Cevher model a sequence of functions by $f_1 = g_1$ and

$$f_{t+1} = \sqrt{1 - \varepsilon}\,f_t + \sqrt{\varepsilon}\,g_{t+1},$$

where the $g_t$ are independent draws from a Gaussian process with kernel $k$ and $\varepsilon \in [0,1]$ is a forgetting rate, so each function keeps a fraction of the previous one and adds fresh variation [91]. Each $f_t$ has covariance $k$, because the variances contributed by the two terms, $(1-\varepsilon)$ and $\varepsilon$ times that of a draw, add to one, and unrolling the recursion for $t' > t$ gives $f_{t'} = (1-\varepsilon)^{(t'-t)/2}f_t$ plus terms independent of $f_t$, so the covariance between $f_t(x)$ and $f_{t'}(x')$ is

$$\mathrm{Cov}\big(f_t(x), f_{t'}(x')\big) = k(x, x')\,(1 - \varepsilon)^{|t - t'|/2},$$

the product of a spatial kernel and a temporal kernel that discounts old evidence geometrically [91, 90]. Inference uses the ordinary posterior of section 4.5 with the kernel matrix multiplied elementwise by the temporal factors. TV-GP-UCB, the upper confidence bound rule under this kernel, has regret bounded by

$$R_T \le \sqrt{C_1T\beta_T\Big(\frac{T}{\tilde N} + 1\Big)\big(\gamma_{\tilde N} + \tilde N^3\varepsilon\big)} + 2$$

for any block length $\tilde N \le T$, where $\gamma_{\tilde N}$ is the information gain of the time-invariant problem over $\tilde N$ steps, the bound splitting the horizon into blocks within which the function changes little [91]. For a fixed $\varepsilon$ regret must grow linearly, since a function that keeps changing cannot be tracked exactly, and the bound is sublinear when $\varepsilon$ shrinks with $T$ [91].

PB2 applies this model to population based training with three choices [90]. Its observation for a member is the change in performance over one interval of $t_{\mathrm{ready}}$ steps, $y_t = F_t(x_t) - F_{t-1}(x_{t-1}) + \varepsilon_t$, so the model learns which hyperparameters produce improvement rather than which produce high levels. Its kernel is the product of a squared exponential over hyperparameters and the temporal kernel above, with forgetting rate $\omega$. And its batch is chosen sequentially by maximising $\mu_{t,1}(x) + \sqrt{\beta_t}\,\sigma_{t,b}(x)$ for $b = 1, \dots, B$, holding the mean fixed at its value before the batch while updating the variance for the members already chosen, the device of GP-BUCB [90, 92]. Its regret is sublinear when consecutive functions are strongly correlated, and its experiments used populations of only four or eight agents for proximal policy optimisation [90]. PB2-Mix extends it to categorical choices as described in section 5.4 [22].

| Population based training | This workspace |
|---|---|
| member of the population | design individual, 256 per generation (`train.py:222`) |
| hyperparameters of a member | link length scales, nominal pose, actuator per group |
| interval between decisions, $t_{\mathrm{ready}}$ | `ea_update_interval`, 480 iterations (`train.py:225`) |
| observation, improvement over the interval | mean return over the interval (`copt_on_policy_runner.py:733`) |
| weights copied from good members to poor ones | weights shared by all members through one conditioned policy |
| batch of 4 or 8 members [90] | batch of 256 designs |

The correspondence is close but not exact. A PB2 member's weights belong to it alone, whereas here every design is served by the same weights, so the coupling of section 8.5 is stronger here than in any setting PB2 was analysed for, and no retrieved work applies a population-based bandit to the morphology of a robot.

## 9. Making the bandit part of proximal policy optimisation

This section answers the follow-up question, whether proximal policy optimisation can be modified so that the design choice becomes part of its own formulation, and which prior works have done so in this or a similar domain. It first derives proximal policy optimisation from the policy gradient theorem, so that the modification can be stated precisely, then shows that a bandit is a one-step instance of the problem the algorithm solves and what the algorithm does on such an instance, then surveys four families of prior work, and closes with a formulation for this workspace.

### 9.1 From the policy gradient theorem to proximal policy optimisation

A Markov decision process consists of states $s$, actions $a$, a transition law $P(s_{t+1} \mid s_t, a_t)$, a reward $r(s_t, a_t)$, an initial state distribution $\rho_0$ and a discount $\gamma \in [0,1)$. A stochastic policy $\pi_\theta(a \mid s)$ with parameters $\theta$ induces trajectories $\tau = (s_0, a_0, s_1, a_1, \dots)$ with probability $p_\theta(\tau) = \rho_0(s_0)\prod_t\pi_\theta(a_t \mid s_t)P(s_{t+1} \mid s_t, a_t)$, and its objective is the expected discounted return $J(\theta) = \mathbb{E}_\tau[\sum_t\gamma^tr_t]$ [93]. Three functions describe a policy's quality, the value $V^\pi(s) = \mathbb{E}[\sum_{l\ge0}\gamma^lr_{t+l} \mid s_t = s]$, the action value $Q^\pi(s,a)$ defined the same way with the first action fixed to $a$, and the advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, the amount by which action $a$ is better than the policy's average action in state $s$ [93].

Applying the score function identity of section 2.5 to trajectories, the transition and initial probabilities do not depend on $\theta$, so $\nabla_\theta\log p_\theta(\tau) = \sum_t\nabla_\theta\log\pi_\theta(a_t \mid s_t)$, and

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t \mid s_t)\,\hat A_t\Big],$$

where the whole-trajectory return has been replaced by an advantage estimate $\hat A_t$, which is permitted because rewards before time $t$ cannot be influenced by the action at time $t$ and because the value $V(s_t)$ may be subtracted as a state-dependent baseline [7, 8]. The usual estimate is generalised advantage estimation,

$$\hat A_t = \sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),$$

where $V$ is a learned value function and $\lambda \in [0,1]$ trades the bias of a short bootstrap, at $\lambda = 0$, against the variance of a long Monte Carlo sum, at $\lambda = 1$ [94, 95].

Following this gradient naively takes steps that may change the policy so much that its data no longer describe it. Trust region policy optimisation controls the step through an identity of Kakade and Langford, which expresses the return of a new policy $\tilde\pi$ through the advantages of the old policy $\pi$,

$$\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tau\sim\tilde\pi}\Big[\sum_t\gamma^tA^\pi(s_t, a_t)\Big],$$

where $\eta$ denotes expected return [93]. Replacing the state distribution of the new policy by that of the old gives a local surrogate $L_\pi(\tilde\pi)$ that can be estimated from old data, and Schulman and colleagues proved that

$$\eta(\pi_{\mathrm{new}}) \ge L_{\pi_{\mathrm{old}}}(\pi_{\mathrm{new}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\alpha^2, \qquad \epsilon = \max_{s,a}|A^\pi(s,a)|,$$

where $\alpha$ is the largest total variation distance between the two policies over states [93]. Improving the surrogate while keeping the policies close therefore guarantees improvement, and the practical algorithm maximises $\mathbb{E}_t[r_t(\theta)\hat A_t]$ with the probability ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)$ subject to an average relative entropy constraint $\mathbb{E}_t[\mathrm{KL}(\pi_{\theta_{\mathrm{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t))] \le \delta$, solved approximately with a conjugate gradient step that is a natural gradient step of section 2.6 [93, 95].

Proximal policy optimisation replaces the constraint by a first-order device [95]. Its clipped surrogate is

$$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big],$$

with $\epsilon$ typically 0.2 [95]. The objective equals the unconstrained surrogate to first order at $\theta_{\mathrm{old}}$, where every ratio is one. When the advantage is positive, raising the probability of the action improves the objective only until the ratio reaches $1+\epsilon$, beyond which the clipped term is flat and the minimum removes the incentive to go further, and when the advantage is negative, lowering the probability helps only until the ratio reaches $1-\epsilon$. Taking the minimum of the clipped and unclipped terms makes the objective a pessimistic lower bound on the unclipped one, ignoring ratio changes when they would make the objective look better and including them when they make it worse [95]. An alternative penalises the relative entropy with a coefficient $\beta$ that is halved when the measured divergence falls below two thirds of a target and doubled when it exceeds one and a half times the target [95]. With a shared network for policy and value, the objective maximised by several epochs of minibatch stochastic gradient ascent on the same batch is

$$L(\theta) = \mathbb{E}_t\Big[L^{\mathrm{CLIP}}_t(\theta) - c_1\big(V_\theta(s_t) - V_t^{\mathrm{targ}}\big)^2 + c_2\,\mathcal{H}\big[\pi_\theta(\cdot \mid s_t)\big]\Big],$$

where $V^{\mathrm{targ}}_t$ is the value target, $\mathcal{H}$ the entropy of the action distribution, and $c_1, c_2$ weights [95]. This is the algorithm the workspace's inner loop runs.

### 9.2 A bandit is a one-step decision process

A contextual bandit is a Markov decision process whose episodes last one step. The state is the context $c$, the action is the arm, and the episode ends after the reward, so the return is the reward, the value is the expected reward of the policy in that context, and the advantage is $A(c, a) = \mu(c, a) - \mathbb{E}_{a'\sim\pi}[\mu(c, a')]$. Proximal policy optimisation can therefore be run on a bandit without change, and its clipped objective becomes

$$L^{\mathrm{CLIP}}(\phi) = \mathbb{E}_i\Big[\min\big(\rho_i(\phi)\,(R_i - b),\ \mathrm{clip}(\rho_i(\phi), 1-\epsilon, 1+\epsilon)\,(R_i - b)\big)\Big], \qquad \rho_i(\phi) = \frac{\pi_\phi(a_i \mid c_i)}{\pi_{\phi_{\mathrm{old}}}(a_i \mid c_i)},$$

where $R_i$ is the reward of the $i$-th sampled arm and $b$ a baseline. At $\phi = \phi_{\mathrm{old}}$ its gradient is the score function estimator of section 2.5, so a single step of the method is a gradient bandit step, and the clipping bounds how far the distribution may move on one batch, a trust region in the sense of section 2.6.

Two published analyses explain how the method behaves on such problems. Wang and colleagues studied proximal policy optimisation on a discrete-armed bandit and showed that its clipping range is unequal across arms [96]. The ratio constraint allows the probability of arm $a$ to change by at most $\pi_{\mathrm{old}}(a)\,\epsilon$ in one update, an amount proportional to its current probability, so an optimal arm that the initial policy disfavours can gain probability only slowly, while a favoured suboptimal arm can gain it quickly, and because the probabilities compete for a fixed total the optimal arm's share may continue to decrease [96]. They concluded that the method is prone to a lack of exploration under a poor initialisation and proposed adapting each arm's clipping range from a relative entropy trust region instead, with a better performance bound [96]. The convergence analysis of exact softmax policy gradient reaches a compatible conclusion from another direction, its constants depending on the smallest probability the optimal arm ever has, which is controlled by the initial distribution, with entropy regularisation accelerating convergence [18]. Both bear directly on a design head initialised, as the incumbent is, with half of each group's probability on the donor's actuator (`usd_generator.py:841`).

The most widely deployed use of proximal policy optimisation on a bandit is the fine-tuning of language models from human preferences. Ouyang and colleagues describe their environment as a bandit environment which presents a random prompt, expects a response, produces a reward determined by a learned reward model and ends the episode [97]. The reward model $r_\theta(x, y)$ for prompt $x$ and response $y$ is trained on human rankings of $K$ responses by the loss

$$\mathcal{L}(\theta) = -\binom{K}{2}^{-1}\mathbb{E}_{(x, y_w, y_l)}\big[\log\sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\big],$$

where $y_w$ is the preferred and $y_l$ the less preferred response of a pair and $\sigma$ the logistic function [97]. The policy is then trained by proximal policy optimisation to maximise

$$\mathbb{E}_{(x,y)\sim\pi_\phi}\Big[r_\theta(x, y) - \beta\log\frac{\pi_\phi(y \mid x)}{\pi^{\mathrm{SFT}}(y \mid x)}\Big] + \gamma\,\mathbb{E}_{x\sim D_{\mathrm{pretrain}}}\big[\log\pi_\phi(x)\big],$$

where $\pi^{\mathrm{SFT}}$ is the supervised reference policy, the relative entropy penalty of weight $\beta$ mitigates over-optimisation of the learned reward model, the last term of weight $\gamma$ mixes in the original training objective, and the value function is initialised from the reward model [97]. Three lessons transfer to design. A learned reward model can stand in for expensive evaluations, as the critic of this workspace can for training runs. A policy optimised against a learned reward model will exploit the model's errors unless held near a reference distribution, which is the same phenomenon Bohlinger and Peters met when maximising a critic over designs (section 8.4). And a relative entropy penalty toward a reference distribution, here the donor-centred design distribution, is one principled guard against it.

### 9.3 Design as the first actions of an augmented decision process

The single-level formulation of section 8.1 makes the design part of the episode. Several works train the resulting augmented process with proximal policy optimisation, and they differ chiefly in how the design actions receive credit.

Transform2Act is the reference construction [98]. The agent's design is a graph $D_t = (V_t, E_t, A_t)$ of joints $V_t$, bones $E_t$ and per-joint attributes $A_t$ such as bone length, size and motor strength, and the state of the augmented process is $s_t = (s^e_t, D_t, \Phi_t)$, the environment state, the design and a stage flag. Each episode has three stages. In a skeleton transform stage of $N_s$ steps a sub-policy chooses for each joint a discrete action, add a child joint, delete the joint or leave it unchanged. In an attribute transform stage of $N_z$ steps a sub-policy chooses for each joint a continuous change drawn from a Gaussian, $z_{u,t+1} = z_{u,t} + a^z_{u,t}$ with $a^z_{u,t} \sim \mathcal{N}(\mu^z_{u,t}, \Sigma^z)$. In an execution stage the design is frozen and a control sub-policy drives the joints. All three sub-policies are graph neural networks sharing one parameter vector and selected by the stage flag, the transform stages receive no reward, and the transform actions are trained only through the future rewards of the execution stage that their design earns [98]. A single value network, also a graph network, takes the stage flag as input and predicts the return from every stage, the environment state being set to zero during the transform stages, and the whole policy is trained by proximal policy optimisation, which its authors chose because its constraint on the change between successive policies prevents large changes to the transform actions and hence to the design in any one update [98]. At test time the most likely transform actions define the final design. A joint-specialised multilayer perceptron supplements the graph network, because weight sharing across joints otherwise makes similar joints choose identical transform actions [98].

Symmetry-aware robot design extends this with a search over symmetry subgroups of the dihedral group, generating designs under the chosen symmetry, and updates its design policy and control policy together by proximal policy optimisation [99]. Pathak and colleagues let primitive limbs choose to link into composite bodies during an episode, the linking being one of the actions of a modular policy trained by proximal policy optimisation [100]. Chen, He and Ciocarlie move hardware parameters, such as those of an underactuated hand they then built, from the environment into the policy's own computational graph, so that they are optimised by the policy gradient alongside the network weights [101].

BodyGen identifies the credit problem that the earlier formulations leave open and repairs it [102]. Only the control stage receives environmental reward, yet a design action affects every later step, while a control action's effect fades, so discounting and bootstrapping treat the two kinds of action inappropriately alike. BodyGen therefore computes advantages differently in the two stages,

$$\hat A_t = \begin{cases}\delta_t + \gamma\lambda\hat A_{t+1}(1 - T_t \vee C_t) & \text{control stage,}\\ U_t - V_\theta(s_t) & \text{design stage,}\end{cases} \qquad U_t = r_t + U_{t+1}(1 - T_t \vee C_t),$$

where $\delta_t$ is the temporal difference error, $T_t$ and $C_t$ flag termination and truncation of the episode, and $U_t$ is the undiscounted sum of the rewards from step $t$ to the end of the episode [102]. The design-stage advantage is thus the whole episode's undiscounted return minus a baseline, which is exactly the advantage of a one-step bandit whose arm is the design and whose reward is the episode return, and each stage has its own value network to avoid conflicting gradients [102]. Its policy and value networks are transformers with a topology-aware position encoding that hashes each limb's path to the root, which keeps the encoding consistent as the morphology grows [102]. Its authors report an average improvement of 60.03 per cent over the baselines they compare [102].

### 9.4 The leader and the follower, Stackelberg proximal policy optimisation

Dai, Wang, Ashley and Schmidhuber observe that every formulation above updates the design while treating the controller as fixed, although the controller will adapt to whatever design is chosen, so that the design update may point in a direction the controller's adaptation then undoes [103]. They model co-design as a Stackelberg game, in which a leader commits first and a follower then best-responds. Writing $\theta_L$ for the design policy's parameters, $\theta_F$ for the controller's, and $J_L$ and $J_F$ for their objectives, the leader solves

$$\max_{\theta_L}J_L\big(\theta_L, \theta_F^\star(\theta_L)\big) \quad\text{subject to}\quad \theta_F^\star(\theta_L) = \arg\max_{\theta_F}J_F(\theta_L, \theta_F),$$

and its total gradient contains, beyond the direct term, a term for its influence on the follower [103],

$$\nabla_{\theta_L}J_L = \nabla_{\theta_L}J_L(\theta_L, \theta_F) + \big(\nabla_{\theta_L}\theta_F^\star(\theta_L)\big)^\top\nabla_{\theta_F}J_L(\theta_L, \theta_F).$$

The response Jacobian follows from the implicit function theorem applied to the follower's optimality condition $\nabla_{\theta_F}J_F(\theta_L, \theta_F^\star) = 0$, which holds for every $\theta_L$, so differentiating it gives

$$\big(\nabla_{\theta_L}\theta_F^\star\big)^\top = -\nabla_{\theta_L\theta_F}J_F\,\big(\nabla^2_{\theta_F}J_F\big)^{-1},$$

where $\nabla_{\theta_L\theta_F}J_F$ is the matrix of mixed second derivatives and $\nabla^2_{\theta_F}J_F$ the follower's Hessian [103]. The objectives are asymmetric. The leader's objective sums immediate rewards for the $T$ design steps, such as material costs, and the follower's subsequent rewards, $J_L = \mathbb{E}[\sum_{t<T}\gamma^tR_L + \sum_{t\ge T}\gamma^{t-T}R_F]$, while the follower's objective contains its own rewards only [103]. The interaction is phase-separated, the leader acting for $T$ steps and the follower thereafter, and the interface between them is the final design state, which is not differentiable with respect to the leader's parameters because design actions are discrete.

The paper's contribution is to estimate every term of the gradient from sampled trajectories despite the non-differentiable interface [103]. The cross derivative is obtained from the surrogate

$$L^F_{L,F}(\theta_L, \theta_F) = c\,\mathbb{E}\Big[\frac{\pi^L_{\theta_L}(a^L \mid s^L)}{\pi^L_{\theta_L^o}(a^L \mid s^L)}\,\gamma^T\,\mathbb{E}\Big[\frac{\pi^F_{\theta_F}(a^F \mid s^F; s^L_T)}{\pi^F_{\theta_F^o}(a^F \mid s^F; s^L_T)}\,A^F(s^F, a^F; s^L_T)\Big]\Big], \qquad c = \frac{T}{1-\gamma},$$

whose mixed second derivative at the behaviour parameters $\theta^o$ equals that of the follower's objective, the log-derivative technique carrying the dependence on the leader through the importance ratio of its own actions [103]. The first-order terms come from the analogous importance-weighted surrogates. The follower's Hessian, which is typically indefinite, is replaced by the Fisher information of the follower's policy, estimated as the Hessian of the average relative entropy between new and old policies, and damped by $\lambda I$, so that $\lambda \to \infty$ recovers the ordinary policy gradient and $\lambda \to 0$ the full Stackelberg gradient [103]. The resulting estimate is

$$\nabla_{\theta_L}\hat J_L = \nabla_{\theta_L}\hat L^L_L - \nabla_{\theta_L\theta_F}\hat L^F_{L,F}\,\big(\nabla^2_{\theta_F}\hat L^F_{\mathrm{KL}} + \lambda I\big)^{-1}\nabla_{\theta_F}\hat L^L_F,$$

with proximal policy optimisation's clipping applied to the sample surrogates, which the authors justify by the surrogates' local equivalence to the true gradients rather than by simple reuse of the device [103]. The inverse is applied to a vector rather than formed, so the cost is that of a few Hessian-vector products. Across their co-design tasks the method outperforms standard proximal policy optimisation in stability and final performance, and its comparisons include BodyGen and Transform2Act [103].

### 9.5 Bandits that steer proximal policy optimisation from outside

A second family keeps proximal policy optimisation unchanged and places a bandit beside it, choosing what the learner trains on. Schaff and colleagues are the co-design instance, proximal policy optimisation for the controller and a score function update for the design distribution (section 8.3) [2]. Three adjacent instances show how such a bandit must be built when its rewards depend on the learner.

Prioritized level replay chooses which procedurally generated level the learner plays next [104]. Each level $l_i$ seen so far carries a score, the average magnitude of the generalised advantage estimate over its latest trajectory,

$$S_i = \frac1T\sum_{t=0}^T\Big|\sum_{k=t}^T(\gamma\lambda)^{k-t}\delta_k\Big|,$$

taken as a measure of how much the learner can still learn from it, and levels are replayed from the mixture

$$P_{\mathrm{replay}}(l_i) = (1 - \rho)\,\frac{h(S_i)^{1/\beta}}{\sum_jh(S_j)^{1/\beta}} + \rho\,\frac{c - C_i}{\sum_j(c - C_j)},$$

where $h(S_i) = 1/\mathrm{rank}(S_i)$, $\beta$ is a temperature, $c$ the number of episodes so far, $C_i$ the episode at which level $i$ was last played and $\rho$ a staleness coefficient [104]. The second term exists because a score computed under an old policy becomes stale as the policy changes, which is the endogenous drift of section 8.5 in another guise, and mixing in levels in proportion to how long ago they were seen keeps every score fresh [104].

Agent57 trains a family of policies differing in their exploration bonus and discount, and chooses which to run in each episode by a non-stationary bandit, a sliding-window upper confidence bound with an additional small probability of a random choice, whose reward is the undiscounted extrinsic return of the episode [105]. Its authors chose a sliding window explicitly because the reward of each arm changes as the agent learns, so a stationary bandit would not adapt [105]. PB2 runs population based bandits over the hyperparameters of proximal policy optimisation itself (section 8.6) [90].

### 9.6 Objectives other than the expected reward

Proximal policy optimisation maximises an expectation, which suits a controller that must perform well on average but not a design search, whose output is the single best design found. Petersen and colleagues called this mismatch the expectation problem and proposed the risk-seeking objective

$$J_{\mathrm{risk}}(\theta;\varepsilon) = \mathbb{E}_{\tau\sim p_\theta}\big[R(\tau) \mid R(\tau) \ge R_\varepsilon(\theta)\big],$$

the expected reward of the best fraction $\varepsilon$ of samples, where $R_\varepsilon(\theta)$ is the $(1-\varepsilon)$-quantile of the reward under the current policy [106]. Its gradient is $\mathbb{E}[(R(\tau) - R_\varepsilon(\theta))\nabla_\theta\log p_\theta(\tau) \mid R(\tau) \ge R_\varepsilon(\theta)]$, estimated from a batch of $N$ samples by

$$\nabla_\theta J_{\mathrm{risk}} \approx \frac{1}{\varepsilon N}\sum_{i=1}^N\big(R(\tau^{(i)}) - \tilde R_\varepsilon\big)\,\mathbb{1}\big[R(\tau^{(i)}) \ge \tilde R_\varepsilon\big]\,\nabla_\theta\log p_\theta(\tau^{(i)}),$$

which is REINFORCE with two changes, a baseline fixed at the empirical quantile $\tilde R_\varepsilon$ and only the top fraction of samples contributing [106]. The authors note that it applies to any batch policy gradient method, proximal policy optimisation included [106]. For a design head it is the simple regret objective of section 3.9 expressed as a policy gradient.

Two further departures from the expectation matter for designs. Generative flow networks learn a policy that generates objects with probability proportional to a positive reward, rather than concentrating on the single best, which produces the diverse batches that a search with few rounds and large batches needs, and they are trained by turning flow consistency equations into a temporal difference style objective [107]. And where a deterministic greedy decoding of the generator is available, the reward of the greedy design is a baseline that Kool and colleagues found more effective than a learned value function for a generator of combinatorial solutions trained by REINFORCE [108].

### 9.7 A proximal policy optimisation formulation for this workspace

The following construction is this document's own, assembled from the pieces above, and is offered as the precise answer to how proximal policy optimisation could be modified so that the design bandit becomes part of its formulation.

The first observation fixes the time scale. In Transform2Act and BodyGen the design is chosen anew in every episode. In this workspace a design is held by its environments for a whole generation of 480 iterations, because changing it means authoring new geometry into the design prototypes and resetting every environment (`copt_on_policy_runner.py:656`, and section 5.1 of `CO_OPTIMISATION.md` at the workspace root), so a design decision lasts for about twelve episodes per environment and its consequence is observed only at the end of the generation. The design choice is therefore a contextual bandit at the level of the generation, not an action within an episode, and the natural modification of proximal policy optimisation runs two clipped objectives at two time scales, the existing one for the controller every iteration and a new one for the design head every generation. The study of porting the workspace to Newton records that per-world model arrays would make per-environment design changes far cheaper (`isaaclab_newton.md` section 10.13), which would bring the per-episode formulation of section 9.3 within reach, but this report does not assume it.

The design head is a policy $\pi_\phi(x \mid c)$ over hybrid designs with the factorised likelihood of section 5.6, a softmax head per actuator group and a Gaussian head over length scales and pose, given a context $c$ that may be empty. At generation $k$ it draws $B = 256$ designs $x_{k,i}$, the runner trains the controller on them as now, and at the end of the generation each design has a fitness $\hat F_{k,i}$. The design advantage follows BodyGen in using the undiscounted return without bootstrapping through design steps [102], standardised within the generation as fitness shaping prescribes [42],

$$\hat A_{k,i} = \frac{\hat F_{k,i} - b_k}{s_k},$$

where $b_k$ is a baseline and $s_k$ the standard deviation of the generation's fitnesses. The baseline may be the generation mean, or, following InstructGPT's initialisation of the value function from the reward model [97], the critic's estimate of the design distribution's average value. The design head is updated by several epochs on the generation's 256 designs of

$$L_D(\phi) = \frac1B\sum_{i=1}^B\min\big(\rho_i(\phi)\hat A_{k,i},\ \mathrm{clip}(\rho_i(\phi), 1-\epsilon_D, 1+\epsilon_D)\hat A_{k,i}\big) + \beta_H\,\mathcal{H}(\pi_\phi) - \beta_{\mathrm{KL}}\,\mathrm{KL}\big(\pi_\phi\,\|\,\pi_{\mathrm{ref}}\big),$$

where $\rho_i(\phi) = \pi_\phi(x_{k,i} \mid c)/\pi_{\phi_{\mathrm{old}}}(x_{k,i} \mid c)$ is the ratio of the design's likelihood under the updated and the sampling head, $\epsilon_D$ the design clipping range, $\beta_H$ an entropy weight and $\beta_{\mathrm{KL}}$ the weight of a relative entropy penalty toward a reference distribution $\pi_{\mathrm{ref}}$ such as the donor-centred initial distribution [95, 97]. Without a network, with rank-based weights in place of the standardised advantage and a natural gradient step in place of the clipped surrogate, the same update reduces to the information-geometric updates that CatCMA performs (sections 4.9 and 5.5), which is the precise sense in which the incumbent is a special case.

Four modifications follow from the literature. The clipping range should be adapted per category or replaced by a relative entropy trust region, or every probability floored as CatCMA's margin does, because otherwise the categorical head inherits the lock-in on the initially favoured actuator that Wang and colleagues proved [96, 45]. The advantage may be computed from the critic rather than from the observed fitness alone, $\bar V(x) = \frac{1}{|\mathcal{S}_0|}\sum_{s\in\mathcal{S}_0}V_\psi(s, x)$ averaged over a bank $\mathcal{S}_0$ of initial states, which has lower variance and exists for designs not sampled, at the cost of the extrapolation that section 8.4 warned of [85, 86]. The design update can anticipate the controller's adaptation through the Stackelberg correction of section 9.4, whose cross derivative is estimable from the same rollouts because the controller's actions are already logged with their probabilities [103]. And where the best design rather than the average is wanted, the risk-seeking estimator of section 9.6 replaces the standardised advantage by the excess over the generation's $(1-\varepsilon)$-quantile on the top fraction only [106].

### 9.8 What this question's literature does not contain

Every retrieved work that places the design inside a proximal policy optimisation objective, Transform2Act, symmetry-aware design, BodyGen, Stackelberg proximal policy optimisation and the self-assembling limbs, evaluates on simulated creatures assembled from generic limbs rather than on a legged robot with identified actuators, and none chooses components from a catalogue [98, 99, 102, 103, 100]. The legged robot works that train a design-conditioned policy with proximal policy optimisation, Bjelonic's and Schaff's, keep the design search outside the clipped objective, the former as Bayesian optimisation after training and the latter as a score function update during it [73, 2]. No retrieved work applies a clipped surrogate to a design distribution at the time scale of a generation, which is the construction of section 9.7, and none combines a design head with a learned critic baseline and a relative entropy anchor to a reference design distribution, although each ingredient is established in the work cited for it.

## 10. Adjacent domains

### 10.1 Hyperparameter optimisation

Hyperparameter search is the adjacent domain most like this one, a choice among configurations whose merit is revealed only by expensive and noisy training. It has also developed the idea that matters most here, that a configuration can be evaluated at several fidelities, cheaply and roughly or expensively and well. Successive halving, introduced to hyperparameter search by Jamieson and Talwalkar, evaluates many configurations on a small budget, keeps the best fraction $1/\eta$ of them, multiplies their budget by $\eta$, and repeats until one remains [26]. Its weakness is the choice of how many configurations to start with, since starting with many gives each too little budget to be judged, and Hyperband hedges by running several brackets of successive halving that start from different trade-offs [109]. With a maximum budget $b_{\max}$ and minimum $b_{\min}$ per configuration and elimination rate $\eta$, it sets $s_{\max} = \lfloor\log_\eta(b_{\max}/b_{\min})\rfloor$ and for each $s = s_{\max}, \dots, 0$ samples

$$n_s = \Big\lceil\frac{s_{\max} + 1}{s + 1}\,\eta^s\Big\rceil$$

configurations and runs successive halving on them from the initial budget $\eta^{-s}b_{\max}$, so that every bracket consumes about the same total budget and the most aggressive bracket evaluates many configurations very cheaply while the most conservative evaluates few at full budget [109, 110]. BOHB replaces Hyperband's random sampling of configurations with a model-based proposal, the tree Parzen estimator, which fits kernel density estimates $l(x) = p(y < \alpha \mid x)$ and $g(x) = p(y > \alpha \mid x)$ to the good and bad configurations relative to a quantile $\alpha$ of the observed losses and proposes the configuration maximising $l(x)/g(x)$, which is equivalent to maximising expected improvement and handles mixed continuous and discrete spaces with cost linear in the data [110]. Multi-fidelity Gaussian process bandits model the objective and its cheap approximations jointly and use the approximations to eliminate low-value regions cheaply, reserving expensive evaluations for a small promising region, with a regret better than that of strategies ignoring the approximations [111]. In the present setting a fidelity is the number of policy iterations or of environments spent on a design, and section 12.4 uses it.

### 10.2 Neural architecture search

Neural architecture search is the domain in which a network that predicts designs is most firmly established. Zoph and Le's controller, described in section 6.6, emits an architecture as a sequence of discrete choices and learns by REINFORCE with a moving-average baseline from the validation accuracy of what it emits [70]. ENAS shares the parameters of every candidate architecture within one large graph, trains its controller by policy gradient, and reduces the cost of the search a thousandfold, because a candidate is evaluated with the shared weights rather than trained from scratch [71]. A design-conditioned locomotion policy plays exactly the role of those shared weights, a single network that serves every design and against which each design is scored.

The same domain supplies the sharpest caution in this survey. Li and Talwalkar found that random search with early stopping performs at least as well as ENAS on two standard benchmarks, and that random search with weight sharing does better still [112]. When evaluation runs through shared weights, the quality of the shared model can dominate the cleverness of the search, so any bandit proposed here must be measured against random search over the same lattice under the same shared policy. Mertan and Cheney make the same connection between weight-sharing supernets and co-design (section 8.5) [88].

### 10.3 Curriculum learning and environment design

A curriculum chooses which tasks a learner trains on, and the choice has repeatedly been cast as a bandit whose reward is the learner's progress. Graves and colleagues fed a measure of learning progress, such as the gain in prediction accuracy on a task, as the reward of a non-stationary bandit choosing the next task [21]. They used Exp3.S, whose distribution is $\pi(i) = (1-\epsilon)\pi^{\mathrm{EXP3}}(i) + \epsilon/N$ over $N$ tasks with weights updated by importance-weighted rewards and mixed by fixed share, because the best task changes as the model learns, and they rescaled every reward to $[-1, 1]$ by clipping it to the 20th and 80th percentiles $q^{\mathrm{lo}}_t$ and $q^{\mathrm{hi}}_t$ of the rewards seen so far and mapping linearly, $r_t = 2(\hat r_t - q^{\mathrm{lo}}_t)/(q^{\mathrm{hi}}_t - q^{\mathrm{lo}}_t) - 1$, so that the step size need not track the changing scale of progress [21]. Matiisen and colleagues framed the choice as a teacher selecting subtasks for a student by the slope of the student's learning curve on each, taking the absolute value of the slope so that a task on which performance is falling, through forgetting, is also revisited, and estimating the slope online, from a window of recent scores by linear regression, or by sampling [113].

Portelas and colleagues turned the choice of continuous environment parameters into what they call a surrogate continuous bandit problem [114]. For each newly sampled parameter $p_{\mathrm{new}}$ with episodic reward $r_{\mathrm{new}}$, they find the nearest previously sampled parameter $p_{\mathrm{old}}$, with reward $r_{\mathrm{old}}$, and define its absolute learning progress as

$$\mathrm{alp}_{\mathrm{new}} = |r_{\mathrm{new}} - r_{\mathrm{old}}|.$$

They fit a Gaussian mixture model, with between two and ten components chosen by the Akaike information criterion, to a window of the most recent 250 pairs of parameter and learning progress, treat each component as an arm whose utility is its mean learning progress, sample new parameters from the component chosen by a bandit, and sample uniformly at random 20 per cent of the time [114]. They studied the method across learners of different embodiment on parameterised variants of a bipedal walker, which makes it the nearest curriculum precedent to choosing robot designs for a shared learner [114].

Environment design methods turn the curriculum into a game. PAIRED trains an adversary network to generate the parameters of environments that maximise the regret of a protagonist agent, measured as the difference between the return of an allied antagonist agent and that of the protagonist, which yields environments that are hard but solvable and induces a natural curriculum [115]. It is a generator network over a parameter space trained against a learning agent, the structure of the proposal with the objective reversed. Active domain randomisation learns where within the randomisation ranges to sample, favouring environment instances on which the agent's rollouts differ most from those in a reference environment [116], and BayRn adapts the parameters of a domain randomisation distribution by Bayesian optimisation of the policy's return on the real system [117]. The lesson these carry is that the designs on which the shared policy trains form a curriculum, and that the curriculum objective, which rewards learnability, differs from the design objective, which rewards final merit.

### 10.4 Selection problems in robotics

Robotics uses bandits wherever a choice among many candidates must be made from noisy trials. Dex-Net 1.0 casts grasp selection as a multi-armed bandit with correlated rewards, sharing evidence between similar grasps so that a large set of candidates can be resolved with few evaluations [118]. Koval and colleagues select the most robust rearrangement trajectory by framing the selection as a multi-armed bandit [119].

Intelligent trial and error is the robotics precedent closest in spirit to using a learned critic as the prior of a design search [120]. Before deployment, MAP-Elites fills a map of behaviours, indexed by a behaviour descriptor such as the fraction of time each of a hexapod's six legs touches the ground, with the best controller found in simulation for each cell and its simulated performance $\mathcal{P}(x)$ [121, 120]. After damage, the robot runs Bayesian optimisation over the map with the simulated performance as the Gaussian process prior mean,

$$\mu_0(x) = \mathcal{P}(x), \qquad \mu_{t+1}(x) = \mathcal{P}(x) + k_t(x)^\top K_t^{-1}\big(P_{1:t+1} - \mathcal{P}(\chi_{1:t+1})\big),$$

where $\chi_{1:t+1}$ are the behaviours tested on the damaged robot, $P_{1:t+1}$ their measured performances, and $K_t$ the kernel matrix with noise on its diagonal, so the model learns only the discrepancy between simulation and reality [120]. It selects the next behaviour by $\arg\max_x\mu_t(x) + \kappa\sigma_t(x)$ and stops when a tested behaviour reaches 90 per cent of the best predicted performance, adapting a damaged legged robot in less than two minutes [120]. The analogue here would be a Gaussian process over designs whose prior mean is the critic's estimate $\bar V(x)$, learning only the critic's error.

### 10.5 Experimental design in the physical sciences

Chemical reaction optimisation faces a hybrid space of categorical choices, such as catalysts, ligands and solvents, beside continuous temperatures and concentrations. Shields and colleagues showed, through a benchmark game against expert chemists and engineers linked to real laboratory experiments, that Bayesian optimisation over such spaces outperforms human decision-making in both the number of experiments needed and the consistency of the outcome [122], which is direct evidence that hybrid bandit methods work on physical design problems of realistic size.

## 11. Modifications required for this use case

### 11.1 The problem stated formally

At generation $k$ the search proposes a batch $\{x_{k,1}, \dots, x_{k,B}\}$ with $B = 256$ and observes for each design the mean return of the episodes completed by its sixteen environments over 480 iterations, excluding episodes shorter than 25 control steps (`copt_on_policy_runner.py:365`, `train.py:271`). Writing $G_{k,i,j}$ for the return of the $j$-th of $N_{k,i}$ counted episodes of design $i$, the observation is

$$\hat F_{k,i} = \frac{1}{N_{k,i}}\sum_{j=1}^{N_{k,i}}G_{k,i,j},$$

an estimate of the merit $F_k(x_{k,i}) = \mathbb{E}[\text{return} \mid \pi_k, x_{k,i}]$ of section 8.5. Section 1.2 showed that $N_{k,i}$ is at most about 192 when episodes run to their full length and larger when they end early. If the returns of one design have standard deviation $\sigma_G$ and are roughly independent, the standard error of the estimate is $\sigma_G/\sqrt{N_{k,i}}$, which varies between designs because designs that fall often complete more but shorter episodes, so the noise is heteroscedastic and a bandit should weight each observation by its own precision rather than treat all designs alike. The aim is a design of high merit under the final policy, a simple regret objective, subject to the policy having trained enough across the neighbourhood of that design to evaluate it fairly, a coverage condition with a cumulative flavour. Every modification below follows from one of these features.

### 11.2 Batch selection

A sequential bandit chooses one arm per round, whereas this loop chooses 256 at once and hears nothing until all have run. Posterior sampling parallelises naturally, since 256 independent posterior draws give 256 sensible and diverse designs, and Kandasamy and colleagues proved that making $n$ evaluations distributed among parallel workers by Thompson sampling is essentially equivalent to making them in sequence [123]. The optimistic rule needs more care, because 256 maximisations of the same upper bound return the same design. GP-BUCB chooses the members of a batch in sequence by

$$x_t = \arg\max_x\;\mu_{\mathrm{fb}[t]}(x) + \beta_t^{1/2}\sigma_{t-1}(x),$$

where $\mathrm{fb}[t]$ is the last round whose feedback is available, so the mean is frozen at the start of the batch while the variance is updated as though each chosen point had been observed, which is possible because the variance does not depend on the outcomes (section 2.4) [92]. Freezing the mean makes the intervals overconfident, and Desautels, Krause and Burdick bounded the shrinkage, $\sigma_{\mathrm{fb}[t]}(x)/\sigma_{t-1}(x) \le \exp(I(f; y_{\mathrm{pending}} \mid y_{\mathrm{observed}}))$, the exponential of the conditional mutual information of the pending observations, so that inflating the confidence parameter to $\beta_t = \exp(2C)\alpha_{\mathrm{fb}[t]}$, where $C$ bounds that information within a batch and $\alpha$ is the sequential parameter, restores validity and gives regret of order $\sqrt{T\gamma_T\exp(2C)\alpha_T}$, larger than the sequential rule's by a constant factor that does not grow with the batch size [92]. Batch neural Thompson sampling provides the neural form [67], and the time-varying parallel EXP3 of PB2-Mix the categorical form [22]. Perchet and colleagues showed that a very small number of batches suffices for near minimax optimal regret [124], so the batch structure costs little provided the batch is diverse.

### 11.3 Drifting arm values

The merit of a design changes as the policy trains, and a useful model separates the change into three parts,

$$F_k(x) = a_k + h(x) + e_k(x),$$

where $a_k$ is a common improvement shared by all designs as the policy becomes generally competent, $h(x)$ is the design's intrinsic merit that the search seeks, and $e_k(x)$ is the endogenous part of section 8.5, which grows with the amount of training the policy has received on designs near $x$. Each part calls for its own remedy.

The common part $a_k$ is removed by any comparison made within a generation. Ranking the generation's fitnesses, as CMA-ES does [44, 43], or standardising them by the generation's mean and standard deviation, as section 9.7 prescribes, leaves $\hat F_{k,i} - \bar F_k$ unchanged by any additive shift common to the generation. Observing improvements rather than levels, as PB2 does [90], removes it as well but requires the same design to be evaluated in consecutive generations.

The design-specific part is learned across generations by a model that forgets old evidence at a controlled rate. In a Gaussian process this is the temporal kernel of section 8.6, $k(x, x')(1-\varepsilon)^{|k - k'|/2}$ between observations made at generations $k$ and $k'$ [91]. For finite arms the discounted upper confidence bound replaces the empirical mean by

$$\bar X_t(\gamma, i) = \frac{1}{N_t(\gamma, i)}\sum_{s=1}^t\gamma^{t-s}X_s\mathbb{1}[I_s = i], \qquad N_t(\gamma, i) = \sum_{s=1}^t\gamma^{t-s}\mathbb{1}[I_s = i],$$

with discount $\gamma \in (0,1)$, and pads it by $2B\sqrt{\xi\log n_t(\gamma)/N_t(\gamma, i)}$, where $B$ bounds the rewards, $\xi$ is a constant and $n_t(\gamma) = \sum_iN_t(\gamma, i)$, while the sliding window variant averages only the last $\tau$ rounds and pads by $B\sqrt{\xi\log\min(t, \tau)/N_t(\tau, i)}$ [125]. Both match a lower bound for abruptly changing rewards up to a logarithmic factor, and any policy with logarithmic regret in the stationary case suffers regret of order at least $T/\log T$ once breakpoints occur [125]. Besbes, Gur and Zeevi tied the best achievable regret to the total variation the means may undergo over the horizon [126]. A neural reward model achieves the same effect by weighting the loss of an observation from generation $k'$ by $(1-\varepsilon)^{k - k'}$ when training at generation $k$, or by taking the generation index as an input.

The endogenous part cannot be removed by forgetting, because it is caused by the search's own allocation. It is countered by keeping the allocation broad, which section 11.8 treats, and by evaluating designs under conditions that equalise their training, which section 11.4 treats.

### 11.4 The reward signal

Four signals are available, and the choice among them is the most consequential decision a bandit here entails. The mean episode return $\hat F_{k,i}$ is the present signal, and it carries the endogenous bias of section 8.5. The critic's estimate of each design's value,

$$\bar V(x) = \frac{1}{|\mathcal{S}_0|}\sum_{s\in\mathcal{S}_0}V_\psi(s, x),$$

averaged over a bank $\mathcal{S}_0$ of initial states with the critic $V_\psi$ observing the design, is smooth, has low variance and exists for designs never built, but it is biased wherever the critic has seen little data, as Bohlinger and Peters found at the boundary of their design space [85, 86]. The improvement over an interval, or the learning progress of the curriculum literature, rewards learnability rather than merit [90, 21, 114]. And an evaluation after a short fine-tuning of the policy on the candidate alone is a multi-fidelity observation in the sense of section 10.1 [111].

Whatever is chosen, any term of the reward that depends on the design directly rather than through behaviour will be learned by the bandit as faithfully as any genuine merit. The audit (section 4.2) shows that a base height penalty with a fixed target makes about five per cent of the quadruped's fitness a deterministic function of leg length, symmetric about the donor, which a search maximises by returning to the design it started from, so the design-invariant reward of the audit's section 4.8 is a precondition for any bandit and not merely for the incumbent. A neural reward model would learn that artefact faster than a search distribution does, because it generalises.

### 11.5 The hybrid acquisition step

A reward model is only half of a bandit, the other half being the maximisation of its acquisition over the design space, which here is a mixed problem in its own right. With the continuous block of two or three dimensions and the catalogue of section 1.2, block coordinate ascent is practical. Starting from a candidate, it enumerates the $M = 15$ actuators of one group with the others held and keeps the best, does the same for each of the $G = 4$ groups in turn, then improves the continuous block by projected gradient ascent from several starting points, and cycles until nothing changes, at a cost of $GM = 60$ model evaluations plus the gradient steps per cycle. Under posterior sampling each of the 256 candidates is obtained by sampling the model once and running this ascent, so the batch is diverse by construction. The alternation is the scheme MiVaBO uses with an exact solver for its discrete block [50]. Where the actuator groups are expected to act nearly independently, a factorised categorical bandit per group is the combinatorial semi-bandit of Chen, Wang and Yuan, in which a super arm composed of several base arms is played and each base arm's outcome is observed [127], and the additive part of the mixed kernel of section 5.3 is its Gaussian process counterpart.

### 11.6 Resolution of the length lattice

The centimetre rounding makes the length block finite, so the finite-set guarantee of section 4.6 holds over $|\mathcal{X}| = 323$ or $144$ length designs, with $\beta_t = 2\log(|\mathcal{X}|t^2\pi^2/(6\delta))$ [34]. Two consequences follow for a bandit. Its kernel length scale must exceed the one centimetre spacing, or the model will mistake the plateaus of the rounded objective for structure [46]. And the quadruped's 144 points lie below the population of 256, so duplicates are certain (audit, section D8). A bandit should treat duplication as a resource. Evaluating a design $r$ times reduces the standard error of its estimated merit by the factor $\sqrt r$, so the choice of how many replicates each promising design receives is a best-arm identification problem of section 3.9, which sequential halving solves within a fixed budget [24], rather than an accident of sampling.

### 11.7 The nominal pose as an arm

The nominal pose differs in kind from the lengths, since it changes the meaning of the policy's output as well as the robot. It enters the action as the offset added to every scaled action, copied from the articulation's default joint positions when the action term is built (`joint_actions.py:193`), it enters the policy's observation through the joint positions measured relative to it (`observations.py:219`), and it enters the joint deviation penalty as the zero of that penalty (`rewards.py:185`). The default joint position is held per environment, so a per-design pose is representable, but the action offset is refreshed only by the respawn pathway (`respawn.py:181`), so any in-place pathway must refresh it as well. Three requirements follow. The policy must observe the pose, since the same action means a different joint target under a different pose and an unobserved design parameter makes the problem partially observable [128]. Every reward term that takes a target from the pose must take it from the design, which the audit (sections 4.7 and 4.8) enumerates. And the pose should be parameterised compactly, the audit (section 4.3) proposing a single knee fold per family from which the other joints follow by two constraints, so that the continuous block grows by one dimension rather than by the number of joints.

### 11.8 An exploration floor

The endogenous drift of section 8.5 rewards whatever is already sampled, so the search must be prevented from abandoning a region before the policy has learned to use it. EXP3's uniform term $\gamma/K$ [20], CatCMA's margin [45, 47] and an entropy bonus on a generator network are three forms of the same floor, and Mei and colleagues' result that the stochastic gradient bandit keeps every probability above order $1/t$ shows that even an unfloored softmax explores eventually, though possibly too slowly for a run of 68 generations [19]. The floor can be sized from the population. With $B$ designs per generation and a floor $q_{\min}$ on each category's probability, each catalogue entry of each group appears $Bq_{\min}$ times per generation in expectation. The margin that the CatCMA paper derives, $q_{\min} = (1 - 0.73^{1/N_{\mathrm{ca}}})/(K - 1)$ with $N_{\mathrm{ca}}$ categorical variables of $K$ categories [45], evaluates to about 0.0054 for the biped's four groups of fifteen and 0.0077 for the quadruped's three groups of fourteen, so with 256 designs each actuator appears about 1.4 and 2.0 times per group per generation in expectation. That is enough to keep a category alive but too few for the controller to learn to use it, so a floor stated in terms of training coverage, guaranteeing a minimum number of environments per catalogue entry per generation, protects the controller's coverage of the actuator space, without which an actuator abandoned early can never be fairly re-evaluated.

### 11.9 Infeasible designs

When the design builder rejects a specification, the generator substitutes the last feasible one and continues (`usd_generator.py:451`), so the requested genotype is credited with the fitness of a different design. For a search that keeps only distribution parameters this is a small distortion, but for a bandit with a reward model it is mislabelled training data, assigned to a region the model will then believe it understands. A bandit must be told of infeasibility, either as a known poor value or through a separate feasibility model, and must be given the design actually built rather than the one requested.

### 11.10 A context for the network

Section 6.8 showed that a generator network is justified chiefly by a context. Candidate contexts here are the terrain difficulty, the commanded velocity band, a payload, or a weighting between tracking and energy, and with any of them the problem becomes a contextual bandit whose kernel or network takes the context as input [40], and whose generator $\pi_\phi(x \mid c)$ amortises the design search across tasks [56]. Without a context the proposal reduces to a reward-model bandit, which is the stronger of the two non-contextual readings.

## 12. Candidate contributions

Each candidate below is specified concretely enough to be implemented, states the published work it builds on, and states what it adds. The novelty claims are bounded by section 13.

### 12.1 Critic-seeded neural Thompson sampling over the hybrid space

The design-conditioned critic is a reward model the pipeline already trains (section 8.4). The proposal is a small ensemble of heads over the critic's design-dependent features, trained on the observed fitness of every generation with the recency weighting of section 11.3 and the precision weighting of section 11.1, used by posterior sampling with the hybrid acquisition of section 11.5 to choose each generation's 256 designs. In outline, at each generation the method computes the critic's feature vector for candidate designs, fits each head on all past fitness observations with a different bootstrap resample, draws one head for each of the 256 slots, maximises that head over the design space by block coordinate ascent, and submits the resulting batch. It supplies the uncertainty-directed exploration that the published critic-as-surrogate methods lack, since Luck and colleagues explore at random and Bohlinger and Peters average their critic heads and guard against extrapolation with a distance penalty rather than exploiting the heads' disagreement [85, 86]. It retains evidence that the incumbent discards (section 4.9), and it handles the actuator catalogue without relaxation. Its risk is the critic's bias in regions it has not visited, which the floor of section 11.8 and the critic-as-prior-mean construction of section 10.4 contain [120].

### 12.2 A time-varying hierarchical bandit for morphology

The second proposal transplants PB2-Mix to morphology, choosing actuators per group by a time-varying parallel EXP3 and the lengths and pose by a time-varying GP-UCB over the mixed kernel of section 5.3, with the generation as the time axis [22, 91]. A variant replaces the hierarchy by Bandit-BO's one Gaussian process per actuator configuration with a temporal kernel added, which suits the case in which the best lengths depend strongly on the actuator [51]. Either inherits a regret analysis for drifting objectives, is cheap at a continuous dimension of three, and replaces CatCMAwM without changing the runner's interface. Its novelty lies in the setting rather than the algorithm, since the analysis of PB2 assumes members with private weights and the shared policy here violates that assumption (section 8.6), so whether the guarantees survive the coupling is an open question worth answering.

### 12.3 Separating the training distribution from the search distribution

The third proposal follows from sections 8.5 and 10.3. The designs the policy trains on form a curriculum, whose right objective is learnability, while the designs the search recommends answer a best-arm question, whose right objective is final merit. Running two bandits, one allocating training environments by absolute learning progress in the manner of ALP-GMM [114] and one identifying the best design from the critic and from replicated evaluations [25, 24], attacks the premature convergence of co-design [87, 88] at its cause, the first-mover advantage, rather than by enlarging the exploration noise. No retrieved work separates the two distributions in morphology and control co-optimisation.

### 12.4 Successive halving within a generation

The fourth proposal moves environments, rather than designs, within a single 480 iteration interval. Each generation starts with many designs on few environments each and, at fixed fractions of the interval, reassigns the environments of the worst designs to the best, which is Hyperband's bracket structure inside a shared policy [109], after ECoDe's use of it with a universal policy network [81]. ECoDe's observation that a shared policy favours designs that received more training (section 7.3) applies with full force, so the survivors' advantage in training must be corrected for, either by running the brackets in reverse order as ECoDe does or by comparing designs through the critic rather than through raw returns [81]. Unequal environment counts also give the survivors more evidence than the eliminated, which the audit (section B3) records as a defect for a ranking search, so the bandit must use precision-weighted estimates rather than raw means (section 11.1).

### 12.5 A generation-level proximal policy optimisation design head

The fifth proposal is the construction of section 9.7, a design head trained by a clipped surrogate at the time scale of the generation, with a standardised advantage, a critic baseline, an entropy bonus, a relative entropy anchor to the donor-centred distribution, a floored categorical head to avoid the lock-in proven for proximal policy optimisation on bandits, and optionally the Stackelberg correction for the controller's adaptation [95, 96, 97, 103]. It answers the follow-up question directly, and it is the most natural route for the proposal's neural network, since a generator network trained this way is the policy-network role of section 6.1 with proximal policy optimisation's stabilising machinery. It is cheap to implement because the clipped surrogate, the entropy bonus and the relative entropy penalty all exist in the vendored algorithm already. Its weakness is the one section 6.8 identified, that without a context the head is a parameterisation of the same family CatCMA maintains, so it should be paired with a context or with the reward model of section 12.1 used as its baseline.

### 12.6 A contextual design generator

The sixth proposal is the proposal of section 1.3 in its most defensible form. A generator $\pi_\phi(x \mid c)$ with a softmax head per actuator group and a Gaussian head over lengths and pose conditioned on the chosen actuators (section 5.6), given a task context $c$ and trained by the design head of section 12.5 with the critic as its baseline, would map a task specification to a design distribution, in the manner of the architecture controllers of section 10.2 [70, 71] and the environment generator of PAIRED [115], but over a robot's body. It is justified only if a context exists (section 6.8), it is the most novel of the six, and it is also the most expensive, since a context multiplies the evaluations required.

### 12.7 Recommendation

The evidence favours beginning with 12.1, whose reward model already exists and whose uncertainty is the element every published critic surrogate lacks, benchmarked against the incumbent CatCMAwM and against random search over the same lattice, which section 10.2 shows to be indispensable [112]. Proposal 12.5 is the cheapest principled route to a trained design network, answers the follow-up question, and is the natural ablation of 12.1 when given the same critic baseline. Proposal 12.2 is the cheapest principled alternative with an existing regret analysis. Proposal 12.3 carries the strongest claim to novelty, and 12.6 should wait for a context to justify it. All six presuppose the design-invariant reward of section 11.4 and the feasibility reporting of section 11.9, and the nominal pose should enter only after the three requirements of section 11.7 are met.

## 13. What the retrieved literature does not contain

Seven absences bound the novelty claims above, each stated against the sources retrieved for this survey rather than against the literature as a whole. No retrieved work in robot co-design treats an actuator catalogue as the arms of a bandit. No retrieved work applies a population-based or time-varying bandit to the morphology of a robot under one shared design-conditioned policy. No retrieved work uses the disagreement of a design-conditioned critic's heads, or any calibrated uncertainty of such a critic, to direct exploration over designs, the one work whose critic has several heads using only their mean. No retrieved work separates the training distribution from the search distribution in concurrent co-design. No retrieved work searches a nominal pose as a design variable jointly with discrete actuators. No retrieved work that places the design inside a proximal policy optimisation objective does so for a legged robot with identified actuators or a component catalogue. And no retrieved work applies a clipped surrogate to a design distribution at the time scale of a generation.

The survey has two limits a reader should weigh. Its sources are those retrieved in one session, so a work published under unexpected terms may have been missed, and several of the most recent co-design works are preprints whose claims have not been peer reviewed, which the bibliography marks wherever the retrieved record stated no venue.

## 14. Bibliography

1. Wang, Y., Chen, Z., Zhang, T., Yin, Q., Chang, Y., Li, Z., Wang, L., Wang, X. Embodied Co-Design for Rapidly Evolving Agents, Taxonomy, Frontiers, and Challenges. arXiv:2512.04770, 2025.

2. Schaff, C., Yunis, D., Chakrabarti, A., Walter, M. R. Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning. IEEE International Conference on Robotics and Automation (ICRA) 2019. arXiv:1801.01432.

3. Hamano, R., Nomura, M., Saito, S., Uchida, K., Shirakawa, S. CatCMA with Margin for Single- and Multi-Objective Mixed-Variable Black-Box Optimization. arXiv:2504.07884, 2025, whose fifth version the record states corresponds to the GECCO 2025 conference paper.

4. Lattimore, T., Szepesvári, C. Bandit Algorithms. Cambridge University Press, 2020. DOI 10.1017/9781108571401.

5. Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Schmidhuber, J. Natural Evolution Strategies. arXiv:1106.4487, 2011, the venue of the published version not established by the retrieved record.

6. Rasmussen, C. E., Williams, C. K. I. Gaussian Processes for Machine Learning. MIT Press, dated 2005 by its Crossref record. DOI 10.7551/mitpress/3206.001.0001.

7. Williams, R. J. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning 8(3-4), pages 229 to 256, 1992. DOI 10.1023/A:1022672621406.

8. Sutton, R. S., Barto, A. G. Reinforcement Learning, An Introduction, second edition. MIT Press, 2018. ISBN 9780262039246. Section 2.8, gradient bandit algorithms.

9. Ha, D. Reinforcement Learning for Improving Agent Design. Artificial Life 25(4), pages 352 to 365, 2019. arXiv:1810.03779.

10. Robbins, H. Some Aspects of the Sequential Design of Experiments. Bulletin of the American Mathematical Society 58(5), pages 527 to 535, 1952. DOI 10.1090/S0002-9904-1952-09620-8.

11. Thompson, W. R. On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika 25(3/4), pages 285 to 294, 1933. DOI 10.2307/2332286.

12. Slivkins, A. Introduction to Multi-Armed Bandits. Foundations and Trends in Machine Learning, 2019, as the arXiv record states. arXiv:1904.07272.

13. Bubeck, S., Cesa-Bianchi, N. Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems. Foundations and Trends in Machine Learning, 2012, as the arXiv record states, the volume not established by the retrieved record. arXiv:1204.5721.

14. Lai, T. L., Robbins, H. Asymptotically Efficient Adaptive Allocation Rules. Advances in Applied Mathematics 6(1), pages 4 to 22, 1985. DOI 10.1016/0196-8858(85)90002-8.

15. Auer, P., Cesa-Bianchi, N., Fischer, P. Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning 47(2-3), pages 235 to 256, 2002. DOI 10.1023/A:1013689704352.

16. Russo, D., Van Roy, B. Learning to Optimize via Posterior Sampling. Mathematics of Operations Research 39(4), pages 1221 to 1243, 2014. DOI 10.1287/moor.2014.0650.

17. Agrawal, S., Goyal, N. Analysis of Thompson Sampling for the Multi-armed Bandit Problem. Conference on Learning Theory (COLT) 2012, PMLR 23. arXiv:1111.1797.

18. Mei, J., Xiao, C., Szepesvári, C., Schuurmans, D. On the Global Convergence Rates of Softmax Policy Gradient Methods. International Conference on Machine Learning (ICML) 2020, as the arXiv record states. arXiv:2005.06392.

19. Mei, J., Zhong, Z., Dai, B., Agarwal, A., Szepesvári, C., Schuurmans, D. Stochastic Gradient Succeeds for Bandits. arXiv:2402.17235, 2024, whose record states that it corrects a version published at ICML 2023.

20. Auer, P., Cesa-Bianchi, N., Freund, Y., Schapire, R. E. The Nonstochastic Multiarmed Bandit Problem. SIAM Journal on Computing 32(1), pages 48 to 77, 2002. DOI 10.1137/S0097539701398375.

21. Graves, A., Bellemare, M. G., Menick, J., Munos, R., Kavukcuoglu, K. Automated Curriculum Learning for Neural Networks. International Conference on Machine Learning (ICML) 2017, PMLR 70. arXiv:1704.03003.

22. Parker-Holder, J., Nguyen, V., Desai, S., Roberts, S. J. Tuning Mixed Input Hyperparameters on the Fly for Efficient Population Based AutoRL. Advances in Neural Information Processing Systems (NeurIPS) 34, 2021. arXiv:2106.15883.

23. Bubeck, S., Munos, R., Stoltz, G. Pure Exploration for Multi-Armed Bandit Problems. arXiv:0802.2655, 2008, the venue of the published version not established by the retrieved record.

24. Karnin, Z., Koren, T., Somekh, O. Almost Optimal Exploration in Multi-Armed Bandits. International Conference on Machine Learning (ICML) 2013, PMLR 28(3), pages 1238 to 1246.

25. Garivier, A., Kaufmann, E. Optimal Best Arm Identification with Fixed Confidence. Conference on Learning Theory (COLT) 2016, as the arXiv record states. arXiv:1602.04589.

26. Jamieson, K., Talwalkar, A. Non-stochastic Best Arm Identification and Hyperparameter Optimization. International Conference on Artificial Intelligence and Statistics (AISTATS) 2016, PMLR 51, pages 240 to 248. arXiv:1502.07943.

27. Abbasi-Yadkori, Y., Pál, D., Szepesvári, C. Improved Algorithms for Linear Stochastic Bandits. Advances in Neural Information Processing Systems (NeurIPS) 24, 2011.

28. Li, L., Chu, W., Langford, J., Schapire, R. E. A Contextual-Bandit Approach to Personalized News Article Recommendation. International World Wide Web Conference (WWW) 2010. DOI 10.1145/1772690.1772758. arXiv:1003.0146.

29. Kleinberg, R., Slivkins, A., Upfal, E. Multi-Armed Bandits in Metric Spaces. ACM Symposium on Theory of Computing (STOC) 2008, pages 681 to 690. DOI 10.1145/1374376.1374475. arXiv:0809.4882.

30. Kleinberg, R. D. Nearly Tight Bounds for the Continuum-Armed Bandit Problem. Advances in Neural Information Processing Systems (NeurIPS) 17, 2004.

31. Agrawal, R. The Continuum-Armed Bandit Problem. SIAM Journal on Control and Optimization 33(6), pages 1926 to 1951, 1995. DOI 10.1137/S0363012992237273.

32. Bubeck, S., Munos, R., Stoltz, G., Szepesvári, C. X-Armed Bandits. Journal of Machine Learning Research 12, pages 1655 to 1695, 2011. arXiv:1001.4475.

33. Kocsis, L., Szepesvári, C. Bandit Based Monte-Carlo Planning. Lecture Notes in Computer Science, pages 282 to 293, 2006. DOI 10.1007/11871842_29.

34. Srinivas, N., Krause, A., Kakade, S. M., Seeger, M. Gaussian Process Optimization in the Bandit Setting, No Regret and Experimental Design. arXiv:0912.3995, 2009, whose record links the journal version, Information-Theoretic Regret Bounds for Gaussian Process Optimization in the Bandit Setting, IEEE Transactions on Information Theory 58(5), pages 3250 to 3265, 2012, DOI 10.1109/TIT.2011.2182033.

35. Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C., Osborne, M. A. Think Global and Act Local, Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. International Conference on Machine Learning (ICML) 2021, PMLR 139. arXiv:2102.07188.

36. Chowdhury, S. R., Gopalan, A. On Kernelized Multi-armed Bandits. International Conference on Machine Learning (ICML) 2017, PMLR 70. arXiv:1704.00445.

37. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., de Freitas, N. Taking the Human Out of the Loop, A Review of Bayesian Optimization. Proceedings of the IEEE 104(1), pages 148 to 175, 2016. DOI 10.1109/JPROC.2015.2494218.

38. Frazier, P. I. A Tutorial on Bayesian Optimization. arXiv:1807.02811, 2018.

39. Snoek, J., Larochelle, H., Adams, R. P. Practical Bayesian Optimization of Machine Learning Algorithms. Advances in Neural Information Processing Systems (NeurIPS) 25, 2012. arXiv:1206.2944.

40. Krause, A., Ong, C. S. Contextual Gaussian Process Bandit Optimization. Advances in Neural Information Processing Systems (NeurIPS) 24, 2011.

41. Flaxman, A. D., Kalai, A. T., McMahan, H. B. Online Convex Optimization in the Bandit Setting, Gradient Descent without a Gradient. ACM-SIAM Symposium on Discrete Algorithms (SODA) 2005. arXiv:cs/0408007.

42. Salimans, T., Ho, J., Chen, X., Sidor, S., Sutskever, I. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864, 2017.

43. Ollivier, Y., Arnold, L., Auger, A., Hansen, N. Information-Geometric Optimization Algorithms, A Unifying Picture via Invariance Principles. Journal of Machine Learning Research 18, pages 1 to 65, 2017. arXiv:1106.3708.

44. Hansen, N. The CMA Evolution Strategy, A Tutorial. arXiv:1604.00772, 2016.

45. Hamano, R., Saito, S., Nomura, M., Uchida, K., Shirakawa, S. CatCMA, Stochastic Optimization for Mixed-Category Problems. Genetic and Evolutionary Computation Conference (GECCO) 2024. DOI 10.1145/3638529.3654198. arXiv:2405.09962.

46. Garrido-Merchán, E. C., Hernández-Lobato, D. Dealing with Categorical and Integer-valued Variables in Bayesian Optimization with Gaussian Processes. Neurocomputing 380, pages 20 to 35, 2020. DOI 10.1016/j.neucom.2019.11.004. arXiv:1805.03463.

47. Hamano, R., Saito, S., Nomura, M., Shirakawa, S. CMA-ES with Margin, Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization. Genetic and Evolutionary Computation Conference (GECCO) 2022. arXiv:2205.13482.

48. Ru, B., Alvi, A. S., Nguyen, V., Osborne, M. A., Roberts, S. J. Bayesian Optimisation over Multiple Continuous and Categorical Inputs. International Conference on Machine Learning (ICML) 2020, PMLR 119. arXiv:1906.08878.

49. Deshwal, A., Belakaria, S., Doppa, J. R. Bayesian Optimization over Hybrid Spaces. International Conference on Machine Learning (ICML) 2021, PMLR 139. arXiv:2106.04682.

50. Daxberger, E., Makarova, A., Turchetta, M., Krause, A. Mixed-Variable Bayesian Optimization. International Joint Conference on Artificial Intelligence (IJCAI) 2020, pages 2633 to 2639. DOI 10.24963/ijcai.2020/365. arXiv:1907.01329.

51. Nguyen, D., Gupta, S., Rana, S., Shilton, A., Venkatesh, S. Bayesian Optimization for Categorical and Category-Specific Continuous Inputs. AAAI Conference on Artificial Intelligence 2020, as the arXiv record states. arXiv:1911.12473.

52. Hausknecht, M., Stone, P. Deep Reinforcement Learning in Parameterized Action Space. arXiv:1511.04143, 2015, the venue of the published version not established by the retrieved record.

53. Xiong, J., Wang, Q., Yang, Z., Sun, P., Han, L., Zheng, Y., Fu, H., Zhang, T., Liu, J., Liu, H. Parametrized Deep Q-Networks Learning, Reinforcement Learning with Discrete-Continuous Hybrid Action Space. arXiv:1810.06394, 2018.

54. Fan, Z., Su, R., Zhang, W., Yu, Y. Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space. arXiv:1903.01344, 2019, the venue of the published version not established by the retrieved record.

55. Li, B., Tang, H., Zheng, Y., Hao, J., Li, P., Wang, Z., Meng, Z., Wang, L. HyAR, Addressing Discrete-Continuous Action Reinforcement Learning via Hybrid Action Representation. International Conference on Learning Representations (ICLR) 2022, as the arXiv record states. arXiv:2109.05490.

56. Amos, B. Tutorial on Amortized Optimization. Foundations and Trends in Machine Learning, as the arXiv record states. arXiv:2202.00665, 2022.

57. Riquelme, C., Tucker, G., Snoek, J. Deep Bayesian Bandits Showdown, An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling. International Conference on Learning Representations (ICLR) 2018, as the arXiv record states. arXiv:1802.09127.

58. Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish, N., Sundaram, N., Patwary, M. M. A., Prabhat, Adams, R. P. Scalable Bayesian Optimization Using Deep Neural Networks. International Conference on Machine Learning (ICML) 2015, PMLR 37. arXiv:1502.05700.

59. Jacot, A., Gabriel, F., Hongler, C. Neural Tangent Kernel, Convergence and Generalization in Neural Networks. Advances in Neural Information Processing Systems (NeurIPS) 31, 2018. arXiv:1806.07572.

60. Zhou, D., Li, L., Gu, Q. Neural Contextual Bandits with UCB-based Exploration. International Conference on Machine Learning (ICML) 2020, PMLR 119. arXiv:1911.04462.

61. Zhang, W., Zhou, D., Li, L., Gu, Q. Neural Thompson Sampling. International Conference on Learning Representations (ICLR) 2021, as the arXiv record states. arXiv:2010.00827.

62. Xu, P., Wen, Z., Zhao, H., Gu, Q. Neural Contextual Bandits with Deep Representation and Shallow Exploration. International Conference on Learning Representations (ICLR) 2022, as the OpenReview record states. arXiv:2012.01780.

63. Ban, Y., Yan, Y., Banerjee, A., He, J. EE-Net, Exploitation-Exploration Neural Networks in Contextual Bandits. International Conference on Learning Representations (ICLR) 2022, as the arXiv record states. arXiv:2110.03177.

64. Lakshminarayanan, B., Pritzel, A., Blundell, C. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. Advances in Neural Information Processing Systems (NeurIPS) 30, 2017. arXiv:1612.01474.

65. Osband, I., Blundell, C., Pritzel, A., Van Roy, B. Deep Exploration via Bootstrapped DQN. Advances in Neural Information Processing Systems (NeurIPS) 29, 2016. arXiv:1602.04621.

66. Osband, I., Wen, Z., Asghari, S. M., Dwaracherla, V., Ibrahimi, M., Lu, X., Van Roy, B. Epistemic Neural Networks. Advances in Neural Information Processing Systems (NeurIPS) 36, 2023. arXiv:2107.08924.

67. Dai, Z., Shu, Y., Low, B. K. H., Jaillet, P. Sample-Then-Optimize Batch Neural Thompson Sampling. Advances in Neural Information Processing Systems (NeurIPS) 35, 2022. arXiv:2210.06850.

68. Foster, D. J., Rakhlin, A. Beyond UCB, Optimal and Efficient Contextual Bandits with Regression Oracles. International Conference on Machine Learning (ICML) 2020, PMLR 119. arXiv:2002.04926.

69. Majzoubi, M., Zhang, C., Chari, R., Krishnamurthy, A., Langford, J., Slivkins, A. Efficient Contextual Bandits with Continuous Actions. Advances in Neural Information Processing Systems (NeurIPS) 33, 2020. arXiv:2006.06040.

70. Zoph, B., Le, Q. V. Neural Architecture Search with Reinforcement Learning. International Conference on Learning Representations (ICLR) 2017. arXiv:1611.01578.

71. Pham, H., Guan, M., Zoph, B., Le, Q., Dean, J. Efficient Neural Architecture Search via Parameters Sharing. International Conference on Machine Learning (ICML) 2018, PMLR 80, pages 4095 to 4104. arXiv:1802.03268.

72. Liao, T., Wang, G., Yang, B., Lee, R., Pister, K., Levine, S., Calandra, R. Data-efficient Learning of Morphology and Controller for a Microrobot. IEEE International Conference on Robotics and Automation (ICRA) 2019, as the arXiv record states. arXiv:1905.01334.

73. Bjelonic, F., Lee, J., Arm, P., Sako, D., Tateo, D., Peters, J., Hutter, M. Learning-based Design and Control for Quadrupedal Robots with Parallel-Elastic Actuators. IEEE Robotics and Automation Letters, 2023. DOI 10.1109/LRA.2023.3234809. arXiv:2301.03509.

74. Wang, H., Fang, Z., Hanna, J., Xiong, X. Efficient and Versatile Quadrupedal Skating, Optimal Co-design via Reinforcement Learning and Bayesian Optimization. arXiv:2603.18408, 2026, the venue not stated by the retrieved record.

75. Chen, C., Xiang, P., Zhang, J., Xiong, R., Wang, Y., Lu, H. Deep Reinforcement Learning Based Co-Optimization of Morphology and Gait for Small-Scale Legged Robot. IEEE/ASME Transactions on Mechatronics 29(4), pages 2697 to 2708, 2024. DOI 10.1109/TMECH.2023.3330427.

76. Hu, J., Whitman, J., Choset, H. GLSO, Grammar-guided Latent Space Optimization for Sample-efficient Robot Design Automation. Conference on Robot Learning (CoRL) 2022, PMLR 205, pages 1321 to 1331. arXiv:2209.11748.

77. Bhatia, J. S., Jackson, H., Tian, Y., Xu, J., Matusik, W. Evolution Gym, A Large-Scale Benchmark for Evolving Soft Robots. Advances in Neural Information Processing Systems (NeurIPS) 34, 2021. arXiv:2201.09863.

78. Belmonte-Baeza, Á., Lee, J., Valsecchi, G., Hutter, M. Meta Reinforcement Learning for Optimal Design of Legged Robots. IEEE Robotics and Automation Letters 7(4), pages 12134 to 12141, 2022. DOI 10.1109/LRA.2022.3211785. arXiv:2210.02750.

79. Zhao, A., Xu, J., Konaković-Luković, M., Hughes, J., Spielberg, A., Rus, D., Matusik, W. RoboGrammar, Graph Grammar for Terrain-Optimized Robot Design. ACM Transactions on Graphics 39(6), 2020. DOI 10.1145/3414685.3417831.

80. Wang, L., Fonseca, R., Tian, Y. Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search. Advances in Neural Information Processing Systems (NeurIPS) 33, 2020. arXiv:2007.00708.

81. Nagiredla, K. R., Semage, B. L., Arun Kumar A., Karimpanal, T. G., Rana, S. ECoDe, A Sample-Efficient Method for Co-Design of Robotic Agents. arXiv:2309.04085, 2023, the venue not stated by the retrieved record.

82. Schneider, R., Honerkamp, D., Welschehold, T., Valada, A. Task-Driven Co-Design of Mobile Manipulators. IEEE Robotics and Automation Letters 10(7), pages 7158 to 7165, 2025. DOI 10.1109/LRA.2025.3573622. arXiv:2412.16635.

83. Radulov, N., Yang, X., Luck, K. S., Pizzuto, G. Tool-Policy Co-Design for Powder Weighing in Laboratory Automation. arXiv:2609.39797, 2026.

84. Huang, X., Dong, J., Zhao, H. Task-Oriented Co-Design and Optimization of Geared Actuators for Robotic Applications. arXiv:2609.22795, 2026.

85. Luck, K. S., Ben Amor, H., Calandra, R. Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning. Conference on Robot Learning (CoRL) 2019, PMLR 100, pages 854 to 869. arXiv:1911.06832.

86. Bohlinger, N., Peters, J. Shape Your Body, Value Gradients for Multi-Embodiment Robot Design. arXiv:2606.00702, 2026.

87. Cheney, N., Bongard, J., SunSpiral, V., Lipson, H. Scalable Co-optimization of Morphology and Control in Embodied Machines. Journal of the Royal Society Interface 15(143), 20170937, 2018. DOI 10.1098/rsif.2017.0937.

88. Mertan, A., Cheney, N. Investigating Premature Convergence in Co-optimization of Morphology and Control in Evolved Virtual Soft Robots. Genetic Programming, 27th European Conference, EuroGP 2024, as the arXiv record states. arXiv:2402.09231.

89. Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., Vinyals, O., Green, T., Dunning, I., Simonyan, K., Fernando, C., Kavukcuoglu, K. Population Based Training of Neural Networks. arXiv:1711.09846, 2017.

90. Parker-Holder, J., Nguyen, V., Roberts, S. J. Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits. Advances in Neural Information Processing Systems (NeurIPS) 33, 2020. arXiv:2002.02518.

91. Bogunovic, I., Scarlett, J., Cevher, V. Time-Varying Gaussian Process Bandit Optimization. International Conference on Artificial Intelligence and Statistics (AISTATS) 2016, PMLR 51. arXiv:1601.06650.

92. Desautels, T., Krause, A., Burdick, J. Parallelizing Exploration-Exploitation Tradeoffs with Gaussian Process Bandit Optimization. International Conference on Machine Learning (ICML) 2012, as the arXiv record states. arXiv:1206.6402.

93. Schulman, J., Levine, S., Moritz, P., Jordan, M. I., Abbeel, P. Trust Region Policy Optimization. International Conference on Machine Learning (ICML) 2015, PMLR 37, pages 1889 to 1897. arXiv:1502.05477.

94. Schulman, J., Moritz, P., Levine, S., Jordan, M. I., Abbeel, P. High-Dimensional Continuous Control Using Generalized Advantage Estimation. International Conference on Learning Representations (ICLR) 2016. arXiv:1506.02438.

95. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.

96. Wang, Y., He, H., Tan, X., Gan, Y. Trust Region-Guided Proximal Policy Optimization. Advances in Neural Information Processing Systems (NeurIPS) 32, 2019. arXiv:1901.10314.

97. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., Lowe, R. Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems (NeurIPS) 35, 2022. arXiv:2203.02155.

98. Yuan, Y., Song, Y., Luo, Z., Sun, W., Kitani, K. Transform2Act, Learning a Transform-and-Control Policy for Efficient Agent Design. International Conference on Learning Representations (ICLR) 2022. arXiv:2110.03659.

99. Dong, H., Zhang, J., Wang, T., Zhang, C. Symmetry-Aware Robot Design with Structured Subgroups. International Conference on Machine Learning (ICML) 2023, PMLR 202. arXiv:2306.00036.

100. Pathak, D., Lu, C., Darrell, T., Isola, P., Efros, A. A. Learning to Control Self-Assembling Morphologies, A Study of Generalization via Modularity. Advances in Neural Information Processing Systems (NeurIPS) 32, 2019. arXiv:1902.05546.

101. Chen, T., He, Z., Ciocarlie, M. Hardware as Policy, Mechanical and Computational Co-Optimization using Deep Reinforcement Learning. Conference on Robot Learning (CoRL) 2020, PMLR 155. arXiv:2008.04460.

102. Lu, H., Wu, Z., Xing, J., Li, J., Li, R., Li, Z., Shi, Y. BodyGen, Advancing Towards Efficient Embodiment Co-Design. International Conference on Learning Representations (ICLR) 2025, as the arXiv record states. arXiv:2503.00533.

103. Dai, Y., Wang, Y., Ashley, D. R., Schmidhuber, J. Efficient Morphology-Control Co-Design via Stackelberg Proximal Policy Optimization. International Conference on Learning Representations (ICLR) 2026, as the arXiv record states. arXiv:2603.15388.

104. Jiang, M., Grefenstette, E., Rocktäschel, T. Prioritized Level Replay. International Conference on Machine Learning (ICML) 2021, as the mlanthology record states. arXiv:2010.03934.

105. Badia, A. P., Piot, B., Kapturowski, S., Sprechmann, P., Vitvitskyi, A., Guo, Z. D., Blundell, C. Agent57, Outperforming the Atari Human Benchmark. International Conference on Machine Learning (ICML) 2020, PMLR 119. arXiv:2003.13350.

106. Petersen, B. K., Landajuela, M., Mundhenk, T. N., Santiago, C. P., Kim, S. K., Kim, J. T. Deep Symbolic Regression, Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients. International Conference on Learning Representations (ICLR) 2021. arXiv:1912.04871.

107. Bengio, E., Jain, M., Korablyov, M., Precup, D., Bengio, Y. Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. Advances in Neural Information Processing Systems (NeurIPS) 34, 2021, as the arXiv record states. arXiv:2106.04399.

108. Kool, W., van Hoof, H., Welling, M. Attention, Learn to Solve Routing Problems. International Conference on Learning Representations (ICLR) 2019, as the arXiv record states. arXiv:1803.08475.

109. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. Hyperband, A Novel Bandit-Based Approach to Hyperparameter Optimization. Journal of Machine Learning Research 18, pages 1 to 52, 2018. arXiv:1603.06560.

110. Falkner, S., Klein, A., Hutter, F. BOHB, Robust and Efficient Hyperparameter Optimization at Scale. International Conference on Machine Learning (ICML) 2018, PMLR 80. arXiv:1807.01774.

111. Kandasamy, K., Dasarathy, G., Oliva, J. B., Schneider, J., Poczos, B. Multi-fidelity Gaussian Process Bandit Optimisation. arXiv:1603.06288, 2016, whose record states that a preliminary version appeared at NIPS 2016.

112. Li, L., Talwalkar, A. Random Search and Reproducibility for Neural Architecture Search. Conference on Uncertainty in Artificial Intelligence (UAI) 2019, as the arXiv record states. arXiv:1902.07638.

113. Matiisen, T., Oliver, A., Cohen, T., Schulman, J. Teacher-Student Curriculum Learning. IEEE Transactions on Neural Networks and Learning Systems 31(9), pages 3732 to 3740, 2020. DOI 10.1109/TNNLS.2019.2934906.

114. Portelas, R., Colas, C., Hofmann, K., Oudeyer, P.-Y. Teacher Algorithms for Curriculum Learning of Deep RL in Continuously Parameterized Environments. Conference on Robot Learning (CoRL) 2019, PMLR 100. arXiv:1910.07224.

115. Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S. J., Critch, A., Levine, S. Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design. Advances in Neural Information Processing Systems (NeurIPS) 33, 2020. arXiv:2012.02096.

116. Mehta, B., Diaz, M., Golemo, F., Pal, C. J., Paull, L. Active Domain Randomization. Conference on Robot Learning (CoRL) 2019, PMLR 100. arXiv:1904.04762.

117. Muratore, F., Eilers, C., Gienger, M., Peters, J. Data-efficient Domain Randomization with Bayesian Optimization. IEEE Robotics and Automation Letters, DOI 10.1109/LRA.2021.3052391, the record stating acceptance at RA-L with ICRA. arXiv:2003.02471.

118. Mahler, J., Pokorny, F. T., Hou, B., Roderick, M., Laskey, M., Aubry, M., Kohlhoff, K., Kröger, T., Kuffner, J., Goldberg, K. Dex-Net 1.0, A Cloud-Based Network of 3D Objects for Robust Grasp Planning Using a Multi-Armed Bandit Model with Correlated Rewards. IEEE International Conference on Robotics and Automation (ICRA) 2016, pages 1957 to 1964. DOI 10.1109/ICRA.2016.7487342.

119. Koval, M. C., King, J. E., Pollard, N. S., Srinivasa, S. S. Robust Trajectory Selection for Rearrangement Planning as a Multi-Armed Bandit Problem. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2015, pages 2678 to 2685. DOI 10.1109/IROS.2015.7353743.

120. Cully, A., Clune, J., Tarapore, D., Mouret, J.-B. Robots that Can Adapt like Animals. Nature 521(7553), pages 503 to 507, 2015. DOI 10.1038/nature14422. arXiv:1407.3501.

121. Mouret, J.-B., Clune, J. Illuminating Search Spaces by Mapping Elites. arXiv:1504.04909, 2015.

122. Shields, B. J., Stevens, J., Li, J., Parasram, M., Damani, F., Martinez Alvarado, J. I., Janey, J. M., Adams, R. P., Doyle, A. G. Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 590(7844), pages 89 to 96, 2021. DOI 10.1038/s41586-021-03213-y.

123. Kandasamy, K., Krishnamurthy, A., Schneider, J., Poczos, B. Parallelised Bayesian Optimisation via Thompson Sampling. International Conference on Artificial Intelligence and Statistics (AISTATS) 2018, PMLR 84, pages 133 to 142. arXiv:1705.09236, under the title Asynchronous Parallel Bayesian Optimisation via Thompson Sampling.

124. Perchet, V., Rigollet, P., Chassang, S., Snowberg, E. Batched Bandit Problems. Annals of Statistics 44(2), pages 660 to 681, 2016. DOI 10.1214/15-AOS1381. arXiv:1505.00369.

125. Garivier, A., Moulines, E. On Upper-Confidence Bound Policies for Switching Bandit Problems. Lecture Notes in Computer Science, pages 174 to 188, 2011. DOI 10.1007/978-3-642-24412-4_16. arXiv:0805.3415, under the title On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems.

126. Besbes, O., Gur, Y., Zeevi, A. Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards. Advances in Neural Information Processing Systems (NeurIPS) 27, 2014. arXiv:1405.3316.

127. Chen, W., Wang, Y., Yuan, Y. Combinatorial Multi-Armed Bandit, General Framework and Applications. International Conference on Machine Learning (ICML) 2013, PMLR 28(1), pages 151 to 159.

128. Ghosh, D., Rahme, J., Kumar, A., Zhang, A., Adams, R. P., Levine, S. Why Generalization in RL is Difficult, Epistemic POMDPs and Implicit Partial Observability. Advances in Neural Information Processing Systems (NeurIPS) 34, 2021. arXiv:2107.06277.
