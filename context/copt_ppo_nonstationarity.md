# copt_ppo_nonstationarity.md — The design swap as a non-stationarity of the PPO objective

Established 2026-08-10, in answer to the question whether reassigning a morphology to an environment index every two hundred and forty iterations is compatible with proximal policy optimisation, and whether it accounts for the training variance that objective 1 of `knowledge_base.md` has left open since the first investigation wave. Revised the same day on the reading of the design generator's quantisation, which withdraws the out of distribution premise of the first draft and sharpens the conclusion rather than weakening it.

## 1. The question, stated precisely

The co-optimisation runner replaces every robot in the scene at a fixed cadence, so that an environment index which carried one set of link lengths for two hundred and forty iterations carries a different set for the next two hundred and forty. A conventional multi-asset run does the opposite, fixing the association for the whole of training. The question is what this substitution does to the estimators that proximal policy optimisation forms, and whether it is a cause of the reward variance the plots record.

The question separates into three that admit different answers. Whether the identity of the environment index that carries a given design enters any quantity PPO computes, which is a question of symmetry and is answered exactly. Whether a single policy update ever mixes transitions gathered under two different populations, which is a question of ordering and is answered by reading the loop. And what the periodic replacement does to the objective, the value baseline, and the sampling distribution, which is where the real effects lie.

The short answer is that the association carries no information PPO consumes, that no update straddles a swap, and that the damage such as it is comes from three mechanisms that have nothing to do with the association, the foremost being an episode phase synchronisation that the swap imposes on all four thousand and ninety six environments at once.

## 2. What the code actually does at a swap

The configuration in force is four thousand and ninety six environments (`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:1469`), twenty five rollout steps per environment per iteration (`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:98`), an episode of twenty seconds at a control period of twenty milliseconds (`limx_base_env_cfg.py:1482,1483,1486`), two hundred and fifty six individuals, an update interval of two hundred and forty iterations and a random design phase of twelve thousand iterations (`scripts/rsl_rl/train.py:202,205,206`), and a ceiling of forty five thousand iterations (`limx_rsl_rl_ppo_cfg.py:138`).

From these the derived quantities follow, and every later argument rests on them. One rollout is one hundred and two thousand four hundred transitions. One episode is one thousand control steps, so forty iterations. One generation is six thousand control steps, so six episodes per environment and twenty four million five hundred and seventy six thousand transitions. Each design is carried by sixteen environments, since the assignment is `env_idx % num_individuals` (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:696`), and therefore accumulates roughly ninety six completed episodes of evidence per generation. The random phase spans fifty generations and the covariance matrix adaptation phase spans one hundred and thirty seven.

The order of operations within an iteration is the decisive structural fact. The rollout runs to completion (`copt_on_policy_runner.py:312-384`), generalised advantage estimation is computed (`:390`), the policy is updated (`:393`), and only then, if the iteration is a multiple of the update interval, is `_update_morphology` called (`:411`) and the observations refreshed (`:413`). The swap therefore falls strictly between one update and the next rollout.

Within `_update_morphology` the simulation is stopped, the population is advanced, the link lengths are authored in place, and every environment is reset by a bare `unwrapped_env.reset()` (`:672`) with the terrain curriculum suppressed across it (`:671,673`). That call resolves `env_ids` to the whole index range (`/ws/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:361-362`) and `_reset_idx` zeroes the episode counter for every one of them (`/ws/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:394`). This single line is the origin of the largest of the three effects, and section 7 is given over to it.

## 3. The formal setting

A design `d` fixes a transition kernel, so the population induces a family of Markov decision processes sharing a state space, an action space, and a reward function, and differing only in their dynamics. This is a contextual Markov decision process in the sense of Hallak, Di Castro and Mannor [1], equivalently a hidden parameter Markov decision process in the sense of Doshi-Velez and Konidaris [2], with the context being the vector of thigh and shank scales. The training objective is the expectation of the per-context return under the distribution the population defines.

```
J_g(theta) = E_{d ~ D_g} [ J_d(theta) ],    D_g = the empirical distribution of the 256 designs of generation g
```

Two properties of this workspace's instance matter. The context is observed rather than hidden, since `link_lengths`, `robot_mass` and `robot_inertia` are members of the `morphologyObs` group (`limx_base_env_cfg.py:417-440`), that group is joined to the policy group in the runner's observation mapping (`limx_rsl_rl_ppo_cfg.py:139-142`), and the critic group carries the same quantities. The problem is therefore not the epistemic partial observability that Ghosh and colleagues identify as the general obstacle to generalisation in reinforcement learning [3], because the agent is told which body it inhabits. Note in passing that the group sets `enable_corruption = True` (`limx_base_env_cfg.py:443`) while none of its three terms declares a noise model, so the flag is inert and the design signal reaches the networks clean.

The second property is that `D_g` is piecewise constant in the iteration index, holding for two hundred and forty iterations and then jumping. The objective is thus not a fixed function being optimised but a staircase, and the analysis below turns on distinguishing a change of the objective from a change of the variance of an estimator of a fixed objective.

### 3.1 The design space is a finite lattice, and the population oversamples it

The third property, which the first draft of this document overlooked and which materially changes section 8, is that the design space is not continuous. `_compute_link_extents` multiplies a cached base box extent by the sampled scale and then rounds the result to two decimal places (`co_optimisation/co_optimisation/runners/usd_generator.py:370-377`), so every design is snapped to a one centimetre lattice before a URDF is ever authored.

The base extents are read from the URDF the generator is pointed at (`usd_generator.py:310`, `scripts/rsl_rl/train.py:198-201`), which declares a thigh box of `0.05 0.032 0.25` and a shank box of `0.025 0.032 0.3` (`exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/tron1/base_robot.urdf:230,270`). With both scale ranges set to the interval 0.75 to 1.25 (`train.py:207-210`), the reachable extents are 0.1875 to 0.3125 metres for the thigh and 0.225 to 0.375 for the shank, so the lattice admits at most thirteen thigh values and about fifteen shank values, an upper bound of roughly one hundred and ninety five distinct designs. The set actually realised is smaller, the user's own count over the recorded populations giving eight thigh values and ten shank values, so about eighty designs, the difference being the neighbourhood of the box boundary that a search distribution centred on the nominal design rarely reaches.

Two consequences follow immediately, and both are structural rather than incidental.

The population of two hundred and fifty six individuals exceeds the realised design space of roughly eighty, so by the pigeonhole principle every generation contains duplicates, with a mean multiplicity above three. Sixteen environments carry each individual, so each distinct design is in fact carried by fifty or more environments, and the population covers a large fraction of the whole reachable space in every single generation.

And during the early random phase the population is very much narrower than that. The growing distribution schedule scales the sampled spread from five percent of the range at generation zero toward the full range at generation fifty (`knowledge_base.md`, growing design distribution). Five percent of the thigh range is about six millimetres, which is below the one centimetre lattice spacing, so the first generations round to a single lattice point and the entire population of two hundred and fifty six individuals is one design repeated. Design diversity does not begin at generation zero, it begins when the growing spread first exceeds the quantisation.

## 4. Result one, the environment index carries no information PPO consumes

Let `pi` be any permutation of the environment indices, applied consistently to every buffer of the rollout storage. The claim is that the PPO gradient is exactly invariant, so that which index holds which design is not a property of the optimisation at all.

The proof is a walk through the four places the environment axis appears. Generalised advantage estimation recurses over the time index alone and indexes the environment axis elementwise (`/ws/rsl_rl/rsl_rl/storage/rollout_storage.py:134-144`), so it commutes with any permutation of that axis. The mini-batch generator flattens the time and environment axes together and draws indices from a single random permutation of the flattened range (`rollout_storage.py:165-167,170-179`), which destroys environment identity before any loss is formed. The losses are means over the flattened batch and the networks are applied row by row, so both are symmetric functions of the batch. And the advantage normalisation takes the mean and standard deviation over the entire flattened tensor (`rollout_storage.py:150-151`), which is likewise permutation invariant.

Only two places in the whole pipeline read the environment index as an identifier. Fitness accumulation maps a completed episode to the individual that owns it (`copt_on_policy_runner.py:353`), and the per-individual boolean masks are built from the same map (`co_optimisation/co_optimisation/algorithms/copt_ppo.py:87-94`). Both belong to the evolutionary bookkeeping rather than to the policy gradient.

The intuition that neither the value function nor the policy depends on the association is therefore correct and provable rather than merely plausible. Permuting the association would leave every logged quantity unchanged in distribution. It is worth adding that the round-robin assignment is not merely harmless but positively desirable, since sixteen environments per design exactly, held fixed across the generation, is a stratified rather than a random allocation, and stratification removes the sampling variance that drawing designs independently per environment would introduce.

## 5. Result two, no update straddles a swap

Because `_update_morphology` is invoked after `update()` and the storage is cleared within that call, the buffer that trains the policy is always filled under one population. There is therefore no stale importance ratio, no bootstrap of a value across a morphology boundary, and no advantage computed from a reward gathered under one body and a value predicted for another. The two failure modes a reader would most reasonably fear are absent, and the record already contained the observation in a narrower form, that the reset falls between rollout windows (`knowledge_base.md` under objective 1).

The corollary is that the swap cannot corrupt an update. It can only change what the next update is estimating, which is the subject of the next section.

## 6. Result three, the swap moves the estimand

### 6.1 What an estimand is, and why the distinction matters

An estimator is a rule for computing a number from data. An estimand is the quantity that rule is trying to recover. Proximal policy optimisation forms a gradient estimator from a rollout, and its estimand is the gradient of the training objective at the current parameters.

```
g_hat = (1 / (N T)) * sum over i, t of  grad log pi_theta(a_it | s_it) * A_hat_it       the estimator
grad J_g(theta) = sum over d of w_g(d) * grad J_d(theta)                                the estimand
```

Two failure modes are possible and they demand different remedies. An estimator can be noisy, meaning it scatters widely around a fixed estimand, and the remedy is more data or a better baseline. Or the estimand itself can move, meaning the estimator is accurate about a quantity that is no longer the one that matters, and the remedy is to slow the motion or to give the learner the means to track it. These look identical on a training curve and they are not the same problem.

The claim of this section is that a heterogeneous but fixed batch composition does not make the estimator noisy. Within a generation the design carried by each environment is deterministic and identical at every iteration, so the between-design dispersion of the per-environment gradient contributions is a fixed property of the batch, not a random quantity resampled each iteration. It changes what the estimator converges to. It does not widen the distribution of the estimator around that value. What the swap does, and the only thing it does at this level of description, is move the estimand.

### 6.2 A worked example from the present problem

Take two designs from the lattice of section 3.1 and follow the consequences through the reward specification.

Let `d_short` have a thigh of 0.21 metres and a shank of 0.25, and let `d_long` have a thigh of 0.29 and a shank of 0.35. Both are interior lattice points, both are reachable, and both will appear in a population once the growing spread has opened. Their standing leg lengths are 0.46 and 0.64 metres respectively, a difference of eighteen centimetres on a machine whose nominal leg is 0.55.

The base height penalty is the informative term. `pen_base_height` uses `base_height_rough_l2` against a target of 0.75 metres (`limx_base_env_cfg.py:1170-1171,1210`), and the record establishes that the standing base height of a design is the leg length plus a fixed offset of 0.20 metres (`knowledge_base.md`, LIPM work stream). So `d_short` stands at 0.66 metres and `d_long` at 0.84, against a target of 0.75 that the nominal design meets exactly.

The same reward term therefore prescribes three different behaviours. The nominal design meets the target at full extension. `d_long` must flex its knees by nine centimetres to reach the target, so its optimal posture is a crouch, and crouching changes the effective inertia, the achievable step length, and the torque each joint must carry. `d_short` cannot reach the target at all, so however it behaves it pays an irreducible penalty of about eighty one square centimetres of squared error, and the gradient of that term with respect to its actions is close to zero because the term is saturated. One design is being told to crouch, another is being told nothing at all by a term that dominates the other's posture.

The velocity tracking term compounds this. A longer leg gives a longer stride at the same cadence, so the two designs reach the same commanded velocity through different gaits, and the action sequence that maximises `rew_lin_vel_xy` for one is not the sequence that maximises it for the other. The per-design gradients `grad J_{d_short}` and `grad J_{d_long}` therefore point in genuinely different directions in parameter space, and this is not a subtlety of the optimiser but a consequence of the reward specification interacting with the geometry.

Now the estimand. Suppose generation `g` weights the short half of the lattice more heavily, because that is where the search distribution happened to sit, and generation `g+1` weights the long half more heavily, because covariance matrix adaptation has moved its mean. The correct gradient at the same parameters `theta` has changed, not because the policy got worse and not because the data got noisier, but because the question changed from how to walk on mostly short legs to how to walk on mostly long ones. Every step the optimiser took in the previous two hundred and forty iterations was a step toward the answer to the previous question.

### 6.3 The reweighting form, and how far the estimand can move

Because the lattice is finite, the population distribution is a set of weights on a fixed set of atoms rather than a density on a continuum. Write the realised lattice as `Lambda` with about eighty members, and write the empirical weight of design `d` in generation `g` as `w_g(d)`, which is the count of individuals carrying `d` divided by two hundred and fifty six.

```
J_g(theta)      = sum over d in Lambda of  w_g(d) * J_d(theta)
grad J_{g+1} - grad J_g = sum over d in Lambda of  [ w_{g+1}(d) - w_g(d) ] * grad J_d(theta)
```

The weight change sums to zero over the lattice, so the estimand moves by a signed reweighting of a fixed collection of per-design gradients, and its displacement obeys a bound in terms of the total variation distance between consecutive populations.

```
|| grad J_{g+1} - grad J_g ||  <=  TV(w_{g+1}, w_g) * max over d, d' of || grad J_d - grad J_d' ||
```

This factorises the problem usefully. The second factor is the diameter of the per-design gradient set, which section 6.2 argues is substantial and which is a property of the reward specification and the design range, fixed for the run. The first factor is how far the population distribution moves in one generation, and it is entirely under the schedule's control.

### 6.4 Why the finite lattice settles the out of distribution question

The out of distribution framing is wrong for this configuration, and it is worth being explicit about why, because the first draft of this document leaned on it.

The support of the population is not growing after the early generations. Two hundred and fifty six individuals drawn onto a lattice of roughly eighty realised points cover most of that lattice in every generation, each distinct design being carried by fifty or more environments. By generation ten or so the critic has seen every design in the reachable space many times over, and by the end of the random phase it has seen each of them for thousands of iterations. There is no novel design arriving at a swap, and therefore no extrapolation demanded of the value function.

What changes at a swap is `w_g`, not `Lambda`. This is a reweighting of a mixture whose components are all familiar, which is a far milder non-stationarity than the arrival of unseen contexts. It is also a reweighting that shrinks over the run, because covariance matrix adaptation concentrates its search distribution, and the record establishes that in the run analysed it concentrated to the point where every individual was identical (`cmaes.md`, and `knowledge_base.md` under objective 3). As `w_g` approaches a point mass the total variation distance between consecutive generations approaches zero, the bound of section 6.3 approaches zero, and the swap stops moving the estimand at all.

### 6.5 The consequence that decides the investigation

The preceding paragraph yields a prediction, and the prediction is what makes this section worth the space it occupies.

If the persistent training variance were caused by the estimand moving, it would have to decay as the design distribution concentrates, because the displacement of the estimand is bounded by a quantity that provably goes to zero. The observed variance does not decay. The record is emphatic on this point, every reward band staying wide or widening to the end of a run by which time the designs are identical (`task_plots.md:201-226`), and it is the founding observation of objective 1.

Therefore the estimand motion of this section is not the dominant cause of the persistent variance. It is real, it is worth bounding, and it matters most in the early generations where the growing distribution is opening fastest, but it cannot explain a variance that survives the collapse of the design distribution to a single point.

What can explain such a variance is a mechanism driven by the swap cadence rather than by the design diversity, one that fires identically whether the two hundred and fifty six individuals are all different or all the same. Section 7 describes exactly such a mechanism, and the reader should note that it would continue to operate at full strength in a run whose population had collapsed to a single design, which is precisely the regime in which the variance was observed to persist.

## 7. Mechanism one, episode phase synchronisation

### 7.1 What phase is, and what episode_length_buf does

Define the phase of an environment as the number of control steps elapsed since it last reset, which is exactly the integer held in `episode_length_buf`. Isaac Lab increments it once per environment step (`manager_based_rl_env.py:201`) and zeroes it for the environments being reset (`:394`). It is not a diagnostic counter. It is the state variable that decides when an episode ends, because the timeout termination is a direct comparison against it.

```python
# /ws/IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:33
return env.episode_length_buf >= env.max_episode_length
```

An environment therefore resets either when it falls, which the record puts at about twelve percent of episodes at convergence, or when its counter reaches one thousand, which accounts for the other eighty eight (`task_plots.md:265-270`). The vector of counters across the four thousand and ninety six environments is the phase profile of the population, and its distribution is what determines what a batch of transitions actually contains.

### 7.2 What init_at_random_ep_len does, and what it does not do

The runner offers one mechanism for shaping that profile, and it uses it once.

```python
# co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:227-230
if init_at_random_ep_len:
    self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
    )
```

`scripts/rsl_rl/train.py:283` sets the flag, so this runs at the top of every training run. It draws an independent uniform integer on the interval from zero to one thousand for every environment.

What it does not do is move any robot. Every environment is still physically at its reset pose when this executes. What it changes is when each environment will next time out, since environment `i` now needs `1000 - tau_i` further steps to reach the ceiling. The first round of timeouts is therefore spread uniformly across the following thousand steps instead of arriving together, and from that point each environment keeps its own beat. The trick works forward in time, by staggering future resets, not by staggering the present state.

Once staggered, the profile stays staggered. Timeouts recur on each environment's own period and falls perturb the phase further, so the population settles into an approximately uniform distribution over the interval from zero to one thousand and remains there. This is the ordinary operating regime of every non-co-optimisation run in this workspace, and it is the regime that the estimators below implicitly assume.

The bare `unwrapped_env.reset()` at `copt_on_policy_runner.py:672` destroys it. It resets every environment, so `_reset_idx` zeroes every counter, and the population returns to a single common phase of zero. Nothing re-randomises it, because the only randomisation is the one at the top of `learn`, which has long since executed.

### 7.3 What a batch samples, and why the profile decides it

The quantity a policy gradient method needs from a batch is a sample from the discounted state occupancy measure of the current policy. Decompose that measure by phase, writing `d_tau` for the distribution of states at phase `tau` and `p` for the distribution of phase across the population.

```
d^pi  =  sum over tau of  p(tau) * d_tau
```

With a uniform profile, `p` is uniform on the interval from zero to one thousand and the batch samples the full occupancy measure, which is what the estimator is supposed to receive. With a synchronised profile, `p` is a point mass at the common phase, and one rollout of twenty five steps covers phases `tau` to `tau + 25`, that is two and a half percent of an episode. The batch is then a sample from `d_tau` rather than from `d^pi`.

The estimator has not become noisy. It has become an accurate estimate of the wrong thing, namely the phase-conditional gradient at whatever slice of the episode the population currently occupies. Successive iterations sweep that slice forward together, so the policy is pulled toward optimising the standing start, then the acceleration, then the cruise, then the approach to timeout, and then the cycle repeats forty iterations later. This is a second and distinct sense in which an estimand moves, and unlike the design reweighting of section 6 it is driven purely by the cadence and is wholly indifferent to how many distinct designs the population holds.

### 7.4 The variance arithmetic

The effect on the logged statistics admits an exact treatment. Let `R_i` be the per-environment contribution to whatever quantity is being averaged, a return, a reward term, or a gradient component, and decompose it into the part explained by phase and the part that is not.

```
R_i = mu(tau_i) + eps_i,    E[eps_i | tau_i] = 0,    Var(eps_i) = sigma_eps^2
sigma_mu^2 = Var over the stationary phase distribution of mu(tau)
```

The idiosyncratic part `eps_i` carries the terrain tile, the sampled command, the action noise and the accumulated divergence of that environment's trajectory. The systematic part `mu(tau)` is the population mean at a given point in the episode, and it is large in a locomotion task for reasons section 7.5 makes concrete.

Under a uniform phase profile the phases are independent across environments, so both components average down.

```
Var( mean of R )  =  ( sigma_mu^2 + sigma_eps^2 ) / N
```

Under a synchronised profile the phase is common, so `mu(tau)` does not average at all. Conditional on the phase only the idiosyncratic part averages, and across iterations the common term sweeps through its full range.

```
Var( mean of R )  =  sigma_mu^2  +  sigma_eps^2 / N
```

The ratio of the two is the design effect of classical sampling theory, and writing `rho` for the fraction of the per-environment variance that phase explains it takes a form that needs no approximation.

```
rho = sigma_mu^2 / ( sigma_mu^2 + sigma_eps^2 )
inflation = 1 + (N - 1) * rho
N_eff = N / ( 1 + (N - 1) * rho )
```

At `N = 4096` the consequences are severe for values of `rho` that are not remotely extreme.

| rho | inflation | N_eff |
|---|---|---|
| 0.001 | 5.1 | 803 |
| 0.01 | 41.9 | 98 |
| 0.05 | 205.8 | 20 |
| 0.10 | 410.5 | 10 |

The reading is that a phase that explains one percent of the variance of a per-environment return is enough to reduce four thousand and ninety six environments to the statistical worth of ninety eight. The parallelism that makes this training setup viable is spent, and it is spent silently, since nothing in the logs reports an effective sample size.

### 7.5 A worked example, the velocity tracking term

The velocity tracking reward makes `rho` concrete, because its dependence on phase can be computed rather than assumed.

```python
# /ws/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:311-315
lin_vel_error = torch.sum(torch.square(
    env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
return torch.exp(-lin_vel_error / std**2)
```

The term is configured with a weight of twenty five and a squared width of 0.16 (`limx_base_env_cfg.py:1138-1142`), and the commanded longitudinal velocity is drawn uniformly on the interval minus one to one at reset, widening to minus one and a half to one and a half under the curriculum (`limx_base_env_cfg.py:122`, `:1304`).

Consider the two ends of an episode. At phase zero the robot has just been placed at its reset pose with a velocity of zero and has just been handed a fresh command. The expected squared error is therefore the second moment of the command, which for a uniform draw on minus one to one is one third. The reward is `exp(-0.333 / 0.16)`, about 0.125, which against the weight of twenty five is roughly 3.1 per step. At cruise, once the policy has accelerated to within about a tenth of a metre per second of the command, the squared error is about 0.01 and the reward is `exp(-0.0625)`, about 0.939, which is roughly 23.5 per step.

The phase-conditional mean of this single term therefore swings by a factor of seven and a half within an episode, and the swing occupies the acceleration transient, which for a biped reaching a metre per second is of the order of twenty five to fifty control steps, that is one to two entire rollouts. It follows that the first iteration after every morphology swap is composed almost entirely of transitions drawn from the acceleration transient, across all four thousand and ninety six environments simultaneously, and that the tracking reward logged for that iteration is depressed by a factor of several relative to the same policy's steady state.

This is corroborated rather than merely predicted. The record notes `rew_lin_vel_xy` oscillating across a band from near zero to about eighteen throughout training while ending at 17.7 (`task_plots.md:218-222`), and the amplitude of that band is of the same order as the phase-conditional swing computed above. The correspondence does not prove the attribution, since the logged channel is an episodic rate rather than a per-step value and other terms contribute, but it establishes that the mechanism is of the right magnitude to be the explanation rather than a footnote to it.

Two further terms inherit the same structure and are worth naming. Termination by falling concentrates in the early part of an episode, where the robot is accelerating from rest against a freshly drawn command, so the fall fraction is also phase-modulated and the termination panel should ripple in antiphase with the tracking reward. And the command is resampled every seven and a half to twelve and a half seconds (`limx_base_env_cfg.py:120`), which is a second clock that the reset also restarts, so each command redraw creates its own smaller tracking transient and those transients are themselves partially aligned across the population for the first cycle after a swap.

### 7.6 Why the lock does not decay within a generation

A reader might reasonably expect the population to shake itself back into a uniform profile within a few episodes, which would confine the effect to a brief transient after each swap. It does not, and the reason is that the dominant termination mode preserves phase rather than perturbing it.

An environment that reaches the timeout resets at exactly one thousand steps, so it re-enters the same phase cohort it left. Only a fall reschedules an environment, and falls account for about twelve percent of episodes at convergence. The fraction of the population still in the original cohort after `k` episodes is therefore about `0.88^k`, which over the six episodes of one generation leaves roughly forty six percent of the four thousand and ninety six environments still exactly phase-locked, whereupon the next swap returns the entire population to a common phase and the process restarts.

The system consequently never reaches the desynchronised regime that a baseline run occupies by default. Worse, the environments that did fall do not scatter uniformly, they form their own secondary cohorts, so the profile is a small number of sharp modes rather than the smooth uniform distribution the estimator assumes. The effect is periodic at exactly the morphology cadence, it is independent of the number of distinct designs in the population, and it therefore survives the collapse of the design distribution that section 6.5 shows must extinguish the estimand-motion explanation.

### 7.7 A downstream consequence, the angular command curriculum

Reading the reset path for this document surfaced a coupling that no prior document records and that is exact rather than approximate.

Curriculum terms are evaluated inside the reset path, `curriculum_manager.compute(env_ids)` running as the first statement of `_reset_idx` (`manager_based_rl_env.py:356`), ahead of `reward_manager.reset(env_ids)` at `:375`. The ordering is deliberate and correct, since it lets a curriculum term read `_episode_sums` for the environments being reset while those sums still hold the episode that has just finished. The command velocity curriculum relies on exactly this, comparing the mean episode sum against a threshold scaled by the term weight and the control period (`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/curriculums.py:435-441`).

The gate is guarded by `env.common_step_counter % interval == 0`, and `common_step_counter` advances once per control step (`manager_based_rl_env.py:202`). The angular command curriculum sets that interval to `250 * 24`, which is six thousand (`limx_base_env_cfg.py:1329`). The morphology cadence is two hundred and forty iterations of twenty five steps, which is also six thousand. The two are not merely commensurate, they are equal, so the only control steps at which the angular gate can fire are precisely the steps at which a morphology swap occurs.

At that step the swap calls a bare reset over the whole index range, so `curriculum_manager.compute` is invoked for all four thousand and ninety six environments while their episode sums hold truncated partial episodes rather than completed ones. Truncated sums are systematically smaller than complete ones, so the gate is systematically starved, and the curriculum that widens the angular velocity command is evaluated on biased evidence at every generation boundary and at no other time. The linear command curricula, at `300 * 24`, and the push curriculum, at `400 * 24`, are affected on every sixth and every eighth swap respectively by the same mechanism.

The terrain curriculum is exempt, because `_update_morphology` sets the suppression flag across its reset (`copt_on_policy_runner.py:671,673`). The suppression does not extend to the command or push curricula, and on this reading it should.

## 8. Mechanism two, the value baseline, corrected for a finite lattice

Advantage is the residual of a return against a fitted value. Where the critic has not fitted a design, its residual carries a constant offset for that design, and the advantage decomposes accordingly.

```
A_i = A_i_centred + b_{d(i)},    b_d = E[R | d] - E[V | d]
```

A constant offset over every transition of a design acts as a score-function weight on every action taken in that body, irrespective of whether the action was good. Its expectation over the policy vanishes, so it introduces no bias, but its second moment adds directly to the variance of the gradient in proportion to the square of the offset. This is the mechanism that generalised advantage estimation exists to suppress by subtracting a state-dependent baseline [6], and it is defeated whenever the baseline is stale with respect to the context.

Global advantage normalisation does not remove it. The normalisation subtracts one mean over the whole flattened batch (`rollout_storage.py:151`), which removes the population average of the offsets and leaves their spread intact. Removing the offset requires centring within each design's own block, which is precisely what `compute_returns_design_wise` implements, disabling the global normalisation and normalising each individual's block against its own statistics (`copt_ppo.py:96-111`).

That method has no caller. A search across the live tree finds it defined at `copt_ppo.py:96` and invoked nowhere, while the runner calls the inherited `compute_returns` at `copt_on_policy_runner.py:390`, which reaches `/ws/rsl_rl/rsl_rl/algorithms/ppo.py:187-192` and requests global normalisation because `normalize_advantage_per_mini_batch` defaults to false (`ppo.py:42`) and no configuration in `exts/bipedal_locomotion/` overrides it. The audit of 2026-07-30 recorded at `knowledge_base.md` that this remedy had landed. It was written, and it was never connected.

The finite lattice of section 3.1 substantially reduces the expected severity of this mechanism, and the first draft of this document overstated it. The offsets `b_d` are not the extrapolation error of a critic meeting a novel design, because after the early generations no design is novel. They are the residual misfit of a critic that has seen every one of roughly eighty designs for thousands of iterations, and such a critic should have small offsets. The explained variance the record reports, between 0.92 and 0.99, is consistent with that reading. The mechanism should therefore be ranked below the phase synchronisation and treated as a correctness matter rather than as a candidate explanation of the observed variance, and the case for wiring the method rests on removing a known residual cheaply rather than on an expectation of a large gain.

One caveat preserves the mechanism's relevance in the early phase. Before the growing spread exceeds the quantisation the population is a single repeated design, and shortly afterwards it is a handful, so the lattice argument does not apply there and the offsets may be genuinely large during the opening generations. That is also the phase in which the design distribution is moving fastest, so the two mechanisms of sections 6 and 8 are aligned in time and neither can be attributed without the measurement of section 10.

## 9. Mechanism three, the objective jumps and the learning rate is thrown back with it

Beyond the value baseline, the adaptive Kullback Leibler schedule is reset at every morphology update during the random phase and once at the transition into covariance matrix adaptation, by `reinit_learning_rate` restoring the configured initial rate (`copt_on_policy_runner.py:647,655`, `copt_ppo.py:347-348`). Fifty such resets occur before iteration twelve thousand. The adaptive schedule must then re-descend from one times ten to the minus three on each occasion, which is consistent with the thrashing the learning rate panel shows, and it means the effective step size is systematically larger in the iterations immediately following a swap, exactly where the batch is most phase-locked and the value function least reliable.

Whether this is a defect or a deliberate re-annealing is a design question rather than a factual one. It is recorded here because the mechanisms compound, a larger step being taken against a batch that samples a narrow slice of the episode.

The wider concern is the one Lyle, Rowland and Dabney identify, that non-stationary prediction targets erode a network's capacity to update its own predictions [4], and that Nikishin and colleagues frame as an over-weighting of early experience [5]. Both are properties of the regime rather than of this implementation, and both argue for reducing the number of discontinuities rather than for any particular patch.

## 10. What the literature prescribes

The closest published precedent is the joint construction and control of Schaff and colleagues, which maintains a distribution over designs, samples a design per episode, gives the policy the design parameters as input so that it may specialise, and optimises with proximal policy optimisation [7]. Three of its choices bear directly on the configuration here. The policy is conditioned on the design explicitly, which this workspace now does through the `morphologyObs` group and did not do when `task_plots.md` was written. The design distribution is a mixture rather than a point, preserving spread against premature collapse, which is the opposite of the collapse `cmaes.md` records. And the controller is trained for one hundred million timesteps before the design distribution is updated at all, against the twenty four and a half million that two hundred and forty iterations supply here, so this configuration advances its design distribution roughly four times more often than the precedent it most resembles.

The wider co-design literature already registered in `literature.md` cluster 4 supports the same reading. Cheney and colleagues identify the invalidation of a co-adapted controller by a morphology change as the central pathology of co-design, which is a statement about the objective moving rather than about estimator noise, and is therefore a statement about section 6 rather than section 7. Luck and colleagues respond to the same difficulty by moving to an off-policy learner whose replay buffer survives a design change, a route unavailable to an on-policy method that discards its data every iteration. Huang and colleagues and Gupta and colleagues establish that one policy can span a morphology family provided it is told which body it occupies, which is the condition this configuration satisfies.

Two further threads bear on the mechanism rather than the setting. The identification of a body from a latent code, as in the rapid motor adaptation of Kumar and colleagues where a privileged environment vector is compressed to a low-dimensional extrinsic and then regressed from proprioceptive history [8], is the architecture the co-optimisation estimator already imitates. And the multi-task literature supplies the vocabulary for what a heterogeneous batch does to a shared parameter vector, the conflict between per-task gradients that Yu and colleagues project away [9], which in this setting is exactly the diameter term in the bound of section 6.3.

Finally, the empirical survey of on-policy implementation choices by Andrychowicz, Raichuk and colleagues is the appropriate authority for the normalisation question, having examined advantage normalisation among some sixty eight such choices across a quarter of a million trained agents [10]. It is the standard against which any change to the normalisation in section 8 should be justified, since the choice is known to interact with batch construction.

## 11. Diagnosis before remedy

None of the mechanisms above can be ranked against the curriculum from the sources alone, and the record is explicit that terrain difficulty, push force, command range expansion and tracking standard deviation all continue to move to the end of training (`task_plots.md:183-190,247-262`), which confounds any attribution of variance to morphology. Three measurements settle the matter and all three can be taken from logs already written.

The first is the swap-aligned average. Take any reward channel, cut it at every multiple of two hundred and forty iterations, and average the segments across generations. A forty iteration ripple that survives the averaging is the phase synchronisation of section 7, its period being the episode length in iterations, and its amplitude relative to the raw band gives that mechanism's share of the variance directly. A flat average exonerates it.

The second is the same average taken over the explained variance channel, which `CoptPPO` logs and `copt.md` section 8 records. A dip in the iterations immediately after each boundary that recovers before the next is the value staleness of section 8, and the depth of the dip bounds the per-design offset. Section 8 predicts this dip should be shallow after the early generations and deep before them, which is itself a testable claim.

The third discriminates section 6 from section 7 directly and is the reason section 6.5 matters. Compare the amplitude of the swap-aligned ripple early in the run, where the design distribution is opening fastest, against its amplitude late in the run, where `cmaes.md` establishes the distribution has collapsed. An amplitude that persists undiminished into the collapsed regime is proof that the cadence and not the design diversity drives it, since a collapsed population presents the same design at every swap.

## 12. Remedies, ordered by expected value against cost

The first is to desynchronise episode phase after the swap, restoring at the reset the randomisation that `learn` performs once at startup, by drawing a uniform initial episode counter for every environment immediately after `unwrapped_env.reset()` at `copt_on_policy_runner.py:672`. It is a single statement and it mirrors an idiom already present at `copt_on_policy_runner.py:227-230`.

Its effect must be stated precisely, since section 7.2 shows the trick works forward in time. It does not unlock the phase at the instant of the swap, because every robot is still physically at its reset pose. What it does is stagger the first post-swap timeout, so the lock is broken permanently after one episode instead of being renewed at every timeout for the whole generation. The locked interval falls from the two hundred and forty iterations of a generation to the forty of a single episode, a sixfold reduction in the time spent in the degraded regime.

One side effect requires handling rather than acceptance. An environment whose counter is drawn near the ceiling times out within a few steps and contributes a spurious short episode, whose return is near zero and whose completion increments `_individual_episode_counts` (`copt_on_policy_runner.py:357`). Since fitness is the mean return per completed episode (`:683-692`), a scatter of one-step episodes would drag the fitness of the individuals unlucky enough to own them, and would do so differently for each individual, which is precisely the kind of noise the covariance matrix adaptation ranking cannot distinguish from signal. The remedy should therefore gate fitness accumulation on episodes that began after the swap, or equivalently ignore completed episodes shorter than some fraction of the maximum, in the same change.

The second is to extend the terrain curriculum suppression at `copt_on_policy_runner.py:671,673` to the command and push curricula, on the finding of section 7.7 that the angular command curriculum is evaluated exclusively at swap boundaries and exclusively on truncated episode sums. This is a correctness fix independent of the variance question and it is cheaper than the first.

The third is to connect `compute_returns_design_wise`, which is already written and exercised by nothing. Section 8 revises its expected benefit downward, so it should be understood as removing a known residual rather than as a fix for the observed variance.

The fourth is to lengthen the generation window, or equivalently to reduce the number of morphology updates over the run. Every mechanism identified here scales with the swap frequency, and the precedent in [7] uses a window four times longer in environment steps. This is the cheapest change of all, being a single constant at `scripts/rsl_rl/train.py:205`, but it trades against the number of covariance matrix adaptation generations the run can afford, presently one hundred and thirty seven, and that budget is already modest for a search over a lattice of roughly eighty points.

The fifth is not to touch the round-robin association, which section 4 proves carries no information the optimiser consumes and which supplies an exactly stratified allocation of environments to designs. Any proposal to shuffle it, to hold it fixed across generations, or to randomise it should be declined on those grounds.

A sixth is recorded because section 3.1 raises it and no other document does. A population of two hundred and fifty six individuals over a realised lattice of roughly eighty designs means every generation evaluates each distinct design about three times under different individual indices, and those duplicate individuals receive different measured fitness purely from sampling noise. Covariance matrix adaptation ranks the individuals rather than the designs, so it is asked to order genotypes that map to identical robots, and whatever order the noise supplies is fed into the recombination as though it carried information. Reducing the population below the size of the reachable lattice, or widening the parameter ranges, or reducing the two decimal rounding at `usd_generator.py:376` that creates the lattice in the first place, would each address this. The rounding is the item the investigation plan already recommended removing and which `knowledge_base.md` records as still outstanding.

Two of these change the behaviour of an existing caller, the first and the second, and one changes it conditionally, the third. Each therefore wants a flag whose default reproduces the present conduct, set explicitly in the co-optimisation configuration so that the choice is recorded in the run's dumped parameters, per the rules governing this workspace.

## 13. Corrections to the existing record

Four entries elsewhere in this directory are overtaken by the readings above and are corrected here rather than edited in place, per the convention of `README.md`. A fifth correction applies to the first draft of this document itself and is recorded in section 6.4 and section 8 at the point of use.

The claim in `knowledge_base.md` under the 2026-07-30 audit, that per-individual advantage normalisation with global normalisation disabled is among the variance remedies that have landed, is withdrawn. The method exists at `copt_ppo.py:96` and has no caller, and the executed path applies global normalisation.

The finding in `task_plots.md:128-165`, that the policy network proper does not see the design and receives only a detached sixteen dimensional latent, no longer describes the configuration. The runner's observation mapping joins `morphologyObs` to the policy group (`limx_rsl_rl_ppo_cfg.py:139-142`), so the actor observes link lengths, mass and inertia directly. The document remains correct as a record of the run it analysed.

The hyperparameters inventoried in `task_plots.md:39-114` are superseded on a further count beyond those the 2026-07-30 audit listed, the iteration ceiling standing at forty five thousand (`limx_rsl_rl_ppo_cfg.py:138`) against the thirty thousand recorded, and the derived generation arithmetic changing with it.

The treatment of the two decimal rounding at `usd_generator.py:376` across the record, which `knowledge_base.md` and `cmaes.md` both characterise as a resolution floor that erases the spread of a converged search, is incomplete rather than wrong. Section 3.1 establishes the stronger consequence, that the rounding makes the design space a finite lattice of at most about one hundred and ninety five points of which roughly eighty are realised, which is smaller than the population of two hundred and fifty six and therefore guarantees duplicate individuals in every generation.

## 14. Bibliography

1. Hallak, A., Di Castro, D., Mannor, S. (2015). Contextual Markov Decision Processes. arXiv:1502.02259.
2. Doshi-Velez, F., Konidaris, G. (2013). Hidden Parameter Markov Decision Processes, A Semiparametric Regression Approach for Discovering Latent Task Parametrizations. arXiv:1308.3513. The venue of the published version was not established by the retrieved sources.
3. Ghosh, D., Rahme, J., Kumar, A., Zhang, A., Adams, R. P., Levine, S. (2021). Why Generalization in RL is Difficult, Epistemic POMDPs and Implicit Partial Observability. Advances in Neural Information Processing Systems 34, arXiv:2107.06277.
4. Lyle, C., Rowland, M., Dabney, W. (2022). Understanding and Preventing Capacity Loss in Reinforcement Learning. ICLR 2022, arXiv:2204.09560.
5. Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., Courville, A. (2022). The Primacy Bias in Deep Reinforcement Learning. ICML 2022, PMLR 162, arXiv:2205.07802.
6. Schulman, J., Moritz, P., Levine, S., Jordan, M. I., Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016, arXiv:1506.02438.
7. Schaff, C., Yunis, D., Chakrabarti, A., Walter, M. R. (2019). Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning. ICRA 2019, arXiv:1801.01432.
8. Kumar, A., Fu, Z., Pathak, D., Malik, J. (2021). RMA, Rapid Motor Adaptation for Legged Robots. Robotics, Science and Systems 2021, arXiv:2107.04034.
9. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C. (2020). Gradient Surgery for Multi-Task Learning. Advances in Neural Information Processing Systems 33, 5824-5836, arXiv:2001.06782.
10. Andrychowicz, M., Raichuk, A., et al. (2020). What Matters In On-Policy Reinforcement Learning, A Large-Scale Empirical Study. arXiv:2006.05990. The full author list was not established by the retrieved sources.
