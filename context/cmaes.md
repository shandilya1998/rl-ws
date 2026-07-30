# cmaes.md — CMA-ES Theory, pycma Behaviour, and Design-Trajectory Statistics (Agent D)

This section supports objective 3 (biased and low-variety CMA-ES sampling)
and contributes the optimisation-theory background for objective 1. It
documents the installed `cma` package behaviour, the precise meaning of
`sigma0` on the unit box, a statistical read of the produced design
trajectory, and a concrete recommended initialisation. All claims about
package behaviour are grounded in the source of the installed library at
`/usr/local/lib/python3.11/dist-packages/cma` (version 4.4.4) and in
Hansen's tutorial arXiv:1604.00772.

## 1. Run configuration under study

Confirmed from `tron1-rl-isaaclab-cozum/scripts/rsl_rl/train.py` lines 200 to 226.

- Generator `GrowingDesignDistCMAESDesignGenerator`, `num_individuals = 64`,
  `ea_update_interval = 120`, `ea_late_start = 8000`.
- `sigma0 = 0.75`, `seed = 42`, `late_start = True`,
  `late_start_it = 8000 / 120 = 66`,
  `max_cma_iter = (max_iterations - 8000) / 120`.
- CMA-ES drives only two parameters, `thigh_length_scale` and
  `shank_length_scale`, each bounded to the range `(0.85, 1.15)` from
  `CMAES_PARAM_RANGES`. The search dimension is therefore `N = 2`.
- The generator constructs `cma.CMAEvolutionStrategy(np.full(2, 0.5), 0.75, opts)`
  with `opts` setting `popsize = 64`, `bounds = [zeros(2), ones(2)]`,
  `seed = 42`, `maxiter = max_cma_iter` (`usd_generator.py` lines 720 to 731).
- Note one latent defect orthogonal to this objective. `train.py` line 209
  instantiates `GrowingDesignDistCMAESDesignGenerator`, yet the import block
  (lines 98 to 102) imports only `CMAESDesignGenerator` and
  `RandomDesignGenerator`. The recorded trajectory exhibits the slow
  distribution growth of `_sample_scales_v2`, so the growing-distribution
  subclass did run, which means the executed file differed from the visible
  import block. This belongs to objective 2 and is flagged here only for
  provenance.

## 2. Meaning of sigma0 and why 0.75 on a [0,1] box is extreme

`sigma0` is the initial step-size, equivalently the initial standard
deviation of the multivariate Gaussian search distribution, expressed in the
same units as the search variables `x0`. At construction the covariance is
`C0 = I`, so the per-coordinate initial sampling standard deviation equals
`sigma0` (subject to the cap in Section 3). The pycma documentation states the
governing rule plainly, namely that the variables should be scaled such that a
single standard deviation on all variables is useful and the optimum is
expected to lie within about `x0 ± 3*sigma0`. The standard recommendation,
repeated across CMA-ES implementations and in Hansen 2016, is that `sigma0`
be roughly one quarter of the width of the search interval in which the
optimum is expected.

Here the search space is the unit hypercube `[0,1]^2`, the initial mean is
`x0 = 0.5`, and `sigma0 = 0.75`. The interval `x0 ± 3*sigma0` evaluates to
`0.5 ± 2.25`, that is `[-1.75, 2.25]`. This overflows the unit box by 225
percent on each side, so the prior places the bulk of its probability mass
outside the feasible region. Measured against the quarter-of-range rule the
appropriate value for a unit box is near `0.25`, whence `0.75` is three times
too large. The intended semantics, namely that `±3 sigma0` should bracket the
plausible optimum inside the domain, are violated by a factor of three to
four. A correctly scaled value would be `sigma0` in `[0.2, 0.3]`, for which
`x0 ± 2 sigma0` spans roughly `[0.0, 1.0]` and covers the box without
overflowing it.

## 3. The hidden maxstd cap (a key pycma nuance)

pycma does not sample with a literal standard deviation of `0.75`. When
`bounds` are set and the user leaves `maxstd` unset, the option
`maxstd_boundrange = 1/3` (`options_parameters.py` line 85) drives a per
coordinate cap

```text
maxstd = (upper - lower) * maxstd_boundrange = 1 * (1/3) = 0.3333...
```

set in `evolution_strategy.py` lines 1156 to 1161. The method
`_stds_into_limits` (`evolution_strategy.py` line 1506) then rescales the
diagonal `sigma_vec` so that no coordinate standard deviation exceeds
`maxstd`. This routine runs once at construction (line 1350) and again at the
end of every `tell` (line 2963). Consequently the effective initial
per-coordinate sampling standard deviation is `min(sigma0, 0.333) = 0.333`,
not `0.75`. pycma silently shrinks the oversized `sigma0` to one third of the
box, yet the result is still the widest distribution the box admits.

The practical effect of `N(0.5, 0.333)` on `[0,1]` is severe boundary
incidence. The point at `+1.5` standard deviations already reaches the upper
bound `1.0`, and roughly 16 percent of the mass on each coordinate falls
beyond each bound before repair. The initial distribution is therefore almost
uniform across the box with heavy accumulation at the bounds, which is the
opposite of a focused local search and is the seed of the boundary bias
analysed next.

## 4. Boundary handling and the bias toward the upper bound

The default boundary handler in cma 4.4.4 is `BoundTransform`, not
`BoundPenalty`. This is set at `options_parameters.py` line 69
(`BoundaryHandler='BoundTransform'`) and instantiated at
`evolution_strategy.py` lines 1129 to 1131, which read
`self.boundary_handler = opts['BoundaryHandler']` followed by instantiation
with `opts['bounds']`. The design generator sets only `bounds` and never
overrides `BoundaryHandler`, so `BoundTransform` is in force. This contradicts
`../plans/CMAES_DESIGN_GENERATOR.md` sections c.6 and b.10, which assert that the
default is `BoundPenalty`. The discrepancy should be corrected in that
document.

`BoundTransform` (`boundary_handler.py` line 454) maps any out-of-bounds
genotype sample back into the feasible box through a smooth piecewise linear
and quadratic transformation, implemented by
`BoxConstraintsLinQuadTransformation`. The internal Gaussian state operates in
an unbounded genotype space, while `ask()` returns repaired in-bounds
phenotypes. The transform saturates, so genotype values that lie far above the
upper bound all map to phenotypes pinned at that bound.

This produces the observed bias toward the upper bound through the following
mechanism. With the mean at `0.5` and an effective standard deviation of
`0.333`, a large fraction of genotype samples land beyond `[0,1]` and are
folded onto the boundary. As the inner-loop fitness rewards longer thigh and
shank, that is scales approaching `1.15`, the rank-based mean update walks the
genotype mean steadily upward, eventually above the upper bound, where it can
sit because the transform keeps the phenotype valid. Many distinct genotypes
then collapse onto the single boundary phenotype, which destroys variety and
pins both parameters at their maximum rather than at any interior optimum.
This is consistent with the recorded convergence to shank z equal to 0.34,
which equals `round(0.30 * 1.15, 2)`, and thigh z equal to 0.28, one
quantisation bin below the rounded upper bound 0.29 equal to
`round(0.25 * 1.15, 2)`.

## 5. Relevant CMAOptions, defaults, and termination

Values below are read from `options_parameters.py` and confirmed against the
constructor and `_CMAStopDict` in `evolution_strategy.py`. `N` is the
dimension, here 2, and `popsize` is 64.

| Option | Default | In this run | Role |
|--------|---------|-------------|------|
| `popsize` | `4 + 3*ln(N)` (≈6 for N=2) | 64 | sample count lambda per generation |
| `bounds` | `[None, None]` | `[0,1]^2` | per-coordinate box |
| `seed` | `time` | 42 | RNG seed |
| `maxiter` | `100 + 150*(N+3)**2 // popsize**0.5` | `max_cma_iter` | hard generation cap |
| `maxfevals` | `inf` | default | evaluation cap |
| `CMA_active` | `True` | default | active negative-weight update |
| `AdaptSigma` | `True` (CSA) | default | cumulative step-size adaptation |
| `CMA_stds` | `None` | default | per-coordinate multipliers on sigma0, held in `sigma_vec`, candidate fix |
| `minstd` | `0` | default | floor on coordinate std |
| `maxstd` | `None` then `bound_range/3 = 0.333` | 0.333 | cap on coordinate std, see Section 3 |
| `tolx` | `1e-11` | default | stop on tiny x and sigma changes |
| `tolfun` | `1e-11` | default | stop on tiny within-generation fitness range |
| `tolfunhist` | `1e-12` | default | stop on tiny historical fitness range |
| `tolflatfitness` | `1`, amended near `3` for these sizes | default | iterations of flat fitness tolerated |
| `tolstagnation` | `int(100 + 100*N**1.5/popsize)` | default | stop on no improvement |
| `tolconditioncov` | `1e14` | default | stop on ill-conditioned C |
| `tolfacupx` | `1e3` | default | stop if sigma grows 1000x, signals sigma0 too small |

`es.stop()` returns a possibly empty dict mapping each fired criterion to its
value, assembled by `_CMAStopDict` (`evolution_strategy.py` lines 4140 to
4269). It becomes non-empty when any of the following hold, namely `ftarget`,
`maxfevals`, `maxiter`, `tolfacupx`, `tolx`, `tolfun`, `tolfunrel`,
`tolfunhist`, `tolstagnation`, `tolxstagnation`, `tolupsigma`, `timeout`,
`noeffectcoord` (a step of `0.2 sigma` no longer changes the mean in some
coordinate), `noeffectaxis` (a step of `0.1 sigma` along a principal axis has
no effect), `tolconditioncov`, `tolflatfitness`, or `callback`. The generator
polls `es.stop()` inside `update_with_fitness` and, once non-empty, sets
`self._terminated` and thereafter returns the incumbent design unchanged,
which is the degenerate fine-tuning branch at `usd_generator.py` lines 761 to
765.

In this run the quantisation analysed in Section 7 makes all 64 fitnesses
identical once the population collapses, so the within-generation fitness range
is zero. This triggers `tolfun` and `tolflatfitness`, which are the most
plausible terminators, after which the search is frozen at the boundary
design.

## 6. How sigma and the covariance evolve, and how they collapse

Cumulative step-size adaptation updates the global step-size by comparing the
length of the conjugate evolution path with its expected length under random
selection,

```text
sigma_{g+1} = sigma_g * exp( (c_sigma / d_sigma) * ( ||p_sigma|| / E||N(0,I)|| - 1 ) ).
```

Positively correlated selection steps lengthen `p_sigma` and grow sigma, while
anti-correlated or short steps shorten it and shrink sigma. The covariance
matrix adapts by the rank-one update from `p_c` and the rank-mu update from
the current selection scatter, with active negative weights contracting C in
unfavourable directions.

The collapse to a single point proceeds as follows. Once the mean reaches the
upper bound, the rank-based selection can no longer pull it further, so the
mean shifts become short and lose temporal correlation. Then `||p_sigma||`
falls below `E||N(0,I)||` and sigma decays geometrically, while the rank-mu and
rank-one updates contract C around the incumbent. The sampling ellipsoid
shrinks toward a point, and the per-individual spread tends to zero, which is
exactly the zero-variety endpoint observed in the final CSV. The
quantisation accelerates this by zeroing the apparent fitness differences,
which removes the selection gradient entirely.

## 7. Statistical analysis of the design trajectory

Method. The CSVs at
`/ws/IsaacLab/logs/rsl_rl/sf_copt/2026-06-26_05-26-28/designs/link_lengths_<iter>.csv`
carry no header. The five columns are `idx, thigh_R, thigh_L, knee_R, knee_L`,
each link a string-encoded `{x,y,z}` dict, and only `z` varies. `thigh_R`
equals `thigh_L` since both are driven by `thigh_length_scale`, and `knee_R`
equals `knee_L`, the shank, driven by `shank_length_scale`. The script
`scratchpad/analyze.py` parses the dicts with `ast.literal_eval` and computes
the per-iteration mean and standard deviation of thigh z and shank z across
the 64 individuals. Baseline thigh z is 0.25 and shank z is 0.30, with range
scale `(0.85, 1.15)`, so thigh z lies in `[0.2125, 0.2875]` and shank z in
`[0.255, 0.345]` before the 2-decimal rounding.

Table, mean and standard deviation of link z over the 64 individuals.

| iter | phase | mean_thigh | std_thigh | mean_shank | std_shank | thigh range | shank range |
|------|-------|-----------|-----------|-----------|-----------|-------------|-------------|
| 119 | random, scale ~0.05 | 0.25000 | 0.00000 | 0.30000 | 0.00000 | 0.2500-0.2500 | 0.3000-0.3000 |
| 959 | random growing, scale ~0.17 | 0.25109 | 0.00437 | 0.30078 | 0.00539 | 0.2400-0.2600 | 0.2900-0.3100 |
| 7799 | end random, scale ~0.99 | 0.24844 | 0.02101 | 0.29375 | 0.02820 | 0.2200-0.2900 | 0.2600-0.3400 |
| 10439 | CMA-ES early | 0.28047 | 0.00211 | 0.34000 | 0.00000 | 0.2800-0.2900 | 0.3400-0.3400 |
| 15119 | CMA-ES | 0.28203 | 0.00402 | 0.34000 | 0.00000 | 0.2800-0.2900 | 0.3400-0.3400 |
| 19799 | CMA-ES | 0.28000 | 0.00000 | 0.34000 | 0.00000 | 0.2800-0.2800 | 0.3400-0.3400 |
| 24959 | CMA-ES | 0.28000 | 0.00000 | 0.34000 | 0.00000 | 0.2800-0.2800 | 0.3400-0.3400 |
| 29999 | CMA-ES final | 0.28000 | 0.00000 | 0.34000 | 0.00000 | 0.2800-0.2800 | 0.3400-0.3400 |

Three findings emerge.

1. Growing spread during 0 to 8000. The standard deviation rises from zero at
   iter 119, through roughly 0.004 to 0.005 at iter 959, to roughly 0.021 for
   thigh and 0.028 for shank at iter 7799. This is the
   `_sample_scales_v2(rng, scale)` widening, where
   `scale = 0.95 * (g / (n - s)) + 0.05` opens from 5 percent of the full
   range to nearly 100 percent across the random phase. At iter 7799 the thigh
   spans 0.22 to 0.29 and the shank spans 0.26 to 0.34, essentially the full
   range. This growing morphology distribution would by itself inflate the
   plotted training variance over the first 8000 iterations, which is relevant
   to objective 1.

2. Collapse of the standard deviation after 8000. Once CMA-ES takes over, the
   spread collapses almost immediately. By iter 10439 the shank standard
   deviation is already zero and the thigh is near 0.002, and from iter 19799
   onward both standard deviations are exactly zero. The population is a single
   repeated design.

3. Drift of the mean to the upper bound. The mean thigh z moves from about
   0.25 to about 0.28, and the mean shank z moves from 0.30 to 0.34. The shank
   value 0.34 equals `round(0.30 * 1.15, 2)`, the rounded upper bound. The
   thigh value 0.28 sits one rounding bin below the rounded upper bound 0.29
   equal to `round(0.25 * 1.15, 2)`, consistent with a CMA-ES mean pushed to a
   thigh scale near 1.12 to 1.14 and then quantised. Both parameters are pinned
   at or just below their maxima rather than at an interior optimum.

Quantisation evidence. At iter 10439 the only distinct thigh values are
`{0.28, 0.29}` and the only distinct shank value is `{0.34}`. From iter 19799
the distinct sets are `{0.28}` and `{0.34}`. The 2-decimal rounding in
`_compute_link_extents` (`usd_generator.py` line 372,
`extents[link] = {key: round(val, 2) for ...}`) bins z to a 1-centimetre grid.
On a baseline of 0.25 to 0.30 metres, one bin is 4 percent of the link length,
which is larger than the entire late-stage CMA-ES spread, so any residual
genotype variety is erased before it reaches the simulator and the fitness.

## 8. Diagnosis, why exploration is poor

The collapse has two compounding causes, an optimisation-theory cause and a
discretisation cause, layered on a scheduling cause.

1. Boundary-dominated initialisation. `sigma0 = 0.75` on a unit box is three to
   four times the recommended value, and even after the maxstd cap to 0.333 the
   initial distribution piles mass on the bounds. Combined with a fitness that
   rewards larger legs, the mean walks to the upper bound and the saturating
   `BoundTransform` keeps it valid there, so the search converges to a corner of
   the box rather than to an interior optimum. This is the single most damaging
   factor, since it determines where the search goes.

2. Quantisation that destroys sub-centimetre variety. The `round(val, 2)` in
   `_compute_link_extents` snaps every individual to a 1-centimetre grid. Late
   CMA-ES proposes variations of a few millimetres, all of which round to the
   same value, so the 64 individuals become identical and the fitness becomes
   flat. A flat fitness removes the selection gradient and triggers `tolfun`
   and `tolflatfitness`, freezing the optimiser.

3. Range too narrow. The CMA-ES range `(0.85, 1.15)` is only plus or minus 15
   percent. Against a 1-centimetre quantisation grid this leaves only a handful
   of distinct thigh values and a handful of distinct shank values, which is
   too coarse a lattice for a covariance-adapting optimiser to navigate.

4. Cold restart after an unrelated random phase. CMA-ES initialises fresh at
   mean 0.5 and sigma 0.75 only after 8000 iterations during which the policy
   was trained on a wide and growing random design distribution. The policy is
   therefore adapted to a broad distribution of morphologies, not to the mean
   design, so the early CMA-ES fitness signal is biased and noisy, which is the
   morphological-innovation-protection problem of Cheney et al. 2018. The
   late-start `toggle` flips to true designs with no warm transfer of the
   distribution that CMA-ES should start from.

5. No noise handling and no restart. The generator uses the bare
   `CMAEvolutionStrategy`, which cannot restart, since
   `evolution_strategy.py` lines 1070 to 1071 raise a `ValueError` if
   `restarts` is requested on the class. Restarts and population growth are
   available only through `cma.fmin` or `cma.fmin2` with `restarts` and
   `incpopsize=2` for IPOP, and the `bipop` option for BIPOP. Noisy fitness is
   handled by `cma.optimization_tools.NoiseHandler`
   (`optimization_tools.py` line 528), which is not wired in. Without these the
   first premature collapse is permanent.

## 9. Recommended initialisation and fixes

Recommended CMA-ES initialisation for the `[0,1]^2` encoding.

- `sigma0 = 0.25`, near one quarter of the unit range, so that `x0 ± 2 sigma0`
  brackets the box without overflowing it. A value in `[0.2, 0.3]` is
  acceptable, and `0.3` is the upper limit imposed anyway by the maxstd cap.
- Initial mean `x0 = np.full(2, 0.5)`, the bias-free centre, which maps to
  scale 1.0 and is the baseline morphology. Retain this, since the centre is
  the correct prior when no informative guess exists. If a warm start from the
  best random-phase designs is desired, seed `x0` with the normalised centroid
  of the top random-phase individuals, or use `es.inject` for that centroid.
- Keep `bounds = [zeros(2), ones(2)]`, and consider setting `BoundaryHandler`
  explicitly to make the choice visible. `BoundTransform`, the current default,
  is the right choice for a tight box with active bounds, so the fix is to
  document it rather than change it.

Range and quantisation fixes, which matter as much as `sigma0`.

- Remove or greatly reduce the `round(val, 2)` quantisation in
  `_compute_link_extents`. Rounding to 4 decimals, or no rounding at all, keeps
  sub-centimetre variety so that the fitness retains a selection gradient and
  CMA-ES does not see flat fitness.
- Widen `CMAES_PARAM_RANGES` beyond plus or minus 15 percent, for example to
  `(0.75, 1.25)` or wider, so the optimum is more likely interior and the
  effective lattice after any rounding is finer.

Algorithmic and scheduling fixes.

- Consider `CMA_stds` to give the two coordinates different initial step-sizes
  if thigh and shank have different sensitivities, since `CMA_stds` multiplies
  `sigma0` per coordinate and is held in `sigma_vec` outside C.
- Wrap `tell` in `cma.NoiseHandler`, or lengthen `ea_update_interval`, so each
  individual completes enough episodes for a low-variance fitness estimate
  before the rank-based update.
- For multimodal gait regimes, route the outer loop through `cma.fmin2` with
  `restarts` and `incpopsize=2` for IPOP, or the `bipop` option, so a premature
  collapse is recovered by a larger restart rather than frozen.
- Increase `popsize` above `num_individuals` only if the environment capacity
  allows, since larger lambda improves global search at the cost of more
  simulation per generation. The present `popsize = 64` is already generous for
  `N = 2`, so the binding constraints here are `sigma0`, the quantisation, and
  the range, not the population size.

## 10. Citations

- Hansen, N. (2016). The CMA Evolution Strategy, A Tutorial. arXiv:1604.00772.
  Source of the `sigma0` quarter-of-range rule, the `x0 ± 3 sigma0` expected
  optimum statement, the CSA step-size update, and the default hyperparameter
  table.
- Hansen, N., Akimoto, Y., Baudis, P. (2024). CMA-ES/pycma on GitHub, package
  `cma` version 4.4.4. https://github.com/CMA-ES/pycma. Behaviour confirmed in
  the installed source under `/usr/local/lib/python3.11/dist-packages/cma`,
  specifically `options_parameters.py` for option defaults including
  `BoundaryHandler='BoundTransform'` and `maxstd_boundrange='1/3'`,
  `evolution_strategy.py` lines 1129 to 1161, 1350, 1506 to 1545, 2963, and
  4140 to 4269 for boundary-handler instantiation, the maxstd cap, and the
  stop criteria, `boundary_handler.py` line 454 for `BoundTransform`, and
  `optimization_tools.py` line 528 for `NoiseHandler`.
- pycma API documentation,
  https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html.
  Source of the statement that variables should be scaled so a single standard
  deviation is useful and the optimum lies within about `x0 ± 3*sigma0`.
- Cheney, N., Bongard, J., SunSpiral, V., Lipson, H. (2018). Scalable
  co-optimization of morphology and control in embodied machines.
  J. R. Soc. Interface 15:20170937. Source of the
  morphological-innovation-protection argument relevant to the cold-restart
  diagnosis.
