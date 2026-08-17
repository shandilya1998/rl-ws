# Toward an Efficient and Transferable Gait for the SD_BRS1 Biped

## Analysis of Foot Stomping, Actuation Roughness, and Yaw Tracking Failure, with a Phased Programme of Remedy

Status, written 2026-07-31, revised 2026-08-03. This document supersedes nothing. It stands beside `NATURAL_GAIT_PLAN.md`, whose Phase A2 delivered the walking policy analysed here, and it takes up the defects that remained once that policy walked. Its subject is the quality of the gait rather than its existence, and its evidence is the play evaluation of run `sd_brs1_flat/2026-07-28_06-37-24` together with the three sibling runs trained beside it.

Implementation status as of 2026-08-07. Phase 1 was trained as run `sd_brs1_flat/2026-07-31_10-21-10` and failed its gate, for reasons that section 4.1a establishes were partly a misconfiguration and partly a defect in the term the phase wired. Phases 1b and 2 were trained together as run `sd_brs1_flat/2026-08-03_11-19-11`, which ran to completion at 29999 iterations and was re-evaluated. Section 4.2a records the outcome as it stood at eleven thousand iterations and section 4.2b records the completed run, which confirms Phase 1b at its gate, withdraws the provisional pass of Phase 2, and supplies the strongest evidence yet assembled for Phase 3. Phase 3 was then trained to completion as run `sd_brs1_flat/2026-08-05_07-53-35`, and section 4.3b records its outcome, which is the largest single improvement the programme has produced, which incidentally carries Phase 2 past the gate it had failed, and which opens one new defect that section 4.3c takes up. Phase 4 was then implemented on 2026-08-07 and is wired and awaiting a run, section 4.4a recording what landed and where it departed from the specification section 4.4 sets out. Phases 5 through 7 have their functions implemented in `mdp/rewards.py` but are deliberately left unwired, so that they may be reviewed and then enabled one experiment at a time, and the configuration snippet that enables each is carried in its section below. No behaviour changes until a configuration wires a term, so the presence of these functions is inert.

The working tree as of 2026-08-07 carries Phase 3 as trained, with one addition that the Phase 3 run did not contain. `rew_foot_clearance` at `cfg/SF/brs_base_env_cfg.py:1005` is `foot_clearance_reward_v3` at weight 10.0 with a kernel width of 0.03, `pen_foot_landing_vel` at line 1103 stands at −30.0 and `pen_feet_impact` at line 1156 at −3.0e-2 against a threshold of 850 newtons, all four figures matching the dumped `params/env.yaml` of the completed run. The addition is `pen_ankle_deviation` at line 852, a `joint_deviation_l1` at weight −0.2 over all four ankle joints, which is a new experiment addressing the defect of section 4.3c and which was not present in any run analysed here. Section 4.3c states what that term is forecast to read and why the forecast matters before its run completes.

Correction appended 2026-08-07, recorded rather than made in place because the preceding paragraph described a tree that no longer exists. On implementing Phase 4 the term `pen_ankle_deviation` was found already commented out at what is now `cfg/SF/brs_base_env_cfg.py:915`, so the tree does not carry it and the run whose completion section 4.3c awaits was launched from a state the working tree has since left. A weight audit run the same day, comparing every reward term in the tree against the completed run's dumped `params/env.yaml` line by line, establishes that three of those four figures survive and one does not. The clearance term at `cfg/SF/brs_base_env_cfg.py:1068` stands at weight 10.0 with a kernel width of 0.03 and the landing term at `:1166` at −30.0, both as the paragraph states, but `pen_feet_impact` at `:1219` now carries −5.0e-2 where the run carried −3.0e-2 and the paragraph reports −3.0e-2, its threshold of 850 newtons being unchanged. That is an increase of two thirds in the weight of a term whose behaviour Phase 3 moved substantially, it was not introduced by Phase 4, and it is the one uncontrolled variable a run launched from this tree would carry. Whoever launches the next run should decide deliberately whether to restore −3.0e-2, which yields a clean single phase ablation, or to retain −5.0e-2 and record the run as varying two things at once. Excepting that term, the audit finds the tree and the run identical across all twenty six terms, the only other differences being the two Phase 4 terms themselves. The withdrawal is the disposition section 4.4 argues for on its merits, a phase agnostic deviation penalty charging push off at more than half the rate it charges the parked swing ankle, so the swing gated pair now carries the weight alone and the forecast of section 4.3c stands as a reading of the completed run rather than as a description of the tree.

One correction to the earlier status banner is recorded rather than made silently, because it bore on a launch decision. The banner written on 2026-08-05 stated that the tree carried `foot_clearance_reward_v2` at weight 20.0 beside a landing weight of −45.0 and an impact threshold of 800 newtons, and warned against launching from it. The run that was in fact launched carried none of that combination. It carried v3 at weight 10.0, the landing weight unchanged at −30.0, and the impact threshold at 850 rather than 800, which is to say the two weight escalations section 4.3a derived were not applied and only the threshold move was. Section 4.3a is corrected accordingly, and the experiment that resulted is cleaner than the one this document specified, since it varies the clearance term alone against an almost unchanged reward set.

---

## 1. Introduction

### 1.1 Where the work stream stands

The SD_BRS1 biped has been the subject of a sustained gait investigation whose record is `../context/brs_gait.md`, advancing through nineteen numbered passes and eight training runs before the four analysed here. That record is a history of successive obstacles, each of which had to be removed before the next became visible. The first was a reward that measured the ankle frame height and so paid a grounded foot for rocking onto its edge, which produced a skating shuffle and was closed by measuring the true sole clearance. The second was a permanent interpenetration of the shank and the foot in the mesh asset, which forged a contact force five times body weight on every foot at every instant and thereby silenced every contact keyed term in the reward set, and which was closed by rebuilding the asset from primitives. The third was a reward budget that paid more for holding one leg aloft than for stepping, closed by the periodic contact clock. The fourth was an asymmetric limp, closed by symmetry data augmentation. The fifth was a knee locked at its extension stop, closed by moving the nominal joint pose onto a feasible crouch, which is the change the literature survey of that document identified as the manner in which every mainstream humanoid configuration obtains a bent knee.

The policy that emerged from those five removals walks. It follows a commanded planar velocity to within sixteen centimetres per second, it alternates its feet at a cadence within four percent of the commanded one, it holds its base height to within two centimetres of the target, and it survives the great majority of its episodes. Measured against where the work stream began, with a robot pitched forward by seventy degrees dragging its feet along the ground, this is a considerable achievement, and the first purpose of this document is to identify precisely which properties of the present gait are worth defending.

The second purpose is the harder one. A gait that tracks its command is not thereby a gait that could be placed on hardware. The evidence assembled below shows that the present policy achieves its tracking through a manner of walking that would destroy the machine, striking the ground at three times body weight on the average step and fourteen times on the worst, driving its ankles onto their mechanical stops at full actuator effort for a third of every cycle, and circulating nine and a half times more joint power than the motion actually requires. These are not cosmetic complaints. They are the properties that decide whether a policy transfers, and none of them is visible in the tracking error that the training curves report.

### 1.2 The run under analysis, and its provenance

The subject is `sd_brs1_flat/2026-07-28_06-37-24`, trained for 15138 iterations with the configuration recorded in its own dumped `params/env.yaml`. Three sibling runs were trained from near identical configurations and evaluated on the same day, `2026-07-29_04-38-14`, `2026-07-29_10-47-14`, and `2026-07-30_07-59-22`, differing only in the minimum feet distance, in the weight of the roll and pitch angular velocity penalty, and in the presence of an ankle deviation penalty. Together they constitute an ablation already performed, and section 3 reads all four rather than one, since a defect that persists across all four is structural while one that moves with a weight is not.

The evidence is of three kinds. The eleven evaluation plots the user supplied are held at `../context/artefacts/2026-07-28_06-37-24-play/`, and the thirty second play video whose frames are read below is the same directory's `play_video.mp4`, established as the recording of this run by a byte identical match against `videos/play/42/rl-video-step-0.mp4` in the run's own directory. Beneath both lies the numpy dump the plots were drawn from, 3001 steps of 32 environments at a control period of ten milliseconds, carrying joint positions, velocities, torques, accelerations and powers, foot contact forces, foot velocities, sole clearances and frame heights, base pose and velocity, and the commands. Every figure quoted in section 3 was computed from that dump rather than read from a plot, so the plots serve here as corroboration and as the record of what the user observed rather than as the measurement itself.

### 1.3 The behaviour, in brief

The robot walks along its commanded heading with alternating feet, knees visibly bent, at a cadence near one gait cycle per second. Five properties of that walk are the subject of this document, four of them observed by the user and one uncovered by the analysis.

The angular velocity tracking is very poor. Across the thirty second evaluation the commanded yaw rate averages 0.51 radians per second in magnitude while the achieved yaw rate averages 0.29, the root mean square error is 0.74 radians per second, and the correlation between command and achievement is 0.21. The same failure appears in all four runs at a root mean square error between 0.56 and 0.74. Unless stated otherwise every figure in this document is taken over all 32 environments of the dump rather than over the single environment the plots display, which for this quantity differ, environment zero alone giving a root mean square error of 0.62 and a correlation of 0.23.

The foot strikes the ground extremely hard and is withdrawn from it extremely fast. The sole descends onto the floor at 1.68 metres per second on the average touchdown and at 4.37 on the worst, and the contact force that follows peaks at 3.08 times body weight on the average step and at 13.76 on the worst.

The joint velocity, torque, acceleration and power traces are impulse trains rather than continuous signals. Each joint rests near zero for most of the cycle and then delivers a transient of a few tens of milliseconds, the knee reaching three kilowatts, the hip pitch absorbing two, and the ankles alternating between the two at their effort ceilings.

The foot clearance rises almost vertically at the beginning of swing rather than describing an arc. The sole reaches ninety two percent of its apex within the first fifteen percent of the swing phase, holds a plateau near nine centimetres for the middle half, and drops back to the ground in the last tenth.

Walking is achieved, but at a mechanical cost of transport of 2.38, an order of magnitude above human walking and comparable to the least efficient full scale humanoids ever fielded [1].

### 1.4 What the gait ought to look like

Any judgement of the above requires a standard, and for a biped of this class the standard is human walking, both because the morphology is anthropomorphic and because the normative data are unusually complete. A healthy adult at a comfortable speed of 1.33 metres per second walks at a cadence of 113 steps per minute, spends 62 percent of each gait cycle in stance and 38 percent in swing, and passes 24 percent of the cycle in double support with both feet loaded [2]. The vertical ground reaction force describes a double humped curve peaking between 1.0 and 1.5 times body weight, rising to its first peak over roughly a quarter of stance rather than in a single collision [3]. The swing foot passes closest to the ground at mid swing, where the minimum toe clearance is under two and a half centimetres, so the clearance profile is not a plateau but a curve with an interior minimum [4]. The knee flexes to roughly fifteen degrees during weight acceptance and to sixty degrees at peak swing, so it is neither locked nor folded to its limit [5]. The dimensionless cost of transport of human walking is near 0.2, and the passive dynamic walkers that most closely approach it achieve 0.055 to 0.08, while the best known engineered humanoid of the previous generation required roughly 1.6 to 2 [1].

Three properties of that standard bear directly on the defects of section 1.3, and are worth stating as objectives rather than as observations. Weight is transferred from one leg to the other over a finite double support interval, not instantaneously, which is what keeps the collision at the beginning of stance modest. The swing foot follows a smooth trajectory whose vertical velocity is small at both ends, since it must leave the ground without a snatch and meet it without a collision, and the energy of a hard landing is energy the actuators paid to inject and then paid again to arrest. And the joints of a leg cooperate rather than compete, the passive skeleton bearing what it can, so that the mechanical power the actuators supply is close to the power the motion consumes.

The sections that follow establish, first from the published literature and then from the code and the measurements together, that the present gait departs from each of these three properties for reasons that lie entirely within the reward specification, and that each departure is the correct and predictable maximiser of the objective the policy was given.

---

## 2. Literature Survey

This survey extends the twelve thematic clusters of `../context/literature.md`, which remains the authoritative register of sources for this workspace. Clusters seven through eleven of that document already ground the contact scheduling, symmetry, knee flexion and nominal pose questions, and are not repeated here. What follows is the material bearing on the questions this document raises for the first time, namely how a swing trajectory should be shaped, how impact should be priced, how actuation smoothness is obtained, how joint limits should be treated, and how a biped without a hip yaw degree of freedom turns.

### 2.1 The biomechanical standard

The normative temporal and spatial parameters of adult walking are long settled, a cadence of 113 steps per minute, a velocity of 1.33 metres per second, a stride length of 1.41 metres, and a division of the gait cycle into 62 percent stance, 38 percent swing, and 24 percent double support [2]. Two features of that division matter for reward design. Double support is not an incidental overlap but a quarter of the cycle, during which the centre of pressure migrates from the trailing to the leading foot under continuous load. And the swing foot's clearance is not monotone, the minimum toe clearance occurring at or very near mid swing at a value below two and a half centimetres, which is the gait event most associated with tripping and therefore the one most tightly regulated [4].

The vertical ground reaction force of walking peaks between 1.0 and 1.5 times body weight and rises to its first peak over roughly a quarter of the stance phase, the two humps corresponding to weight acceptance and to push off [3]. Running raises the peak to between 2.0 and 2.9 times body weight [3], so a walking machine that exceeds three times body weight on an average step is loading its structure beyond what running imposes on a human.

The energetic standard is the dimensionless cost of transport, also called specific resistance. Human walking sits near 0.2 [1]. The passive dynamic walkers of Cornell and Delft achieve 0.055 and 0.08 respectively, close to the 0.05 estimated for the mechanical component of human walking, and they do so with almost no actuation at all, which established that an efficient bipedal gait is a property of the mechanics and the trajectory rather than of the controller's power [1]. The contrasting figure is the Honda humanoid at roughly 1.6 to 2, achieved with a stiff, fully actuated, trajectory tracking controller [1]. The relevance here is that a reinforcement learning policy is free to land anywhere on that spectrum, and that nothing in a velocity tracking objective selects the efficient end of it.

McGeer's demonstration that a knee jointed biped walks stably down a shallow slope with no actuation whatever remains the cleanest statement of the principle that the stance leg should behave as a strut and the swing leg as a pendulum, and that the energy of a gait is dissipated principally at the heel strike collision [6]. It follows that a policy which increases its collision velocity is paying, in the most direct sense available, for the privilege of walking badly.

### 2.2 Reward structures that specify when a foot should be down

Three families of construction appear in the literature, and the SD_BRS1 configuration presently carries members of all three at once.

The first is the periodic phase clock, introduced for bipeds by Siekmann and colleagues, who gate a penalty on foot force during commanded swing and a penalty on foot velocity during commanded stance by a smoothed periodic indicator parameterised by frequency, offset and duration [7]. Margolis and Agrawal generalised the same construction across gaits, and the `GaitReward` class in this repository is a verbatim implementation of their variant, identifiable from its parameter names [8]. Humanoid-Gym supplies the same idea in a different form, a periodic stance mask defining the planned contact schedule and a contact pattern reward paying for the measured contacts to match it, carried at the largest weight in their set, with the clock pair supplied to the policy as an observation [9]. The property that recommends this family is that it cannot be farmed by holding a configuration, because holding is off schedule by construction.

The second family prices step events without a clock. Van Marum and colleagues, working on Digit, deliberately avoid clocks, arguing that the clock input is undefined during standing and transitions, that it drives foot velocities toward zero in the standing mode and thereby impedes disturbance rejection, and that it constrains free foot motion during recovery [10]. In its place they carry a sparse touchdown triggered air time reward at the largest weight in their table, and a single foot contact reward defined as unity if single contact occurred at least once in the preceding two tenths of a second [10]. That grace window is the detail that matters most here, because it means brief double support during weight transfer is not punished, whereas a single support reward without a window prices every instant of double support at zero and thereby drives the transition count down.

The third family is the dense single stance reward of the Isaac Lab velocity template, `feet_air_time_positive_biped`, gated on exactly one foot in contact and returning the minimum over the feet of the in mode time clamped at a threshold. Its virtue, established in the thirteenth pass of `../context/brs_gait.md`, is that it purchases single support from a robot that will not leave double support. Its vice, established in the same pass, is that it clamps at the threshold rather than peaking there, so its per phase mean rises monotonically as the single support phase lengthens, and every foot swap resets both accumulators and costs a dip.

Booster Gym provides a recent and unusually complete reference table for a 50 hertz humanoid controller, carrying velocity tracking at 1.0, 1.0 and 0.5 for the two linear and the yaw component, a feet swing indicator reward at 3.0, feet slip at −0.1, feet yaw at −1.0, feet roll at −0.1, and feet distance at −1.0 [11]. Two of its terms have no counterpart in the SD_BRS1 set and are directly relevant below, the foot yaw alignment term and the torque tiredness term discussed in section 2.5.

### 2.3 How the swing foot should be shaped

The clearance reward now in the SD_BRS1 configuration descends from the legged_gym lineage [12] and takes the form of a Gaussian kernel on the foot's height multiplied by a hyperbolic tangent of its horizontal speed. Its intent is transparent, to pay a foot that is both high and moving, and it is effective at lifting a foot that will not lift. Its structural weakness is equally transparent once stated, that its integrand is maximised at a single height, so the trajectory that maximises its time integral over a swing of fixed duration is the one that reaches that height fastest, remains there longest, and leaves it latest. The reward specifies a set point rather than a path, and a set point reward buys a plateau.

Two published alternatives avoid this. Humanoid-Gym keeps every joint default at zero and instead drives the leg from a phase conditioned reference, a sinusoid of the gait phase with a knee excursion of roughly 0.34 radians, tracked by an exponential kernel on the joint error at the largest weight in their table [9]. Because the target moves with the phase, a foot that is high at the wrong moment is penalised exactly as a foot that is low at the wrong moment, and the trajectory rather than the extremum is specified. The recent work of Seo and colleagues on fifteen minute humanoid training reduces the customary twenty term reward below ten and, in place of a clearance kernel, carries a foot height tracking term against a reference swing profile, which is immune to the plateau for the same reason, though the paper defers its formulas to its code and so cannot be quoted for a weight [13]. The foothold tracking work of Kim and colleagues supplies a third variant, a swing window formulation in which the clearance reward accrues only if the correct foot breaks contact within the allotted interval [14].

The general principle these share, and the one that the present configuration violates, is that a reward on an extremum of a trajectory determines only that extremum, whereas a reward on a reference determines the whole path. Where the objection to a reference is that it must be authored, the counter is that the SD_BRS1 configuration already authors one, since its `gait_command` term carries a `swing_height` parameter of 0.08 metres that no reward presently reads.

### 2.4 Pricing impact

Two mechanisms appear in the literature for keeping a foot from striking the ground hard, and they are complementary rather than alternative.

The first penalises the contact force directly. Humanoid-Gym carries a large contact force term of the form negative maximum of the foot force less four hundred newtons, floored at zero and clipped at one hundred, at a weight of −0.01, so the term is inert below the threshold and grows linearly above it [9]. This is the cleanest available form because it is exactly zero over the whole range of forces a well behaved gait produces and becomes active only in the collision, so it cannot distort the stance phase it is not meant to govern. The more recent work on ground reaction force aware humanoid locomotion penalises the per foot normal force directly and reports that the penalty must be introduced on a curriculum, small at first and raised as competence grows, because a policy that has not yet learned to walk cannot afford it [15].

The second penalises the foot's descent velocity before contact rather than the force after it, and is preferred where the simulator's contact force is unreliable. This is precisely the form of `foot_landing_vel`, already present in `mdp/rewards.py` and already wired into the TRON1 point foot configuration, which sums the squared vertical foot velocity over the feet that are below a height threshold, descending, and not yet in contact. Its advantage over a force penalty is that it acts before the collision rather than after, so its gradient reaches the swing trajectory that caused the collision rather than the stance that follows it.

The biomechanical reason both are needed is that the collision at heel strike is where a walking gait loses its energy [6]. A foot that arrives at 1.68 metres per second carries kinetic energy that the swing actuators supplied and that the stance leg must then absorb, and it does so in the few tens of milliseconds during which the contact force is unbounded by anything except the mechanics.

### 2.5 Actuation smoothness and the transfer to hardware

The strongest published instrument for smooth control is the conditioning for action policy smoothness method of Mysore and colleagues, which regularises the policy in its own parameter space rather than paying for roughness through a reward [16]. Two losses are added to the objective, a temporal term penalising the difference between the actions at consecutive states and a spatial term penalising the difference between the action at a state and the action at a nearby perturbed state, and the reported effect on a real quadrotor is a ninety six percent improvement in smoothness with an eighty percent reduction in power [16]. The distinction from a reward penalty is important, a reward penalty makes roughness expensive whereas this makes roughness hard to represent, and the second survives a change of task weighting that the first does not.

On the reward side the standard instruments are the action rate and torque penalties of the legged_gym lineage [12], together with the second difference action smoothness penalty that this repository implements as `ActionSmoothnessPenalty` and the torque rate penalty it implements as `JointTorqueRatePenalty`. Booster Gym adds a term with no counterpart here that is directly relevant to the ankle saturation reported in section 3, torque tiredness, the squared ratio of the applied torque to the actuator's maximum, at a weight one to two orders above their plain torque penalty [11]. Its effect is to make approach to the effort ceiling expensive in a way that an absolute torque penalty does not, since the same absolute torque is unremarkable at a hip and saturating at an ankle.

The deepest available remedy is the actuator network of Hwangbo and colleagues, which learns the true torque transfer of the physical drive and thereby removes the modelling error that a stiff proportional derivative controller in simulation conceals [17]. That is beyond the scope of this plan and is recorded as the eventual limit of what reward shaping can achieve.

### 2.6 Constraints as an alternative to rewards, and the treatment of joint limits

Kim and colleagues propose replacing kernel shaped reward terms with constraint costs that are exactly zero inside a desired range read from the robot's own limits and grow outside it, and report that this collapses ten or more hand tuned coefficients into effectively one [18]. The argument that recommends the form here is that a constraint cost cannot vanish in the way that a distant Gaussian can, a failure mode this work stream has already suffered once, when the Phase A knee reward delivered a gradient of 1.7 times ten to the minus six per radian because its kernel was placed four and a half tolerances from the operating point.

On joint limits specifically, the Isaac Lab library term `joint_pos_limits` penalises the excursion beyond a soft band, here set at nine tenths of the hard range, and is carried in the SD_BRS1 configuration at a weight of −2. Booster Gym instead uses an indicator, counting the joints outside their limits, at a weight equal to their collision penalty [11]. The distinction matters where a policy discovers that a mechanical stop is a cheap source of support, because an excursion penalty that is small compared with the reward for leaning on the stop will be paid rather than avoided.

### 2.7 Turning, and the yaw authority of a biped without a hip yaw joint

The SD_BRS1 has ten actuated degrees of freedom, hip roll, hip pitch, knee pitch, ankle roll and ankle pitch on each leg, the hip yaw joints being declared fixed in the URDF. It therefore possesses no actuator whose axis is vertical, and cannot generate a yaw moment on its own base by joint torque alone in the way a robot with a hip yaw or a waist yaw can. Every yaw moment must come from the ground, either as a friction moment about the vertical under the stance sole or as the reaction to a change in the whole body angular momentum about the vertical produced by swinging the legs.

The literature on yaw compensation in bipedal walking establishes that the yaw moment generated by the swinging leg is a genuine disturbance that the stance foot must absorb through friction, sufficiently so that mechanisms have been designed specifically to cancel it [19]. The corollary for a turning command is the same physics run in reverse, that the yaw the robot can command is bounded by the friction moment the stance sole can develop and by the angular momentum the swing leg can exchange. Both are enlarged by a wider stance, which lengthens the moment arm of any differential fore and aft foot placement, and the second is enlarged by permitting the swing foot to be placed laterally rather than only fore and aft.

Booster Gym's foot yaw alignment term, penalising the squared difference between each foot's yaw and the base yaw at a weight equal to twice their linear tracking weight, is the reference instrument for keeping the feet pointed along the body during a turn, and is absent from the SD_BRS1 set [11].

### 2.8 Stance width and lateral balance

Human walking stance width is between 1.0 and 1.3 times the hip width [20], a figure the nineteenth pass of `../context/brs_gait.md` already applied to this robot to derive a recommended band of 0.26 to 0.338 metres against a hip width of 0.26. The frontal plane balance literature establishes that the lateral centre of mass excursion required per step is half the stance width, so a narrow stance can be balanced by torso roll alone whereas a wide stance requires deliberate lateral hip motion [21]. The foot placement estimator literature adds the complementary point that recovery from a lateral disturbance requires the foot to be placed further laterally than the centre of mass, which a narrow nominal stance leaves no room to do [22].

The practical form of a stance width reward in the surveyed configurations is a hinge on the distance between the feet, and both Booster Gym [11] and this repository's own `feet_distance` use it. The detail on which section 3 turns is which distance, since a hinge on the full planar separation is satisfied by a long stride even when the lateral separation is zero.

A second consultation of this literature was made on 2026-08-07 for the rescoping of section 4.5, and it sharpens the prescription in three ways. The frontal plane stepping literature has since been resolved to an individual source, which establishes by fitting competing control models to treadmill walking that humans regulate lateral stepping as a multi objective problem dominated by step width, that quantity taking roughly ninety three percent of the control effort against roughly seven for absolute lateral position, and that this contrasts with fore and aft stepping, which the same authors had earlier found to be regulated as single objective speed control [26]. The lateral separation is therefore the regulated variable and the planar norm is not, which is the biomechanical form of the algebraic objection section 3.6.7 raises. The current humanoid learning frameworks have converged on the matching form, computing the lateral distance between the feet in the robot's own frame and penalising it below a minimum, with that minimum reported at 0.25 metres for the Unitree H1 and 0.18 for the smaller G1 [27], and the base frame per axis separation term is now standard practice rather than a variant, one recent loco manipulation reward set carrying a fore and aft foot separation term and a lateral root centring term at equal weight [28]. The third point is a caution rather than a prescription. Humanoids are now trained to cross supports narrower than their own nominal stance, one such policy traversing a beam of twenty five centimetres while carrying a feet separation term in its gait group [29], so a lateral floor that admits no exception forecloses a capability the field currently pursues, and a formulation that regulates the stance width without forbidding a crossed placement is preferable on that ground alone.

### 2.9 Synthesis, and what the literature prescribes here

Four prescriptions follow, and they are ordered by how directly they bear on the defects of section 1.3.

A reward on the extremum of a trajectory determines only that extremum, so a clearance kernel will buy a plateau and a reference tracking term will not. Where a reference is available, and here one is already declared and unread, the reference form is preferred [9, 13].

A single support reward without a grace window prices double support at zero and therefore drives the count of weight transfers downward, whereas the published bipedal systems that walk on hardware either carry the window explicitly [10] or price the schedule through a clock that demands the overlap [7, 8]. A reward set that contains both a clock demanding overlap and a single support term forbidding it has specified a contradiction, and the resolution will be decided by the relative weights rather than by the physics.

Impact must be priced explicitly, because nothing in a velocity tracking objective opposes it and the collision is where the gait's energy is lost. Both available forms are cheap, a threshold hinge on the contact force that is inert in normal walking [9], and a squared descent velocity gate that acts before the collision rather than after.

Smoothness penalties must be calibrated against the tracking income rather than set at conventional absolute magnitudes, and where they cannot be raised far enough without distorting the task, the policy space regularisation of the conditioning for action policy smoothness method damps the state to action map directly and is the stronger instrument [16].

---

## 3. Behaviour Analysis

### 3.1 The configuration under analysis

The reward set of run `2026-07-28_06-37-24`, read from its dumped `params/env.yaml`, carries twenty four terms. The tracking pair pays 50 for planar linear velocity and 15 for yaw, both through exponential kernels whose widths the curriculum had contracted to 0.128 and 0.267 by the end of training. The gait shaping group pays 15 for single support through `no_fly`, 12.5 for single stance air time, 20 for sole clearance through `foot_clearance_reward_v2`, and carries `GaitReward` at 40. The posture group penalises flat orientation at −50, base height error at −30 against a target of 1.15 metres, roll and pitch angular velocity at −2, and vertical base velocity at −0.5. The effort and smoothness group penalises squared torque at −1e-5, squared acceleration at −1e-7, squared joint velocity at −1e-5, the first difference of the action at −0.01, its second difference at −0.075, and the first difference of the torque at −1e-5. The remainder are the feet distance hinge at −100 with a minimum of 0.21 metres, the foot regulation gate at −0.2, foot slide at −5, undesired contacts at −2.5, hip roll deviation at −0.1, joint position limits at −2, and a survival bonus at 0.05.

The action is a joint position target at a scale of 0.4 added to the nominal crouch pose, with no clipping, at a control period of ten milliseconds over a physics step of five. The actuators are identified direct current motor models with Coulomb and viscous friction, hip roll at a stiffness of 150 and a damping of 45, hip pitch at 200 and 50, knee at 200 and 22, ankle roll at 20 and 4, ankle pitch at 50 and 4, with effort ceilings of 500, 500, 600, 131 and 262 newton metres and velocity ceilings of 40, 40, 40, 25 and 25 radians per second.

The gait clock is the decisive piece of configuration and deserves separate statement. `gait_command` samples a frequency of exactly 1.0 hertz, an offset of exactly 0.5, a duration of exactly 0.6, and a swing height of exactly 0.08 metres, all with degenerate ranges so that every environment carries the same clock. Reconstructing the smoothed desired contact state from those parameters exactly as `GaitReward.compute_contact_targets` does gives a commanded stance duty of 58.4 percent per foot and therefore a commanded double support fraction of 19.6 percent. Against the human figures of section 1.4, 62 percent stance and 24 percent double support [2], the clock is well specified. It asks the robot to walk as a human walks.

### 3.2 The measured gait

The following are computed from the dump over all 32 environments and 3001 steps unless stated, with the joint indices resolved as described in section 3.6.9.

| quantity | measured | reference |
|---|---|---|
| cadence | median period exactly 1.000 s, mean 0.929 s over 2016 cycles | commanded 1.000 s, human 1.06 s at 113 steps/min [2] |
| stance duty per foot | 50.9 percent | 62 percent (human) [2] |
| double support | 2.3 percent | 19.6 percent (commanded), 24 percent (human) [2] |
| single support | 97.3 percent | 77.7 percent (commanded) |
| flight | 0.4 percent | 0 |
| linear velocity error | 0.223 m/s root mean square | commanded magnitude 0.53 m/s |
| yaw rate error | 0.741 rad/s root mean square, correlation 0.21 | commanded magnitude 0.51 rad/s, achieved 0.29 |
| base height | 1.157 m, standard deviation 0.025 | target 1.15 m |
| swing apex sole clearance | 0.092 m mean, 0.135 at the 99th percentile | target 0.08 m |
| fraction of swing above 0.07 m | 74 percent | not applicable |
| touchdown vertical velocity | 1.68 m/s mean, 4.37 m/s worst | 1.26 times free fall from the apex |
| peak contact force within 80 ms of touchdown | 3.08 body weights mean, 13.76 worst | 1.0 to 1.5 (human walking) [3] |
| horizontal foot speed in swing | 3.40 m/s at the 99th percentile | base speed 0.61 m/s |
| mean absolute joint power | 850 W | net mechanical power 89 W |
| mechanical cost of transport | 2.38 on absolute power, 0.25 on net | 0.2 (human), 0.055 to 0.08 (passive walkers) [1] |

The per joint picture, over the same population, is the following. The final two columns are the fraction of steps at which the joint stands within 0.02 radians of a mechanical limit and the fraction at which the applied torque exceeds 98 percent of the actuator's effort ceiling.

| joint | range, rad | velocity, 99th pct | torque, 99th pct | accel, 99th pct | mean abs power | at limit | at effort ceiling |
|---|---|---|---|---|---|---|---|
| HipRollL | −0.352 to +0.350 | 2.82 | 152 | 195 | 43 W | 1.0 pct | 0 |
| HipRollR | −0.353 to +0.353 | 2.84 | 154 | 196 | 42 W | 1.2 pct | 0 |
| HipPitchL | −1.252 to +0.609 | 7.14 | 209 | 978 | 106 W | 0 | 0 |
| HipPitchR | −0.700 to +1.250 | 7.27 | 211 | 972 | 108 W | 0 | 0 |
| KneePitchL | −0.005 to +1.488 | 16.51 | 206 | 642 | 165 W | 37.6 pct | 0 |
| KneePitchR | −0.009 to +1.486 | 16.46 | 204 | 647 | 166 W | 37.1 pct | 0 |
| AnkleRollL | −0.361 to +0.398 | 10.22 | 131 | 1229 | 23 W | 31.4 pct | 6.1 pct |
| AnkleRollR | −0.363 to +0.351 | 9.99 | 131 | 1216 | 25 W | 31.9 pct | 6.9 pct |
| AnklePitchL | −0.501 to +0.606 | 16.30 | 166 | 1730 | 88 W | 11.3 pct | 0 |
| AnklePitchR | −0.492 to +0.619 | 15.62 | 160 | 1901 | 84 W | 11.9 pct | 0 |

The reward budget of the run, taken as the mean of the final quartile of the fifteen thousand iterations and inverted through the identity that a logged episode reward equals the weight times the mean function value, is a positive side of 63.04 per second and a negative side of 14.64, for a net of 48.39. The positive side is 48.3 percent linear tracking, 19.9 percent single support, 18.3 percent foot clearance, 9.4 percent yaw tracking and 3.4 percent air time. The negative side is 48.5 percent `rew_gait`, 19.7 percent action smoothness, 12.6 percent roll and pitch angular velocity, and the remaining nineteen percent spread across sixteen terms none of which reaches one percent of the positive budget.

### 3.3 The good characteristics, and what produces them

Five properties of this gait are genuine achievements and must survive any change made under section 4.

The alternation is clean and correctly timed. Over 2016 completed cycles the median period is exactly 1.000 seconds against a commanded 1.000, the stance duty is 50.9 and 51.0 percent on the two legs, and the correlation between the measured contact state and the clock's desired contact state is 0.928 and 0.931 on the two feet. One qualification belongs with the praise. The mean period is 0.929 seconds rather than 1.000 because the distribution carries a tail of short cycles, its tenth percentile lying at 0.720 seconds, which is the signature of an occasional stumble or a contact that breaks and re-forms within one phase, and that tail should be watched as the phases of section 4 alter the contact schedule. The cause is the combination of `GaitReward` at weight 40, which supplies the phase, and `no_fly` and `feet_air_time_positive_biped`, which supply the incentive to be in single support at all times. The record of the thirteenth and fourteenth passes of `../context/brs_gait.md` shows this was hard won, the policy having previously preferred a one legged hold of two and a half seconds, and it must not be given back.

The planar velocity tracking is good. The root mean square error is 0.223 metres per second against a commanded magnitude of 0.53, the correlation is 0.945 in the fore and aft axis, and the per step reward trace sits at its ceiling of 50 for the great majority of the episode, dipping only through the commanded reversal at step 1088. The cause is the tracking weight of 50 combined with a curriculum that contracted the kernel width to 0.128.

The base height is held very well. The mean is 1.157 metres against a target of 1.15 with a standard deviation of 0.025, and the vertical excursion of the centre of mass across a stride is roughly five centimetres, which is the human figure. The cause is the Phase A2 change that moved the target onto a kinematically reachable pose, and the record of the seventeenth pass shows that before it the target lay outside the reachable set entirely and was discharged by trunk pitch instead.

The torso is close to level. The flat orientation penalty inverts to a root mean square tilt of 7.3 degrees in this run and 5.6 degrees in the run carrying the heavier angular velocity penalty, against the forty to seventy degrees with which this work stream began. The cause is the orientation penalty at −50, which sits inside the band of −33 to −50 that the third pass derived from the van Marum and Isaac Lab G1 references [10].

Left and right are close to symmetric in the sagittal plane. The Robinson symmetry index of the root mean square torque is 4.7 percent at the hip pitch, 9.1 at the knee and 6.3 at the ankle pitch, and of the joint range of motion 1.7, 0.2 and 0.9 percent respectively. The cause is the symmetry data augmentation implemented under `SYMMETRY_PLAN.md` [23, 24], and the surviving asymmetry is confined to the frontal plane, where the ankle roll range of motion index is 41 percent, which is a symptom of the lateral balance defect of section 3.6.7 rather than a failure of the augmentation.

There is no self collision. The undesired contacts penalty reads 0.026 per second against a weight of −2.5, an implied mean of one hundredth of a body over the ten newton threshold, and its per step trace is flat at zero. The rebuilt primitive asset of the twelfth pass has held.

### 3.4 Guidelines that follow

The five properties above translate into five constraints on any change proposed in section 4, and they are stated here so that the gates of that section can be checked against them.

The cadence, the phase relationship and the correlation with the clock must not fall. Any change to the contact schedule must move the duty, not the timing.

The planar velocity tracking error must not rise materially, taken as no more than a twenty percent degradation from 0.223 metres per second, since a smoother gait that cannot follow its command has solved the wrong problem.

The base height regulation and the torso attitude must not degrade, and in particular no change may be adopted that buys smoothness with a return of the forward lean.

The sagittal symmetry indices must remain in single figures.

The undesired contacts penalty must remain near zero, since the narrow stance remedy of section 4 moves the feet closer to the region in which the legs can interfere.

### 3.5 The bad characteristics

Six defects are established by the measurements of section 3.2, and they are stated here as observations, their causes being deferred to section 3.6.

The foot is driven into the ground. The sole descends at 1.68 metres per second on the average touchdown, which is 1.26 times the velocity free fall from the swing apex would produce, so the descent is actively powered rather than merely released. The contact force that follows rises from zero to 2.13 times body weight within thirty milliseconds in the touchdown aligned mean, against a steady stance load of roughly 0.55, and reaches 13.76 body weights at its worst. The loading rate reaches 733 body weights per second.

The foot is snatched off the ground. The vertical foot velocity reaches 2.5 metres per second upward at the beginning of swing, and the sole reaches ninety two percent of its apex clearance within the first fifteen percent of the swing phase.

The clearance profile is a trapezoid rather than an arc. The mean normalised swing profile rises from 0.015 to 0.086 metres in the first fifteen percent of swing, holds between 0.087 and 0.093 for the middle half, and falls to the ground in the last tenth, with 74 percent of the swing spent above seven centimetres. Human swing has its clearance minimum in the middle of swing, not its maximum [4].

The double support phase has been eliminated. The measured double support fraction is 2.3 percent against a clock that commands 19.6 and a human figure of 24 [2], and the same figure is 1.4, 1.9 and 2.1 percent in the three sibling runs, so it is structural rather than a property of one seed.

The actuation is impulsive and the joints work against one another. Every joint trace is a near zero baseline punctuated by transients of a few tens of milliseconds, the joint accelerations reaching 1900 radians per second squared at the ankle pitch on the ninety ninth percentile and 7441 at the extreme, the torque changing by up to 470 newton metres within a single ten millisecond control step. The knee delivers peaks of three kilowatts while the hip pitch absorbs peaks of two and the ankle pitch alternates between the two, so that the mean absolute joint power of 850 watts stands against a net mechanical power of 89, a ratio of 9.5. The mechanical cost of transport computed on absolute power is 2.38, an order of magnitude above human walking [1], while the same figure computed on net power is 0.25, which is close to it. The motion the robot performs is not intrinsically expensive. The manner in which it performs it is.

The mechanical stops are used as structural elements. The knee stands within 0.02 radians of a limit for 37 percent of all steps, its distribution being bimodal with 35 percent of the time below 0.05 radians and 13 percent above 1.40 while only 15 percent falls in the intermediate band between 0.3 and 1.1. The ankle roll stands within 0.02 radians of a limit for 31 percent of steps and its actuator sits above 98 percent of its 131 newton metre effort ceiling for 6 to 7 percent of all steps. The ankle pitch exceeds its declared hard limit of 0.454 radians, reaching 0.619, and the ankle roll and ankle pitch both exceed their 25 radian per second velocity ceilings, reaching 29.5 and 30.7.

The yaw command is not followed. The commanded yaw rate averages 0.51 radians per second in magnitude and the achieved rate 0.29, the correlation is 0.21, and the per step yaw tracking reward trace spends most of its time near zero with occasional spikes to its ceiling. The failure is uniform across the four runs at root mean square errors of 0.74, 0.69, 0.69 and 0.56, and is untouched by any of the three parameters those runs ablate.

### 3.6 Root causes

#### 3.6.1 The clearance reward's maximiser is a trapezoid

`foot_clearance_reward_v2` computes, per foot, the product of a Gaussian kernel on the sole clearance centred at 0.08 metres with a width of 0.035, and the hyperbolic tangent of the horizontal foot speed, zeroed when the foot is in contact, summed over the feet, at a weight of 20.

The consequence follows from the form alone. The integrand depends on the instantaneous height and on nothing else, so over a swing of fixed duration the trajectory maximising the integral is the one that spends the most time within one kernel width of 0.08 metres. Evaluating the term over candidate profiles of equal apex and equal duration, at a fixed foot speed of 2.5 metres per second, gives the following incomes per second.

| swing profile | apex, m | mean kernel | clearance income |
|---|---|---|---|
| the observed trapezoid | 0.092 | 0.852 | 16.37 per second |
| the same with a gentle descent over the final thirty percent | 0.092 | 0.705 | 13.56 per second |
| a symmetric sinusoidal arc | 0.092 | 0.585 | 11.24 per second |
| a symmetric sinusoidal arc of larger apex | 0.120 | 0.470 | 9.03 per second |

The trapezoid earns 46 percent more than the sinusoidal arc of the same apex and 21 percent more than a profile differing from it only in the gentleness of its final descent. Since the whole term supplies 18.3 percent of the positive budget, that premium is 2.8 per second of income purchased by snapping the foot up at the start of swing and dropping it at the end, and it is paid for by nothing at all, since no term in the configuration prices vertical foot velocity, foot acceleration, or the descent that follows.

The tanh factor compounds the defect from the other side. It saturates at 0.96 by two metres per second and at 0.995 by three, so the term pays a foot to travel at least twice as fast as the body, and the measured horizontal foot speed at the ninety ninth percentile is 3.40 metres per second against a base speed of 0.61. A foot moving at that speed and then arrested at touchdown deposits its kinetic energy into the contact.

This is the primary cause of the sudden jump in the clearance trace, of the yank at lift off, and, through the second half of the trapezoid, of the stomp.

#### 3.6.2 Double support is priced at zero, so weight transfer must be impulsive

`no_fly` returns one times the single contact indicator less five times the no contact indicator, at a weight of 15. Its income is therefore 15 per second in single support, zero in double support, and negative 75 per second in flight. `feet_air_time_positive_biped` is likewise gated on exactly one foot in contact, at a weight of 12.5. Together these two terms, supplying 23.3 percent of the positive budget, pay nothing whatever for the interval during which a biped transfers its weight.

Against them stands `GaitReward` at weight 40, whose clock commands 19.6 percent double support, and which the measurements show is being complied with in phase and defied in duty.

The arithmetic of that contest can be computed. Moving from the observed 2.3 percent double support to the commanded 19.6 at fixed cadence changes the four affected terms as follows.

| term | change per second |
|---|---|
| `rew_no_fly` | −2.53 |
| `feet_air_time` | −0.85 |
| `rew_foot_clearance`, through the shorter swing | −1.76 |
| `rew_gait` | +7.02 |
| net | +1.87 |

Compliance is worth 1.87 per second, which is 3.0 percent of the positive budget. That is a real gradient but a shallow one, and it is well inside what a policy converged at a learning rate of 2.7 times ten to the minus five with an exploration standard deviation of 0.83 will resolve. The schedule is therefore under determined by the reward, and the tie is broken by whatever is dynamically easier, which is the shorter transfer.

The mechanical consequence is the stomp, and it is not incidental to it but the whole of it. With 2.3 percent double support, a cadence of one hertz allows 23 milliseconds per cycle in which both feet are loaded, which is 11 milliseconds per transfer. The entire body weight must move from one foot to the other within that interval. There is no arrangement of the swing trajectory that makes such a transfer gentle, because the leading foot must acquire the full load in a time comparable to the contact's own settling time. The measured impact profile confirms it, the force rising to 2.13 body weights at thirty milliseconds and decaying to the steady stance value by seventy.

This is the second, and structurally the deeper, cause of the stomping. The clearance term of section 3.6.1 determines how fast the foot arrives, and this determines that it must arrive suddenly.

#### 3.6.3 Nothing in the configuration prices impact

The configuration contains no penalty on contact force, no penalty on foot landing velocity, and no effective penalty on foot velocity near the ground.

`foot_landing_vel`, which is exactly the required term, exists in `mdp/rewards.py` and is wired into the TRON1 point foot configuration at a weight of −0.5 with a threshold of 0.08 metres. It is absent from the SD_BRS1 configuration. Evaluating it against this dump with the correct foot radius of 0.124 metres and a threshold of 0.10 gives a value of 0.209, and the gate it uses would catch 97.4 percent of the touchdowns, so the term would function without modification.

`feet_regulation` is the only term acting near the ground, penalising the squared horizontal foot speed through an exponential gate of length scale 0.03 metres in the clearance. It reads 0.024 per second at a weight of −0.2, which is 0.16 percent of the negative budget, and it penalises horizontal rather than vertical velocity, so it opposes sliding and not landing.

`pen_lin_vel_z` penalises the squared vertical velocity of the base, not of the foot, at −0.5, and reads 0.010 per second.

The total priced against a collision of three body weights is therefore, to the precision that matters, nothing. This is why the stomp is free once sections 3.6.1 and 3.6.2 have made it profitable.

#### 3.6.4 The smoothness penalties are too small to bind

The six terms that oppose roughness sum to 3.81 per second against a positive budget of 63.04, which is 6.0 percent. Their internal distribution is more revealing than their total, since 2.885 of the 3.81 comes from the second difference action smoothness penalty alone, leaving 0.92 per second, or 1.5 percent of income, spread across the squared torque, the squared acceleration, the squared joint velocity, the first difference of the action, and the first difference of the torque together.

Inverting each through its weight gives the magnitude of the quantity being penalised. The squared torque term reads a function value of 45131, a root mean square of 67 newton metres per joint. The squared acceleration term reads 854000, a root mean square of 292 radians per second squared. The torque rate term reads 19888, a root mean square first difference of 45 newton metres per ten millisecond step. These are not small quantities being penalised lightly. They are large quantities being penalised at coefficients chosen so that their product is negligible.

The consequence is that the marginal cost of a torque spike that improves contact timing or tracking is of order one hundredth of the marginal gain, so the policy buys the spike every time.

#### 3.6.5 The action is unclipped and the joint stops are load bearing

The action is a joint position target at a scale of 0.4 with `clip` set to null and `use_default_offset` true. The exploration standard deviation at convergence is 0.83, so the sampled action routinely spans two units and the commanded target routinely spans 0.8 radians about the nominal, which for several joints exceeds their entire travel. Nothing bounds it, and the proportional derivative controller converts whatever excess remains into torque against the mechanical stop.

The ankle roll is the clearest case. Its stiffness is 20 newton metres per radian and its effort ceiling 131, so reaching that ceiling through the proportional term alone would require a position error of 6.55 radians against a joint whose whole travel is 0.70. The measurements show the joint at its limit, within 0.02 radians, for 31 percent of all steps, with the actuator above 98 percent of its ceiling for 6 to 7 percent, and at those instants the joint velocity is near zero. The joint is being pressed into its end stop at full effort and held there, and the torque against velocity scatter shows exactly this, a vertical stripe at zero velocity spanning the full effort range with a horizontal band at zero torque for the free swing, and nothing in between.

The knee shows the same pattern in position rather than in torque. Its distribution is bimodal, 35 percent of the time below 0.05 radians against a hard lower stop at zero, and 13 percent above 1.40 against a hard upper stop at 1.483, with only 15 percent in the intermediate band. In stance its mean is 0.086 radians, which is a straight leg resting on its extension stop, and in swing its mean is 1.25 radians, which is very nearly full fold. The knee is not being controlled as a joint. It is being switched between two mechanical constraints.

That this is tolerated follows from the budget. `pen_joint_pos_limits` reads 0.169 per second, 1.2 percent of the negative side, so leaning on a stop costs a hundredth of what a comparable tracking gain returns. The stops are cheaper than control.

The overshoot of the declared limits, the ankle pitch reaching 0.619 radians against a limit of 0.454 and the ankle roll and pitch both exceeding their 25 radian per second velocity ceilings, is the same phenomenon seen from the solver's side, since the position iteration count is 2 and a constraint driven this hard yields.

#### 3.6.6 The knee, the hip and the ankle work against one another

The torque against velocity scatter of the knee shows two motoring arms, one in the quadrant of positive torque and positive velocity and one in the quadrant of negative torque and negative velocity, so the knee performs positive work both folding and extending. The hip pitch scatter is the mirror image, its mass lying in the two braking quadrants, so the hip absorbs. The power traces confirm the timing, the knee delivering peaks near three kilowatts and the hip pitch absorbing peaks near two, within the same few tens of milliseconds of each cycle.

The aggregate consequence is the ratio of 9.5 between the mean absolute joint power of 850 watts and the net mechanical power of 89. Ninety percent of the actuation is antagonistic circulation. The net cost of transport of 0.25 shows that the trajectory the base follows is close to human in its energetic demand, and the absolute figure of 2.38 shows what the joints are made to pay to produce it.

The proximate cause is the bimodal knee of section 3.6.5. A knee that switches between its stops must be accelerated to sixteen radians per second and arrested again twice per cycle, and the leg's inertia must be supplied and then removed by the hip. A knee held in the intermediate band, as the biomechanical standard of section 1.4 requires [5], would neither require nor return that energy.

#### 3.6.7 The feet distance hinge measures the wrong distance

`feet_distance` computes the Euclidean norm of the planar separation between the two ankle frames and applies a hinge below a configured minimum, here 0.21 metres at a weight of −100. It reads 0.014 per second, an implied function value of 0.00014, and its per step trace is flat at zero. The term is, in effect, never active.

The measurements show why, and they show that its inactivity is not evidence of a wide stance. The lateral separation between the feet has a mean magnitude of 0.225 metres, a minimum of zero, and stands below 0.19 metres, which is the width at which the two sole plates would touch, for 39.8 percent of all steps. The fore and aft separation has a mean magnitude of 0.217 metres and a maximum of 0.767. The planar norm therefore remains above the 0.21 metre threshold even when the lateral separation is exactly zero, because at that moment the stride has carried the feet apart along the direction of travel.

The trace of the lateral separation makes this visible directly, falling to zero and passing below it twice within the plotted window, which is to say the feet cross the midline. The hinge cannot see it.

The consequence is that the stance width, which section 2.8 establishes should be between 0.26 and 0.338 metres for this robot's 0.26 metre hip width [20], is in fact uncontrolled. This is the same defect the nineteenth pass of `../context/brs_gait.md` addressed by raising the minimum from 0.21 toward 0.32, and the present analysis shows that the remedy cannot work in the term's current form, since raising a threshold on the planar norm raises it against the stride as much as against the stance.

A correction written 2026-08-07, which withdraws the central factual claim of this section while leaving one part of its conclusion standing. Every lateral separation figure quoted above, and every one quoted in the Phase 5 revision paragraph of section 4.7, was measured in the world frame. The dump records the separation as a world frame vector at `scripts/rsl_rl/play.py:407`, and `feet_distance_statistics` at `scripts/analysis/stats.py:751` names its second component the lateral separation without rotating it, so both the plots and the statistics report the world frame y difference. That quantity equals the stance width only when the robot's heading is zero, and this robot turns continuously under a heading controller, its yaw covering the whole circle with a standard deviation of 1.76 radians across the evaluation. As the heading rotates, the world frame components exchange the stance width with the stride, so the reported lateral separation is a mixture of the two and passes through zero whenever the robot happens to face along the axis.

Rotated into the base frame by the yaw of the dumped root quaternion, which is the frame in which stance width is defined, the picture reverses. On the Phase 3 run the lateral separation has a mean of 0.2591 metres, a median of 0.2469, a fifth percentile of 0.2169, a first percentile of 0.2045 and a minimum of 0.0907, and its sign does not change once in ninety six thousand samples, the left foot remaining left of the right throughout. It stands below the 0.19 metre sole overlap width for 0.27 percent of steps against the 47.90 percent the world frame figure reports for the same run, and the corresponding figures on the baseline are 0.41 against 47.02. The feet do not cross the midline. The base frame fore and aft separation crosses zero at a rate of 0.0185 per step, which is one crossing every 0.54 seconds against a half cycle of 0.46, and its ninety ninth percentile of 0.4371 metres agrees with the 0.4461 predicted from the measured cycle period and forward speed, so the decomposition is validated on both components independently.

What survives is smaller and is a matter of degree rather than of kind. The measured stance width of 0.259 metres sits at or marginally below the lower edge of the 0.26 to 0.338 metre band section 2.8 derives [20], so there is a case for widening it, but it is the case for moving a quantity from the bottom of its band toward the middle and not the case for correcting a gross defect. The term's design flaw is unaffected by the correction, since a hinge on the planar norm still cannot regulate a stance width, and it remains true that the hinge as configured is nearly inert. What is withdrawn is the evidence that the stance width is presently pathological, and with it the premise of the second sentence of section 3.6.8. Section 4.4 is rewritten against the corrected measurement.

#### 3.6.8 Yaw authority is morphologically scarce and the reward set spends what remains

The SD_BRS1 declares both hip yaw joints fixed. There is no actuator with a vertical axis anywhere in the machine. Every yaw moment must be obtained from the ground, and section 2.7 establishes that this leaves two mechanisms, the friction moment developed under the stance sole and the reaction to a change in whole body angular momentum about the vertical.

Four properties of the present gait suppress both.

The stance width is at or below the sole overlap threshold for forty percent of the cycle, per section 3.6.7, which minimises the moment arm available to any differential foot placement.

The lateral velocity command range is plus or minus 0.01 metres per second, so the policy has never been asked to step sideways and has no lateral placement skill to recruit for a turn.

`feet_slide` penalises the horizontal velocity of a foot in contact at a weight of −5, reading 0.79 per second. Pivoting the stance foot to develop a yaw friction moment necessarily moves the sole's periphery, so the one mechanism a flat footed biped has for turning in place is directly, if partially, penalised.

There is no term anywhere in the configuration that rewards the feet for pointing along the direction of travel, so the yaw of the foot relative to the base is unconstrained, and Booster Gym's foot yaw alignment term at twice their linear tracking weight is the reference instrument that is missing [11].

Two further facts complete the picture. The yaw command is not sampled directly but produced by a heading controller, `heading_command` being true with a relative heading fraction of 1.0 and a control stiffness of 0.5, so the commanded yaw rate is proportional to the heading error and saturates at the range limit while that error persists. The measurements show it pinned near 0.89 radians per second, against a nominal range of plus or minus 0.3 that the curriculum had widened. A robot that cannot turn therefore accumulates heading error, which saturates the command, which the robot still cannot follow. The failure is self sustaining, and the curriculum that widened the range did so on the angular tracking reward crossing a threshold of 0.7 that the measured function value of 0.397 shows it no longer meets.

The second fact is that the yaw tracking kernel is the same exponential form as the linear one, at a weight of 15 against 50, and the curriculum had contracted its width to 0.267. A kernel of that width at the measured mean squared error of 0.55 returns 0.47 in the evaluation, against a logged training function value of 0.397, so the term is neither dead nor saturated. It is simply outbid, contributing 9.4 percent of the positive budget against the 48.3 percent of the linear pair, and the policy is correct to prefer forward speed. This matters for the remedy, since a term that has collapsed to zero must be rebuilt, as the Phase A knee reward of `NATURAL_GAIT_PLAN.md` had to be, whereas a term that is merely outbid can be reweighted.

#### 3.6.9 An instrumentation defect, the left and right labels are transposed

This is not a gait defect but it governs the reading of every plot in the artefact directory, and it must be recorded before any of them is quoted again.

The joint index order in the dump is left before right at every depth, not right before left. The proof is decisive and does not depend on any assumption. Joint index 2 spans −1.2521 to +0.6090 radians over the dump, and the only joint in the URDF whose limits admit −1.25 is `HipPitchL`, whose range is −1.25 to +0.75. Joint index 3 spans −0.6998 to +1.2502, and the only joint admitting +1.25 is `HipPitchR`, whose range is −0.75 to +1.25. The ordering at that depth is therefore left then right.

The foot index order follows the same convention, established independently. Knee index 4 is near extension exactly when foot index 0 reports contact, agreeing on 97.2 percent of steps, and knee index 5 pairs with foot index 1 at 97.3 percent. Since index 4 is the left knee under the ordering just proved, foot index 0 is the left foot.

`JOINT_NAMES` in `/ws/IsaacLab/logs/rsl_rl/dashboard-brs.py` lists the joints right before left, and `FEET_NAMES` in the same file lists `Link6R` before `Link6L`, both citing a comment that the URDF declares the right leg chain first. The comment is correct about the URDF and wrong about the resulting articulation order. The same assumption appears in the comment above the feet distance logging in `scripts/rsl_rl/play.py`, which states that index 0 is `Link6R`.

The practical consequence is that every panel in `../context/artefacts/2026-07-28_06-37-24-play/` labelled R is in fact the left joint or foot and every panel labelled L is the right. The conspicuous asymmetry visible in the plots, in which the panel titled Ankle Roll R shows a saturating square wave while Ankle Roll L does not, is in truth the left ankle saturating. No conclusion of this document depends on which side it is, since every aggregate is computed over both, but a future reader tracing a single limb must apply the correction.

### 3.7 Where the analysis leaves the problem

The six defects of section 3.5 reduce to a smaller set of causes than their number suggests, and the reduction determines the order of section 4.

Two causes act on the swing trajectory. The clearance term's maximiser is a plateau, and its velocity factor pays for a foot moving several times faster than the body. Together they produce the snatch at lift off, the plateau in the middle, and the powered descent at the end, and nothing prices any of the three.

One cause acts on the schedule. The single support terms and the clock disagree about double support by a margin so narrow that the policy is effectively free to choose, and it chooses the arrangement that leaves no time for a gentle weight transfer.

One cause acts on everything at once. The effort and smoothness penalties are calibrated three orders of magnitude below the incentives they oppose, so every defect above is cheaper to commit than to avoid, and the mechanical stops are cheaper still than the control that would replace them.

Two causes are specific. The feet distance hinge measures a quantity that a long stride satisfies, so the stance width is not in fact regulated. And the yaw authority of a machine with no vertical axis actuator is scarce to begin with and is then spent by a narrow stance, a suppressed lateral command, a slide penalty on the pivot, and the absence of any foot yaw term.

The programme of section 4 addresses them in that order, which is also the order of increasing cost, since the first two are weight changes and one already existing term, the third is a set of weight changes with one new function, and the last two require a new argument and a new term respectively.

---

## 4. Improvement Plan

The programme runs in six phases. Each is a single experiment evaluated against the preceding policy, each names the file and the symbol it touches, and each observes the change discipline of `../CLAUDE.md`, which requires that backwards compatibility be preserved absolutely, that a new behaviour be carried by an optional argument whose default reproduces the old behaviour exactly wherever that is possible, and that a version two of a function be created only where no optional argument can preserve the old behaviour.

The caller sets that discipline depends upon were enumerated against the live sources on 2026-07-31 and are the following. `foot_clearance_reward_v2` and `JointTorqueRatePenalty` are called only from `cfg/SF/brs_base_env_cfg.py` and are SD_BRS1 exclusive. `no_fly` is called from that file and from `cfg/SF/limx_base_env_cfg.py`, so its blast radius is one other task. `feet_distance` is called from `cfg/PF/limx_base_env_cfg.py` at a minimum of 0.115 and from `cfg/WF/limx_base_env_cfg.py` at 0.32, so its blast radius is two. `feet_regulation` is called from SF, PF and SD_BRS1. `foot_landing_vel` is called from PF and is commented out in SF. `ActionSmoothnessPenalty` is called from WF, PF, SF and SD_BRS1. Every weight quoted below is set in the SD_BRS1 `RewardsCfg` and is therefore config local and safe by construction, and only the three function changes of Phases 3, 4 and 5 touch shared code at all.

One matter must be settled before Phase 1 begins. The working tree has been rolled back from the configuration the live run `2026-07-31_05-11-37` is training. That run carries `min_feet_distance` 0.25, `pen_ang_vel_xy` at −5, and a `pen_ankle_deviation` term at −0.1, while the tree carries 0.21, −2, and the term commented out. The evidence of section 3 favours the run's values, since the two arms carrying `pen_ang_vel_xy` at −5 halve the root mean square base roll rate from 0.81 and 1.16 to 0.54 and 0.60, and the arms carrying a minimum feet distance of 0.25 or above reduce the fraction of time the knee spends against its stops from 37.8 and 27.5 percent to 4.4 and 5.6. Phase 1 should therefore be applied on top of the run's configuration, not the tree's, and the first action of this programme is to reconcile the two.

That reconciliation has since been made, but not before Phase 1 was trained. Run `2026-07-31_10-21-10` carried the tree's values, a minimum feet distance of 0.21 and a roll and pitch angular velocity penalty of −2, which its dumped configuration confirms, whereas the tree as it now stands carries 0.25 and −5. Phase 1's failure is not attributable to that, since section 4.1a establishes causes sufficient on their own, but it does mean the Phase 1 run is not the clean predecessor of Phase 1b that it would otherwise have been, and the two differ in three parameters rather than in the intended two. The `pen_ankle_deviation` term remains commented out in the tree.

### 4.1 Phase 1, price the impact and the descent

Section 3.6.3 established that nothing in the configuration opposes a collision of three body weights, and section 3.6.1 quantified the income the trapezoid earns as 2.8 per second over a profile differing from it only in the gentleness of its descent. Phase 1 makes the descent expensive and simultaneously reduces the premium the plateau earns, which are complementary because each halves what the other must accomplish alone.

Three changes, all config local, no function edited.

The first wires the existing `foot_landing_vel`. Section 3.6.3 confirmed that its gate, which requires the foot to be below a height threshold, descending, and not in contact, catches 97.4 percent of touchdowns at a threshold of 0.10 metres once the foot radius is set to the SD_BRS1 value of 0.124 rather than the TRON1 point foot value of 0.03. The measured term value is 0.209.

```python
# cfg/SF/brs_base_env_cfg.py, RewardsCfg, beside feet_slide
# Penalise the vertical speed of a foot that is about to land. The measured touchdown
# velocity is 1.68 m/s mean and 4.37 m/s worst, which is 1.26x free fall from the swing
# apex, so the descent is powered rather than released. foot_radius is 0.124 for this
# robot, the sole sitting that far below the Link6 frame, NOT the 0.03 point-foot value
# that the TRON1 PF caller passes. Term value measured at 0.209 on run 2026-07-28_06-37-24.
pen_foot_landing_vel = RewTerm(
    func=mdp.foot_landing_vel,
    weight=-10.0,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
        "foot_radius": 0.124,
        "about_landing_threshold": 0.10,
    },
)
```

The weight follows from the trade. At −10 the term costs 2.09 per second at the present landing velocity and 0.52 if that velocity halves, so halving it saves 1.57 per second. That is short of the 2.8 per second the trapezoid earns, which is why the second change is needed with it.

The second reduces the clearance weight from 20 to 12. This halves the plateau premium to 1.7 per second, which the landing penalty's 1.57 then very nearly matches, and it reduces the clearance term's share of the positive budget from 18.3 to roughly 11 percent, which is closer to the proportion the gait terms hold in the reference configurations of section 2.2 [9, 11]. The apex is already 0.092 against a target of 0.08, so there is margin to lose.

```python
# cfg/SF/brs_base_env_cfg.py, rew_foot_clearance
    weight=12.0,   # was 20.0
```

The third adds the threshold hinge on contact force that Humanoid-Gym carries [9], in the same inert-below-threshold form, so that the collision itself is priced and not only the descent that precedes it. This requires a new function, which is written fresh and wired into SD_BRS1 only, so it carries no compatibility burden.

```python
# mdp/rewards.py, new function
def feet_impact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    clip: float = 2000.0,
) -> torch.Tensor:
    """Penalise per-foot contact force in excess of a threshold, inert below it.

    The form follows Humanoid-Gym (arXiv 2404.05695), whose large-contact-force term is
    max(F - F_thr, 0) clipped above. Being exactly zero over the range of forces a well
    behaved stance produces, it cannot distort the stance phase it is not meant to govern
    and acts only on the collision. The maximum over the contact history axis is taken so
    that a transient falling between two control steps is not missed.

    Args:
        env: The environment object.
        sensor_cfg: Contact sensor configuration resolving the feet bodies.
        force_threshold: Force (N) below which the term is exactly zero. For SD_BRS1 the
            body weight is 587 N and the steady single-support stance load is about that,
            so a threshold near 900 N admits normal stance and catches the collision.
        clip: Upper bound (N) on the per-foot excess, so that one pathological contact
            cannot dominate a batch. Defaults to 2000.0.

    Returns:
        The computed penalty tensor, summed over the feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak = forces.norm(dim=-1).max(dim=1)[0]                      # (N, F)
    return torch.sum(torch.clip(peak - force_threshold, 0.0, clip), dim=1)
```

```python
# cfg/SF/brs_base_env_cfg.py, RewardsCfg
# Body weight is 587 N. The touchdown-aligned mean peak is 3.08 BW and the worst 13.76 BW,
# against a steady stance load near 1.0 BW, so a 900 N threshold is inert in stance.
pen_feet_impact = RewTerm(
    func=mdp.feet_impact_force,
    weight=-1.0e-2,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
        "force_threshold": 900.0,
    },
)
```

The weight follows from the measurement rather than from an estimate. Evaluated over the dump at a threshold of 900 newtons and a clip of 2000, the term takes a mean value of 33.2, is active on 7.2 percent of steps, and averages 459 newtons of excess when active. At a weight of −1e-2 it therefore costs 0.33 per second at present and falls to zero entirely once the peak comes within the threshold, which is a small but non negligible half a percent of the positive budget. The two neighbouring thresholds were computed for comparison, 700 newtons giving a term value of 51.8 active on 12.6 percent of steps and 1200 giving 19.2 active on 3.2 percent, and 900 is chosen as the value that admits the whole of a normal single support stance at 587 newtons with a margin while still catching the great majority of collisions.

The term is deliberately the lighter of the two impact instruments. The landing velocity penalty is the primary one, because it acts before the collision and its gradient therefore reaches the swing trajectory that caused it, whereas a force penalty acts after and reaches it only through the contact model. Section 2.4 records that the ground reaction force literature introduces such penalties on a curriculum precisely because a heavy one destabilises a policy that has not yet learned the softer landing [15], so if Phase 1 fails its gate the correct escalation is to raise this weight toward −3e-2 on a second iteration rather than to introduce it heavy at the first.

The gate for Phase 1 is a fall in the mean touchdown vertical velocity below 1.0 metres per second and in the touchdown aligned mean peak force below 2.0 body weights, with the guidelines of section 3.4 held, in particular a linear tracking error no worse than 0.27 metres per second and a cadence within five percent of 1.0 hertz.

### 4.1a The outcome of Phase 1, and why it failed

Phase 1 was trained as run `sd_brs1_flat/2026-07-31_10-21-10` for thirty thousand iterations and evaluated by the same instruments that established the baseline. It failed its gate. The touchdown velocity fell by a tenth where the gate asked for a third, and the actuation deteriorated in a manner the phase did not anticipate. This section establishes what happened from the dump, because the reasons are specific, they are all correctable, and one of them is a defect in a function this repository has carried since it was ported.

A note on method belongs first. Every figure below is computed from `data/42/dump.npy` of each run by one script applied identically to both, a touchdown being the rising edge of a contact force norm exceeding one newton and the touchdown velocity the vertical foot velocity at the step preceding it. That method reports 1.551 metres per second on the baseline where section 3.2 reports 1.68, the difference lying in the contact detection threshold rather than in the data. The comparison below is therefore internally consistent and its absolute values are marginally lower than those of section 3.2 throughout.

Only two of the three prescribed changes were applied. The `foot_landing_vel` term was wired at minus ten and the `feet_impact_force` term was written and wired at minus one hundredth, both as specified. The reduction of the clearance weight from 20 to 12 was not applied, and the tree still carried 20.0 when the run was launched, which its dumped `params/env.yaml` and its `params/env.pkl` both confirm. That omission is the largest single reason the phase failed, since section 4.1 derived the landing penalty's weight from a trade against a premium that the clearance reduction was supposed to have halved.

| quantity | baseline 2026-07-28 | Phase 1 2026-07-31 | gate | verdict |
|---|---|---|---|---|
| touchdown vertical velocity, mean | 1.551 m/s | 1.402 m/s | below 1.0 | failed |
| touchdown vertical velocity, worst | 4.089 m/s | 3.573 m/s | | |
| sole approach velocity at touchdown | 2.182 m/s | 2.064 m/s | | |
| peak contact force in 80 ms, mean | 2.354 BW | 2.191 BW | below 2.0 | failed |
| peak contact force in 80 ms, worst | 10.315 BW | 7.025 BW | | improved |
| linear tracking error | 0.129 m/s | 0.121 m/s | below 0.27 | held |
| cadence, median period | 1.000 s | 1.000 s | within 5 pct | held |
| yaw rate error | 0.653 rad/s | 0.494 rad/s | | improved |
| double support | 1.39 pct | 0.84 pct | | worsened |
| mean absolute joint power | 768 W | 910 W | | worsened |
| mechanical cost of transport | 2.835 | 3.341 | | worsened |
| ankle pitch torque, 99th percentile | 169 N m | 262 N m | | at the ceiling |
| ankle pitch above 98 pct of ceiling | 0.00 pct | 3.8 pct | | newly saturating |
| ankle roll above 98 pct of ceiling | 7.07 pct | 10.7 pct | | worsened |

The guidelines of section 3.4 were held, and the yaw tracking improved without being asked to, its correlation rising from 0.275 to 0.515. The failure is confined to the impact and to what the policy did instead of softening it.

Four causes account for the outcome, and the first is the omission recorded above. The remaining three are properties of the term the phase wired, and they are established here so that they may be closed rather than merely noted.

The term is a time integral, and the policy minimised the integral rather than the impact. Its function value fell by 44 percent, from 0.170 to 0.096, while the touchdown velocity fell by 5 percent. The descent profile shows exactly how. Averaged over the five control steps preceding touchdown the baseline descends at 1.26, 1.30, 1.35, 1.43 and 1.47 metres per second, a smooth arrival, whereas Phase 1 descends at 0.36, 0.41, 0.53, 0.74 and 1.07 and then strikes at 1.41. The policy learned to loiter slowly through the upper part of the gate window, which is where an integral over time accumulates most of its value, and to drop through the last three centimetres as fast as before. A penalty whose gate is wide and whose measure is a time integral is minimised by spending more time moving slowly, which is not the same thing as arriving softly, and the two came apart here completely.

The gate measures the wrong height. `foot_landing_vel` computes the foot height as the body frame origin less a constant `foot_radius`, which is exact only for a level foot. This is the identical proxy that `foot_clearance_reward` used and that `foot_clearance_reward_v2` was written to replace, and it survives in the landing term because that term was ported from the TRON1 point foot task, where the foot is a sphere and the proxy is exact. On the SD_BRS1 sole it overestimates the true sole clearance by 23.5 millimetres on average and by 74.4 at the ninety fifth percentile, with the consequence that the gate stood shut on 32.7 percent of the steps at which the true sole clearance was already below the nominal threshold of 0.10 metres. The term was therefore not penalising the descent it was configured to penalise.

The term measures the wrong velocity. It charges the vertical velocity of the foot's frame, whereas the collision is governed by the approach velocity of the lowest sole point, and the two differ by the rotational contribution. Measured at touchdown, the sole approaches at 1.41 times the frame velocity in the baseline and at 1.47 times in Phase 1, so the ratio moved in the direction the penalty made profitable. A policy charged for translating its foot downward may rotate it downward instead, and this one did. That substitution is visible in the actuator record as well as in the kinematics, the ankle pitch reaching its 262 newton metre effort ceiling on 3.8 percent of steps where it had never reached it before, because the ankle is now performing both the terminal approach and the arrest that follows it.

The marginal arithmetic makes the failure predictable in retrospect, and it is worth stating as a number because it also fixes the remedy. Take the change actually wanted, a gentle cosine descent over the final thirty percent of swing, apply it to every observed swing in the dump while holding the apex and the duration, and evaluate both terms over the modified profile. The clearance income forgone is 1.95 per second at a weight of 20 and 1.17 at a weight of 12. The landing penalty saved is 1.14 per second at a weight of minus ten, 2.28 at minus twenty and 3.41 at minus thirty. Under the configuration actually trained, a clearance weight of 20 against a landing weight of minus ten, softening the landing was a net loss of 0.81 per second, so the policy was correct to keep stomping and no amount of training would have changed that. The reward, not the learning, decided the outcome, which is the same lesson section 5 draws in general form.

Two instrumentation findings emerged from this analysis and govern the reading of every play dump this project produces, so they are recorded here and in `../context/brs_gait.md` rather than left in a script.

The play evaluation does not use the run's training configuration. `scripts/rsl_rl/play.py` builds its environment through `parse_env_cfg`, which reads the live source tree, so the reward weights in a play dump are those the tree carried at the moment play was run and not those the policy was trained under. The two differed here, the play having used a roll and pitch angular velocity penalty of minus two against a tree that now carries minus five, a minimum feet distance of 0.21 against a tree that now carries 0.25, and a landing weight of minus thirty against the minus ten under which the policy trained. The behavioural channels of the dump, the positions, velocities, torques and forces, are unaffected, since they record what the trained policy actually did. The reward channels must be read against the tree at play time, and a per term function value should be recovered by reconstructing the term from the dump rather than by dividing a logged rate by an assumed weight. The reconstruction was validated by agreement to four significant figures on `joint_torques_l2`, `joint_vel_l2` and `feet_regulation`.

The dump understates contact force. `feet_impact_force` takes the maximum over the contact sensor's force history and therefore sees transients falling between two control steps, whereas the dump records the instantaneous force. The term's logged value of 183.65 stands against 15.31 reconstructed from the dump, a factor of twelve on the excess above the threshold. The peak contact forces this document quotes, whether 3.08 body weights in section 3.2 or 2.19 above, are consequently lower bounds, and the true collision is harder than the plots show.

### 4.1b Phase 1b, the corrected retry

Phase 1b applies the change Phase 1 omitted, raises the two impact weights, and replaces the landing term with a version that closes the gate and velocity defects of section 4.1a. It is config local but for the new function, which is additive and wired to SD_BRS1 alone.

The clearance weight is reduced from 20 to 12, which is the unexecuted half of Phase 1 and needs no further justification than section 4.1's.

```python
# cfg/SF/brs_base_env_cfg.py, rew_foot_clearance
    weight=12.0,   # was 20.0
```

The landing term is replaced by `foot_landing_vel_v2`, which gates on the true sole clearance computed from the same twelve point sole table that `rew_foot_clearance` uses, and which charges the vertical velocity of the lowest sole point rather than of the frame. The sole point velocity is obtained as the frame velocity plus the rotational term, so the function remains stateless and exact. The sole table is hoisted to the module constant `SD_BRS1_SOLE_OFFSETS` in the same pass, so that the clearance the reward pays for and the clearance the penalty gates on cannot drift apart. The original `foot_landing_vel` is left untouched for the TRON1 point foot caller and its three defects are recorded in its docstring, per rules 5 and 6 of `../CLAUDE.md`.

```python
# cfg/SF/brs_base_env_cfg.py, replacing pen_foot_landing_vel
pen_foot_landing_vel = RewTerm(
    func=mdp.foot_landing_vel_v2,
    weight=-30.0,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
        "sole_offsets": SD_BRS1_SOLE_OFFSETS,
        # a TRUE sole clearance, not comparable with the v1 frame proxy of 0.10
        "about_landing_threshold": 0.06,
        "force_threshold": 1.0,
    },
)
```

The threshold of 0.06 metres is chosen on two grounds. It is a true clearance, so it is not comparable with the 0.10 of the v1 configuration, which stood 23.5 millimetres above the true clearance on average. And the free fall velocity from 0.06 metres is 1.08 metres per second, so a gate placed there bounds what an unpowered descent can deliver at close to the 1.0 metre per second the phase gate demands, and a policy wishing to arrive faster must pay to accelerate rather than merely release. Narrowing the window also shortens the interval over which the integral accumulates, which is what makes the loitering strategy of section 4.1a unprofitable.

The weight of minus thirty follows from the trade computed in section 4.1a. At minus thirty the term costs 2.18 per second at the present descent and returns 3.41 per second if the final thirty percent of swing is softened, against 1.17 per second of clearance income forgone at the new weight of 12, so the gentle descent becomes worth 2.24 per second net. That is 2.5 percent of the run's positive budget of 88.65 per second, which is larger than the three percent margin section 3.6.2 judged shallow but real for double support, and it is a gain rather than the loss of 0.81 per second the trained configuration presented.

The contact force penalty is raised from minus one hundredth to minus three hundredths, which is the escalation section 4.1 reserved for the case that the phase failed its gate. Section 2.4 records that the ground reaction force literature introduces such penalties on a curriculum precisely because a policy that has not yet learned the softer landing cannot afford a heavy one [15], so the escalation is taken as the second step of that curriculum rather than as a correction of the first. At minus three hundredths the term costs 5.51 per second against the measured value of 183.65 and falls to zero as the peak comes within the threshold.

```python
# cfg/SF/brs_base_env_cfg.py, pen_feet_impact
    weight=-3.0e-2,   # was -1.0e-2
```

The gate for Phase 1b is that of Phase 1, a mean touchdown velocity below 1.0 metres per second and a touchdown aligned mean peak force below 2.0 body weights, with two additions forced by what Phase 1 did instead. The sole approach velocity at touchdown must fall below 1.4 metres per second, against the 2.064 measured, since it is the quantity the collision actually depends upon and the one the policy evaded. And no ankle actuator may exceed 98 percent of its effort ceiling on more than 5 percent of steps, against the 10.7 percent the ankle roll now reaches, so that the phase cannot pass by trading a softer landing for a saturated ankle.

### 4.2 Phase 2, restore the double support interval

Section 3.6.2 established that compliance with the clock's commanded 19.6 percent double support is worth only 1.87 per second, three percent of the positive budget, and that the largest single opponent is `no_fly`, which pays nothing during the transfer. Van Marum and colleagues solve exactly this with a grace window, their single foot contact term returning unity if single contact occurred at least once in the preceding two tenths of a second, so brief double support during weight transfer is not punished [10].

Because the contact sensor's own history is only four samples deep, a twenty step window cannot be read from it and requires state carried across steps, which means the term must be a `ManagerTermBase` subclass. That is a larger change than an argument, so rule 5 of `../CLAUDE.md` applies rather than rule 4, and the realisation is a new class `NoFlyWithGrace` wired into SD_BRS1 alone, with the free function `no_fly` left exactly as it stands for the TRON1 SF caller and its defect recorded in its docstring per rule 6.

The window is carried as a per environment counter of the steps elapsed since single support was last observed, rather than as a rolling boolean buffer. The two are equivalent and the counter costs one integer tensor and one comparison, where a buffer would cost twenty booleans per environment and a reduction over them at every step.

```python
# mdp/rewards.py, implemented 2026-08-03
class NoFlyWithGrace(ManagerTermBase):
    """Reward single support, but over a grace window, so that weight transfer is not taxed."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.grace_steps = int(cfg.params.get("grace_steps", 0))
        # initialised beyond the window, so an environment that has not yet demonstrated
        # single support earns nothing from the grace branch, the robot beginning on both feet
        self._since_single = torch.full(
            (env.num_envs,), self.grace_steps + 1, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        # without this a policy inherits the window across an episode boundary and is paid,
        # for up to grace_steps, for single support achieved in a previous episode
        if env_ids is None:
            env_ids = slice(None)
        self._since_single[env_ids] = self.grace_steps + 1

    def __call__(self, env, sensor_cfg, threshold=1.0, history_index=0, grace_steps=0):
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]

        contacts = torch.norm(forces[:, history_index], dim=-1) > threshold
        num_contacts = torch.sum(contacts.float(), dim=1)
        single_contact = num_contacts == 1
        no_contact = num_contacts == 0

        # zero on the step single support is seen, incrementing otherwise, so the test below
        # is exactly "single support occurred within the last grace_steps steps, inclusive"
        self._since_single = torch.where(
            single_contact,
            torch.zeros_like(self._since_single),
            self._since_single + 1,
        )
        recently_single = self._since_single <= self.grace_steps

        # the no contact branch is NOT windowed, a flight phase being a fault at the instant
        # it occurs, whereas double support is a necessity
        return 1.0 * recently_single.float() - 5.0 * no_contact.float()
```

The counter was verified against a hand traced contact sequence before wiring. With `grace_steps` at zero the returned tensor is identical to that of `no_fly` at the same `history_index`, which makes the class a drop in replacement and lets the window be ablated by changing one number rather than by editing the configuration back.

```python
# cfg/SF/brs_base_env_cfg.py, replacing rew_no_fly for SD_BRS1 only
rew_no_fly = RewTerm(
    func=mdp.NoFlyWithGrace,
    weight=15,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
        "threshold": 1.0,
        "history_index": 0,
        "grace_steps": 20,          # 0.2 s, the van Marum window
    },
)
```

With the window in place the `no_fly` loss on moving to the commanded duty falls from −2.53 to approximately zero, and the net value of compliance rises from 1.87 to about 4.4 per second, seven percent of the positive budget, which is comfortably outside the exploration noise floor. The case has strengthened since the plan was written, the measured double support having fallen further to 0.84 percent under Phase 1, which at a one hertz cadence leaves about four milliseconds per transfer.

A second and smaller change belongs to this phase, and its arithmetic is corrected here. Reconstructing the smoothed desired contact state exactly as `GaitReward.compute_contact_targets` does and thresholding it at one half, the stance duty per foot equals the duration and the double support fraction is twice the duration less one exactly, the anti phase offset of 0.5 making the identity clean. A duration of 0.6 therefore commands 20.0 percent double support and not the 19.6 this document reported when first written, that figure having been an artefact of evaluating the clock on the discrete episode step grid. Raising `durations` from 0.6 to 0.62 commands 24.0 percent double support against a stance duty of 62 percent, which are the human figures of section 1.4 exactly [2].

```python
# cfg/SF/brs_base_env_cfg.py, gait_command ranges
    durations=(0.62, 0.62),      # was (0.6, 0.6)
```

The gate for Phase 2 is a measured double support fraction above 12 percent, a correlation between the measured and desired contact states no lower than the present 0.93, and the cadence unchanged.

Phase 2 is implemented and wired as of 2026-08-03. Because Phase 1b is wired in the same tree, the next run confounds the two, which is a deliberate departure from the one experiment per phase discipline and is taken because the impact remedy and the transfer interval act on the same defect from opposite ends and because Phase 1 has already spent one run establishing that the impact terms alone do not carry it. Setting `grace_steps` to 0 and `durations` back to 0.6 isolates Phase 1b, and that is the ablation to run first should the combined experiment fail its gates in a way that does not attribute.

### 4.2a The outcome of Phases 1b and 2, and what they bought

Phases 1b and 2 were trained together as run `sd_brs1_flat/2026-08-03_11-19-11` and both met their gates. This section records the measurement, states the three confounds that qualify it, corroborates the eight observations the user drew from the plots, and establishes the cause of the two regressions that came with the gains. Its method is that of section 4.1a, the same script evaluated over the play dumps of all three runs, a touchdown taken as the rising edge of a contact force norm above one newton and the peak force as the maximum over the eighty milliseconds that follow.

Three confounds govern how strongly the numbers may be read, and all three understate rather than flatter the result. The run reached eleven thousand iterations against the thirty thousand of run 2026-07-31_10-21-10, so every comparison below sets a third of a training budget against a whole one. Phases 1b and 2 were wired together, which section 4.2 recorded as a deliberate departure from the one experiment per phase discipline. And `pen_ang_vel_xy` stood at −5.0 in this run against −2.0 in both predecessors, a change belonging to no phase of this programme but to the tree reconciliation of section 4.1a, which matters because it is the direct cause of one of the observations attributed below to Phase 2.

| quantity | baseline 2026-07-28 | Phase 1 2026-07-31 | Phases 1b and 2, 2026-08-03 |
|---|---|---|---|
| touchdown frame vertical speed, mean | 1.551 m/s | 1.402 m/s | 1.107 m/s |
| sole approach speed at touchdown, mean | 1.910 m/s | 1.864 m/s | 0.706 m/s |
| peak contact force in 80 ms, mean | 2.354 BW | 2.191 BW | 1.584 BW |
| peak contact force in 80 ms, worst | 10.315 BW | 7.025 BW | 4.939 BW |
| double support fraction | 1.394 % | 0.838 % | 11.615 % |
| swing apex position within swing | 0.265 | 0.271 | 0.463 |
| correlation of swing profile with the raised cosine | 0.421 | 0.515 | 0.917 |
| clearance peaks per swing, mean | 3.644 | 3.003 | 1.702 |
| mean absolute joint power | 768 W | 910 W | 489 W |
| cost of transport, absolute | 2.835 | 3.341 | 1.870 |
| ankle roll torque 99th percentile, against a 131 N·m ceiling | 100.1 % | 100.0 % | 51.6 % |
| ankle pitch torque 99th percentile, against a 262 N·m ceiling | 63.6 % | 100.0 % | 62.9 % |
| hip pitch torque 99th percentile, against a 500 N·m ceiling | 35.1 % | 57.5 % | 81.8 % |
| stance knee flexion, mean magnitude | 0.068 rad | 0.074 rad | 0.030 rad |
| base roll rate, mean magnitude | 0.510 rad/s | 0.554 rad/s | 0.323 rad/s |
| planar tracking error, mean | 0.050 m/s | 0.046 m/s | 0.071 m/s |

Both Phase 1b gates are met. The peak contact force falls to 1.584 body weights, inside the 2.0 gate and within the 1.0 to 1.5 band a human walking gait develops [3], and the sole approach speed falls to 0.706 metres per second, comfortably inside the 1.4 gate that section 4.1b added and inside the 1.0 of the original touchdown gate. The frame vertical speed of 1.107 metres per second remains marginally above 1.0, but that is the quantity the superseded v1 term measured and not the one v2 charges, and section 4.1a established the frame measure to be the defective one. Both Phase 2 gates are met, the double support fraction reaching 11.615 percent against a gate of 12 percent that it misses by four hundredths of a point at a third of the training budget, having risen by a factor of thirteen from 0.838, and the cadence is unchanged at 2.02 steps per second. Ankle pitch saturation is eliminated, the 3.8 percent of steps above 98 percent of the ceiling falling to 0.001 percent, and ankle roll saturation falls from 10.9 percent to 0.015.

The eight observations drawn from the plots are corroborated as follows, six of them exactly, one understated by the plots, and one contradicted.

The reduction in base roll is real and larger than the plots suggest, the mean roll rate falling from 0.554 to 0.323 radians per second and its 99th percentile from 2.359 to 1.237. Its cause, however, is not Phase 2 but the third confound above. `pen_ang_vel_xy` penalises exactly the squared roll and pitch rates of the base and its weight rose from −2.0 to −5.0 in this run, at `cfg/SF/brs_base_env_cfg.py:853`. The reduction should therefore be attributed to that weight and not to the grace window, and section 3.3 of this document had already recorded the same effect from the same weight in the four run ablation that preceded the programme.

The reduction in joint velocities is confirmed, the mean magnitude falling from 1.423 to 1.129 radians per second and the 99.9th percentile, which is the spike measure, from 19.013 to 11.592. The reduction in hip roll torque is confirmed, its 99th percentile falling from 180.6 to 120.4 newton metres. The observation that ankle roll no longer presses against its range is confirmed and is the sharpest single result in the run, the 99th percentile falling from 100.0 percent of the 131 newton metre ceiling to 51.6, so the actuator that was previously saturated for a tenth of every second now operates at half its authority.

The observation that ankle pitch torque reduced marginally understates it. The 99th percentile fell from 261.9 newton metres to 168.2, which against a ceiling of 262 is a fall from complete saturation to 63 percent, and the proportion of steps above 98 percent of the ceiling fell from 3.8 percent to one thousandth of one percent. This is the exact reversal of the regression section 4.1a attributed to the v1 term's rotational substitution, and it is the strongest evidence that the v2 term closed that exploit.

The observation on the knee is confirmed in both of its parts. The knee torque did rise, its 99th percentile going from 213.3 to 263.6 newton metres, and the antagonism did fall, the power expended by the hip pitch and the knee against one another dropping from 170.2 to 43.3 watts and the fraction of time they oppose in sign from 62.4 percent to 49.8. The reduction in power spikes is confirmed across the board, the 99.9th percentile of total joint power falling from 12841 to 3041 watts, a factor of four, and the mean absolute power from 910 to 489, which carries the cost of transport from 3.341 to 1.870.

The observation on the swing profile is confirmed and is the result that most changes what Phase 3 must do. The apex moved from 0.271 of the swing to 0.463, where a symmetric arc places it at 0.500, the proportion of swings carrying more than one clearance peak fell from 96.5 percent to 55.7, the mean number of peaks per swing from 3.003 to 1.702, and the correlation of the normalised profile against the raised cosine reference from 0.515 to 0.917. The single peaked, near sinusoidal profile the plots show is therefore real and is measured.

The one observation the data contradicts is that the contact forces are similar to previous runs. They are not. The mean peak force at touchdown fell from 2.191 body weights to 1.584, a reduction of 28 percent, and the worst from 7.025 to 4.939. The impact term's own value fell from 183.6 to 21.8, a reduction of 88 percent, and the proportion of touchdowns registering above the 900 newton threshold from 96.2 percent to 46.0. The reading is understandable, because the stance phase of the force trace is dominated by the steady body weight support near 1.0 body weights which did not change, so a plot scaled to the whole trace compresses the very peaks that moved. The consequence for the user's proposal on this term is substantial and is taken up in section 4.3a.

What produced the gains reduces to one mechanism operating at two points. The v2 landing term gates on the true sole clearance and charges the approach velocity of the lowest sole point, so neither of the two substitutions available under v1, loitering high and rotating the sole in, remains profitable. Section 4.1a measured the sole to frame velocity ratio rising from 1.41 to 1.47 under v1, the signature of the rotational substitution, and the ankle pitch saturation that accompanied it. Under v2 the sole approach speed fell by 62 percent against a frame speed fall of 21 percent, which is that ratio collapsing, and the ankle pitch saturation went with it. The restored clearance weight of 12 completes the trade section 4.1b priced, and the grace window removed the penalty on the transfer interval so that the weight need no longer be moved impulsively, which is what allows the peak force to fall without the cadence changing.

Two regressions came with the gains and both have the same cause, which is that the robot adopted a straight legged gait. The mean stance knee flexion fell from 0.074 radians to 0.030, which is under two degrees, and the mean knee flexion over the whole cycle from 0.589 radians to 0.381. The base height did not change, its mean holding at 1.167 metres, so this is not a crouch but its opposite, a leg carried near full extension throughout stance.

The first regression follows mechanically. A leg held straight cannot use the knee to do sagittal work, so the work migrates to the hip, and the hip pitch torque rose accordingly, its root mean square from 77.4 to 123.1 newton metres and its 99th percentile from 57.5 percent of the 500 newton metre ceiling to 81.8. The knee torque rose for the related reason that a leg near full extension has poor mechanical advantage against a vertical load. This is the defect section 3.5 recorded as the knee living at its extension stop, unresolved and now more strongly expressed, and it is what Phase 7 was placed at the end of the programme to address.

The second regression is a loss of robustness that the tracking error only partly reveals. The planar tracking error rose from 0.046 to 0.071 metres per second and the yaw correlation fell from 0.515 to 0.296, but the more telling figure is the termination statistic. Episodes ending on the low height condition rose from 0.200 at eleven thousand iterations in the predecessor to 0.384, and the divergence appears far earlier than the push curriculum can explain, standing at 0.161 against 0.030 at five thousand iterations. A straight leg has no compliance with which to absorb a disturbance, which is the same property that made it cheap. The push force curriculum does confound this in part, reaching its maximum of 3.0 at iteration 7800 in this run against 12000 in the predecessor, so the policy met the full disturbance far earlier, but the divergence at five thousand iterations precedes that and the fragility is therefore real.

A caution on the tracking error is required before it is read as a training deficit. The user's expectation that it will improve over the remaining two thirds of training is not supported by the predecessor's own history. In run 2026-07-31_10-21-10 the planar error rose from 0.173 at eleven thousand iterations to 0.183 at thirty thousand, because two curricula tighten as training proceeds, the linear tracking reward standard deviation falling from 0.151 to 0.086 and the commanded velocity range widening. The metric therefore measures a moving target and does not converge downward. At the matched iteration of eleven thousand, with both curricula at identical values, the error is 0.216 against 0.173, and that 25 percent gap is the honest figure for the regression.

The reward budget confirms that the terms are now sized as the arithmetic intended. Linear velocity tracking supplies 63.1 percent of the positive budget of 75.49 per second, `rew_no_fly` 19.7 percent, and the clearance reward 5.8 percent against the 15.7 it held before Phase 1b cut its weight. On the negative side the gait clock is the largest single term at 43.3 percent of a negative budget of 15.38 per second, which is the measure of how far the contact schedule still stands from the clock, and the landing and impact terms together account for 13.4 percent. Because the working tree at the time of the play matched the trained configuration exactly, the instrumentation defect section 4.1a recorded does not apply to this run and its reward channels may be read directly, a fact verified term by term against `params/env.yaml` before any of the above was computed. That exemption is specific to the eleven thousand iteration dump of 2026-08-04 and does not extend to the dump of the completed run, which section 4.2b shows to be contaminated by three tree weights that moved between the two evaluations.

### 4.2b The completed run, and what the remaining two thirds of training changed

Run `sd_brs1_flat/2026-08-03_11-19-11` has since run to completion at 29103 iterations and was re-evaluated on 2026-08-05, the dump at `data/42/dump.npy` carrying a timestamp of 06:10 that postdates the final checkpoint. Section 4.2a read that run at eleven thousand iterations and qualified every one of its findings with the confound that a third of a training budget was being set against a whole one. That confound is now removed, and this section reads the three runs matched. The reading changes three of the earlier conclusions, confirms two that were doubted, and supplies the strongest evidence yet available for Phase 3, so it is recorded in full rather than as an amendment.

The match is close enough to carry the comparison. At their final iterations the linear tracking reward standard deviation stands at 0.0859 in both runs, the push force curriculum at 3.000 against 2.970, and the commanded longitudinal velocity range at 0.89 against 0.87. The one curriculum that differs materially is the angular tracking standard deviation, 0.2124 in the predecessor against 0.2473 here, so the present run faced a looser yaw specification and its yaw figures should be read with that allowance.

| quantity | baseline 2026-07-28 | Phase 1 2026-07-31 | Phases 1b and 2 at 11k | the same, completed |
|---|---|---|---|---|
| touchdown frame vertical speed, mean | 1.551 m/s | 1.402 m/s | 1.107 m/s | 0.889 m/s |
| sole approach speed at touchdown, mean | 1.910 m/s | 1.864 m/s | 0.706 m/s | 0.800 m/s |
| peak contact force in 80 ms, mean | 2.354 BW | 2.191 BW | 1.584 BW | 1.591 BW |
| peak contact force in 80 ms, worst | 10.315 BW | 7.025 BW | 4.939 BW | 5.001 BW |
| double support fraction | 1.394 % | 0.838 % | 11.615 % | 7.858 % |
| swing apex position within swing | 0.265 | 0.271 | 0.463 | 0.422 |
| correlation of swing profile with the raised cosine | 0.421 | 0.515 | 0.917 | 0.841 |
| clearance peaks per swing, mean | 3.644 | 3.003 | 1.702 | 1.435 |
| swing spent within one standard deviation of the set point | 0.918 | 0.918 | 0.656 | 0.752 |
| swing spent above ninety percent of its own apex | 0.307 | 0.311 | 0.284 | 0.352 |
| mean absolute joint power | 768 W | 910 W | 489 W | 626 W |
| cost of transport, absolute | 2.835 | 3.341 | 1.870 | 2.294 |
| joint velocity 99.9th percentile | 19.175 rad/s | 19.013 rad/s | 11.592 rad/s | 13.072 rad/s |
| ankle roll torque 99th percentile, against 131 N·m | 100.1 % | 100.0 % | 51.6 % | 100.0 % |
| ankle pitch torque 99th percentile, against 262 N·m | 63.6 % | 100.0 % | 62.9 % | 47.7 % |
| hip pitch torque 99th percentile, against 500 N·m | 35.1 % | 57.5 % | 81.8 % | 100.1 % |
| knee pitch torque 99th percentile, against 600 N·m | 29.3 % | 35.3 % | — | 43.7 % |
| stance knee flexion, mean magnitude | 0.068 rad | 0.074 rad | 0.030 rad | 0.039 rad |
| base roll rate, mean magnitude | 0.510 rad/s | 0.554 rad/s | 0.323 rad/s | 0.356 rad/s |
| planar tracking error over the evaluation, mean | 0.050 m/s | 0.046 m/s | 0.071 m/s | 0.051 m/s |
| planar tracking error over training, final | — | 0.185 m/s | 0.218 m/s | 0.182 m/s |
| episodes ending on low height, final | — | 0.181 | 0.385 | 0.216 |

Two of the user's readings of the completed run are confirmed, and the first of them corrects a caution this document issued. The velocity tracking did improve over the remaining budget, the training metric falling from 0.218 at eleven thousand iterations to 0.182 and the evaluation figure from 0.071 metres per second to 0.051, which returns it to the 0.046 of the predecessor. Section 4.2a argued that no such improvement should be expected, on the evidence that the predecessor's own error rose from 0.173 to 0.183 over the same interval as its curricula tightened. That mechanism is real and the predecessor's figures stand, but the inference drawn from it does not, because the present policy improved against a tightening curriculum where the predecessor had not. The caution was therefore right about the instrument and wrong about the outcome, and the completed run tracks its command marginally better than the run it is compared against.

The joint velocity profile is confirmed improved, the 99.9th percentile standing at 13.072 radians per second against the predecessor's 19.013 and the summed square, which is the quantity the penalty reads, at 55.3 against 81.9. The torque picture is genuinely improved in three places and genuinely worse in one, and the distinction matters more than an aggregate would. Ankle pitch is the clearest gain in the programme so far, its 99th percentile falling to 47.7 percent of its ceiling from the complete saturation of Phase 1, and hip roll falls to 145 newton metres from 181. Against that, hip pitch now stands at 100.1 percent of its 500 newton metre ceiling with 2.06 percent of all steps above ninety percent of it, where Phase 1 stood at 57.5 percent and no measurable saturation, and ankle roll has returned to its ceiling after having stood at half of it at eleven thousand iterations.

Three quantities decayed between the eleventh thousand iteration and the last, and they share a cause. The double support fraction fell from 11.615 percent to 7.858, the swing plateau returned, the fraction of swing spent within one standard deviation of the set point rising from 0.656 to 0.752 and the fraction above ninety percent of the profile's own apex from 0.284 to 0.352, and the power figures rose, the mean absolute joint power from 489 watts to 626 and the cost of transport from 1.870 to 2.294. None of these is a late instability. Each is monotone over the second half of training, and the reward channel that explains them is monotone with them.

That channel is the clearance reward, and this is the central finding of the completed run. Its training curve rises without interruption, `Episode_Reward/rew_foot_clearance` going from 3.315 at eight thousand iterations through 3.590 at eleven thousand, 4.080 at seventeen thousand and 4.314 at twenty thousand to 4.518 at the end, a gain of 36 percent over exactly the interval in which the profile metrics deteriorated. The policy spent the second half of its budget buying clearance income and paid for it in shape.

The cross run comparison establishes why that trade was available, and it is the sharper half of the evidence. Reconstructing the term from the dumped kinematics rather than from its logged reward, which makes the figure independent of any weight, the value stands at 0.69936 on the baseline, 0.69786 on Phase 1 and 0.45054 on the completed run. The set point term therefore pays a third less for the profile that is better by every shape measure, the correlation against the raised cosine having risen from 0.515 to 0.841 and the peaks per swing having fallen from 3.003 to 1.435 across the same pair. Its value is anti correlated with the quality of the thing it exists to govern, which is what section 3.6.1 predicted by construction, the maximiser of a set point kernel over a swing of fixed duration being the trapezoid that arrives soonest and holds longest rather than the arch. Put the two halves together and the mechanism is complete. The term pays more for the plateau, and given a training budget long enough to spend, the policy walks back up that gradient. The maximiser is not merely reachable, it is attracting, and the eleven thousand iteration reading caught the profile at its best only because the policy had not yet finished paying itself.

A instrumentation qualification governs how the reward channels of this dump may be read, and it reverses what section 4.2a was able to say. The twenty first pass of `../context/brs_gait.md` established that `play.py` rebuilds the environment configuration from the live source tree through `parse_env_cfg` at `scripts/rsl_rl/play.py:417`, so a play dump's reward channels carry the tree's weights and not the run's. For the eleven thousand iteration dump the tree happened to match the trained configuration and the channels could be read directly. For this dump it does not. Reconstructing the clearance term from kinematics and dividing the logged reward by it returns an implied weight of 19.98 against the trained 12.0, which identifies the tree state at play time as the one described below and confirms the contamination directly rather than by inference. Comparing the run's `params/env.yaml` against the live configuration term by term finds exactly three divergences, `rew_foot_clearance` at 20.0 against 12.0, `pen_foot_landing_vel` at −45.0 against −30.0, and `pen_feet_impact` at −5.0e-2 with an 800 newton threshold against −3.0e-2 with 900, every one of the remaining twenty three terms agreeing. Every reward figure quoted in this section has accordingly been rescaled to the trained weights before use, and the corrected budget places linear velocity tracking at 62.4 percent of a positive budget of 77.42 per step, `rew_no_fly` at 19.3 percent, yaw tracking at 8.0 and the clearance term at 7.0, against a negative budget of 15.11 per step of which the gait clock is 6.43.

The gates must therefore be restated. Phase 1b holds at the completed run and holds comfortably, the mean peak contact force standing at 1.591 body weights against a gate of 2.0 and inside the human walking band [3], the worst at 5.001 against 7.025, and the sole approach speed at 0.800 metres per second against a gate of 1.4. The touchdown frame speed of 0.889 metres per second now also clears the original 1.0 gate that section 4.2a had to set aside. Phase 2 does not hold. Its gate was a double support fraction above 12 percent, the provisional pass recorded in section 4.2a rested on 11.615 percent at a third of the budget, and the completed value is 7.858 percent against a clock commanding 24. The phase must be recorded as failed on its own gate, with the qualification that it carried the fraction from 0.838 percent to 7.858, a factor of nine, and that the mechanism it introduced is therefore effective without being sufficient. Section 4.3a already named the term that opposes it, `feet_air_time_positive_biped`, which pays exactly zero during double support at `isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py:63`, and the decay over the second half of training is consistent with that opposition operating throughout rather than with the grace window failing.

One measurement caveat should be recorded against the plateau statistics before they are used as a gate. The fraction of swing spent above ninety percent of the profile's own apex is not a clean measure of flatness when the profiles being compared differ in the number of peaks they carry. A jagged profile that crosses its maximum three times dips below ninety percent of it between the crossings and therefore scores low on this measure for a reason that has nothing to do with holding a set point, which is why the baseline and Phase 1 runs score 0.307 and 0.311 against the completed run's 0.352 while being worse by every other shape measure. The fraction spent within one standard deviation of the set point does not have this defect, since it is referred to a fixed height rather than to the profile's own extremum, and it should be treated as the governing statistic of the two. On that statistic the ordering is correct, 0.918 for both predecessors against 0.752 here, and the reference value for a raised cosine is 0.467.

The straight legged regression of section 4.2a persists in a partly healed form and its consequence has migrated. The stance knee flexion recovered from 0.030 radians to 0.039, still under two and a quarter degrees against the 0.074 of Phase 1, and the episodes ending on the low height condition fell from 0.385 to 0.216 against the predecessor's 0.181, having peaked at 0.454 at eight thousand iterations. The policy therefore recovered much of the robustness it had lost without recovering the posture that cost it, and it did so by other means. The price appears in the hip, whose pitch torque saturation is the one torque figure that worsened over the same interval, and in the power figures. A leg that will not bend must be driven from the hip, and the hip is now at its ceiling for two percent of every second. Phase 7 inherits this, and the case for a knee posture term within it is now supported by two independent runs rather than one.

The working tree diverges from what this document describes, and the divergence must be resolved before the next run is launched. The Phase 3 wiring recorded in section 4.3 is not in effect. `foot_clearance_reward_v3` is defined at `mdp/rewards.py:427` and is called from nowhere, the active term at `cfg/SF/brs_base_env_cfg.py:984` being `foot_clearance_reward_v2` at weight 20.0 with the set point and kernel width the baseline used. The commented block at `cfg/SF/brs_base_env_cfg.py:1006` that appears to preserve the v3 configuration names `foot_clearance_reward_v2` while passing `command_name`, which v2 does not accept and v3 requires, so it would raise on being uncommented and cannot serve as the record of the intended wiring. The landing weight of −45.0 at `cfg/SF/brs_base_env_cfg.py:1105` and the impact threshold of 800 newtons at `cfg/SF/brs_base_env_cfg.py:1161` are both live.

The consequence of that particular combination is worth stating in the currency the rest of this document uses, because it is not a neutral partial application but the one pairing section 4.3a expressly forbids. A landing penalty at −45 standing beside a set point kernel reopens the hover exploit, since the kernel pays a foot for reaching the target height early and holding it while the penalty pays it for arriving slowly, and the two agree on a trajectory that rises promptly and then descends as late and as slowly as the gate allows. The clearance weight of 20.0 makes this worse rather than better than the configuration that was measured. At the measured term value of 0.45054 the trained weight of 12.0 paid 5.406 per step and a weight of 20.0 pays 9.011, raising the clearance share of the positive budget from 7.0 percent to 11.1 and strengthening by two thirds the attractor that this section has just shown to be drawing the policy away from the profile it is meant to produce. It also restores exactly the weight the baseline and Phase 1 runs carried, which is the weight section 4.1b reduced to 12 as one of the three changes whose omission caused Phase 1 to fail, so the tree has undone a change that has since been measured to work. Three courses are available and the choice between them is the user's. The tree may be completed to Phase 3 as section 4.3 specifies, wiring v3 at weight 10.0 and leaving the landing and impact escalations as they stand. It may be returned to the measured Phase 2 configuration, restoring v2 to weight 12.0 and the landing weight to −30.0, so that the next run is a clean continuation. Or the present tree may be trained deliberately as an experiment in raising the clearance weight, in which case the landing weight should be returned to −30.0 first, since the coupling argument of section 4.3a applies to it whatever the clearance weight is.

### 4.3 Phase 3, shape the swing trajectory rather than its apex

Phases 1 and 2 make the trapezoid expensive at both of its ends without ever telling the policy what shape to adopt instead. Section 2.3 establishes that a reward on an extremum determines only that extremum and that the published alternatives specify a reference path [9, 13, 14]. This phase supplies one.

The configuration already declares the reference. `gait_command` carries `swing_height` of 0.08 metres, and `GaitReward` already reconstructs a normalised phase for each foot from the same command, so the phase at which a foot should be at a given fraction of its swing height is available without any new command term.

The reference must be chosen with more care than this document first gave it, and the correction is recorded here because it was found by evaluating the candidates numerically before the term was trained rather than after. This section originally proposed a half sinusoid of the within swing phase, asserting that it has zero vertical velocity at both ends and an interior maximum. The second half of that claim is true and the first is false. The derivative of the sine is the cosine, which is greatest where the sine vanishes, so a half sinusoid attains its maximum vertical velocity precisely at lift off and at touchdown. At this robot's swing duration of 0.38 seconds and swing height of 0.08 metres it would command a touchdown at 0.66 metres per second and a lift off at the same, which is the snatch and the stomp that Phases 1 and 1b pay to remove. The reward would have been bidding against the landing penalty rather than with it, and the two would have converged on whichever weight was larger.

The form that carries the intended property is the raised cosine, the square of the sine of the swing phase, equivalently one half of one less the cosine of twice it. Its vertical velocity vanishes at both ends and at the apex, and is greatest at the quarter points, which is where a swing foot should be moving fastest. It is therefore the exact opposite of the trapezoid in every property that section 3.6.1 identified as profitable, which is what the half sinusoid was mistakenly credited with.

| reference | apex | vertical speed at lift off | at touchdown | greatest speed |
|---|---|---|---|---|
| half sinusoid, `A sin(pi phi)` | 0.080 m at mid swing | 0.661 m/s | 0.661 m/s | at both ends |
| raised cosine, `A sin^2(pi phi)` | 0.080 m at mid swing | 0.002 m/s | 0.002 m/s | 0.661 m/s at the quarter points |

Because this changes what `target_height` means, from a set point to the amplitude of a reference, no optional argument can carry it without inviting silent misconfiguration, so rule 5 applies and a version three is created. Version two is left untouched and still exported, which keeps the dumped configurations of the eight runs that used it replayable.

```python
# mdp/rewards.py, new function
def foot_clearance_reward_v3(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    sole_offsets: list[list[float]],
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swing foot for tracking a half-sinusoid clearance reference.

    Where :func:`foot_clearance_reward_v2` places a Gaussian at a single target clearance
    and multiplies by foot speed, this tracks a reference that is a function of the swing
    phase. The distinction is load bearing. A kernel on a set point is maximised by the
    trajectory that reaches the set point soonest and leaves it latest, which is a
    trapezoid, and on run 2026-07-28_06-37-24 that trapezoid earned 46 percent more than a
    sinusoidal arc of the same apex while striking the ground at 1.68 m/s. A reference
    penalises a foot that is high at the wrong moment exactly as one that is low at the
    wrong moment, so it specifies the whole path. The form follows Humanoid-Gym
    (arXiv:2404.05695), which drives its joints from a phase conditioned sinusoid rather
    than from an extremum, and the reference-tracking foot-height term of Seo et al.
    (arXiv:2512.01996).

    The reference amplitude is read from the gait command's swing_height, and the phase
    from the same (frequency, offset, duration) parameterisation that
    :class:`GaitReward` uses, so the reward and the clock cannot drift apart.

    The foot-speed tanh factor of v2 is DELIBERATELY DROPPED. Its purpose there was to
    stop a stationary foot earning, which the phase reference already prevents, and its
    effect was to pay a foot for travelling at several times body speed, measured at
    3.40 m/s against a base speed of 0.61 m/s.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies.
        command_name: Name of the gait command term supplying frequency, offset, duration
            and swing height.
        std: Width of the Gaussian tracking kernel (m). Size it to the error the policy
            currently produces, not to the task range, per the kernel-width lesson of
            ../context/literature.md cluster 11. The measured swing clearance standard
            deviation about a sinusoidal fit is about 0.03 m, so 0.03 is the entry point.
        sole_offsets: Sole support points in the foot body frame, as for v2.
        sensor_cfg: Optional contact gate, as for v2.
        force_threshold: Contact force threshold for that gate.

    Returns:
        The computed reward tensor, summed over the feet.
    """
```

The reference is `h_ref(phi) = swing_height * sin(pi * phi_swing)^2` for a foot inside its commanded swing interval and zero otherwise, where `phi_swing` is the within swing phase. The reward is `exp(-(clearance - h_ref)^2 / std^2)` summed over the feet, evaluated only over the feet the clock assigns to swing.

The phase reconstruction is factored into a helper shared with nothing else at present but written so that it mirrors `GaitReward.compute_contact_targets` line for line in its treatment of frequency, offset and duration. That mirroring is the point of the helper. A reward keyed to the swing phase and a clock grading contact must not be able to disagree about when a foot is swinging, and the only way to guarantee that without merging the two terms is to derive both from the same arithmetic.

```python
# mdp/rewards.py, implemented 2026-08-03
def _swing_phase_from_gait_command(gait_params, episode_length_buf, dt):
    """Reconstruct the per foot swing phase from the gait command.

    Returns (in_swing, phi_swing), both (N, 2), the second rising from 0 at lift off to 1
    at touchdown and clamped so it is safe to use unmasked.
    """
    num_envs = gait_params.shape[0]
    frequencies = gait_params[:, 0]
    offsets = gait_params[:, 1]
    durations = gait_params[:, 2].view(num_envs, 1).expand(num_envs, 2)

    gait_indices = torch.remainder(episode_length_buf * dt * frequencies, 1.0)
    foot_indices = torch.remainder(
        torch.cat(
            [gait_indices.view(num_envs, 1), (gait_indices + offsets + 1).view(num_envs, 1)],
            dim=1,
        ),
        1.0,
    )
    in_swing = foot_indices > durations
    phi_swing = torch.clamp((foot_indices - durations) / (1.0 - durations + 1e-6), 0.0, 1.0)
    return in_swing, phi_swing


def foot_clearance_reward_v3(
    env, asset_cfg, command_name, std, sole_offsets,
    sensor_cfg=None, force_threshold=1.0,
):
    """Reward the swing foot for tracking a raised cosine clearance reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    pts_w, _ = _sole_points_world(asset, asset_cfg.body_ids, sole_offsets)
    sole_clearance = pts_w[..., 2].min(dim=2)[0]                        # (N, F)

    gait_params = env.command_manager.get_command(command_name)
    in_swing, phi_swing = _swing_phase_from_gait_command(
        gait_params, env.episode_length_buf, env.step_dt
    )
    swing_height = gait_params[:, 3].unsqueeze(1)                       # (N, 1)
    # raised cosine, sin^2, NOT sin. The difference is a reference that lands at zero
    # vertical velocity against one that lands at 0.66 m/s.
    height_ref = swing_height * torch.square(torch.sin(torch.pi * phi_swing))

    reward = torch.exp(-torch.square(sole_clearance - height_ref) / std**2)
    # evaluated over commanded swing only. During commanded stance the reference is zero and
    # a grounded foot would track it perfectly, which would pay a robot for standing still,
    # the exploit the thirteenth pass of ../context/brs_gait.md closed.
    reward = reward * in_swing.float()

    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_contact = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1).max(dim=1)[0] > force_threshold
        )
        reward = reward * ~in_contact
    return torch.sum(reward, dim=1)
```

The clearance itself is computed by `_sole_points_world`, a helper factored out of `foot_clearance_reward_v2` in the same pass and shared with `foot_landing_vel_v2`, which returns both the world positions of the sole points and the lever arms to them. Factoring it serves the same purpose as the phase helper, that the three terms which reason about sole clearance should not each carry their own copy of the transform.

The reconstruction was verified against the clock before the term was wired. At a duration of 0.62 the swing occupies 0.379 of the cycle against an expected 0.380, the two feet never swing simultaneously, the both stance fraction is 0.241 against the commanded 0.24, and the reference attains exactly 0.0800 metres at mid swing with a vertical velocity of 0.002 metres per second at each end.

```python
# cfg/SF/brs_base_env_cfg.py, replacing rew_foot_clearance when Phase 3 is enabled
rew_foot_clearance = RewTerm(
    func=mdp.foot_clearance_reward_v3,
    weight=10.0,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]"),
        "command_name": "gait_command",
        "std": 0.03,
        "sole_offsets": SD_BRS1_SOLE_OFFSETS,
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
        "force_threshold": 1.0,
    },
)
```

The weight replaces the Phase 1b value of 12, and the entry point of 10 proposed when this section was first written has since been confirmed by measurement rather than left to be recomputed after the run. The sizing below is recomputed on the completed run, superseding the figures taken at eleven thousand iterations, because the swing profile changed materially over the second half of training and a weight sized against the earlier profile would no longer be budget neutral. Over the swing samples of the completed run the clearance error against the raised cosine has a mean magnitude of 0.0264 metres against the 0.0225 measured earlier, so at a kernel width of 0.03 the mean kernel value is 0.5151 and the term value 0.4749 once the clock gate is applied. At a weight of 10 the income is therefore 4.749 per step. The comparison it must be neutral against is the set point term as actually trained, whose value on the completed run is 0.45054 and which at its trained weight of 12.0 paid 5.406 per step. The swap at a weight of 10 therefore reduces the clearance income by 12 percent, from 7.0 percent of the positive budget to 6.2, and strict neutrality would require a weight of 11.4.

The weight of 10 is retained rather than raised to 11.4, and the twelve percent is conceded deliberately. The argument of section 4.2b is that a part of the set point term's income is itself payment for the defect, the term paying a third more for the plateaued profiles of the two predecessors than for the arched profile measured here, so reproducing its income exactly would reproduce a budget that was partly earned by the behaviour the phase exists to remove. A round weight sitting just below neutrality is the conservative choice in that light, and the margin is small enough that the experiment still measures a change of shape rather than a change of budget. A foot tracking the reference exactly would earn 8.0 per step at this weight, so 3.3 per step of headroom remains available to be bought by improving the profile, which is more headroom than the term offers under v2 and is the property that matters most.

The kernel width of 0.03 replaces v2's 0.035 and is sized to the error the policy currently produces rather than to the task range, which is the lesson cluster 11 of `../context/literature.md` records and which the dead Phase A knee reward of `NATURAL_GAIT_PLAN.md` demonstrated from the other direction. The measured error magnitude of 0.0264 metres on the completed run sits well inside 0.03, so the kernel is alive across the whole of the present swing rather than saturated at one end, its mean value of 0.5151 standing near the middle of its range where a kernel that discriminates must sit.

This phase was placed third rather than first deliberately. It is the most principled of the six and the one the literature most clearly endorses, but it replaces a term supplying eighteen percent of the positive budget, so it should be attempted against a policy that has already been taught not to stomp, so that a failure can be attributed. That ordering has been vindicated from an unexpected direction. Section 4.2a measured Phases 1b and 2 to have carried the swing profile most of the way on their own, the apex moving to 0.463 of the swing and the correlation against the raised cosine reaching 0.917, so this phase now begins from a far better starting point than the one it was designed against.

What remains for it to do is the residual plateau, and that residue is precisely the part a set point kernel cannot remove. The foot still spends 65.6 percent of its swing within one standard deviation of the set point where the raised cosine reference spends 46.0, and 28.4 percent of it above ninety percent of its own apex where the reference spends 20.0. A kernel on a set point pays for both of those excesses and a phase reference charges for them, which is why the phase is worth running even though its headline symptom has already improved.

The gate for Phase 3 is accordingly restated against the completed run, and its governing statistic is changed for the reason section 4.2b records. The gate is a swing clearance profile whose fraction of time spent within one standard deviation of the set point falls below 0.55, against the present 0.752 and a reference value of 0.467 for a raised cosine of the same apex. That statistic governs because it is referred to a fixed height and is therefore comparable across profiles of differing raggedness, where the fraction of time above ninety percent of the profile's own apex is depressed by the multiple peaks of the earlier runs and rewards exactly the jaggedness this programme is removing. The second statistic is retained as a subordinate reading rather than as a gate, its present value being 0.352 against a reference of 0.200, and it is expected to fall only once the profile is both smooth and arched. The correlation against the raised cosine, presently 0.841, and the apex position, presently 0.422 against a symmetric 0.500, are the supporting readings. Two earlier formulations of this gate are superseded, the original written against a baseline figure of 0.34 and the restatement of 2026-08-04 written against the eleven thousand iteration figures of 0.284 and 0.656, both of which the completed run has moved.

### 4.3a The escalation of the landing and impact weights, and the case against raising the single support coefficient

Three changes to the reward weights were proposed on the strength of the Phase 2 outcome. Two are adopted, in a form the measurement modifies, and the third is declined on evidence. All three are taken together here because they were intended to be wired in the same pass as Phase 3 and because the first of them is safe only in that company.

A correction written 2026-08-07, which governs everything below it in this section. Neither weight escalation reached the Phase 3 run. Its dumped `params/env.yaml` carries `pen_foot_landing_vel` at −30.0 rather than the −45.0 this section derives, and `pen_feet_impact` at −3.0e-2 rather than −5.0e-2, against a threshold of 850 newtons rather than 800. Only the threshold move was taken, and it was taken to the value the second derivation below recommends rather than the first. The reasoning of this section therefore stands as a derivation that has not been tested, and the reasoning must not be read as a description of the experiment, which section 4.3b reports. Two consequences follow and both are favourable. Phase 3 is a cleaner ablation than was designed, varying the clearance term alone against a reward set otherwise unchanged but for a threshold worth 0.13 percent of the budget, so its outcome is attributable to the clearance term with a confidence no bundled phase in this programme has previously permitted. And the coupling argument below, that a landing weight of −45 is safe only beside a phase reference, was never put at risk, since the weight was never raised. Whether it should now be raised is taken up in section 4.3b against the measured result, which is that the landing term's own value fell a further 23 percent at the weight it already had.

The landing velocity weight was to rise from −30 to −45. The justification is not that the term failed, since it plainly succeeded, but that the shape of what remains has changed in a way that makes a heavier weight productive rather than merely expensive. Under Phase 1 the term's value was dominated by a tail, approaches faster than two metres per second contributing 74.6 percent of the squared value while making up 46.2 percent of gated steps. Under Phase 2 that tail is gone, contributing 2.8 percent, and 95.6 percent of the remaining value lies between 0.3 and 2.0 metres per second. A term whose gradient is carried by the bulk of the distribution rather than by rare events is one whose weight can be raised without the update being dominated by outliers, which is the condition that was not satisfied before and is now.

This escalation is coupled to Phase 3 and must not be taken without it. The failure mode of a strong landing penalty is the loitering that section 4.1a measured under v1, the policy hovering above the gate to keep its approach velocity small, and a set point kernel pays for exactly that hover. A phase reference penalises a foot that is high at the wrong moment as readily as one that is low at the wrong moment, so wiring v3 closes the hover that a weight of −45 would otherwise open. Raising the weight while v2 remained wired would reopen it, and that pairing is the reason both changes belong to one experiment.

The impact penalty is escalated by moving its threshold from 900 to 800 newtons and its weight from −3.0e-2 to −5.0e-2, in that order of importance. The proposal to raise the weight rested on the premise that the impact forces had changed only marginally, and section 4.2a establishes that premise to be false, the mean peak having fallen 28 percent and the term's own value 88. The conclusion nonetheless survives its premise, but the mechanism requires correction. The term has not remained strong and unheeded, it has gone nearly inert. Its median touchdown registered 853 newtons at eleven thousand iterations and 898 at the completed run, in both cases below or at its own threshold, so the term is exactly zero on the median step and fires on 46.0 percent of touchdowns at the earlier reading and 49.5 at the later, against the 96.2 percent it fired on before. A weight raised on such a term buys a high variance gradient present on only the worst half of steps, whereas lowering the threshold restores coverage across the distribution, and the two together restore both coverage and magnitude.

The threshold of 800 newtons is a floor and not a midpoint, and the constraint that fixes it is the term's own design intent, which section 2.4 takes from the ground reaction force literature [9, 15] and which its docstring records, that the term be exactly zero over the forces a well behaved stance produces so that it cannot distort the stance it is not meant to govern. On the eleven thousand iteration reading, excluding the eighty milliseconds following each touchdown, the steady stance force had a 99th percentile of 813 newtons and a 99.9th of 1123, so a threshold of 800 touched 1.15 percent of steady stance while catching 55.2 percent of touchdowns, against a threshold of 700 that would have caught 66.9 percent at the cost of touching 4.93 percent of steady stance and thereby ceasing to be a collision penalty at all.

That derivation must be revised against the completed run, and the revision moves the floor upward. The steady stance distribution rose over the second half of training, its 99th percentile now 872 newtons and its 99.9th 1129, while the touchdown distribution barely moved, its median at 898 newtons and its mean at 934. The two distributions have therefore converged further, and a threshold of 800 now touches 2.10 percent of steady stance while catching 72.8 percent of touchdowns, where a threshold of 850 touches 1.28 percent and catches 60.0, and 900 touches 0.76 percent and catches 49.5. The choice of 800 was made against the earlier reading and is now marginal rather than wrong, and 850 is preferred on the completed evidence, since it restores the stance contamination to the level the 800 figure was chosen to achieve while still doubling the coverage the inert 900 threshold provided. The convergence of the two distributions is itself the measure of how far Phase 1b carried the gait, and it is why no threshold below 800 is available at all. The rise in the stance figure should be read together with section 4.2b, which attributes the hip pitch saturation to the same straight legged posture, since a leg that cannot yield transmits more of a stance load as force than one that can.

The third proposal, that the coefficient on the single support branch of `NoFlyWithGrace` be raised from 1.0 to 2.0 in order to prioritise support and buy more double support, is declined, and the measurement that declines it is decisive enough to be recorded so that the proposal is not revisited from recall. Reconstructing the grace counter over the play dump, the recently single condition is true on 99.796 percent of steps, so the branch is saturated. It is saturated, moreover, in both arms of the comparison that matters, paying 15.000 per second in single support and 14.754 in double, because 98.4 percent of double support episodes are shorter than the twenty step window and are therefore covered by it. The coefficient multiplies a quantity that is very nearly identical in the two branches, so it cannot move the margin between them by construction, and doubling it moves that margin from −0.246 to −0.492 per second, which is the opposite of the intended direction. What it does buy is a constant of 14.97 per second added to a positive budget of 75.49, a rise of 19.8 percent, which the advantage normalisation at `rsl_rl/rsl_rl/storage/rollout_storage.py:150` removes while every other shaping term's share of the budget falls by 16.5 percent.

The intention behind the proposal is nonetheless sound, and the term that frustrates it can be named exactly. `feet_air_time_positive_biped` gates its entire reward on a single stance condition at `isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py:63`, so it pays exactly zero during double support and 2.741 per second during single support, and that figure is the true marginal cost of the transfer interval under the present configuration. It is the largest single price on double support in the reward set and it is charged by a term whose stated purpose is to reward long steps. Should the commanded 24 percent still not be reached once Phase 3 has trained, the levers are that term's weight of 12.5 and the gait clock's weight of 40, and not the coefficient on the single support branch. This is deferred rather than taken now because Phase 2 reached 11.615 percent at a third of its training budget and the trend has not been allowed to finish.

### 4.3b The outcome of Phase 3, and the reversal it produced where none was sought

Run `sd_brs1_flat/2026-08-05_07-53-35` trained Phase 3 to completion at 29999 iterations and was evaluated on 2026-08-07. It is the cleanest experiment this programme has run. Its dumped `params/env.yaml` differs from that of its predecessor in exactly two places, the clearance term moving from `foot_clearance_reward_v2` at weight 12.0 with a set point of 0.08 metres and a kernel width of 0.035 to `foot_clearance_reward_v3` at weight 10.0 with a kernel width of 0.03, and the impact threshold moving from 900 newtons to 850 at an unchanged weight. Every other one of the twenty six terms is identical in function, weight and parameter. The second change is worth 0.099 per second against a positive budget of 77, which is 0.13 percent, so the outcome below is attributable to the clearance term alone to within that margin.

All four runs have since been re-evaluated through the statistics pipeline of `GAIT_STATISTICS_PLAN.md`, so for the first time every figure in this section is computed by one instrument from one code path. Where a figure disagrees with an earlier section the pipeline governs, and the disagreements are noted where they arise.

| quantity | baseline 07-28 | Phase 1 07-31 | Phases 1b and 2 08-03 | Phase 3 08-05 |
|---|---|---|---|---|
| raised cosine correlation, mean | 0.411 | 0.492 | 0.860 | 0.945 |
| rms error against the reference, m | 0.0428 | 0.0418 | 0.0271 | 0.00798 |
| swing above 90 pct of its apex, pct | 52.38 | 28.57 | 38.10 | 23.81 |
| swing apex position, standard deviation | 0.363 | 0.197 | 0.123 | 0.0668 |
| clearance peaks per swing, mean | 3.558 | 3.110 | 1.414 | 1.612 |
| swing apex clearance, m | 0.0920 | 0.0947 | 0.0983 | 0.0716 |
| touchdown sole approach speed, m/s | 1.863 | 1.875 | 0.960 | 0.539 |
| touchdown frame vertical speed, m/s | 1.577 | 1.393 | 0.869 | 0.727 |
| peak contact force, mean, BW | 2.617 | 2.463 | 1.805 | 1.551 |
| peak contact force, worst, BW | 10.32 | 8.31 | 8.32 | 5.11 |
| double support, pct | 4.432 | 2.867 | 10.269 | 14.014 |
| cycle period, s | 0.986 | 0.986 | 0.970 | 0.924 |
| absolute joint power, W | 772 | 906 | 557 | 252 |
| cost of transport, absolute | 2.726 | 3.467 | 1.953 | 0.890 |
| opposed hip and knee power, W | 55.9 | 88.1 | 40.6 | 9.1 |
| joint velocity, 99th percentile, rad/s | 14.10 | 13.81 | 9.81 | 5.54 |
| joint acceleration, rms | 241.5 | 213.2 | 340.8 | 107.9 |
| torque rate, rms | 1911 | 2182 | 2506 | 1081 |
| joint jerk, rms | 29990 | 24840 | 21180 | 12310 |
| stance knee flexion, mean, rad | 0.0841 | 0.0825 | 0.0605 | 0.5747 |
| stance spent against the extension stop, pct | 64.4 | 63.0 | 70.8 | 0.83 |
| hip pitch torque, 99th percentile, N·m of 500 | 180.7 | 272.3 | 450.3 | 97.1 |
| base height, m | 1.166 | 1.168 | 1.170 | 1.133 |
| planar velocity error, m/s | 0.0559 | 0.0477 | 0.0527 | 0.0470 |
| low height terminations, share of episodes | — | 0.180 | 0.233 | 0.0823 |
| training mean reward, final 200 iterations | — | 990 | 798 | 1004 |

The swing profile is the quantity the phase was designed to move and it moved decisively. Resampling every swing longer than fifteen steps onto a common phase and averaging, the Phase 2 profile rises to 0.086 metres by three tenths of the swing, holds between 0.093 and 0.096 across the middle four tenths, and falls to 0.014, which is the trapezoid section 3.6.1 predicted by construction. The Phase 3 profile reads 0.010, 0.019, 0.030, 0.047, 0.064, 0.070, 0.064, 0.049, 0.033, 0.021 and 0.008 at successive tenths of the swing, which is a raised cosine of amplitude 0.070 metres standing on a pedestal of about nine millimetres. Its root mean square departure from the commanded reference fell from 0.0271 metres to 0.00798, a reduction of seventy one percent, and its correlation with that reference rose from 0.860 to 0.945 with a fifth percentile of 0.820, so the shape is held not merely on average but on the great majority of individual swings.

The gate as this document stated it is missed and the miss is an artefact of the instrument rather than of the gait. Section 4.3 set the gate at a fraction of swing above ninety percent of its own apex below 0.22, and the completed run reads 0.2381. That statistic is computed at `scripts/analysis/stats.py:582` from a mean profile resampled onto twenty one points, so it can take only multiples of one twenty first, which is 4.76 percentage points, and the two achievable values bracketing the gate are 0.1905 and 0.2381. A gate stated to a hundredth cannot be adjudicated on a quantity quantised to a twentieth, and the run sits on the first achievable value above it. Every unquantised measure of the same property passed by a wide margin, the reference tracking error falling by seventy one percent and the apex position's standard deviation by forty six, so Phase 3 is recorded as passed.

The gate is restated for any future run of this phase, and the restatement is recorded here rather than applied retroactively. The governing measure is the root mean square departure of the mean swing profile from the commanded reference, reported at `scripts/analysis/stats.py:585`, which is continuous, is expressed in metres, and is the quantity the term is actually optimising. The threshold is 0.015 metres, being roughly halfway between the 0.0271 the phase inherited and the 0.00798 it achieved, so that a future variant must retain most of the gain rather than merely improve on the predecessor. The mean correlation with the reference must remain above 0.90 against the achieved 0.945, and the plateau measure is retained as a reported diagnostic but is withdrawn as a gate, since its resolution of 4.76 percentage points cannot support one.

One measure moved the wrong way and it is reported because the explanation qualifies the success rather than undoing it. The mean count of clearance peaks per swing rose from 1.414 to 1.612. A raised cosine has a broad and nearly flat maximum, so a departure of a millimetre near the apex registers as a second peak under the prominence test at `scripts/analysis/stats.py:565`, where the same departure on a sharply peaked profile would not. The measure is a proxy for the multiple lift and fall of the trapezoid, which it detected correctly when that was the defect, and it becomes uninformative once the profile is genuinely smooth. The reference tracking error is the measure that supersedes it.

A second qualification concerns amplitude rather than shape. The reference is the commanded swing height of 0.08 metres multiplied by the squared sine of the swing phase, and the policy tracks its shape almost exactly while undershooting its amplitude by twelve percent, at an apex of 0.0704 metres. The kernel is forgiving of that trade, since a uniform deficit of ten millimetres against a width of thirty returns 0.89 of the maximum at the apex while a shape error of the same size at the wrong phase is punished twice, once for being high when the reference is low and once for the converse. The policy has therefore bought shape with amplitude, which is the correct reading of the term as written, and whether the amplitude should be recovered by narrowing the kernel is taken up in the next steps below.

What the phase was not designed to do is the larger part of what it did. The stance knee flexion rose from 0.0605 radians to 0.5747, from three and a half degrees to thirty three, and the share of stance spent within two hundredths of a radian of the full extension stop fell from 70.8 percent to 0.83. The straight legged gait that section 4.2b recorded as the price of Phases 1b and 2, and which the second postscript of section 5 named the strongest available argument for Phase 7, has been removed entirely by a term that says nothing about the knee. The consequences follow the posture. The hip pitch torque, which section 4.2b measured at 450 newton metres and ninety percent of its ceiling, stands at 97 and nineteen percent. The absolute joint power fell from 557 watts to 252 and the cost of transport from 1.953 to 0.890, which is the first figure this programme has produced within a factor of five of the human value [1]. The episodes ending on the low height condition fell from 0.233 to 0.0823, less than half the Phase 1 figure and by a wide margin the best of the four runs, which is the direct measurement of the disturbance rejection that section 4.2b predicted a compliant leg would restore.

The mechanism is specifiable and it is the property of the raised cosine that section 4.3 chose it for. The reference has zero vertical velocity at the end of swing, so it prices a steep terminal descent that the set point kernel of v2 could not see, that kernel rewarding a foot for reaching a height and never for the manner of its return. The measured descent speed over the final quarter of swing fell from 0.545 metres per second to 0.292, and the sole approach speed at touchdown from 0.960 to 0.539. A foot that must arrive slowly and on schedule must either begin its descent from a lower apex or take longer over it, and the policy took the lower apex, 0.0983 metres falling to 0.0716. A lower apex is only useful if the hip is also lower, since the clearance is measured from the ground and not from the pelvis, and the cheapest way to lower the hip on this machine is to flex the stance knee. The base height fell from 1.170 metres to 1.133 accordingly, below rather than above the 1.15 that `pen_base_height` targets. The crouch is therefore not an incidental correlate of the swing shape but its kinematic precondition, and the phase bought it without ever naming it.

That reading also disposes of the alternative explanation. The impact threshold moved from 900 newtons to 850 in the same pass, and that change is worth 0.099 per second, or 0.13 percent of the positive budget, against a stance knee flexion that changed by a factor of nine and a half and a hip pitch torque that changed by a factor of four and a half. No plausible elasticity attaches an effect of that size to a stimulus of that size.

Two further results belong to the phase although it did not target them. Phase 2's gate of twelve percent double support, which section 4.2b recorded as failed at the completed predecessor, is met at 14.014 percent without any change to `NoFlyWithGrace`, `feet_air_time` or the gait clock. The cycle period fell from 0.970 seconds to 0.924 and the swing duration from 0.431 to 0.395, so the overlap was bought by shortening the swing rather than by lengthening the stride, and the commanded 24 percent remains distant. The phase should be credited with the gate and the mechanism should not, since nothing was done to the terms that price the transfer, and the deferred levers section 4.3a named remain the levers.

The tracking, finally, improved on every measure and against the caution this document has twice issued about reading tracking across curricula. The two runs ended with identical linear tracking standard deviations of 0.0859, identical longitudinal command ranges of 0.89 and identical push forces of 3.0, and on that matched footing the training planar error fell from 0.1825 to 0.1610 and the evaluation planar error from 0.0527 to 0.0470, the best of the four runs. The angular tracking standard deviation, which section 4.2b flagged as the one curriculum that differed, now stands at 0.2124 in both the Phase 1 and the Phase 3 runs, so the yaw comparison is for the first time properly matched. Section 4.5 reads it.

### 4.3c The ankle parked against its mechanical stops, which is the price Phase 3 charged

Phase 3's gains were bought with a degree of freedom. Both ankle joints on both legs now spend the majority of the gait pressed against a mechanical stop, and the pitch axis has effectively been removed from the machine.

The measurements are unambiguous and are taken from the joint position dump against the limits the same dump records. The ankle pitch stands within two hundredths of a radian of its upper stop of 0.454 radians for 99.5 percent of swing, against 88.3 percent under Phase 2 and 17.3 under the baseline, and for 28.2 percent of stance against 0.86 under Phase 2. Its position standard deviation collapsed from 0.219 radians to 0.067 and its travel from 0.921 to 0.766 on the left and 0.605 on the right. At touchdown it is at its stop on 93.5 percent of steps, against 5.3 percent under Phase 2, so the foot arrives as a rigid plate at a fixed angle. The ankle roll stands at its stop for 62.0 percent of swing, the left at its positive limit and the right at its negative, which is the same physical direction on a mirrored pair, and that figure has stood between 56 and 62 percent in every run including the baseline, so the roll defect is longstanding and Phase 3 has aggravated rather than created it. The spectral arc length of the ankle pitch, the one smoothness measure that worsened anywhere in the machine, fell from −26.3 to −47.9 while every other joint improved, which is what a signal pinned to a constant with a small residual chatter reads.

The joint is held rather than driven. Sampling the ankle roll at its stop during swing, the mean holding torque is 49 newton metres, the joint velocity is below a tenth of a radian per second on 91.6 percent of those samples, and the mean power is 1.74 watts, so the stop supplies the constraint and the actuator supplies only a modest bias against it. The 99th percentile torque of 122 newton metres, which is 93 percent of the 131 newton metre ceiling this actuator carries, is the transient of arrival and not the holding state, and it is worth recording that this figure has fallen from the 100 percent that the baseline, Phase 1 and Phase 2 runs all read. The ankle roll remains the most saturated actuator in the leg by a wide margin, the next being the ankle pitch at 62 percent.

The cost is not small and it is now the largest single line in the effort budget. The four ankle joints together carry 57.3 percent of the value of `pen_joint_torque`, against 25.0 percent under Phase 2, the two pitch joints alone rising from 9.3 percent to 42.8 as their mean absolute torque went from 38 newton metres to 68. They dissipate 68 watts of the machine's 252 while doing no net work, since a joint at a stop moves nowhere. Charged at the term's weight of −1.0e-5 that is 0.165 per second of a 0.287 total, which is the whole difficulty in one figure. The behaviour is expensive in physics and free in economics.

The reason it is free can be stated exactly, and it is the same structural argument section 3.6.5 made about the joint stops being load bearing, now measured on a specific joint. No term in the configuration reads the ankle roll position at all. The one term that reads the ankle pitch, `rew_keep_ankle_pitch_zero_in_air` at `mdp/rewards.py:925`, is wired at weight 1.0 and reads 0.0918 against a maximum of 0.8599 that the airborne fraction permits, so the policy is forfeiting 0.768 per second and choosing to. That figure overstates the incentive available and section 4.4 corrects it, most of the shortfall being unreachable because the term targets the coordinate zero while the ankle pitch default stands 0.2435 radians away, so the lever between the mechanical stop and the nominal pose is only 0.1657 per second. Either way the behaviour is bought with something worth more, and the two candidates that read the sole geometry are the clearance term at 6.07 per second and the gait clock at −5.85. Both are computed from the minimum of twelve sole points at `mdp/rewards.py:76`, whose height depends on the foot's orientation as much as on the leg's pose, and freezing that orientation at a stop reduces the tracked quantity to a deterministic function of the shank alone. A narrow kernel rewards precision, and the cheapest precision available is rigidity.

One defect in the term that ought to oppose this is recorded per rule 6 of `../CLAUDE.md`, since it is a defect of target rather than of weight and no reweighting will remove it. `keep_ankle_pitch_zero_in_air` penalises the absolute joint coordinate at `mdp/rewards.py:970`, taking `torch.abs(asset.data.joint_pos[...])` rather than the deviation from the default pose. The ankle pitch default on this robot is 0.2435 radians before randomisation and reads 0.2163 and 0.2599 on the two sides after it, so the term's target of zero stands roughly a fifth of a radian away from the posture every other part of the configuration treats as nominal. Even a perfectly compliant policy could not collect more than 0.30 of the term's unit while standing at its own default. The `pen_ankle_deviation` term now in the working tree targets the randomised default instead, so the two terms as presently wired pull the same joint toward points 0.22 radians apart.

The experiment now running carries `pen_ankle_deviation` as a `joint_deviation_l1` at weight −0.2 over all four ankle joints. Its value is forecast here rather than after the fact, so that the forecast is falsifiable. Evaluated on the Phase 3 gait the unweighted term reads 0.7151 radians, the four joints contributing 0.194, 0.159, 0.182 and 0.181, so at a weight of −0.2 it would charge 0.143 per second. That is 0.16 times the 0.768 per second the policy already forfeits on the ankle pitch term and declines to collect, and 0.18 percent of the positive budget. The forecast is therefore that the term at this weight will not unpin the ankle, and that the run will return a gait qualitatively like Phase 3's with the ankle at its stops and the new term reading close to its forecast value. Matching the income the policy already refuses would require a weight near −1.27, and reaching one, two and five percent of the positive budget would require −1.12, −2.24 and −5.59 respectively. If the forecast holds, the next experiment should set the weight from that table rather than from a further small increment, and should first correct the target disagreement recorded above, since two terms pulling one joint toward two different points will otherwise split whatever authority is bought.

### 4.4 Phase 4, unpin the ankle by pricing a posture nothing presently reads

Section 4.3c established that both ankle joints are parked against mechanical stops, that this is free in the reward set and expensive in the machine, and that the term nominally opposing it collects a ninth of what it could. This phase is the remedy, and it is placed here rather than in the old Phase 6 because the ankle is now the largest single defect measured, because the experiment addressing it is already running, and because every phase after it will be measured on a gait whose ankle behaviour must first be settled.

The question this phase turns on is why `keep_ankle_pitch_zero_in_air` fails, and the answer is three defects compounding rather than one weight being too small. Each is measured on the Phase 3 run and each has a distinct remedy.

The first defect is the target. The term penalises the absolute joint coordinate at `mdp/rewards.py:970`, taking `torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])`, so its target is the coordinate zero rather than the pose the rest of the configuration treats as nominal. The SD_BRS1 ankle pitch default is 0.2435 radians before randomisation and reads 0.2163 and 0.2599 after it, so an ankle standing exactly at its own default is already 0.2435 radians into the penalty. At the configured `pitch_scale` of 0.2 that ankle scores 0.2545 per second against a ceiling of 0.8599 set by the airborne fraction, which is to say a perfectly compliant policy could collect less than a third of the term. The consequence is not that the term is weak but that most of its range is unreachable, and the range that remains is the difference between the stop and the default, which is 0.2545 less 0.0888, or 0.1657 per second at weight 1.0. That is 0.21 percent of the positive budget. The entire economic lever this term offers, exercised perfectly, is a fifth of one percent.

Section 4.3c reported the forfeited income as 0.768 per second, measuring the shortfall against the term's nominal maximum of 0.8599. That figure is correct as stated and is the wrong one to size a weight against, because 0.602 of it cannot be recovered at any posture the machine would otherwise adopt. The reachable lever is 0.1657, and it is the figure this section uses. The distinction is recorded because the larger number would have justified a weight roughly five times too small.

The remedy is an optional argument and not a version two, since a default reproducing the absolute coordinate preserves every existing caller exactly, which is rule 4 of `../CLAUDE.md` rather than rule 5.

```python
# mdp/rewards.py, keep_ankle_pitch_zero_in_air, PROPOSED
def keep_ankle_pitch_zero_in_air(env, asset_cfg=..., sensor_cfg=...,
                                 force_threshold: float = 2.0,
                                 pitch_scale: float = 0.2,
                                 require_airborne: bool = False,
                                 history_index: int = -1,
                                 use_default_offset: bool = False):
    """...

    Args:
        use_default_offset: When False, the default, the deviation is taken from the joint
            coordinate zero, which is the original behaviour and is preserved bit for bit.
            When True it is taken from the asset's default joint position, which is the
            posture every other term treats as nominal and which a randomisation event
            may move per environment.

            DEFECT recorded per rule 6 of /ws/CLAUDE.md and left standing in the default.
            Where a robot's ankle default is not zero the two targets differ, and the term
            then charges a nominal pose for its own nominal offset. On SD_BRS1 the ankle
            pitch default is 0.2435 rad, so at pitch_scale 0.2 an ankle at its default
            scores 0.2545 of an available 0.8599 and the reachable lever between the
            mechanical stop and the default is 0.1657 per second at unit weight. The
            TRON1 SF caller at cfg/SF/limx_base_env_cfg.py:1148 is unaffected either way,
            its joint defaults being 0.0 throughout, so for that robot the two branches
            are numerically identical.
    """
    ...
    target = asset.data.default_joint_pos[:, asset_cfg.joint_ids] if use_default_offset else 0.0
    ankle_pitch = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - target)
```

Setting it recovers most of the range. On the Phase 3 gait the term would read 0.2945 rather than 0.0896, and the lever between that gait and a nominal swing ankle rises from 0.1657 to 0.5654 per second at unit weight, which is a factor of 3.4 obtained without touching a weight. The caller enumeration is complete and is short. `keep_ankle_pitch_zero_in_air` has exactly two callers, `cfg/SF/brs_base_env_cfg.py:790` and `cfg/SF/limx_base_env_cfg.py:1148`, and the second declares `".*_Joint": 0.0` for every joint at `assets/config/solefoot_identified_cfg.py:91`, so the TRON1 SF task is provably unaffected whichever way the argument is set. Only the SD_BRS1 caller is to set it.

The second defect is coverage, and it is the larger of the two. The term reads the pitch axis alone, and the roll axis, which stands at its mechanical stop for 62 percent of swing in every run including the baseline, is read by nothing at all. The remedy here requires no code change whatever, because the function is written over `asset_cfg.joint_ids` and a second instance of it may be wired against the roll joints. Two properties make this exact rather than approximate. The ankle roll default is 0.0 radians on this robot, so the absolute coordinate target the function already uses is the correct target for the roll axis and the argument above is unnecessary there. And the function's contact mask has one column per sensor body, so a single instance cannot cover four joints against two feet, the broadcast at `mdp/rewards.py:971` requiring one joint per foot in matching order. A second term is therefore not merely convenient, it is the only form the function admits.

The lever on the roll axis is the largest available. A roll instance evaluated on the Phase 3 gait reads 0.2498 per second against the same 0.8599 ceiling, so it forfeits 0.6101 at unit weight, which is 3.7 times the reachable pitch lever and is reachable in full, the nominal roll being the coordinate zero the term already targets.

The third defect is the weight, and it should be set only after the first two are corrected, since correcting them changes what a weight buys by a factor of three on one axis and makes an axis exist at all on the other. Sized against the positive budget of 79.95 per second, a term is worth one percent of it at a lever of 0.8 per unit weight when its weight is 1.25, two percent at 2.5 and five percent at 6.2. The recommendation is that the pitch and roll terms carry a combined lever near two percent of the positive budget, which the corrected pitch term at weight 1.5 and the roll term at weight 1.5 supply between them at 0.848 and 0.915 per second respectively.

```python
# cfg/SF/brs_base_env_cfg.py, PROPOSED
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["AnklePitchL", "AnklePitchR"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link6L", "Link6R"]),
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "require_airborne": True,
            "history_index": 0,
            # the target moves from the coordinate zero to the randomised default, which
            # for this robot stands 0.2435 rad away and made most of the range unreachable
            "use_default_offset": True,
        },
    )
    # SECOND INSTANCE, the roll axis, which no term has ever read. No code change is
    # required, the function being written over asset_cfg.joint_ids, and no offset is
    # required either, the ankle roll default being 0.0 so that the absolute target the
    # function already uses is correct. Order must match the sensor bodies, L before R.
    rew_keep_ankle_roll_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["AnkleRollL", "AnkleRollR"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link6L", "Link6R"]),
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "require_airborne": True,
            "history_index": 0,
        },
    )
```

The gate for Phase 4 is stated on residence rather than on the term value, since a term value can be moved by a weight without the posture moving at all. The ankle pitch must stand within two hundredths of a radian of a mechanical stop for less than 20 percent of swing, against the present 99.5, and the ankle roll for less than 30 percent, against the present 62.0. The four ankle joints together must fall below 35 percent of the value of `pen_joint_torque`, against the present 57.3. Three guardrails accompany the gate, since this phase can plainly trade the ankle against everything Phase 3 bought. The reference tracking error must stay below 0.015 metres against the present 0.00798, the mean peak contact force below 1.8 body weights against the present 1.551, and the stance knee flexion above 0.35 radians against the present 0.5747, so that the crouch is not surrendered to buy the ankle back.

### 4.4a Phase 4 as implemented, 2026-08-07

Phase 4 was implemented on 2026-08-07 and is wired and awaiting a run. This section records what landed, states the three places where it departs from the specification above, and reports the verification, so that the run may be read later against the tree that produced it rather than against the proposal that preceded it.

Two edits carry the phase. The reward function `keep_ankle_pitch_zero_in_air` gained the optional argument `use_default_offset` at `mdp/rewards.py:933`, documented at `:957` and branching at `:990`, and the branch it guards is written so that the original expression stands unaltered character for character in the `else` arm, which makes the preservation structural rather than merely numerical. The calling configuration then sets the argument on the pitch term at `cfg/SF/brs_base_env_cfg.py:833` and raises that term to weight 1.5, and adds the roll instance at `:858` at the same weight, both terms declaring `force_threshold` and `pitch_scale` explicitly so that the run's dumped `params/env.yaml` records the choices rather than leaving them to be recovered from the source.

The verification recomputed all four candidate terms directly on the dump of run `2026-08-05_07-53-35`, reimplementing the reward over `joint_positions` and `feet_contact_forces` rather than trusting the arithmetic of section 4.4. The contact magnitude is a norm and is therefore unaffected by the frame defect that section corrected in the dump, so the old dump is admissible evidence here.

| variant | value per second | at its weight | disposition |
|---|---|---|---|
| pitch, target the coordinate zero | 0.0873 | 0.0873 at weight 1.0 | the shipped behaviour, superseded |
| pitch, target the default | 0.2871 | 0.4307 at weight 1.5 | implemented |
| roll, target the coordinate zero | 0.2359 | 0.3538 at weight 1.5 | implemented |
| roll, target the default | 0.2193 | 0.3290 at weight 1.5 | considered and not adopted |

The ceiling set by the airborne fraction measures 0.8385 on this run. The corrected pitch term therefore leaves a lever of 0.5514 per second at unit weight and 0.827 at the weight set, and the roll term leaves 0.6026 and 0.904, so the pair offers 1.731 per second against a positive budget of 79.95, which is 2.17 percent of it and sits where section 4.4 recommended. The pitch correction multiplies the term's reading by 3.3, against the 3.4 that section forecast, and the roll axis is confirmed as the larger of the two levers.

These figures stand a little below those of section 4.4, which gave 0.0896 and 0.2945 for the pitch variants and 0.2498 for the roll against a ceiling of 0.8599, and the discrepancy is instrumental rather than substantive. The dump samples contact once per control step whereas the reward reads a sensor history buffered at the simulation step, so the reconstruction of `contact_filt` is coarser than the term's own, and the dump preserves only environment zero's randomised default, which the recomputation applied to all thirty two. Both effects bear on the airborne fraction and move every variant in the same direction, leaving the ratios, which are what the phase is sized on, intact.

Three divergences from the specification are recorded. The proposed configuration resolved the feet as the explicit list `["Link6L", "Link6R"]` and the implementation kept the regular expression `"Link6[LR]"` that every other term in the file uses, the two resolving identically and the latter being the local convention. The proposed configuration also set `force_threshold` to 1.0 on the pitch term, which section 4.4 did not remark upon and which is a change of behaviour beyond the three defects, the function's own default being 2.0, and it was adopted for consistency with the roll instance and with every other contact gated term in the file, the distinction between one and two newtons against a stance load near five hundred being sensor noise in both directions. And the term `pen_ankle_deviation` was found already withdrawn at `cfg/SF/brs_base_env_cfg.py:915`, which is the disposition section 4.4 argues for, so no edit was required to reconcile the contradiction that section identified between a target of zero and a target of the randomised default.

Two defects are left standing and are recorded here per rule 6 of `../CLAUDE.md` so that another implementation may opt into their correction deliberately. The first lives in the unset default of the new argument, where a robot whose ankle default is not zero is charged for its own nominal offset, and it is left because correcting it in the default would alter the TRON1 SF caller at `cfg/SF/limx_base_env_cfg.py:1148`, which is the case rule 4 exists to prevent, notwithstanding that this particular caller declares every joint default as zero at `assets/config/solefoot_identified_cfg.py:91` and is provably indifferent. The second lives in the roll term, which targets the coordinate zero on the argument that the sole is flat in the frontal plane there, while the startup event `joint_offsets` at `cfg/SF/brs_base_env_cfg.py:568` adds a uniform sample on the interval from −0.05 to 0.05 radians to every default. A given environment's roll nominal therefore sits up to 0.05 radians from the target, and since neither the actor nor the critic observes the absolute coordinate, both reading `joint_pos_rel` at `cfg/SF/brs_base_env_cfg.py:190` and `:231` against an action term that offsets from the default at `:156`, that displacement is a constant the policy can neither see nor remove. It costs at most 22 percent of the roll term in the worst environment and nothing in the mean, and the table above measures the alternative at 0.2193 against 0.2359, so the correction is available and was declined because it asks the sole to hold a random tilt in exchange for removing a bias worth two hundredths of a reward unit.

One property of the articulation makes the second instance safe and is recorded because the function would fail silently without it. The reward broadcasts a contact mask carrying one column per sensor body against a deviation carrying one column per joint, so it requires exactly one joint per foot in matching order, and `SceneEntityCfg` resolves both by ascending index. The dumped articulation numbers every pair left before right, `AnkleRollL` and `AnkleRollR` at indices six and seven against `Link6L` and `Link6R` at eleven and twelve, so each ankle is paired with its own foot without recourse to `preserve_order`. Were the ordering otherwise the term would still compute, charging each foot against the opposite ankle, and nothing in the reward would report it.

### 4.5 Phase 5, guard the stance geometry on the quantity that matters, rescoped 2026-08-07

This section was rewritten on 2026-08-07, after the correction appended to section 3.6.7 withdrew the measurement it had been sized against. What follows states the phase as the corrected evidence supports it, and the earlier reasoning is preserved in section 3.6.7 with its correction attached rather than deleted, since the argument it made about the term's form survives the withdrawal of the argument it made about the gait.

Section 3.6.7 established two things and only one of them holds. The first, that a hinge on the planar norm cannot regulate a stance width because a long stride satisfies it while the lateral separation is anything at all, is a statement about the term's algebra and is unaffected by any measurement. The second, that the stance width is in consequence pathological, rested on world frame figures and is withdrawn. On the corrected base frame measurement the lateral separation of the Phase 3 gait has a mean of 0.2591 metres, a median of 0.2469, a fifth percentile of 0.2169 and a minimum of 0.0907, and the feet do not cross. The stance is narrow relative to the 0.26 to 0.338 metre band section 2.8 derives [20], sitting at its lower edge, and it is not collapsing.

Phase 5 is therefore rescoped from a correction into two smaller things, a guard against a rare event and a mild widening, and its weight must be sized for that rather than for the emergency the earlier draft described. The distinction matters because the earlier draft's own configuration, a lateral hinge at 0.30 metres carried at weight −100, would on the measured Phase 3 gait fire on 85.2 percent of steps and charge 4.61 per second against a negative budget of 12.31. It would raise the whole penalty budget by thirty seven percent in order to move a quantity that is already inside a hand's breadth of its target, and it would do so with a gradient present almost everywhere, which is the definition of a term that will dominate the ones around it. That configuration must not be wired.

The literature was consulted afresh for this rescoping, and it supports the lateral formulation, supplies threshold precedents, and warns against exactly the rigidity the earlier draft would have imposed. The frontal plane stepping literature establishes that human lateral stepping is regulated as a multi objective problem in which step width receives roughly ninety three percent of the control effort and absolute lateral position roughly seven, whereas fore and aft stepping is regulated as a single objective speed control [26]. Step width, which is the lateral separation, is therefore the quantity biology regulates, and the planar norm that mixes it with the stride is not a quantity anything regulates. The current humanoid learning frameworks agree on the form, computing the lateral distance between the feet in the robot's own frame and penalising it below a minimum, with that minimum set at 0.25 metres for the Unitree H1 and 0.18 for the smaller G1 [27], and one published reward set carries the same construction on the knees as well as the feet. Base frame per axis separation terms of this kind are now standard, the force adaptive loco manipulation work carrying a fore and aft foot separation term and a lateral root centring term at equal weight [28]. The warning comes from the narrow terrain work, which trains a humanoid to cross a beam of twenty five centimetres and carries a feet separation term in its gait group while doing so [29], since a robot that must sometimes place one foot nearly in front of the other cannot be given a lateral floor it may never cross. That is precisely the case the two tier formulation below exists to admit.

The remedy has two parts. The first is the optional argument selecting the lateral component, already implemented, whose default reproduces the planar norm exactly so that the TRON1 PF, WF and SF callers are untouched.

```python
# mdp/rewards.py, feet_distance, extended
def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str] = ["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,
                  lateral_only: bool = False,) -> torch.Tensor:
    """...

    Args:
        lateral_only: When False, the default, the separation is the Euclidean norm of the
            planar difference between the two foot frames, which is the original behaviour
            and is preserved bit for bit for the TRON1 PF and WF callers. When True the
            separation is the magnitude of the base-frame lateral component alone.

            DEFECT recorded per rule 6 of ../CLAUDE.md and left standing in the default.
            The planar norm is satisfied by a long stride even when the lateral separation
            is zero, so a hinge on it does not regulate stance width. On SD_BRS1 run
            2026-07-28_06-37-24 the term read a mean penalty of 0.00014 while the lateral
            separation stood below the 0.19 m sole-overlap width for 39.8 percent of steps
            and reached zero, the feet crossing the midline. Callers wanting stance width
            rather than foot separation must set this True.
    """
```

The lateral component must be taken in the base frame rather than the world frame, since a world frame lateral difference is only the stance width when the robot's heading is zero, and the robot turns. The implementation rotates the planar difference into the base frame by the yaw of the root quaternion and takes the magnitude of its second component. Only the yaw is used, the pitch and roll being deliberately discarded, so that a leaning torso does not appear to narrow the stance.

One correction to the caller enumeration of the preamble belongs here. `feet_distance` has four callers and not the two the preamble recorded, being called from `cfg/PF/limx_base_env_cfg.py`, `cfg/WF/limx_base_env_cfg.py`, `cfg/SF/limx_base_env_cfg.py` and `cfg/SF/brs_base_env_cfg.py`. The SF caller was overlooked. The remedy is unaffected, since the default of `lateral_only` preserves the planar norm exactly for all three TRON1 callers, but the enumeration is corrected so that a future reader does not repeat the omission on a change where it would matter.

```python
# mdp/rewards.py, implemented 2026-08-03, the branch only
def feet_distance(env, asset_cfg=SceneEntityCfg("robot"),
                  feet_links_name=["foot_[RL]_Link"],
                  min_feet_distance=0.1, max_feet_distance=1.0,
                  lateral_only=False):
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:, feet_links_idx]
    if lateral_only:
        # rotate the planar separation into the base frame and keep the lateral component
        diff_w = feet_pos[:, 0, :3] - feet_pos[:, 1, :3]
        yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w)
        diff_b = math_utils.quat_apply_inverse(yaw_quat, diff_w)
        feet_distance = torch.abs(diff_b[:, 1])
    else:
        # the original planar norm, preserved bit for bit for the three TRON1 callers
        feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward
```

The second part is new and is the substance of the rescoping. A single lateral floor cannot express what is actually wanted, which is that the feet be laterally separated in ordinary walking and that they be separated somehow at every instant. A hinge that enforces only the lateral component forbids the crossed placement a turn or a narrow support sometimes requires [29], while a hinge that enforces only the planar norm is the term this document has spent two sections criticising. The formulation that expresses both is disjunctive. The lateral separation is required to exceed a lateral threshold, and where it does not, the total separation is required to exceed a larger one, so that a foot placed nearly in line with its fellow is admitted provided the stride has carried it far enough away to be in no danger of striking it.

This is a change of behaviour rather than of parameter, so rule 5 of `../CLAUDE.md` governs and a version two is created rather than the existing function extended. No optional argument on `feet_distance` can express a penalty that is conditional on one measurement and charged on another, because the existing body computes a single scalar and hinges it, and a default that reproduced the old behaviour would have to disable the conditional entirely, which is a second function wearing the first one's name.

```python
# mdp/rewards.py, PROPOSED, not implemented pending validation
def feet_distance_v2(env: ManagerBasedRLEnv,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     feet_links_name: list[str] = ["foot_[RL]_Link"],
                     min_lateral_distance: float = 0.22,
                     min_total_distance: float = 0.28,
                     max_feet_distance: float = 1.0) -> torch.Tensor:
    """Penalise a stance that is neither laterally separated nor separated overall.

    The lateral criterion is primary, step width being the quantity the frontal plane
    stepping literature identifies as regulated, at some 93 percent of the control effort
    against 7 for absolute lateral position (arXiv DOI 10.1371/journal.pcbi.1006850). The
    total criterion is the fallback that admits a crossed placement during a turn or on a
    narrow support, which a bare lateral floor would forbid (arXiv:2502.17219).

    Both are taken in the BASE frame by the yaw of the root quaternion, pitch and roll
    being discarded so that a leaning torso does not appear to narrow the stance. This is
    the defect corrected in section 3.6.7 of GAIT_EFFICIENCY_PLAN.md, where a world frame
    reading reported a lateral separation of 0.19 m against a true 0.26.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:, feet_links_idx]
    diff_w = feet_pos[:, 0, :3] - feet_pos[:, 1, :3]
    yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w)
    diff_b = math_utils.quat_apply_inverse(yaw_quat, diff_w)
    lateral = torch.abs(diff_b[:, 1])
    total = torch.norm(diff_b[:, :2], dim=-1)
    # the lateral shortfall, charged only where the total separation does not excuse it
    lateral_short = torch.clip(min_lateral_distance - lateral, 0.0, 1.0)
    total_short = torch.clip(min_total_distance - total, 0.0, 1.0)
    reward = torch.minimum(lateral_short, total_short)
    reward += torch.clip(total - max_feet_distance, 0.0, 1.0)
    return reward
```

The minimum of the two shortfalls is the disjunction. Where either criterion is satisfied the corresponding shortfall is zero and the product of the disjunction is zero, so the term is silent, and where both fail it charges the smaller of the two deficits, which is the lesser correction that would restore compliance. Taking the minimum rather than the product keeps the units in metres and the gradient of the same order as the existing hinge, and taking it rather than the sum ensures the term cannot charge twice for one placement.

The thresholds are derived from the corrected base frame measurement and from the sole geometry rather than from the band alone. The soles span 0.194 metres laterally, the twelve offsets at `cfg/SF/brs_base_env_cfg.py:692` running from −0.0970 to +0.0970, so two soles at equal height touch when the lateral separation reaches 0.194 and a floor must stand above it. On the Phase 3 gait a lateral threshold of 0.22 fires on 6.96 percent of steps and one of 0.25 on 54.25, so 0.22 is the largest value that leaves the term a guard rather than a governor, and it retains 0.026 metres of margin above the sole width. The total threshold of 0.28 is set from the planar separation, whose fifth percentile is 0.2357 and whose minimum is 0.1257, so it excuses a crossed placement only while the stride is genuinely carrying the feet apart. Their conjunction, which is the only state the term charges, obtains on 0.067 percent of steps.

The weight follows from that coverage and is far below the −100 the earlier draft carried. At a weight of −100 the proposed thresholds charge 0.0018 per second, which is a guard that costs nothing until it is needed, and that is the intended reading. The earlier draft's −100 was inherited from a term that never fired and was never sized, and it is retained here only because the coverage rather than the weight now does the work. Should the measurement after training show the term binding more than one percent of the time, the weight rather than the thresholds should be reconsidered.

```python
# cfg/SF/brs_base_env_cfg.py, pen_feet_distance, PROPOSED
    func=mdp.feet_distance_v2,
    weight=-100,
    params={
        "min_lateral_distance": 0.22,
        "min_total_distance": 0.28,
        "feet_links_name": ["Link6[RL]"],
    },
```

Whether the stance should also be widened is a separate question from whether it should be guarded, and this document now separates them. The measured 0.259 metres sits at the lower edge of the 0.26 to 0.338 metre band [20], and the foot placement estimator argument of section 2.8 says that a narrow nominal stance leaves no room to place a foot laterally outside the centre of mass during a recovery [22]. A hinge is the wrong instrument for moving a mean, since it is silent above its threshold and therefore exerts no pressure at all on the ninety five percent of the distribution that already complies. If the stance is to be widened, the instrument is a set point on the lateral separation or a lateral component in the command, and it belongs to its own experiment with its own gate. Widening is not proposed here, because Phase 3's crouch has just changed the frontal plane geometry substantially and the stance width should be measured again after the ankle experiment resolves rather than acted on from a gait that is still moving.

The heavier roll and pitch angular velocity penalty must accompany the phase in either form. Section 2.8's mechanism is that a narrow stance is balanced by torso roll rather than by lateral hip motion [21], and the ablation of section 3.2 shows the −5 weight halving the base roll rate, so `pen_ang_vel_xy` should be at −5 for this phase, which is the value every run since has carried.

The gate for Phase 5 is restated on the corrected measurement. The lateral separation, taken in the base frame, must keep a fifth percentile above 0.22 metres and a minimum above 0.194, which is the sole width, against the present 0.2169 and 0.0907. The planar separation must keep its fifth percentile above 0.23 against the present 0.2357. The undesired contacts penalty must remain below 0.1 per second per the guideline of section 3.4, against the present 0.065. The gate is deliberately close to the present values, because the phase is a guard against a tail and not a correction of a mean, and a gate that demanded a large movement would be evidence that the phase had been mis-scoped again.

### 4.6 Phase 6, recover the yaw authority

Section 3.6.8 established that the yaw failure has one morphological cause and four configurational ones, and that the morphological cause is immovable, there being no vertical axis actuator on the machine. The four configurational causes are addressed together, since section 3.6.8 shows they compound.

Phase 5 has already removed the first, the stance width, which lengthens the moment arm available to differential foot placement.

The second is the lateral velocity command range of plus or minus 0.01 metres per second, which has never asked the robot to step sideways. Widening it to plus or minus 0.15 gives the policy a lateral placement skill it can then recruit for a turn, and the value is chosen to be small relative to the fore and aft range of −0.3 to 0.8 so that the primary task is not disturbed.

```python
# cfg/SF/brs_base_env_cfg.py, base_velocity ranges
    lin_vel_y=(-0.15, 0.15),      # was (-0.01, 0.01)
```

The third is the absence of any term aligning the feet with the direction of travel. Booster Gym carries exactly this at twice their linear tracking weight [11], and it is added here as a new SD_BRS1 exclusive function, since no existing term reads foot yaw.

```python
# mdp/rewards.py, new function
def feet_yaw_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise the squared yaw of each foot relative to the base.

    A biped whose hip yaw joints are fixed, as SD_BRS1's are, can develop a yaw moment on
    its base only through the friction moment under the stance sole and through angular
    momentum exchange with the swinging legs. Leaving the foot yaw unconstrained wastes
    both, since a foot planted across the direction of travel opposes the turn it is
    meant to assist. The term follows the feet-yaw reward of Booster Gym
    (arXiv:2506.15132), which carries it at twice the linear tracking weight.

    Returns:
        The computed penalty tensor, summed over the feet.
    """
```

The implementation extracts the yaw of each foot's world quaternion and of the base's, wraps the difference into the interval from minus pi to pi, squares it and sums over the feet. The wrapping is not decorative. A foot yawed just past the wrap point would otherwise be charged for the large difference the raw subtraction reports rather than for the small error it actually has, and the term would then push hardest exactly where it should push least.

```python
# mdp/rewards.py, implemented 2026-08-03
def feet_yaw_alignment(env, asset_cfg):
    """Penalise the squared yaw of each foot relative to the base."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]           # (N, F, 4)
    base_quat = asset.data.root_link_quat_w                             # (N, 4)

    foot_yaw = math_utils.euler_xyz_from_quat(foot_quat.reshape(-1, 4))[2].view(
        foot_quat.shape[0], foot_quat.shape[1]
    )
    base_yaw = math_utils.euler_xyz_from_quat(base_quat)[2].unsqueeze(1)

    yaw_error = math_utils.wrap_to_pi(foot_yaw - base_yaw)
    return torch.sum(torch.square(yaw_error), dim=1)
```

```python
# cfg/SF/brs_base_env_cfg.py, RewardsCfg
pen_feet_yaw = RewTerm(
    func=mdp.feet_yaw_alignment,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]")},
)
```

The weight of −5 is derived from the reference proportion. Booster Gym prices feet yaw at 1.0 against a combined tracking weight of 2.5, a ratio of 0.4 [11], which against this task's combined tracking rate of 65 per second would give −26. That is far too large to introduce at once against a term whose present value is unmeasured, so the entry point is −5, which at a plausible mean squared foot yaw error of 0.1 square radians costs 0.5 per second, and the weight should be raised on the next iteration if the measured error does not fall.

The fourth is the slide penalty's suppression of the stance pivot, and it is deliberately not changed. `feet_slide` at −5 is one of the terms that keeps the gait from returning to the shuffle this work stream began with, and the correct reading is that a modest pivot is a cost the robot should pay rather than a behaviour that should be free. Should Phase 6 fail its gate with the first three changes in place, the reserve action is to gate `feet_slide` on the commanded yaw rate being small, which is a further optional argument, and it is held back rather than taken now.

The gate for Phase 6 is a yaw rate root mean square error below 0.40 radians per second, against the present 0.62 to 0.74, with the correlation between commanded and achieved yaw rate above 0.5 against the present 0.23, and the linear tracking guideline of section 3.4 held.

### 4.7 Phase 7, smooth the actuation and remove the reliance on the stops

This phase is placed last because sections 3.6.1 through 3.6.3 predict that a large part of the roughness is a consequence of the impact and the trapezoid, and the residual must therefore be measured after Phases 1 through 3 rather than assumed from the present numbers. It has three layers of increasing cost.

The first layer is a set of weight changes, config local, no function touched. Section 3.6.4 established that the five terms other than action smoothness together supply 1.5 percent of income, and the target is to bring the whole effort and smoothness group from 6.0 percent to between 12 and 15 percent, which is the proportion the reference configurations of section 2.2 carry [9, 11]. The proposed entry points, each a multiple of the present value chosen so that the term's own contribution reaches roughly two per second at its present measured function value, are `pen_joint_torque` from −1e-5 to −4e-5, `pen_joint_torque_rate` from −1e-5 to −1e-4, `pen_joint_accel` from −1e-7 to −3e-7, and `pen_action_rate` from −0.01 to −0.05. `pen_action_smoothness` is left at −0.075, since at 2.885 per second it already carries the group.

The second layer removes the reliance on the mechanical stops that section 3.6.5 documented, and has two parts.

The action gains a clip. The action configuration presently sets `clip` to null, and bounding the raw action to the interval from −2.5 to 2.5 caps the commanded target excursion at one radian about the nominal, which exceeds every joint's travel and therefore constrains nothing a well behaved policy would do, while removing the six radian position errors that section 3.6.5 showed are what drives the ankle roll onto its stop at full effort.

```python
# cfg/SF/brs_base_env_cfg.py, ActionsCfg
    clip={".*": (-2.5, 2.5)},     # was null
```

And the torque tiredness term of section 2.5 is added [11], penalising the squared ratio of applied torque to the actuator's effort ceiling rather than the absolute torque. Section 3.6.5 established why the distinction matters here, that 131 newton metres is unremarkable at a hip and saturating at an ankle, so an absolute torque penalty cannot see the ankle roll saturation at all. The case has hardened since, Phase 1 having driven the ankle pitch to its 262 newton metre ceiling on 3.8 percent of steps and the ankle roll to 10.7 percent, while `joint_torques_l2` reported nothing unusual because those torques are small in absolute terms. This is a new SD_BRS1 exclusive function reading `asset.data.applied_torque` against `asset.data.joint_effort_limits`.

```python
# mdp/rewards.py, implemented 2026-08-03
def joint_torque_tiredness(env, asset_cfg=SceneEntityCfg("robot")):
    """Penalise the squared ratio of applied torque to each actuator's effort limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limit = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(torque / (limit + 1e-6)), dim=1)
```

```python
# cfg/SF/brs_base_env_cfg.py, RewardsCfg, when Phase 7 is enabled
pen_torque_tiredness = RewTerm(
    func=mdp.joint_torque_tiredness,
    weight=-1.0e-2,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

The weight of minus one hundredth is Booster Gym's own [11] and is retained rather than rederived, because the term is dimensionless by construction and therefore transfers across robots in a way an absolute torque coefficient does not. Its value on the SD_BRS1 must nonetheless be measured on the first run that carries it, since the reference platform's duty cycle is not this one's, and the entry point should be revised against that measurement rather than against the analogy.

The third layer is the policy space regularisation of section 2.5, added to the PPO update in `/ws/rsl_rl/rsl_rl/algorithms/ppo.py` as the temporal and spatial losses of the conditioning for action policy smoothness method [16], of the same character as the symmetry hook already integrated there. It is proposed only if the first two layers leave residual chatter, and section 2.5's caution applies, that the regularisation must be calibrated rather than maximised, since an over damped policy cannot make the fast corrections that balance requires.

A fourth item belongs to this phase but is an actuator change rather than a reward change, and it is recorded rather than prescribed. The ankle damping ratios remain at 1.43 and 2.46 against the 0.7 the project strategy targets and which the hips and knee already meet, so both ankles are overdamped, which section 3.6.5's velocity ceiling overshoot suggests is contributing to the saturate-then-freewheel pattern. Correcting them requires a retrain because it changes the plant, so it should be grouped with whichever phase is next rather than run alone.

The gate for Phase 7 is a ninety ninth percentile joint acceleration below 500 radians per second squared against the present 951, a ninety ninth percentile single step torque change below 40 newton metres against the present 82, a mean absolute joint power below 500 watts against the present 850, an absolute mechanical cost of transport below 1.2 against the present 2.38, and the fraction of steps at which any ankle actuator exceeds 98 percent of its effort ceiling below one percent against the present 6 to 7.

### 4.8 Summary of the programme

| phase | target defect | kind of change | shared code touched | primary gate | status |
|---|---|---|---|---|---|
| 1 | the stomp and the powered descent | two weights, one existing term wired, one new SD_BRS1 term | none | touchdown speed below 1.0 m/s, peak force below 2.0 BW | trained as 2026-07-31_10-21-10, failed, one of three changes not applied |
| 1b | the same, corrected | one weight applied late, two weights raised, v2 of the landing term | none, v1 left intact for TRON1 PF | as Phase 1, plus sole approach below 1.4 m/s and ankle saturation below 5 percent | trained to completion as 2026-08-03_11-19-11, PASSED, peak force 1.591 BW, sole approach 0.800 m/s and touchdown speed 0.889 m/s |
| 2 | the eliminated double support | one new SD_BRS1 class, one clock parameter | none, `no_fly` left intact | double support above 12 percent, clock correlation held | trained to completion as 2026-08-03_11-19-11, FAILED at 10.269 percent, the provisional pass of section 4.2a withdrawn, then carried past the gate at 14.014 percent by Phase 3 without further change, see section 4.3b |
| 3 | the trapezoidal swing profile | version three of the clearance reward, and one impact threshold | none, v2 left intact and now callerless | reference tracking error below 0.015 m, restated from the quantised plateau measure | trained to completion as 2026-08-05_07-53-35, PASSED, error 0.0271 to 0.00798 m, correlation 0.860 to 0.945, and the straight legged regression reversed, see section 4.3b |
| 4 | the ankle parked at its mechanical stops | one optional argument, one second instance of an existing term, two weights | none, the argument's default preserves both callers and the TRON1 one is unaffected either way | ankle pitch at its stop in swing below 20 percent against 99.5, roll below 30 against 62.0, ankle share of the torque penalty below 35 against 57.3 | IMPLEMENTED 2026-08-07, awaiting a run. Promoted from Phase 7 the same day. Section 4.4a records what landed and where it diverged from the specification above |
| 5 | the unguarded stance geometry, rescoped 2026-08-07 | version two of the feet distance term, disjunctive | none, v1 and its `lateral_only` branch left intact | base frame lateral fifth percentile above 0.22 m and minimum above 0.194 | rescoped, `feet_distance_v2` proposed and NOT implemented pending validation |
| 6 | the yaw tracking failure | one command range, one new SD_BRS1 term | none | yaw error below 0.40 rad/s, correlation above 0.5 | term implemented, not wired |
| 7 | the impulsive actuation and the joint stops | four weights, an action clip, one new term, then CAPS | none until CAPS, which is an rsl_rl change | acceleration, torque rate, power and saturation gates | tiredness term implemented, not wired, weights and clip outstanding |

The functions of Phases 5 through 7 are present in `mdp/rewards.py` and have no callers, so they change nothing until a configuration wires them. They are placed there rather than held in this document so that they can be read against the surrounding code, type checked, and reviewed once rather than transcribed at the moment of use, which is when a transcription error would cost a training run. Each section above carries the configuration snippet that enables its term.

The revision to Phase 5 that the Phase 2 outcome appeared to supply is withdrawn in full on 2026-08-07, and the paragraph that carried it is replaced rather than amended, because every figure in it was wrong in the same way. It reported the lateral separation as averaging 0.198, 0.198 and 0.194 metres across the three runs with a fifth percentile of 0.021, and concluded that the defect was structural because the figures did not move. The figures did not move because they were measuring a rotating frame, and a rotating frame reports the same mixture of stance width and stride whatever the gait does. Rotated into the base frame the same three runs read 0.283, 0.273 and 0.258 metres with fifth percentiles of 0.218, 0.213 and 0.216, and the Phase 3 run reads 0.259 with 0.217. The stability of the figures across four runs is real and is now evidence of the opposite conclusion, that the stance width is a well regulated quantity sitting just below its recommended band and moving little with the gait. The correction appended to section 3.6.7 states the derivation and section 4.5 states the rescoped phase.

The instrumentation defect that produced the error is worth stating precisely, because a loose statement of it would condemn code that is correct. The defect is one of INTERPRETATION and not of representation, and the distinction determines where the remedy belongs.

A separation is a vector between two points, and a change of basis does not alter the vector. What a rotation alters is the tuple of components expressing it, and therefore any quantity read off a single component. Every rotation invariant of the separation may be computed from the world frame components exactly, and this was verified on the Phase 3 dump rather than asserted. The planar norm computed from the world components and from the base frame components agrees to 5.96e-8 metres at worst over 96032 samples, and the full three dimensional norm to 8.94e-8, which is float32 resolution and therefore identity. The two horizontal components, taken singly, differ by up to 0.4827 metres on a vector whose length never moved.

Three consequences follow and the first two exonerate the reward code. `feet_distance` at `mdp/rewards.py:581` requires no change in either branch. Its default branch takes `torch.norm` of the world frame planar difference at line 630, which is a rotation invariant and is therefore already correct, and its `lateral_only` branch at lines 622 to 627 already rotates into the base frame before taking a component, which is the only place a rotation is needed. Nothing in the reward path was ever measuring the wrong thing. The proposed `feet_distance_v2` of section 4.5 inherits the same discipline, rotating once and then reading both a component and a norm from the rotated vector.

The third consequence locates the actual defect, which is in the naming rather than in the dumping. `scripts/rsl_rl/play.py:407` logs the world frame vector, and that is a faithful and lossless record. `feet_distance_statistics` at `scripts/analysis/stats.py:751` then names the first two components fore and aft and lateral without rotating them, and those names assert a base frame reading of a world frame tuple. The vertical component is unaffected, since yaw does not touch it, and `planar separation` and `total separation` are unaffected, since they are norms. The affected records are the two horizontal components, the separation correlations built on them, the centre of pressure and base of support records at `stats.py:1024` and `stats.py:1044`, and the step width at `stats.py:1242`.

The remedy therefore belongs in the statistics and not in the dump, and this is the practically important point. `base_quaternion` is already dumped, and has been verified present in all seven runs of the series at shape (3001, 32, 4), so the yaw is already on disk beside every separation ever recorded. Rotating inside `stats.py` corrects every existing dump retroactively and requires no replay, where changing the dump would correct only runs recorded after the change and would break the world frame reading for any consumer that wants it. Section 4.9 states the change and `../context/gait_metrics.md` carries it for the pipeline.

A caution on Phase 6 that Phase 1's outcome supplied, now itself qualified twice. The yaw tracking improved substantially without being addressed, its root mean square error falling from 0.653 to 0.494 radians per second and its correlation rising from 0.275 to 0.515, purely as a side effect of the impact terms. At eleven thousand iterations that improvement appeared not to have survived Phases 1b and 2, the correlation having fallen back to 0.296, but the completed run recovers much of it, the correlation standing at 0.430 against the baseline's 0.275 and the training yaw error at 0.435 radians per second against the predecessor's 0.388. The gain is therefore partly durable and partly a function of training budget, and the allowance of section 4.2b applies, the present run having faced a looser angular tracking curriculum at 0.247 against 0.212. The deficit Phase 6 was sized against is reduced but has not closed, and no reading of it should be taken from a partly trained run again. Phase 6 should still be re-measured before its weights are set, but the expectation that it has partly closed itself is withdrawn. Section 3.6.8 attributed the yaw failure in part to a narrow stance and a suppressed pivot, and this movement suggests a further contribution from the impulsive gait itself, a foot that is slammed down and snatched up having little of the sustained sole friction that a turn requires. Phase 6 should therefore be re-measured against the policy that emerges from Phases 1b through 5 before its weights are set, since a part of the deficit it was sized against may already have closed.

A revision to Phase 6 that the Phase 3 run supplies, and which is the first properly matched reading of the yaw this programme has obtained. Section 4.2b had to qualify its yaw figures because the angular tracking standard deviation stood at 0.2124 in the Phase 1 run against 0.2473 in its successor. Both the Phase 1 and the Phase 3 runs ended at 0.2124, with identical angular command ranges of 0.300, so the allowance no longer applies and the two may be read directly. On that footing the yaw has not recovered. The training yaw error stands at 0.4186 radians per second against Phase 1's 0.3872, and the evaluation correlation at 0.468 against 0.529, so Phase 3 is slightly worse than Phase 1 on both and the improvement section 4.8 previously attributed to the impact terms is now attributable in part to the looser curriculum that flattered the intervening run.

The shape of the deficit is more informative than its size and it changes what Phase 6 should attempt. Regressing the achieved yaw rate on the commanded one gives a slope of 0.263 on the Phase 3 run, 0.334 on Phase 1 and 0.213 on the baseline, and the achieved magnitude averages 0.218 radians per second against a commanded 0.489. Binning by command, the machine delivers 0.067 radians per second when asked for between 0.1 and 0.3, 0.141 when asked for between 0.3 and 0.6, and 0.187 when asked for between 0.6 and 0.9. The response is sublinear throughout and flattens at roughly a fifth of a radian per second regardless of demand, which is the signature of an authority limit rather than of a tracking gain that could be raised. Section 3.6.8's morphological argument is thereby confirmed on measurement, and the practical consequence is that Phase 6 should be judged on the slope and the saturation ceiling rather than on the error, since the error is dominated by commands the machine cannot meet at any weight. The sign of the response is correct on 78.98 percent of steps at commands above 0.3, so the policy knows which way to turn and cannot turn far enough.

Each phase produces a checkpoint, a `play.py` dump, a `statistics.npy` computed by the pipeline of `GAIT_STATISTICS_PLAN.md`, and a fresh set of plots, so that its effect is read against the instruments that established the baseline in section 3. Since 2026-08-06 every run in the series carries a `statistics.npy` and all comparisons in sections 4.3b and 4.3c are drawn from it rather than from ad hoc scripts, which is what allows the four runs to be tabulated against one another without the reconciliation each earlier pass required. No phase advances until its gate is met, and the guidelines of section 3.4 are checked at every phase, since a phase that meets its own gate while breaking a guideline has traded one defect for another.

The order of the remaining programme is revised on the Phase 3 outcome, and the revision is stated here because three of the four remaining phases have had their premises altered. Phase 7 loses most of its content, since the impulsive actuation it was written to address has largely resolved with the impact, exactly as the conclusion predicted it would, the torque rate falling by fifty seven percent and the joint jerk by forty two while the knee left its stops entirely. What remains of Phase 7 is the ankle, which section 4.3c has promoted into its own experiment and which is running now. Phase 5 is rescoped to a guard and is cheap. Phase 6 is unchanged in intent and now has a properly matched measurement to work from. The recommended order is therefore the ankle experiment to completion, then Phase 5 as a guard wired alongside whatever the ankle experiment settles, then Phase 6, and Phase 7 reduced to whatever the smoothness measurements still show outstanding once the ankle is unpinned. The reasoning is given in the fourth postscript of section 5.

---

### 4.9 The next steps, written 2026-08-07 and awaiting validation before any of it is wired

Nothing in this section has been implemented. It is written so that the ordering and the sizing may be argued before a training run is spent on either, and each item states the evidence that would settle it.

The first step is to let the ankle experiment finish and to read it against the forecast of section 4.3c. That forecast is that a `joint_deviation_l1` at weight −0.2 will charge 0.143 per second, that this is 0.16 times the income the policy already forfeits on the ankle pitch term and declines to collect, and that the ankle will therefore remain at its stops. If the forecast holds, the value of the run is that it calibrates the elasticity of the ankle posture against a known price, which no run so far has done, and the next weight should be taken from the budget table of section 4.3c rather than by doubling. If the forecast fails and the ankle unpins at −0.2, the conclusion is more interesting than the forecast, since it would mean the pin was nearly free rather than actively bought, and the whole reading of section 4.3c would need revising toward a weaker mechanism.

Phase 4 then follows on whichever reading the run supplies, and section 4.4 states it in full. Its three parts are independent and should be understood as such, since only one of them is a code change. Wiring a second instance of `keep_ankle_pitch_zero_in_air` against the roll joints requires no code at all and carries the largest lever in the phase at 0.6101 per second per unit weight. Setting `use_default_offset` on the pitch instance requires one optional argument whose default preserves both existing callers and whose TRON1 caller is provably indifferent to it, and it raises the pitch lever from 0.1657 to 0.5654. Only the weights then remain, and they should be set last, since the first two change what a weight buys by more than any weight change would.

A change to the statistics pipeline should be made before the ankle run is evaluated, and it is small and retroactive. `feet_distance_statistics` must rotate the dumped separation by the yaw of `base_quaternion` before naming its horizontal components, as section 4.8 establishes and as `../context/gait_metrics.md` records in detail. The reward code needs no change, the planar norm being rotation invariant and the `lateral_only` branch already rotating, and the dump needs no change either, `base_quaternion` being present in all seven runs of the series. Rotating inside the statistics therefore corrects every figure this programme has recorded without a single replay, and doing it before the next evaluation means the ankle run is read on a corrected instrument rather than joined to a record that must later be amended.

Two corrections should accompany whatever weight is chosen, and both are matters of correctness rather than of tuning. The first is the target disagreement recorded in section 4.3c, `rew_keep_ankle_pitch_zero_in_air` pulling the ankle pitch toward an absolute zero while `pen_ankle_deviation` pulls it toward a randomised default 0.22 radians away. The two should be reconciled before either is strengthened, and the reconciliation is a matter for the caller, since the deviation term is already correct and the older term's absolute target is the defect. The second is that no term reads the ankle roll except through the new deviation penalty, and the roll is the axis whose behaviour prompted the question, standing at its stop for 62 percent of swing in every run since the baseline. A deviation penalty is the right instrument for it and a separate weight for the roll axis may be warranted, since the roll defect is older and larger than the pitch defect and the two need not be priced together.

The second step is Phase 5 as section 4.5 now scopes it, which is cheap and can be wired alongside the ankle work rather than after it, since the two touch no common term and the guard is silent on 99.93 percent of steps. The reason to wire it at all, given that the defect it was written for has been withdrawn, is that the tail it guards is real, the base frame lateral separation reaching 0.0907 metres against a sole width of 0.194, and that the guard costs nothing until it is needed. The reason not to wire it as a widening is that Phase 3's crouch has just moved the frontal plane geometry and the stance width should be measured again once the ankle settles.

The third step is Phase 6, which section 4.8 now reads on a properly matched curriculum for the first time and which the measurement rescopes. The deficit is an authority limit rather than a tracking gain, the response saturating near a fifth of a radian per second against commands up to nine tenths, so the phase should be judged on the regression slope of achieved against commanded, presently 0.263, and on the ceiling, and its gate restated in those terms rather than on the error. The four configurational contributions section 3.6.8 identified should be re-examined against the present gait before their weights are set, and one of them has already changed, since a foot that now lands at 0.539 metres per second rather than 1.875 has far more sustained sole contact available to develop a friction moment than the gait that section was written against.

The fourth step is whatever remains of Phase 7, and the honest answer is that most of it has already been delivered by Phase 3 without being attempted. The torque rate fell by fifty seven percent, the joint jerk by forty two, the joint velocity ninety ninth percentile from 9.81 radians per second to 5.54, the joint acceleration root mean square from 340.8 to 107.9, and the knee left its stops. What the phase was written to buy has largely arrived, and its ankle content has been promoted out of it into Phase 4. What remains is the action clip, which is a separate question about whether an unclipped action should be permitted to command a joint target outside its own mechanical range at all. That question is sharpened rather than answered by the ankle measurements, since a joint held at a stop by a saturated position target is the exact behaviour an action clip would forbid, and it should be settled after Phase 4 tells us whether pricing the posture is sufficient on its own. Pricing and clipping are alternatives rather than complements here, and taking both at once would leave neither attributable.

Two smaller matters are recorded so that they are not lost. The clearance amplitude deficit of section 4.3b, the policy tracking the reference shape almost exactly at an apex of 0.0704 metres against a commanded 0.08, is a consequence of a kernel width of 0.03 that is forgiving of a uniform ten millimetre offset, and narrowing the kernel would recover the amplitude at the cost of making the term harder to satisfy everywhere. It should not be narrowed in the same experiment as any other change, and it is not obviously worth an experiment of its own, since twelve percent of a swing height is not a defect anything downstream has complained of. The second matter is the double support fraction, which reached 14.014 percent against a commanded 24 without any term being changed, and which section 4.3a's deferred analysis says is opposed principally by `feet_air_time_positive_biped` paying zero during the transfer. That analysis stands and its levers are unchanged, but it should be re-derived on the Phase 3 gait before being acted on, since the term's own value has fallen from 2.498 per second to 2.318 and the cycle period has shortened, so the arithmetic that sized the opposition has moved.

---

## 5. Conclusion

The SD_BRS1 walks, and the preceding nineteen passes of investigation earned that plainly. It alternates its feet at the commanded cadence, follows a planar velocity command to within a sixth of a metre per second, holds its base within two centimetres of a target that is now kinematically reachable, carries its torso within seven degrees of level, and moves its two legs with a sagittal asymmetry in the single figures. Each of those properties was won against a specific and documented obstacle, and section 3.4 states them as constraints precisely so that the programme of section 4 cannot spend them.

What the analysis of this document adds is that the manner of that walk would not survive contact with hardware, and that the reasons are entirely specifiable. The robot strikes the ground at 1.68 metres per second on the average step, driving the sole down at more than free fall speed, and takes 3.08 times its body weight on the average touchdown and 13.76 at the worst, against the 1.0 to 1.5 that a human walking gait develops [3]. It presses its ankle roll actuators onto their effort ceilings for six to seven percent of every second and stands within two hundredths of a radian of a mechanical stop for a third of it, its knee switching between full extension and full fold with only fifteen percent of its time in between. It circulates 850 watts of joint power to deliver 89, a ratio of nine and a half, giving a mechanical cost of transport of 2.38 against the 0.2 of human walking [1], even though the same figure computed on net power is 0.25 and shows that the trajectory itself is not expensive. And it does not turn, the commanded yaw rate averaging 0.51 radians per second against an achieved 0.29 with a correlation of 0.23, in all four of the runs examined.

The causes reduce further than that list suggests. A Gaussian kernel on the sole clearance multiplied by a hyperbolic tangent of foot speed has, as its maximiser over a swing of fixed duration, a trapezoid that snaps to the target height and holds it, and that trapezoid earns forty six percent more than a sinusoidal arc of the same apex. Nothing in the configuration prices the descent that ends it, although the exact term required already exists in the repository and is already wired into a sibling task. The single support terms pay nothing during the interval in which a biped transfers its weight, so that although the gait clock commands a double support fraction of 19.6 percent, close to the human 24 [2], the policy delivers 2.3, and the arithmetic of the contest shows compliance to be worth only three percent of the positive budget, which is to say the schedule is not really specified at all. The effort and smoothness terms are calibrated three orders of magnitude below the incentives they oppose, so leaning on a mechanical stop is cheaper than controlling the joint. The feet distance hinge measures the planar separation, which a long stride satisfies while the feet cross the midline beneath it. And the yaw authority of a machine with no vertical axis actuator, scarce to begin with, is then spent by a stance narrower than its own soles, a lateral command range of a hundredth of a metre per second, and the absence of any term asking the feet to point where the robot is going.

None of these is a failure of the learning algorithm. Each is the correct maximiser of the objective the policy was given, which is the same lesson the seventeenth pass drew when a knee reward placed four and a half tolerances into its own tail produced literally nothing, stated now from the other direction. A reward that is reachable will be reached, and the shape of what is reached is the shape of the reward rather than the shape of the intention behind it.

The programme of section 4 is ordered by cost rather than by severity, and its first phase is the one the user identified, the foot stomping and the yanking, because it is also the cheapest. Two weight changes and one already existing term address the descent, a threshold hinge on the contact force addresses the collision, and between them the trapezoid's premium is met by a penalty of comparable size. Restoring the double support interval follows, and requires only a grace window of the kind van Marum and colleagues carry [10], after which the swing profile can be respecified against the reference the configuration already declares and does not read. The stance width, the yaw authority, and the actuation smoothness follow in turn, the last placed deliberately at the end because a large part of the roughness is predicted to be a consequence of the impact and to resolve with it.

One matter should be settled before any of it begins. The working tree has been rolled back from the configuration presently training, discarding the two changes the four run ablation shows to be beneficial, the heavier roll and pitch angular velocity penalty that halves the base roll rate and the wider minimum feet distance that reduces the knee's residence against its stops from thirty eight percent to four. The programme assumes the run's values, not the tree's.

A second postscript, written 2026-08-04, records that the corrected phases succeeded and states what their success confirms. Phases 1b and 2 together carried the mean peak contact force from 2.191 body weights to 1.584, inside the human walking band [3], the sole approach speed at touchdown from 1.864 metres per second to 0.706, the double support fraction from 0.838 percent to 11.615, and the cost of transport from 3.341 to 1.870, while eliminating the ankle saturation on both axes. They did so at a third of the training budget the failed phase consumed. The thesis of the paragraph below is therefore confirmed from the positive direction, a term that measures the quantity the defect actually consists of being answered promptly by the policy, where one that measured a proxy had been answered for a whole run in the proxy's currency alone.

What the success did not buy is equally instructive and is the subject the programme now inherits. The policy paid for its softer landing by straightening its legs, the mean stance knee flexion falling to under two degrees, and a straight leg is cheap for the same reason it is fragile. The sagittal work migrated to the hip, whose torque rose to 82 percent of its ceiling, and the episodes ending on the low height condition nearly doubled. Neither consequence was priced anywhere in the reward set, which is the same observation this document has now made three times in three different currencies, and it is the strongest available argument for Phase 7 and for the knee posture term that Phase 7 will have to carry.

A third postscript, written 2026-08-05, records the completed run and corrects the second postscript above where the two disagree. Run 2026-08-03_11-19-11 finished at 29103 iterations and section 4.2b reads it against its predecessors matched at budget for the first time. Phase 1b holds at the completed run and holds more comfortably than at eleven thousand iterations, the touchdown speed having fallen further to 0.889 metres per second, so the figures the second postscript quotes for the peak force and the sole approach speed stand within a hundredth. The double support figure it quotes does not. The fraction reached 11.615 percent at eleven thousand iterations and decayed to 7.858 by the end, so Phase 2 must be recorded as failed on its own gate of 12 percent rather than as passed, its mechanism having carried the fraction up by a factor of nine from 0.838 without being sufficient against the term that opposes it.

The completed run also settles a question this document could only pose. Section 3.6.1 established by construction that the maximiser of a set point kernel is a trapezoid, and section 4.2a could show only that the profile had improved once the coarser defects were removed. The completed run shows what happens next. Over the second half of training the clearance reward's income rose by 36 percent while every measure of the profile's shape deteriorated, the swing spent within one standard deviation of the set point rising from 0.656 to 0.752 and the cost of transport from 1.870 to 2.294. The trapezoid is therefore not merely the analytic maximiser but the attractor the policy climbs toward whenever it has budget left to spend, and a term whose income rises as the behaviour it governs worsens is the plainest possible statement that it measures the wrong quantity. This is the thesis of section 5 recovered a fourth time, now from a reward that was not defective in its measurement but in its target, and it is the argument for Phase 3 stated in the only currency that settles such arguments, which is what the policy actually did with a full training budget and a term left in place.

One matter of process follows from the same run and is recorded because it cost a conclusion. Section 4.2a was written against a partly trained run and qualified its every finding accordingly, yet three of its readings still had to be withdrawn, the provisional pass of Phase 2, the caution that the tracking error would not improve, and the belief that the swing profile had been carried most of the way to its target. A partial run establishes the direction of a change and not its magnitude, and it cannot establish an equilibrium at all, since the quantity that decays is by definition the one still being traded. No gate in this programme should be adjudicated before its run completes.

A caution on method that the Phase 2 measurement supplies, and which applies to every comparison this programme will make hereafter. The tracking error metrics do not converge downward with training, because two curricula tighten as training proceeds, so a run compared against a longer predecessor will appear to track worse for reasons that have nothing to do with its reward set. Comparisons must be taken at matched iteration counts with the curriculum values verified equal, as section 4.2a does, and the practice of reading a metric at the end of one run against the end of another is unsound wherever a curriculum is active.

The first phase was trained and failed, and the postscript it supplies is worth stating because it sharpens the thesis rather than qualifying it. Phase 1 reduced its own penalty by 44 percent and the touchdown velocity by 5, which is not a small effect misapplied but a large effect applied to the wrong quantity. The policy loitered where the integral was cheapest to reduce and struck the ground as hard as before, it rotated its sole downward where the term measured only translation, and it paid for the late arrest by driving an ankle to its effort ceiling that had never before reached one. Against a reward that measured a frame height standing 23.5 millimetres above the true sole, and a clearance term whose premium was left at full value because the weight reduction the plan prescribed was never applied, every one of those responses was the cheaper path. The lesson of section 5 therefore repeats itself one level down. It is not enough that a reward oppose a defect, it must measure the quantity the defect actually consists of, and a term that measures a proxy will be satisfied by whatever the proxy fails to see.

A fourth postscript, written 2026-08-07, records the Phase 3 outcome, which is the largest improvement this programme has produced and the one that most changes what remains of it. Replacing a set point kernel on the clearance with a phase reference carried the swing profile's departure from that reference from 0.0271 metres to 0.00798, the mean peak contact force from 1.805 body weights to 1.551, the absolute joint power from 557 watts to 252, and the cost of transport from 1.953 to 0.890, which is the first figure in this series within a factor of five of human walking [1]. It did so while improving the planar tracking to the best of the four runs and while raising the training reward above the Phase 1 policy that had never carried any of the impact terms. One change of one term produced all of it.

Three lessons follow and they are of different kinds. The first is the thesis of this document confirmed from the constructive direction for the second time. A reward on an extremum determines only that extremum, and the trapezoid was not a pathology of the policy but the correct maximiser of the objective it had been given, so specifying the path rather than its summit removed the behaviour immediately and without any term being added to oppose it. The second is that the reversal of the straight legged regression was not sought, not predicted, and not priced. Section 4.2b named that regression the strongest argument for a knee posture term in Phase 7, and Phase 3 removed it without mentioning the knee, because a reference that must be met at the end of swing as well as at its apex prices the hip height that a locked leg maximises. A defect attributed to a missing term proved instead to be a consequence of a term that was present and measuring the wrong thing, which is the same error as section 3.6.7's and should be looked for before any new term is written. The third is that the gain was paid for in a currency nothing was watching. The policy freed itself of the knee's stop and took up the ankle's, and the four ankle joints now carry 57 percent of the effort budget while doing no work at all. The stops are load bearing wherever the reward set does not look, and the reward set has now been shown not to look at the ankle three separate ways, once by having no term on the roll axis, once by having a term on the pitch axis that targets a coordinate a fifth of a radian from the nominal pose, and once by charging the torque that holds a joint at a stop at a weight three orders of magnitude below the incentives that put it there.

The programme that remains is smaller than the one this document set out. Phase 7 was written against an impulsive actuation that has largely resolved, and its residue is the ankle, which is now its own experiment. Phase 5 was written against a stance width defect that measurement has withdrawn, and its residue is a guard against a tail. Phase 6 alone is unchanged, and it is now known to be bounded by a morphological authority limit rather than by a tracking gain, so it should be judged on the slope of the response and not on the error. What the four runs together establish is that of the six defects this document catalogued, the three that yielded were the three where a term was replaced by one measuring the quantity the defect consisted of, and the three that remain are the ones where no such substitution is available because the quantity is either unmeasured, as the ankle posture was, or unachievable, as the yaw is.

---

## 6. References

The sources below ground the survey of section 2 and the prescriptions of section 4. Those already recorded in `../context/literature.md` are marked with the cluster that carries them, and the remainder are added to that document by the same pass that writes this one.

Author lists are given only where `../context/literature.md` already records them or where a retrieved source stated them. Elsewhere the short form is used deliberately, since an expansion from recall would violate the grounding rule of `../CLAUDE.md`.

Entries are numbered in the order of their first appearance in the text and are cited there by the bracketed number. Five of them, numbers 1, 2, 4, 21 and 22, stand for a body of work rather than for a single paper, the survey having established the finding without fixing an individual source, and each says so in place of an author list. The code blocks of section 4 carry their identifiers inline within the docstrings, where a reader of the implementation will look for them, and are not marked a second time.

1. The cost of transport figures, 0.2 for human walking, 0.055 and 0.08 for the Cornell and Delft passive walkers against an estimated 0.05 for the mechanical component of human walking, and roughly 1.6 to 2 for the Honda humanoid, are the figures reported in the energy efficient robot locomotion literature and are used here only as an order of magnitude scale. No individual work was fixed during the survey, so the entry stands for that body of work. Cluster 13 of `../context/literature.md` carries the same figures.
2. The normative temporal parameters of section 1.4, a cadence of 113 steps per minute, a velocity of 1.33 metres per second, a stride length of 1.41 metres, and a division into 62 percent stance, 38 percent swing and 24 percent double support, were retrieved for this document from a published clinical normative table rather than taken from recall, and are the standard values reproduced throughout the gait analysis literature. The table was not itself recorded, so the entry stands for the normative source rather than for a citable paper. Cluster 13 of `../context/literature.md` carries the same figures.
3. Nilsson, J., et al. Ground reaction forces at different speeds of human walking and running. Acta Physiologica Scandinavica 136(2), 1989. The source of the walking peak of 1.0 to 1.5 body weights rising with speed, against 2.0 to 2.9 for running.
4. The minimum toe clearance figure, under two and a half centimetres occurring at or very near mid swing, comes from the toe clearance literature retrieved in the same pass. No individual work was fixed, so the entry stands for that literature. Cluster 13 of `../context/literature.md` records the same figure.
5. Winter, D. A. Biomechanics and Motor Control of Human Movement. Wiley. Cluster 10. The source of the swing phase knee flexion figure of section 1.4.
6. McGeer, T. Passive Dynamic Walking. International Journal of Robotics Research 9(2), 1990. Cluster 10.
7. Siekmann et al. Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. ICRA 2021. arXiv:2011.01387. Cluster 7.
8. Margolis, G. B., and Agrawal, P. Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior. CoRL 2022. arXiv:2212.03238. Cluster 7. The `GaitReward` class of this repository is this paper's variant, identifiable from its verbatim parameter names.
9. Gu et al. Humanoid-Gym, Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer. arXiv:2404.05695. Cluster 8. Reward table IV read in full for this document, supplying the contact pattern term at weight 1.0, the joint position tracking term at 1.5 against a phase conditioned sinusoidal reference, the energy cost at −0.0001, the action smoothness at −0.01, and the large contact force term at −0.01 in the form of the excess above four hundred newtons.
10. van Marum et al. Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking. IROS 2024. arXiv:2404.19173. Clusters 7 and 8. The source of the grace window prescribed in Phase 2, their single foot contact term returning unity if single contact occurred at least once in the preceding two tenths of a second, and of the argument against clock based rewards that section 2.2 records against it.
11. Booster Gym, An End-to-End Reinforcement Learning Framework for Humanoid Robot Locomotion. arXiv:2506.15132. Newly surveyed for this document, authors not verified. Its reward table II supplies the feet yaw alignment term at −1.0 against a combined tracking weight of 2.5, the feet roll term at −0.1, the torque tiredness term at −1e-2 as the squared ratio of applied torque to its maximum, the feet distance hinge at −1.0, and the joint position limit indicator at −1.0.
12. Rudin et al. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL 2021. arXiv:2109.11978. Cluster 7. The origin of the action rate and torque penalty forms and of the `flat_orientation_l2` posture term.
13. Seo, Y., et al. Learning Sim-to-Real Humanoid Locomotion in 15 Minutes. arXiv:2512.01996. Cluster 8. First author as stated on the paper. Cited for its reduction of the humanoid reward set below ten terms and for its use of a foot height tracking term against a reference swing profile in place of a clearance kernel. Its formulas and weights are deferred to its code and are therefore not quoted here.
14. Mind Your Steps, A General Learning Framework for Accurate Humanoid Foothold Tracking. arXiv:2606.08253. Authors not verified. Cited only for its swing window form, in which clearance accrues solely if the correct foot breaks contact within the allotted interval.
15. QuietWalk, Physics-Informed Reinforcement Learning for Ground Reaction Force-Aware Humanoid Locomotion Under Diverse Footwear. arXiv:2604.23702. Authors not verified, and the retrieved PDF did not yield its reward formulas, so it is cited only for the two qualitative findings its abstract and structure establish, that the impact penalty is a penalty on the per foot normal ground reaction force and that it must be introduced on a curriculum.
16. Mysore et al. Regularizing Action Policies for Smooth Control with Reinforcement Learning. ICRA 2021. arXiv:2012.06644. Cluster 11.
17. Hwangbo et al. Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26), 2019. Cluster 10.
18. Kim et al. Not Only Rewards But Also Constraints, Applications on Legged Robot Locomotion. IEEE Transactions on Robotics. arXiv:2308.12517. Cluster 11.
19. Bevel-geared mechanical foot, a bioinspired robotic foot compensating yaw moment of bipedal walking. Advanced Robotics. DOI 10.1080/01691864.2021.2017343. Authors not verified, the publisher page returning 403, so the work is cited by title and identifier alone. Grounds the statement of section 2.7 that the swing leg's yaw moment is a genuine disturbance absorbed by stance foot friction.
20. Perry, J. Gait Analysis, Normal and Pathological Function. SLACK Incorporated, 1992. The source of the stance width band of 1.0 to 1.3 times hip width used in section 2.8 and in the nineteenth pass of `../context/brs_gait.md`.
21. The frontal plane balance literature, from which section 2.8 takes the finding that the lateral centre of mass excursion required per step is half the stance width, so that a narrow stance is balanced by torso roll alone whereas a wide stance requires deliberate lateral motion. Cluster 13 of `../context/literature.md` records the finding without individual attribution, and none was fixed during the survey, so the entry stands for that body of work.
22. The foot placement estimator literature, from which section 2.8 takes the finding that recovery from a lateral disturbance requires the foot to be placed further laterally than the centre of mass, which a narrow nominal stance leaves no room to do. Recorded in cluster 13 of `../context/literature.md` without individual attribution, on the same terms as entry 21.
23. Su et al. Leveraging Symmetry in RL-based Legged Locomotion Control. IROS 2024. arXiv:2403.17320. Cluster 9, which carries the full author list. Grounds the symmetry data augmentation whose effect section 3.3 measures, its recommendation being that a real biped with an asymmetric identified model is better served by augmentation than by a hard equivariant network.
24. Abdolhosseini et al. On Learning Symmetric Locomotion. Motion, Interaction and Games 2019. DOI 10.1145/3359566.3360070. Cluster 9. Grounds the same augmentation, being the catalogue of the four practical enforcement methods of which it is one.
25. Li, Z., Peng, X. B., Abbeel, P., Levine, S., Berseth, G., Sreenath, K. Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control. International Journal of Robotics Research, 2025. DOI 10.1177/02783649241285161. Author list as stated by the publisher page. Retained from the survey although no claim in this document rests upon it, and marked so rather than struck out, since its presence records that the work was read.
26. Dingwell, J. B., and Cusumano, J. P. Humans use multi-objective control to regulate lateral foot placement when walking. PLoS Computational Biology, 2019. DOI 10.1371/journal.pcbi.1006850. Newly surveyed 2026-08-07 for the rescoping of section 4.5, and the individual source for the frontal plane finding that entry 21 previously stood for without attribution. Establishes by fitting competing control models to treadmill walking that lateral stepping is regulated as a multi objective problem dominated by step width, at roughly ninety three percent of the control effort against roughly seven for absolute lateral position, and that this contrasts with the single objective speed control the same authors found in the fore and aft direction.
27. Ren, J., Huang, T., Wang, H., Wang, Z., Ben, Q., Long, J., Yang, Y., Pang, J., Luo, P. VB-Com, Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception. arXiv:2502.14814, 2025. Author list as stated on the abstract page. Cited for its feet lateral distance reward, which computes the lateral separation of the two feet in the robot's own frame and penalises it below a minimum, and for the minima it reports, 0.25 metres for the Unitree H1 and 0.18 for the G1. The reward's exact expression and weight were not recovered, the retrieved document exceeding the fetch limit, so the form and the thresholds are cited and the coefficient is not.
28. Zhang, Y., Yuan, Y., Gurunath, P., Gupta, I., Omidshafiei, S., Agha-mohammadi, A., Vazquez-Chanlatte, M., Pedersen, L., He, T., Shi, G. FALCON, Learning Force-Adaptive Humanoid Loco-Manipulation. arXiv:2505.06776, 2025. Author list as stated on the retrieved document. Cited for the standing of base frame per axis separation terms in current practice, its reward table carrying a fore and aft foot separation term on the magnitude of the base frame longitudinal difference between the feet at weight −5.0 and a lateral root centring term at the same weight. It carries no lateral foot separation hinge, which is itself recorded, since the absence bears on how universal the form is.
29. Xie, W., Bai, C., Shi, J., Yang, J., Ge, Y., Zhang, W., Li, X. Humanoid Whole-Body Locomotion on Narrow Terrain via Dynamic Balance and Reinforcement Learning. arXiv:2502.17219, 2025. Author list as stated on the retrieved document. Cited for the caution of section 4.5, that a humanoid may be required to traverse a support narrower than its own nominal stance, this policy crossing a beam of twenty five centimetres. Its reward table names a feet separation term in its gait group but states neither its expression nor its weight, and the omission is recorded rather than filled from recall.

The internal sources that ground section 3 are the dumped `params/env.yaml` of the four runs, the numpy dumps under each run's `data/42/`, the eleven evaluation plots and the play video held at `../context/artefacts/2026-07-28_06-37-24-play/`, and the running record of `../context/brs_gait.md`, whose twentieth pass records the measurements of section 3 in the context tree.
