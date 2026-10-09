"""Convert the wide TensorBoard scalar table for
quadruped_flat/2026-09-01_07-22-37 into the physical quantities each reward
term integrates, and into a reward budget. Adapted from the identical script
used for the 2026-08-25_10-15-10 run (weights unchanged, see brief), with
three additions this run's investigation needs that the original script did
not compute: pen_joint_vel_l2 and pen_joint_accel converted to rms per-joint
velocity/acceleration, pen_action_rate converted to rms action rate, and an
explicit sign/fraction-negative tally for Episode_Reward/feet_air_time, since
feet_air_time is now feet_air_time_v2 (threshold_min=0.0) and can go
unboundedly negative per the brief. All conversions use
raw = logged * episode_length_s / (weight * dt), dt=0.02s, episode_length_s=20.0,
i.e. Reward manager arithmetic value = func() * weight * dt, and the logged
Episode_Reward/<term> is the per-episode accumulated weighted sum divided by
episode_length_s.

Term semantics (confirmed against
/ws/tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/mdp/rewards.py
and IsaacLab's own isaaclab/envs/mdp/rewards.py):
  rew_lin_vel_xy, rew_ang_vel_z   exp(-error^2/std^2) tracking kernel, std=0.4 both
  keep_balance                     stay_alive, 1.0 while alive
  pen_base_height                  base_height_rough_l2: (clearance-target)^2, target 0.33
  pen_lin_vel_z                    v_z^2
  pen_ang_vel_xy                   sum_{x,y} w^2
  pen_joint_torque                 sum_{12 joints} tau^2
  pen_joint_accel                  sum_{12 joints} qddot^2
  pen_action_rate                  sum_{12 actions} (a_t - a_{t-1})^2
  pen_joint_pos_limits              sum of soft-limit violation (abs, one-sided)
  pen_joint_vel_l2                 sum_{12 joints} qdot^2
  pen_action_smoothness            ActionSmoothnessPenalty (2nd-difference-based, custom class)
  pen_flat_orientation             sum_{x,y} g_b^2 (projected gravity)
  pen_undesired_contacts           count of non-foot bodies with contact force > 10 N
  pen_abad_deviation                sum_{4 abad joints} |q - q_default|  (L1)
  feet_air_time (v2)               per-touchdown: +t if t<=0.4s, else (0.4-t) [unbounded negative]; 0 if never lands
  feet_slide                       sum over feet in contact of foot xy speed
"""
import csv
import gzip
import math

PATH = "/ws/context/artefacts/quadruped_flat-2026-09-01_07-22-37/tb_scalars_wide.csv.gz"
with gzip.open(PATH, "rt") as f:
    rows = list(csv.DictReader(f))

DT = 0.02
EPL = 20.0
N_JOINTS = 12
N_ABAD = 4
N_ACTIONS = 12

W = {'feet_air_time': 2.0, 'feet_slide': -0.25, 'keep_balance': 0.05, 'pen_abad_deviation': -2.0,
     'pen_action_rate': -0.05, 'pen_action_smoothness': -0.075, 'pen_ang_vel_xy': -0.5,
     'pen_base_height': -5.0, 'pen_flat_orientation': -1.0, 'pen_joint_accel': -2.5e-7,
     'pen_joint_pos_limits': -2.0, 'pen_joint_torque': -2.0e-4, 'pen_joint_vel_l2': -5.0e-5,
     'pen_lin_vel_z': -2.0, 'pen_undesired_contacts': -2.5, 'rew_ang_vel_z': 12.5, 'rew_lin_vel_xy': 25.0}


def series(tag):
    return [float(r['Episode_Reward/' + tag]) for r in rows]


def win(a, b, tag):
    v = series(tag)[a:b]
    return sum(v) / len(v)


def eplen(a, b):
    v = [float(r['Train/mean_episode_length']) for r in rows[a:b]]
    return sum(v) / len(v)


def raw(tag, a, b):
    return win(a, b, tag) * EPL / (W[tag] * DT)


if __name__ == "__main__":
    n = len(rows)
    print("N rows (iterations):", n)

    fat = series('feet_air_time')
    n_neg = sum(1 for v in fat if v < 0)
    print("feet_air_time: n<0 = %d / %d = %.4f fraction negative (whole run)" % (n_neg, n, n_neg / n))
    for label, (a, b) in [('early 0-200', (0, 200)), ('mid ~4918-5118', (4918, 5118)), ('late 9737-10037', (9737, 10037))]:
        seg = fat[a:b]
        nn = sum(1 for v in seg if v < 0)
        print("  %s: n<0=%d/%d=%.3f  mean=%.5f  min=%.5f max=%.5f" % (label, nn, len(seg), nn/len(seg), sum(seg)/len(seg), min(seg), max(seg)))

    for label, (a, b) in [('early it 0-200', (0, 200)), ('mid it 4918-5118', (4918, 5118)), ('late it 9737-10037', (9737, 10037))]:
        print('==', label)
        NSTEP = eplen(a, b)
        print("  mean episode length %.2f steps" % NSTEP)
        ra = raw('feet_air_time', a, b)
        print('  feet_air_time raw per-episode touchdown-score sum (NOT seconds under v2) %9.4f' % ra)
        r = raw('pen_lin_vel_z', a, b)
        print('  mean v_z^2 %8.5f  rms v_z %6.4f m/s' % (r / NSTEP, math.sqrt(r / NSTEP)))
        r = raw('pen_flat_orientation', a, b)
        print('  mean |g_xy|^2 %8.5f  |g_xy| %6.4f  tilt %5.2f deg' % (r / NSTEP, math.sqrt(r / NSTEP), math.degrees(math.asin(min(1, math.sqrt(r / NSTEP))))))
        r = raw('pen_ang_vel_xy', a, b)
        print('  mean w_xy^2(sum x,y) %7.4f -> rms per-axis %5.4f rad/s' % (r / NSTEP, math.sqrt(r / NSTEP / 2)))
        r = raw('pen_base_height', a, b)
        print('  rms base height error %6.4f m' % math.sqrt(r / NSTEP))
        r = raw('pen_abad_deviation', a, b)
        print('  mean sum|abad| %6.4f rad  per joint %6.4f rad = %4.2f deg' % (r / NSTEP, r / NSTEP / N_ABAD, math.degrees(r / NSTEP / N_ABAD)))
        r = raw('pen_joint_pos_limits', a, b)
        print('  mean soft-limit violation summed %7.5f rad' % (r / NSTEP))
        r = raw('pen_undesired_contacts', a, b)
        print('  mean non-foot bodies in contact %6.4f  (pct of steps %4.1f)' % (r / NSTEP, 100 * r / NSTEP))
        r = raw('pen_joint_torque', a, b)
        print('  mean sum tau^2 %8.4f N2m2 -> rms tau/joint %5.4f Nm' % (r / NSTEP, math.sqrt(r / NSTEP / N_JOINTS)))
        r = raw('pen_joint_vel_l2', a, b)
        print('  mean sum qdot^2 %8.4f -> rms qdot/joint %6.4f rad/s' % (r / NSTEP, math.sqrt(r / NSTEP / N_JOINTS)))
        r = raw('pen_joint_accel', a, b)
        print('  mean sum qddot^2 %10.2f -> rms qddot/joint %8.3f rad/s2' % (r / NSTEP, math.sqrt(r / NSTEP / N_JOINTS)))
        r = raw('pen_action_rate', a, b)
        print('  mean sum d_action^2 %8.5f -> rms d_action/dim %6.5f' % (r / NSTEP, math.sqrt(r / NSTEP / N_ACTIONS)))
        r = raw('feet_slide', a, b)
        print('  mean sliding-foot speed sum m/s %6.4f' % (r / NSTEP))
        r = raw('rew_lin_vel_xy', a, b)
        print('  mean lin tracking kernel %6.4f (1.0 perfect)' % (r / NSTEP))
        r = raw('rew_ang_vel_z', a, b)
        print('  mean ang tracking kernel %6.4f' % (r / NSTEP))
        r = raw('keep_balance', a, b)
        print('  alive fraction of nominal episode %6.4f' % (r / NSTEP))
