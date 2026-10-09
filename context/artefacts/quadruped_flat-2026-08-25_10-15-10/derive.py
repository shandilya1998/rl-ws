import csv, math
rows=list(csv.DictReader(open('tb_scalars_wide.csv')))
DT=0.02; EPL=20.0; NSTEP=1000
W={'feet_air_time':2.0,'feet_slide':-0.25,'keep_balance':0.05,'pen_abad_deviation':-2.0,
   'pen_action_rate':-0.05,'pen_action_smoothness':-0.075,'pen_ang_vel_xy':-0.5,
   'pen_base_height':-5.0,'pen_flat_orientation':-1.0,'pen_joint_accel':-2.5e-7,
   'pen_joint_pos_limits':-2.0,'pen_joint_torque':-2.0e-4,'pen_joint_vel_l2':-5.0e-5,
   'pen_lin_vel_z':-2.0,'pen_undesired_contacts':-2.5,'rew_ang_vel_z':12.5,'rew_lin_vel_xy':25.0}
def win(a,b,tag):
    v=[float(r['Episode_Reward/'+tag]) for r in rows[a:b]]
    return sum(v)/len(v)
def eplen(a,b):
    v=[float(r['Train/mean_episode_length']) for r in rows[a:b]]
    return sum(v)/len(v)
def raw(tag,a,b):  # raw per-episode sum of the unweighted term value
    return win(a,b,tag)*EPL/(W[tag]*DT)
for label,(a,b) in [('early it 400-600',(400,600)),('mid it 4900-5100',(4900,5100)),('late it 16968-17267',(16968,17268))]:
    print('==',label)
    NSTEP=eplen(a,b); print("  mean episode length %.1f steps"%NSTEP)
    ra=raw('feet_air_time',a,b); print('  feet_air_time raw sum s      %9.2f  min landings/episode %.0f'%(ra, abs(ra)/0.25))
    r=raw('pen_lin_vel_z',a,b);  print('  mean v_z^2 %8.4f  rms v_z %6.3f m/s'%(r/NSTEP, math.sqrt(r/NSTEP)))
    r=raw('pen_flat_orientation',a,b); print('  mean |g_xy|^2 %8.5f  |g_xy| %6.4f  tilt %5.2f deg'%(r/NSTEP, math.sqrt(r/NSTEP), math.degrees(math.asin(min(1,math.sqrt(r/NSTEP))))))
    r=raw('pen_base_height',a,b); print('  rms base height error %6.4f m'%math.sqrt(r/NSTEP))
    r=raw('pen_abad_deviation',a,b); print('  mean sum|abad| %6.4f rad  per joint %6.4f rad = %4.2f deg'%(r/NSTEP, r/NSTEP/4, math.degrees(r/NSTEP/4)))
    r=raw('pen_joint_pos_limits',a,b); print('  mean soft-limit violation summed %7.5f rad'%(r/NSTEP))
    r=raw('pen_undesired_contacts',a,b); print('  mean non-foot bodies in contact %6.4f  (pct of steps %4.1f)'%(r/NSTEP, 100*r/NSTEP))
    r=raw('pen_joint_torque',a,b); print('  mean sum tau^2 %8.3f N2m2 -> rms tau/joint %5.3f Nm'%(r/NSTEP, math.sqrt(r/NSTEP/12)))
    r=raw('pen_ang_vel_xy',a,b); print('  mean w_xy^2 %7.4f -> rms %5.3f rad/s'%(r/NSTEP, math.sqrt(r/NSTEP)))
    r=raw('feet_slide',a,b); print('  mean sliding foot speed sum %6.4f m/s'%(r/NSTEP))
    r=raw('rew_lin_vel_xy',a,b); print('  mean tracking kernel %6.4f (1.0 perfect)'%(r/NSTEP))
    r=raw('rew_ang_vel_z',a,b); print('  mean ang kernel      %6.4f'%(r/NSTEP))
    r=raw('keep_balance',a,b); print('  alive fraction of nominal episode %6.4f'%(r/NSTEP))
