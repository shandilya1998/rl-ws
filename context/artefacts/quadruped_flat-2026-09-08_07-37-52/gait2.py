import numpy as np, json, sys
BASE='/ws/IsaacLab/logs/rsl_rl/quadruped_flat/{}/data/42/'
FEET=['FL','FR','RL','RR']
EPLEN=20.0   # episode_length_s ; per-term rewards.npy series are RATES per second
NSTEP=1000

def analyse(run):
    p=BASE.format(run)
    d=np.load(p+'dump.npy',allow_pickle=True).item()
    r=np.load(p+'rewards.npy',allow_pickle=True).item()
    dt=d['step_dt']; out={'run':run,'dt':dt}
    f=np.linalg.norm(d['feet_contact_forces'],axis=-1)   # (T,E,4)
    contact=f>1.0
    T,E,_=contact.shape
    c=contact[50:]
    out['T']=T; out['E']=E
    out['duty_global']=dict(zip(FEET,c.mean(axis=(0,1)).round(4).tolist()))
    duty_env=c.mean(axis=0)                              # (E,4)
    nsup=c.sum(axis=-1)
    hist=np.bincount(nsup.ravel(),minlength=5)/nsup.size
    out['support_hist']={str(k):round(float(v),4) for k,v in enumerate(hist)}
    out['mean_feet_grounded']=round(float(nsup.mean()),4)
    dead=(duty_env<0.05).sum(axis=1)
    out['envs_by_n_dead_feet']={str(k):int((dead==k).sum()) for k in range(5)}
    out['duty_per_env_sorted_mean']=np.sort(duty_env,axis=1).mean(axis=0).round(4).tolist()
    out['duty_env_min']=round(float(duty_env.min()),4)
    out['n_env_feet_duty_lt_0.05']=int((duty_env<0.05).sum())
    out['n_env_feet_duty_lt_0.15']=int((duty_env<0.15).sum())

    # ---- air interval statistics -------------------------------------
    airs=[];holds=[];stances=[]
    for e in range(E):
        for k in range(4):
            s=contact[:,e,k]
            idx=np.flatnonzero(np.diff(s.astype(np.int8)))
            edges=np.concatenate(([0],idx+1,[T]))
            for a,b in zip(edges[:-1],edges[1:]):
                L=(b-a)*dt
                if not s[a]:
                    (holds if b==T else airs).append(L)
                else:
                    if b!=T: stances.append(L)
    airs=np.array(airs);holds=np.array(holds);stances=np.array(stances)
    out['n_air_intervals']=int(airs.size)
    if airs.size:
        out['air_pct']={str(q):round(float(np.percentile(airs,q)),4) for q in (5,25,50,75,90,99)}
        out['air_mean']=round(float(airs.mean()),4)
        out['air_frac_under_0.15']=round(float((airs<0.15).mean()),4)
        out['air_frac_over_0.45']=round(float((airs>0.45).mean()),4)
        out['landings_per_env']=round(float(airs.size/E),1)
    if stances.size:
        out['stance_median']=round(float(np.median(stances)),4)
    out['n_unterminated_air_runs']=int(holds.size)
    if holds.size:
        out['hold_mean_s']=round(float(holds.mean()),3)
        out['hold_max_s']=round(float(holds.max()),3)
        out['hold_over_5s']=int((holds>5.0).sum())

    # ---- feet_air_time_v2 payoff, thresholds from the run config -------
    tmin,tmax=0.15,0.45; cap=tmax-tmin
    if airs.size:
        a=airs-tmin
        pay=np.where(a>cap,cap-a,a)
        out['v2_pay_mean_per_landing']=round(float(pay.mean()),4)
        out['v2_pay_frac_negative']=round(float((pay<0).mean()),4)
        out['v2_pay_total_per_env_episode']=round(float(pay.sum()/E),3)

    # ---- gait coordination -------------------------------------------
    # normalised cross-correlation of the four binary contact trains, per env,
    # searched over lag; a trot puts the diagonal pair at lag 0 and the
    # lateral/fore-aft pairs at half a stride.
    def best_lag(x,y,maxlag):
        x=x-x.mean(); y=y-y.mean()
        if x.std()<1e-6 or y.std()<1e-6: return None,None
        n=len(x); best=(-2,0)
        for L in range(-maxlag,maxlag+1):
            if L>=0: a_,b_=x[L:],y[:n-L] if L>0 else y
            else: a_,b_=x[:n+L],y[-L:]
            if len(a_)<50: continue
            cval=float(np.dot(a_,b_)/(len(a_)*x.std()*y.std()))
            if cval>best[0]: best=(cval,L)
        return best[0],best[1]
    pairs={'FL-FR':(0,1),'RL-RR':(2,3),'FL-RL':(0,2),'FR-RR':(1,3),
           'FL-RR':(0,3),'FR-RL':(1,2)}
    # stride period per env from the dominant foot's contact autocorrelation
    corr={k:[] for k in pairs}; lags={k:[] for k in pairs}
    periods=[]
    for e in range(E):
        sig=c[:,e,:].astype(float)
        live=[k for k in range(4) if 0.05<sig[:,k].mean()<0.95]
        if len(live)<2: continue
        # stride period: autocorr peak of the first live foot beyond 5 steps
        x=sig[:,live[0]]-sig[:,live[0]].mean()
        ac=np.correlate(x,x,'full')[len(x)-1:]
        ac=ac/ (ac[0]+1e-12)
        srch=ac[5:150]
        if srch.size: periods.append((np.argmax(srch)+5)*dt)
        for name,(i,j) in pairs.items():
            if i in live and j in live:
                cv,L=best_lag(sig[:,i],sig[:,j],60)
                if cv is not None:
                    corr[name].append(cv); lags[name].append(L*dt)
    out['stride_period_s']=round(float(np.median(periods)),4) if periods else None
    out['pair_corr']={k:(round(float(np.mean(v)),3) if v else None) for k,v in corr.items()}
    out['pair_best_lag_s']={k:(round(float(np.median(v)),3) if v else None) for k,v in lags.items()}
    # zero-lag correlation, the direct synchrony measure
    zl={}
    for name,(i,j) in pairs.items():
        acc=[]
        for e in range(E):
            a_=c[:,e,i].astype(float); b_=c[:,e,j].astype(float)
            if a_.std()<1e-6 or b_.std()<1e-6: continue
            acc.append(float(np.corrcoef(a_,b_)[0,1]))
        zl[name]=round(float(np.mean(acc)),3) if acc else None
    out['pair_corr_zero_lag']=zl

    # ---- impact / landing diagnostics ---------------------------------
    fz=np.abs(d['feet_contact_forces'][50:,:,:,2])
    fn=np.linalg.norm(d['feet_contact_forces'][50:],axis=-1)
    out['impact']={
      'peak_force_p50':round(float(np.percentile(fn,50)),2),
      'peak_force_p95':round(float(np.percentile(fn,95)),2),
      'peak_force_p99':round(float(np.percentile(fn,99)),2),
      'max_force':round(float(fn.max()),2),
      'frac_over_160N':round(float((fn>160.0).mean()),5),
      'mean_excess_over_160N':round(float(np.clip(fn-160.0,0,2000).sum(axis=-1).mean()),3),
    }
    # touchdown vertical speed: |vz| at last airborne sample before a landing
    vz=d['feet_velocities'][:,:,:,2]
    tdv=[]
    for e in range(E):
        for k in range(4):
            s=contact[:,e,k]
            td=np.flatnonzero((~s[:-1])&(s[1:]))+1
            td=td[td>50]
            if td.size: tdv.append(np.abs(vz[td-1,e,k]))
    if tdv:
        tdv=np.concatenate(tdv)
        out['touchdown_vz']={'mean':round(float(tdv.mean()),4),
                             'p50':round(float(np.percentile(tdv,50)),4),
                             'p95':round(float(np.percentile(tdv,95)),4)}
    # ---- clearance ----------------------------------------------------
    h=d['feet_sole_clearances'][50:] if 'feet_sole_clearances' in d else d['feet_frame_heights'][50:]
    sw=~c
    out['clearance']={
      'key':'feet_sole_clearances' if 'feet_sole_clearances' in d else 'feet_frame_heights',
      'mean_airborne':round(float(h[sw].mean()),4) if sw.any() else None,
      'p90_airborne':round(float(np.percentile(h[sw],90)),4) if sw.any() else None,
      'per_foot_mean':{FEET[k]:round(float(h[:,:,k].mean()),4) for k in range(4)},
      'per_foot_max':{FEET[k]:round(float(h[:,:,k].max()),4) for k in range(4)},
    }
    # ---- base ---------------------------------------------------------
    blv=d['base_linear_velocity'][50:]; bav=d['base_angular_velocity'][50:]
    clv=d['commanded_linear_velocity'][50:]; cav=d['commanded_angular_velocity'][50:]
    pg=d['base_projected_gravity'][50:]
    tilt=np.degrees(np.arccos(np.clip(-pg[:,:,2]/np.linalg.norm(pg,axis=-1),-1,1)))
    bz=d['base_com_position'][50:,:,2]
    out['base']={
      'lin_track_err_mean':round(float(np.linalg.norm(clv-blv[:,:,:2],axis=-1).mean()),4),
      'ang_track_err_mean':round(float(np.abs(cav-bav[:,:,2]).mean()),4),
      'cmd_speed_mean':round(float(np.linalg.norm(clv,axis=-1).mean()),4),
      'actual_speed_mean':round(float(np.linalg.norm(blv[:,:,:2],axis=-1).mean()),4),
      'rms_vz':round(float(np.sqrt((blv[:,:,2]**2).mean())),4),
      'tilt_deg_mean':round(float(tilt.mean()),3),
      'tilt_deg_p95':round(float(np.percentile(tilt,95)),3),
      'height_mean':round(float(bz.mean()),4),
      'height_p2p':round(float((bz.max(axis=0)-bz.min(axis=0)).mean()),4),
    }
    # ---- joints -------------------------------------------------------
    v=d['joint_velocities'][50:]; pw=np.abs(d['joint_powers'][50:]); tq=d['joint_torques'][50:]
    jn=d['joint_names']
    groups={'abad':[i for i,n in enumerate(jn) if 'abad' in n],
            'hip':[i for i,n in enumerate(jn) if n.startswith('hip')],
            'knee':[i for i,n in enumerate(jn) if 'knee' in n]}
    g={}
    for name,ids in groups.items():
        vv=v[:,:,ids]
        zc=(np.diff(np.sign(vv),axis=0)!=0).sum(axis=0)/(vv.shape[0]*dt)/2.0
        g[name]={'rms_vel':round(float(np.sqrt((vv**2).mean())),3),
                 'zc_freq_hz':round(float(zc.mean()),3),
                 'rms_torque':round(float(np.sqrt((tq[:,:,ids]**2).mean())),3),
                 'total_abs_power_W':round(float(pw[:,:,ids].mean(axis=(0,1)).sum()),3)}
    out['joints']=g
    out['total_mech_power_W']=round(float(pw.sum(axis=-1).mean()),3)
    # per-joint power, to see if two knees carry everything
    pj=pw.mean(axis=(0,1))
    out['power_per_joint']={jn[i]:round(float(pj[i]),3) for i in np.argsort(-pj)[:6]}

    # ---- reward budget, per EPISODE = rate * episode_length_s ----------
    terms={k:v for k,v in r.items() if k!='_weights'}
    tot=terms.pop('total_reward')
    budget={}
    for k,series in terms.items():
        budget[k]=round(float(series[50:].mean()*EPLEN),4)
    out['reward_budget_per_episode']=dict(sorted(budget.items(),key=lambda kv:-abs(kv[1])))
    out['total_reward_per_episode']=round(float(tot[50:].mean()*NSTEP),2)
    out['weights']={k:v for k,v in r['_weights'].items()}
    out['terminations']={'terminated':int(d['episode_terminated'].sum()),
                         'time_outs':int(d['episode_time_outs'].sum())}
    return out

runs=sys.argv[1:]
res=[analyse(r) for r in runs]
print(json.dumps(res,indent=1))
