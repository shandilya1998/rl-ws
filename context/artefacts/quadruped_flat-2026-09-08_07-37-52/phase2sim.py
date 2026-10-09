import numpy as np, json
BASE='/ws/IsaacLab/logs/rsl_rl/quadruped_flat/{}/data/42/'
FEET=['FL','FR','RL','RR']
DT=0.02; EP=20.0; SPE=int(EP/DT)   # 1000 steps per episode

def sim(run, ceiling=0.45, weight=-0.1, tmin=0.15, tmax=0.45, w_air=5.0):
    d=np.load(BASE.format(run)+'dump.npy',allow_pickle=True).item()
    f=np.linalg.norm(d['feet_contact_forces'],axis=-1)
    contact=f>1.0                       # (T,E,4)
    T,E,_=contact.shape
    cmd=np.linalg.norm(d['commanded_linear_velocity'],axis=-1)>0.1   # (T,E)
    # reconstruct current_air_time with a reset at each episode boundary
    cur=np.zeros((T,E,4)); acc=np.zeros((E,4))
    for t in range(T):
        if t % SPE == 0: acc[:]=0.0
        acc = np.where(contact[t], 0.0, acc+DT)
        cur[t]=acc
    excess=np.clip(cur-ceiling,0.0,None)          # (T,E,4)
    func=excess.sum(axis=2)*cmd                   # (T,E)
    ep_per_foot=(excess*cmd[:,:,None]).mean(axis=(0,1))*EP*weight
    ep_total=float(func.mean()*EP*weight)
    # per-foot feet_air_time_v2 contribution
    air_pay=np.zeros(4); n_land=np.zeros(4)
    for e in range(E):
        for k in range(4):
            s=contact[:,e,k]
            td=np.flatnonzero((~s[:-1])&(s[1:]))+1
            for i in td:
                # air time = length of the airborne run ending at i
                j=i-1
                while j>=0 and not s[j]: j-=1
                a=(i-1-j)*DT
                p=a-tmin
                p=(tmax-tmin)-p if p>(tmax-tmin) else p
                air_pay[k]+=p; n_land[k]+=1
    air_ep=air_pay/E*(EP/ (T*DT))*w_air   # scale sum-over-dump to one episode
    return dict(run=run,
        duty={FEET[k]:round(float(contact[50:,:,k].mean()),4) for k in range(4)},
        hold_pen_per_episode=round(ep_total,3),
        hold_pen_per_foot={FEET[k]:round(float(ep_per_foot[k]),3) for k in range(4)},
        frac_steps_any_excess=round(float((excess>0).any(axis=2).mean()),4),
        air_v2_per_foot_per_episode={FEET[k]:round(float(air_ep[k]),4) for k in range(4)},
        air_v2_total_per_episode=round(float(air_ep.sum()),4),
        landings_per_foot_per_episode={FEET[k]:round(float(n_land[k]/E*(EP/(T*DT))),1) for k in range(4)},
    )

for r in ['2026-09-08_07-37-52','2026-09-09_11-10-10']:
    print(json.dumps(sim(r),indent=1))
