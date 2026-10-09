import numpy as np
BASE='/ws/IsaacLab/logs/rsl_rl/quadruped_flat/{}/data/42/'
FEET=['FL','FR','RL','RR']; DT=0.02; EP=20.0; SPE=1000
def runs_from(mask,T):   # mask True == airborne
    out=[];held=[]
    for e in range(mask.shape[1]):
        for k in range(mask.shape[2]):
            s=mask[:,e,k]
            idx=np.flatnonzero(np.diff(s.astype(np.int8)))
            edges=np.concatenate(([0],idx+1,[T]))
            for a,b in zip(edges[:-1],edges[1:]):
                if s[a]:
                    (held if b==T else out).append((b-a)*DT)
    return np.array(out),np.array(held)

for run in ['2026-09-08_07-37-52','2026-09-09_11-10-10']:
    d=np.load(BASE.format(run)+'dump.npy',allow_pickle=True).item()
    T=d['feet_sole_clearances'].shape[0]
    clr=d['feet_sole_clearances']
    frc=np.linalg.norm(d['feet_contact_forces'],axis=-1)
    print("="*70); print(run)
    print(f"  history_length={d['contact_force_history_length']}  smear={d['contact_force_history_length']*DT:.2f}s")
    for label,air in [("force>1N (smeared)", frc<=1.0),
                      ("clearance>5mm",  clr>0.005),
                      ("clearance>10mm", clr>0.010)]:
        a,h=runs_from(air,T)
        duty=(~air[50:]).mean(axis=(0,1))
        pf={FEET[k]:round(float((~air[50:,:,k]).mean()),4) for k in range(4)}
        if a.size:
            print(f"  {label:<20} air n={a.size:>6} med={np.median(a):.4f} mean={a.mean():.4f} "
                  f"p90={np.percentile(a,90):.4f} <0.15={100*(a<0.15).mean():.1f}% >0.45={100*(a>0.45).mean():.2f}%")
            print(f"  {'':<20} duty per foot {pf}   held-to-end n={h.size} max={h.max() if h.size else 0:.2f}s")
    # clearance percentiles while force says contact
    incon=frc>1.0
    print(f"  clearance when force>1N: p50={np.percentile(clr[incon],50):.4f} p95={np.percentile(clr[incon],95):.4f}")
