import csv,numpy as np
runs=["2026-09-02_11-30-45","2026-09-08_07-37-52","2026-09-09_11-10-10"]
lbl={"2026-09-02_11-30-45":"A' base","2026-09-08_07-37-52":"B clear","2026-09-09_11-10-10":"C +imp"}
def load(r):
    rows=list(csv.reader(open(f"tb/{r}/tb_scalars_wide.csv"))); hdr=rows[0]
    d={}
    for i,name in enumerate(hdr):
        if i==0: continue
        s=[];v=[]
        for row in rows[1:]:
            if row[i]!='': s.append(int(row[0])); v.append(float(row[i]))
        if s: d[name]=(np.array(s),np.array(v))
    return d
D={r:load(r) for r in runs}
print("RUN LENGTHS (last logged iteration)")
for r in runs:
    s,v=D[r]["Train/mean_reward"]; print(f"  {lbl[r]:<10} {s[-1]}")
LAST=min(D[r]["Train/mean_reward"][0][-1] for r in runs)
print(f"\nCommon horizon = {LAST}\n")

def rmed(v,w=101):
    out=np.empty_like(v)
    for i in range(len(v)):
        a=max(0,i-w//2); b=min(len(v),i+w//2+1); out[i]=np.median(v[a:b])
    return out

print("Train/mean_reward, ROLLING MEDIAN (w=101), spike-insensitive")
CK=[1000,2000,3000,4000,5000,6000,7000,7600]
print(f"  {'iter':<10}"+"".join(f"{c:>9}" for c in CK)+f"{'  ret%':>9}")
for r in runs:
    s,v=D[r]["Train/mean_reward"]; m=rmed(v)
    row=[m[np.searchsorted(s,c)] for c in CK]
    print(f"  {lbl[r]:<10}"+"".join(f"{x:>9.2f}" for x in row)+f"{100*row[-1]/max(row):>8.1f}%")

print("\nSPIKES: iterations where mean_reward < 0.5 x rolling median")
for r in runs:
    s,v=D[r]["Train/mean_reward"]; m=rmed(v)
    bad=np.flatnonzero(v < 0.5*m)
    print(f"  {lbl[r]:<10} n={len(bad):<5} rate={100*len(bad)/len(v):.2f}%  worst={v[bad].min() if len(bad) else 0:.1f}  "
          f"iters={s[bad][:12].tolist()}{'...' if len(bad)>12 else ''}")

print("\nOPTIMISER HEALTH at common horizon %d (Phase 1 target)"%LAST)
print(f"  {'run':<10}{'noise_std':>11}{'entropy':>10}{'ent*coef':>10}{'surrogate':>11}{'ratio':>9}{'lr':>11}")
for r in runs:
    def g(t):
        s,v=D[r][t]; i=np.searchsorted(s,LAST); return float(v[min(i,len(v)-1)])
    e=g("Loss/entropy"); su=abs(g("Loss/surrogate"))
    print(f"  {lbl[r]:<10}{g('Policy/mean_noise_std'):>11.4f}{e:>10.3f}{0.001*e:>10.4f}{su:>11.5f}{(0.001*e/su if su>1e-9 else float('inf')):>9.1f}{g('Loss/learning_rate'):>11.2e}")
print("\n  run A (2026-09-01) for reference: noise_std 1.594 RISING, entropy 21.496, ent*coef/surrogate = 26.2")

print("\nEpisode_Reward/feet_air_time  x20 = per-episode units")
CK2=[1000,3000,5000,7000]
for r in runs:
    s,v=D[r]["Episode_Reward/feet_air_time"]
    row=[v[np.searchsorted(s,c)]*20 for c in CK2]
    print(f"  {lbl[r]:<10}"+"".join(f"{x:>10.3f}" for x in row))
