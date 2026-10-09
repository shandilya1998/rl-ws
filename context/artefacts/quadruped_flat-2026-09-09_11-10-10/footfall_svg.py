import numpy as np
BASE='/ws/IsaacLab/logs/rsl_rl/quadruped_flat/{}/data/42/'
FEET=['FL','FR','RL','RR']; COL=['#1b6ca8','#e07a2f','#2e8b57','#a83232']
runs=[('2026-09-08_07-37-52','Run B, phase 1 + foot clearance (phase 3)'),
      ('2026-09-09_11-10-10','Run C, run B + landing velocity and impact penalties')]
ENV=0; T0,T1=200,700; W,H=1180,470; L,R=78,150; ROW=34; GAP=118
out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     f'<text x="{L}" y="26" font-size="15" font-weight="bold" fill="#111">Footfall pattern, filled bar = foot loaded above 1 N</text>',
     f'<text x="{L}" y="45" font-size="12" fill="#555">Environment {ENV}, 10 s window from t = {T0*0.02:.0f} s. Duty percentages are over all 32 environments and the full 60 s dump.</text>']
y0=72
for run,title in runs:
    d=np.load(BASE.format(run)+'dump.npy',allow_pickle=True).item()
    dt=d['step_dt']
    f=np.linalg.norm(d['feet_contact_forces'],axis=-1)
    c=(f>1.0); duty=c[50:].mean(axis=(0,1)); ce=c[:,ENV,:]
    sx=(W-L-R)/float(T1-T0)
    out.append(f'<text x="{L}" y="{y0-9}" font-size="13" font-weight="bold" fill="#111">{title}</text>')
    for k in range(4):
        yy=y0+k*ROW
        out.append(f'<rect x="{L}" y="{yy}" width="{W-L-R}" height="{ROW-9}" fill="#f2f2f2"/>')
        seg=ce[T0:T1,k]; i=0
        while i<len(seg):
            if seg[i]:
                j=i
                while j<len(seg) and seg[j]: j+=1
                out.append(f'<rect x="{L+i*sx:.2f}" y="{yy}" width="{max((j-i)*sx,0.7):.2f}" height="{ROW-9}" fill="{COL[k]}"/>')
                i=j
            else: i+=1
        out.append(f'<text x="{L-8}" y="{yy+ROW-15}" font-size="12.5" font-weight="bold" fill="{COL[k]}" text-anchor="end">{FEET[k]}</text>')
        note=" DEAD" if duty[k]<0.02 else ""
        out.append(f'<text x="{W-R+8}" y="{yy+ROW-15}" font-size="11.5" fill="{COL[k]}">duty {duty[k]*100:5.1f}%{note}</text>')
    for s in range(0,11,2):
        x=L+(s/10.0)*(W-L-R)
        out.append(f'<line x1="{x:.1f}" y1="{y0+4*ROW-9}" x2="{x:.1f}" y2="{y0+4*ROW-4}" stroke="#999"/>')
        out.append(f'<text x="{x:.1f}" y="{y0+4*ROW+9}" font-size="10.5" fill="#666" text-anchor="middle">{T0*0.02+s:.0f}s</text>')
    y0+=4*ROW+GAP-46
out.append('</svg>')
open('footfall.svg','w').write("\n".join(out))
print("wrote footfall.svg", len("\n".join(out)), "bytes")
