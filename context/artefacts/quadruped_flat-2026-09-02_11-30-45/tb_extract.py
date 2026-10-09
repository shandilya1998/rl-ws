import sys, csv, json
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

path = sys.argv[1]
outdir = sys.argv[2]

ea = event_accumulator.EventAccumulator(path, size_guidance={'scalars': 0})
ea.Reload()
tags = sorted(ea.Tags().get('scalars', []))

# gather per-tag step->value
data = {}
tag_info = []
all_steps = set()
for t in tags:
    events = ea.Scalars(t)
    d = {}
    for e in events:
        d[e.step] = e.value
    data[t] = d
    all_steps.update(d.keys())
    steps_sorted = sorted(d.keys())
    tag_info.append({
        'tag': t,
        'n_points': len(events),
        'step_min': steps_sorted[0] if steps_sorted else None,
        'step_max': steps_sorted[-1] if steps_sorted else None,
    })

all_steps = sorted(all_steps)

# write wide csv
wide_path = f"{outdir}/tb_scalars_wide.csv"
with open(wide_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step'] + tags)
    for s in all_steps:
        row = [s]
        for t in tags:
            v = data[t].get(s, '')
            row.append(v)
        w.writerow(row)

# write tag summary
summary_path = f"{outdir}/tag_summary.csv"
with open(summary_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tag', 'n_points', 'step_min', 'step_max'])
    for ti in tag_info:
        w.writerow([ti['tag'], ti['n_points'], ti['step_min'], ti['step_max']])

print(f"path={path}")
print(f"num_tags={len(tags)}")
print(f"num_steps_seen={len(all_steps)}")
print(f"step_range={all_steps[0]}..{all_steps[-1]}")
print(f"wrote {wide_path}")
print(f"wrote {summary_path}")
print("TAGS:")
for t in tags:
    print(" ", t)
