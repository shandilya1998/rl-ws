import csv, sys, math

def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows

def col(rows, tag):
    # return list of (step, value) for rows where tag is non-empty, sorted by step
    out = []
    for row in rows:
        v = row.get(tag, '')
        if v == '' or v is None:
            continue
        out.append((int(row['step']), float(v)))
    out.sort(key=lambda x: x[0])
    return out

def quantiles(series):
    # series: list of (step, value) assumed contiguous per-iteration (step 0..N-1)
    n = len(series)
    idxs = [0, int(round(0.25*(n-1))), int(round(0.5*(n-1))), int(round(0.75*(n-1))), n-1]
    return [(series[i][0], series[i][1]) for i in idxs]

def sign_history(series):
    # find contiguous runs of sign (neg/pos/zero) and report step ranges where negative
    neg_ranges = []
    cur_start = None
    prev_neg = False
    for step, v in series:
        isneg = v < 0
        if isneg and not prev_neg:
            cur_start = step
        if not isneg and prev_neg:
            neg_ranges.append((cur_start, prev_step))
        prev_neg = isneg
        prev_step = step
    if prev_neg:
        neg_ranges.append((cur_start, prev_step))
    return neg_ranges

def stats(series):
    vals = [v for _, v in series]
    return min(vals), max(vals), sum(vals)/len(vals)

def main():
    label = sys.argv[1]
    path = sys.argv[2]
    rows = load(path)
    with open(path) as f:
        header = f.readline().strip().split(',')
    tags = header[1:]

    reward_tags = sorted(t for t in tags if t.startswith('Episode_Reward/'))
    print(f"=== {label} : {path} ===")
    print(f"n_rows_total={len(rows)}")

    print("\n--- Episode_Reward tags: quantile table (raw TB value = rate, units per second) ---")
    print(f"{'tag':40s} {'it@0%':>8s} {'v@0%':>12s} {'it@25%':>8s} {'v@25%':>12s} {'it@50%':>8s} {'v@50%':>12s} {'it@75%':>8s} {'v@75%':>12s} {'it@100%':>8s} {'v@100%':>12s} {'min':>12s} {'max':>12s} {'mean':>12s}")
    reward_series = {}
    for t in reward_tags:
        s = col(rows, t)
        reward_series[t] = s
        q = quantiles(s)
        mn, mx, mean = stats(s)
        line = f"{t:40s}"
        for st, v in q:
            line += f" {st:8d} {v:12.6f}"
        line += f" {mn:12.6f} {mx:12.6f} {mean:12.6f}"
        print(line)

    print("\n--- Sign history (step ranges where value < 0) ---")
    for t in reward_tags:
        s = reward_series[t]
        negs = sign_history(s)
        mn, mx, mean = stats(s)
        if not negs:
            verdict = "NEVER NEGATIVE"
        elif len(negs) == 1 and negs[0][0] == s[0][0] and negs[0][1] == s[-1][0]:
            verdict = "ALWAYS NEGATIVE"
        else:
            verdict = f"NEGATIVE for {len(negs)} range(s)"
        print(f"{t:40s} min={mn:12.6f} max={mx:12.6f} {verdict}")
        if negs:
            for a, b in negs[:20]:
                print(f"    neg range: step {a} .. {b}")
            if len(negs) > 20:
                print(f"    ... and {len(negs)-20} more ranges")

    # per-episode contribution: value * 20.0 (episode_length_s)
    EPL_S = 20.0
    print("\n--- Per episode reward contribution (raw_TB_value * 20.0 s) at 25pct and 100pct ---")
    n = len(reward_series[reward_tags[0]])
    idx25 = int(round(0.25*(n-1)))
    idx100 = n-1
    budget25 = []
    budget100 = []
    for t in reward_tags:
        s = reward_series[t]
        step25, v25 = s[idx25]
        step100, v100 = s[idx100]
        budget25.append((t, step25, v25*EPL_S))
        budget100.append((t, step100, v100*EPL_S))
    print("\n25pct budget ranked by |contribution| desc:")
    for t, st, c in sorted(budget25, key=lambda x: -abs(x[2])):
        print(f"  {t:40s} it={st:7d} per_episode_contribution={c:12.4f}")
    print("\n100pct budget ranked by |contribution| desc:")
    for t, st, c in sorted(budget100, key=lambda x: -abs(x[2])):
        print(f"  {t:40s} it={st:7d} per_episode_contribution={c:12.4f}")
    tot25 = sum(c for _, _, c in budget25)
    tot100 = sum(c for _, _, c in budget100)
    print(f"\nsum of per-episode contributions at 25pct = {tot25:.4f}")
    print(f"sum of per-episode contributions at 100pct = {tot100:.4f}")

    # Train / Loss / Policy tags
    other_tags = ['Train/mean_reward', 'Train/mean_episode_length', 'Loss/value_function',
                  'Loss/surrogate', 'Loss/entropy', 'Policy/mean_noise_std', 'Loss/learning_rate']
    print("\n--- Train/Loss/Policy trajectories: quantile table ---")
    print(f"{'tag':30s} {'it@0%':>8s} {'v@0%':>12s} {'it@25%':>8s} {'v@25%':>12s} {'it@50%':>8s} {'v@50%':>12s} {'it@75%':>8s} {'v@75%':>12s} {'it@100%':>8s} {'v@100%':>12s} {'min':>12s} {'argmin_it':>10s} {'max':>12s} {'argmax_it':>10s}")
    for t in other_tags:
        if t not in tags:
            print(f"{t:30s} NOT PRESENT")
            continue
        s = col(rows, t)
        if not s:
            print(f"{t:30s} EMPTY")
            continue
        q = quantiles(s)
        vals = [v for _, v in s]
        mn = min(vals); mx = max(vals)
        argmin_it = [st for st, v in s if v == mn][0]
        argmax_it = [st for st, v in s if v == mx][0]
        line = f"{t:30s}"
        for st, v in q:
            line += f" {st:8d} {v:12.6f}"
        line += f" {mn:12.6f} {argmin_it:10d} {mx:12.6f} {argmax_it:10d}"
        print(line)

    # decline-after-peak check for Train/mean_reward
    t = 'Train/mean_reward'
    if t in tags:
        s = col(rows, t)
        vals = [v for _, v in s]
        peak_idx = vals.index(max(vals))
        peak_step, peak_val = s[peak_idx]
        final_step, final_val = s[-1]
        print(f"\nTrain/mean_reward peak={peak_val:.4f} at step {peak_step}, final={final_val:.4f} at step {final_step}, decline_from_peak={peak_val-final_val:.4f} ({100*(peak_val-final_val)/abs(peak_val) if peak_val else 0:.2f}% of peak)")

    # Curriculum / Metrics / Episode_Termination
    print("\n--- Curriculum/Metrics/Episode_Termination tags: quantile table ---")
    ct_tags = sorted(t for t in tags if t.startswith('Curriculum/') or t.startswith('Metrics/') or t.startswith('Episode_Termination/'))
    for t in ct_tags:
        s = col(rows, t)
        if not s:
            print(f"{t:45s} EMPTY")
            continue
        q = quantiles(s)
        vals = [v for _, v in s]
        mn = min(vals); mx = max(vals)
        line = f"{t:45s}"
        for st, v in q:
            line += f" it={st:6d} v={v:10.5f} |"
        line += f" min={mn:10.5f} max={mx:10.5f}"
        print(line)
        # detect change points (value changes)
        changes = []
        prev = None
        for st, v in s:
            if prev is not None and v != prev:
                changes.append((st, prev, v))
            prev = v
        if changes:
            print(f"    {len(changes)} change point(s), first 10:")
            for st, pv, nv in changes[:10]:
                print(f"      at it {st}: {pv:.6f} -> {nv:.6f}")
            if len(changes) > 10:
                print(f"      ... and {len(changes)-10} more")

if __name__ == '__main__':
    main()
