"""Diagnostic v1 analysis (metric-generic). Unit of analysis = prompt (seed-averaged). Paired deltas vs lambda=0,
category summaries (mean/std/median/95% prompt-level bootstrap CI, 10,000 resamples, fixed seed), pre-declared
hypothesis decision rules, figures. Metrics: CLAP (clap_scores.csv) and, if present, ELSA (elsa_scores.csv)."""
import csv, json, os, sys
import numpy as np
ROOT = "outputs/diagnostic_v1"; LAMBDAS = [0.0, 0.5, 1.0, 2.0]; CATS = ["DISCRETE_SINGLE", "DIFFUSE_SINGLE", "DISCRETE_DISCRETE", "DISCRETE_DIFFUSE"]
CAT_SHORT = {"DISCRETE_SINGLE": "A", "DIFFUSE_SINGLE": "B", "DISCRETE_DISCRETE": "C", "DISCRETE_DIFFUSE": "D"}
BOOT_N, BOOT_SEED = 10000, 20261022


def load_metric(path, col):
    rows = list(csv.DictReader(open(path)))
    d = {}
    for r in rows:
        if r.get(col, "") == "": continue
        d.setdefault(r["prompt_id"], {}).setdefault(float(r["lambda"]), {})[int(r["seed"])] = float(r[col])
    cat = {r["prompt_id"]: r["category"] for r in rows}
    return d, cat


def boot_ci(x, rng):
    x = np.asarray(x, float); n = len(x)
    idx = rng.integers(0, n, size=(BOOT_N, n)); means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(x, rng):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "median": float(np.median(x)), "ci95_boot": boot_ci(x, rng), "values": [float(v) for v in x]}


def analyze(name, d, cat):
    rng = np.random.default_rng(BOOT_SEED)
    prompt_level = {p: {l: float(np.mean(list(d[p][l].values()))) for l in LAMBDAS if l in d[p]} for p in d}
    seed_delta = {p: {l: {s: d[p][l][s] - d[p][0.0][s] for s in d[p][l]} for l in LAMBDAS if l in d[p]} for p in d}
    prompt_delta = {p: {l: float(np.mean(list(seed_delta[p][l].values()))) for l in seed_delta[p]} for p in d}
    cats = [c for c in CATS if any(cat[p] == c for p in d)]          # exploratory per-event metrics exist only for some categories
    out = {"metric": name, "bootstrap": {"n_resamples": BOOT_N, "seed": BOOT_SEED, "unit": "prompt (seed-averaged)"}, "prompt_level": prompt_level, "prompt_delta": prompt_delta,
           "seed_level": {p: {str(l): d[p][l] for l in d[p]} for p in d}, "category": {}, "categories_present": cats}
    for c in cats:
        ps = sorted(p for p in d if cat[p] == c)
        out["category"][c] = {"prompts": ps, "abs": {str(l): summarize([prompt_level[p][l] for p in ps], rng) for l in LAMBDAS},
                              "delta": {str(l): summarize([prompt_delta[p][l] for p in ps], rng) for l in LAMBDAS if l != 0.0}}
    # between-category contrasts of prompt-level deltas (difference of means, bootstrap over prompts within each category independently)
    def contrast(c1, c2):
        res = {}
        for l in LAMBDAS[1:]:
            a = np.array([prompt_delta[p][l] for p in out["category"][c1]["prompts"]]); b = np.array([prompt_delta[p][l] for p in out["category"][c2]["prompts"]])
            ia = rng.integers(0, len(a), size=(BOOT_N, len(a))); ib = rng.integers(0, len(b), size=(BOOT_N, len(b)))
            diff = a[ia].mean(1) - b[ib].mean(1)
            res[str(l)] = {"mean_diff": float(a.mean() - b.mean()), "ci95_boot": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
        return res
    if cats != CATS:
        return out                                                      # hypotheses/contrasts need all four categories
    out["contrasts"] = {"B_minus_A": contrast("DIFFUSE_SINGLE", "DISCRETE_SINGLE"), "C_minus_A": contrast("DISCRETE_DISCRETE", "DISCRETE_SINGLE"), "D_minus_C": contrast("DISCRETE_DIFFUSE", "DISCRETE_DISCRETE")}
    # ---- pre-declared decision rules (structured interpretation aids only) ----
    A, B, C, D = (out["category"][c]["delta"] for c in CATS)
    h = {}
    a05 = A["0.5"]; h["H1"] = {"rule": "A lambda=0.5 mean delta > 0; strongly if 95% CI excludes 0", "mean_delta_A_0.5": a05["mean"], "ci": a05["ci95_boot"],
                               "status": "strongly supported" if a05["mean"] > 0 and a05["ci95_boot"][0] > 0 else ("weakly supported" if a05["mean"] > 0 else "not supported"),
                               "secondary": {l: {"mean": A[l]["mean"], "ci": A[l]["ci95_boot"]} for l in ["1.0", "2.0"]}}
    b_mono = B["0.5"]["mean"] >= B["1.0"]["mean"] >= B["2.0"]["mean"]; b_lower = all(B[l]["mean"] < A[l]["mean"] for l in ["0.5", "1.0", "2.0"])
    h["H2"] = {"rule": "B more negative with stronger lambda AND B delta lower than A", "B_delta": {l: B[l]["mean"] for l in ["0.5", "1.0", "2.0"]}, "A_delta": {l: A[l]["mean"] for l in ["0.5", "1.0", "2.0"]},
               "B_minus_A": out["contrasts"]["B_minus_A"], "monotone_more_negative": bool(b_mono), "B_lower_than_A_all_lambda": bool(b_lower),
               "status": "supported" if (b_mono and b_lower) else ("partially supported" if (b_mono or b_lower) else "not supported")}
    c_lower = [C[l]["mean"] < A[l]["mean"] for l in ["0.5", "1.0", "2.0"]]
    h["H3"] = {"rule": "C consistently loses more alignment than A as lambda increases", "C_delta": {l: C[l]["mean"] for l in ["0.5", "1.0", "2.0"]}, "C_minus_A": out["contrasts"]["C_minus_A"],
               "C_lower_than_A_per_lambda": dict(zip(["0.5", "1.0", "2.0"], map(bool, c_lower))), "status": "supported" if all(c_lower) else ("partially supported" if any(c_lower) else "not supported")}
    d_lower = {l: D[l]["mean"] < C[l]["mean"] for l in ["0.5", "1.0", "2.0"]}
    h["H4"] = {"rule": "D degrades more strongly than C, particularly at lambda=1 or 2", "D_delta": {l: D[l]["mean"] for l in ["0.5", "1.0", "2.0"]}, "D_minus_C": out["contrasts"]["D_minus_C"],
               "D_lower_than_C_per_lambda": d_lower, "status": "supported" if (d_lower["1.0"] and d_lower["2.0"]) else ("partially supported" if (d_lower["1.0"] or d_lower["2.0"]) else "not supported")}
    out["hypotheses"] = h
    # ---- pattern classification ----
    sign = {c: {l: out["category"][c]["delta"][l]["mean"] for l in ["0.5", "1.0", "2.0"]} for c in CATS}
    improves = {c: all(v > 0 for v in sign[c].values()) for c in CATS}; degrades = {c: all(v < 0 for v in sign[c].values()) for c in CATS}
    if improves["DISCRETE_SINGLE"] and degrades["DIFFUSE_SINGLE"] and degrades["DISCRETE_DISCRETE"] and degrades["DISCRETE_DIFFUSE"] and sign["DISCRETE_DIFFUSE"]["2.0"] < sign["DISCRETE_DISCRETE"]["2.0"]: pat = 1
    elif all(degrades.values()): pat = 2
    elif all(improves.values()): pat = 3
    else: pat = 4
    out["pattern"] = {"pattern": pat, "category_signs": sign, "all_lambda_improves": improves, "all_lambda_degrades": degrades}
    return out


def main():
    results = {}
    clap_d, cat = load_metric(f"{ROOT}/clap_scores.csv", "clap_cosine"); results["CLAP"] = analyze("CLAP", clap_d, cat)
    for extra in [("fg_event_clap", "CLAP_fg_event_exploratory"), ("bg_event_clap", "CLAP_bg_event_exploratory")]:
        dd, cc = load_metric(f"{ROOT}/clap_scores.csv", extra[0])
        if dd: results[extra[1]] = analyze(extra[1], dd, cc)
    elsa_path = f"{ROOT}/elsa_scores.csv"
    if os.path.exists(elsa_path):
        ed, ec = load_metric(elsa_path, "elsa_score"); results["ELSA"] = analyze("ELSA", ed, ec)
    json.dump(results, open(f"{ROOT}/hypothesis_summary.json", "w"), indent=2)
    # category_summary.csv
    with open(f"{ROOT}/category_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "category", "lambda", "n_prompts", "mean_abs", "std_abs", "median_abs", "ci95_abs_lo", "ci95_abs_hi", "mean_delta", "std_delta", "median_delta", "ci95_delta_lo", "ci95_delta_hi"])
        for m, R in results.items():
            for c in R["categories_present"]:
                for l in LAMBDAS:
                    a = R["category"][c]["abs"][str(l)]; dl = R["category"][c]["delta"].get(str(l))
                    w.writerow([m, c, l, a["n"], a["mean"], a["std"], a["median"], *a["ci95_boot"]] + ([dl["mean"], dl["std"], dl["median"], *dl["ci95_boot"]] if dl else ["", "", "", "", ""]))
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for m, R in results.items():
        tag = m.lower()
        if R["categories_present"] != CATS: continue                    # category figures only for full-design metrics
        for kind in ["abs", "delta"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for c in R["categories_present"]:
                ls = LAMBDAS
                ys = [R["category"][c][kind][str(l)]["mean"] if (kind == "abs" or l != 0.0) else 0.0 for l in ls]
                err = [(R["category"][c][kind][str(l)]["mean"] - R["category"][c][kind][str(l)]["ci95_boot"][0], R["category"][c][kind][str(l)]["ci95_boot"][1] - R["category"][c][kind][str(l)]["mean"]) if (kind == "abs" or l != 0.0) else (0, 0) for l in ls]
                ax.errorbar(ls, ys, yerr=np.array(err).T, marker="o", capsize=3, label=f"{CAT_SHORT[c]} {c}")
            if kind == "delta": ax.axhline(0, color="gray", lw=.8)
            ax.set_xlabel("lambda"); ax.set_ylabel(f"{'mean ' + m if kind == 'abs' else 'Delta ' + m + ' vs lambda=0'} (prompt-level, 95% boot CI)"); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
            fig.savefig(f"{ROOT}/diagnostic_{'delta_' if kind == 'delta' else ''}{tag}{'' if kind == 'delta' else '_by_category'}.png", dpi=120); plt.close(fig)
    if "CLAP_fg_event_exploratory" in results:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, lab in [("CLAP_fg_event_exploratory", "foreground event phrase"), ("CLAP_bg_event_exploratory", "background event phrase")]:
            R = results[m]["category"]["DISCRETE_DIFFUSE"]; ys = [R["abs"][str(l)]["mean"] for l in LAMBDAS]
            err = [(R["abs"][str(l)]["mean"] - R["abs"][str(l)]["ci95_boot"][0], R["abs"][str(l)]["ci95_boot"][1] - R["abs"][str(l)]["mean"]) for l in LAMBDAS]
            ax.errorbar(LAMBDAS, ys, yerr=np.array(err).T, marker="o", capsize=3, label=lab)
        ax.set_xlabel("lambda"); ax.set_ylabel("CLAP cosine to event phrase (exploratory)"); ax.set_title("Category D, event-phrase CLAP (not an event detector)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(f"{ROOT}/foreground_background_clap_exploratory.png", dpi=120); plt.close(fig)
    for m, R in results.items():
        print(f"\n== {m} ==  pattern {R.get('pattern', {}).get('pattern', 'n/a')}")
        for c in R["categories_present"]:
            print(f"  {CAT_SHORT[c]} {c:18s} abs:", {l: round(R['category'][c]['abs'][str(l)]['mean'], 4) for l in LAMBDAS}, " delta:", {l: (round(R['category'][c]['delta'][str(l)]['mean'], 4), [round(x, 4) for x in R['category'][c]['delta'][str(l)]['ci95_boot']]) for l in LAMBDAS[1:]})
        for k, v in R.get("hypotheses", {}).items(): print(f"  {k}: {v['status']}")


if __name__ == "__main__":
    main()
