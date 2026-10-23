"""Sharp Softmax analysis: prompt-level (seed-averaged) CLAP by beta and category with 10,000-resample prompt-level
bootstrap CIs (fixed seed), paired deltas vs beta=1, attention-entropy response, comparison with the Diagnostic V1
Entmax conditions, pre-registered hypothesis rules, figures. Reuses the Diagnostic V1 bootstrap helpers."""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_diagnostic_v1 import boot_ci, summarize, BOOT_N, BOOT_SEED, CATS, CAT_SHORT

ROOT = "outputs/sharp_softmax_v1"; DROOT = "outputs/diagnostic_v1"; BETAS = [1.0, 1.25, 1.5, 2.0]; LAMBDAS = [0.0, 0.5, 1.0, 2.0]; LOG77 = math.log(77)


def load(path, cond, col):
    rows = list(csv.DictReader(open(path))); d, cat = {}, {}
    for r in rows:
        if r.get(col, "") in ("", None): continue
        d.setdefault(r["prompt_id"], {}).setdefault(float(r[cond]), {})[int(r["seed"])] = float(r[col]); cat[r["prompt_id"]] = r["category"]
    return d, cat


def prompt_level(d, conds):
    pl = {p: {c: float(np.mean(list(d[p][c].values()))) for c in conds if c in d[p]} for p in d}
    base = conds[0]
    pdl = {p: {c: float(np.mean([d[p][c][s] - d[p][base][s] for s in d[p][c]])) for c in conds if c in d[p]} for p in d}
    return pl, pdl


def category_stats(pl, pdl, cat, conds, rng):
    out = {}
    for c in CATS:
        ps = sorted(p for p in pl if cat[p] == c)
        out[c] = {"prompts": ps, "abs": {str(k): summarize([pl[p][k] for p in ps], rng) for k in conds},
                  "delta": {str(k): summarize([pdl[p][k] for p in ps], rng) for k in conds[1:]}}
    return out


def contrast(stats_a, stats_b, pdl, conds, rng):
    res = {}
    for k in conds[1:]:
        a = np.array([pdl[p][k] for p in stats_a["prompts"]]); b = np.array([pdl[p][k] for p in stats_b["prompts"]])
        ia = rng.integers(0, len(a), size=(BOOT_N, len(a))); ib = rng.integers(0, len(b), size=(BOOT_N, len(b))); diff = a[ia].mean(1) - b[ib].mean(1)
        res[str(k)] = {"mean_diff": float(a.mean() - b.mean()), "ci95_boot": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    return res


def main():
    rng = np.random.default_rng(BOOT_SEED)
    # ---- Sharp Softmax CLAP ----
    d, cat = load(f"{ROOT}/clap_scores.csv", "beta", "clap_cosine"); pl, pdl = prompt_level(d, BETAS); S = category_stats(pl, pdl, cat, BETAS, rng)
    fg, _ = load(f"{ROOT}/clap_scores.csv", "beta", "fg_event_clap"); bg, _ = load(f"{ROOT}/clap_scores.csv", "beta", "bg_event_clap")
    fpl, fpdl = prompt_level(fg, BETAS); bpl, bpdl = prompt_level(bg, BETAS)
    catD = {p: "DISCRETE_DIFFUSE" for p in fg}
    SD = {"fg": category_stats(fpl, fpdl, catD, BETAS, rng)["DISCRETE_DIFFUSE"], "bg": category_stats(bpl, bpdl, catD, BETAS, rng)["DISCRETE_DIFFUSE"]}
    # ---- entropy (sample level -> prompt level) ----
    ent_rows = list(csv.DictReader(open(f"{ROOT}/attention_entropy.csv")))
    ent = {}
    for r in ent_rows: ent.setdefault(r["prompt_id"], {}).setdefault(float(r["beta"]), {})[int(r["seed"])] = {k: float(r[k]) for k in ["mean_attention_entropy", "normalized_attention_entropy", "effective_support"]}
    ent_pl = {p: {b: {k: float(np.mean([ent[p][b][s][k] for s in ent[p][b]])) for k in ["mean_attention_entropy", "normalized_attention_entropy", "effective_support"]} for b in BETAS} for p in ent}
    ent_by_beta = {str(b): {k: float(np.mean([ent_pl[p][b][k] for p in ent_pl])) for k in ["mean_attention_entropy", "normalized_attention_entropy", "effective_support"]} for b in BETAS}
    ent_by_cat = {c: {str(b): {k: float(np.mean([ent_pl[p][b][k] for p in ent_pl if cat[p] == c])) for k in ["normalized_attention_entropy", "effective_support"]} for b in BETAS} for c in CATS}
    # counterfactual probes on the dense (beta=1) trajectory, from beta_1.00 manifests
    probes = {}
    for r in ent_rows:
        if float(r["beta"]) != 1.0: continue
        for k, v in r.items():
            if k.startswith("probe_") and v not in ("", None): probes.setdefault(k, []).append(float(v))
    probe_means = {k: float(np.mean(v)) for k, v in probes.items()}
    # ---- Entmax (Diagnostic V1) for comparison ----
    dd, dcat = load(f"{DROOT}/clap_scores.csv", "lambda", "clap_cosine"); dpl, dpdl = prompt_level(dd, LAMBDAS); DS = category_stats(dpl, dpdl, dcat, LAMBDAS, rng)
    dfg, _ = load(f"{DROOT}/clap_scores.csv", "lambda", "fg_event_clap"); dbg, _ = load(f"{DROOT}/clap_scores.csv", "lambda", "bg_event_clap")
    dfpl, dfpdl = prompt_level(dfg, LAMBDAS); dbpl, dbpdl = prompt_level(dbg, LAMBDAS)
    DSD = {"fg": category_stats(dfpl, dfpdl, catD, LAMBDAS, rng)["DISCRETE_DIFFUSE"], "bg": category_stats(dbpl, dbpdl, catD, LAMBDAS, rng)["DISCRETE_DIFFUSE"]}
    dgs = json.load(open(f"{DROOT}/generation_summary.json"))["runs"]
    supp = {str(l): float(np.mean([r["mean_entmax_support_cond"] for r in dgs.values() if r["lambda"] == l and r["mean_entmax_support_cond"] is not None])) for l in LAMBDAS[1:]}
    base_equal = all(abs(pl[p][1.0] - dpl[p][0.0]) < 1e-9 for p in pl)
    # ---- hypotheses (pre-registered rules, applied to CLAP) ----
    A, B, C, D = (S[c]["delta"] for c in CATS)
    H = {}
    # NOTE: keys below are the category comparisons requested in the task's section 21 (A tolerance, B vs A, C vs A, D asymmetry);
    # the task's hypotheses H1-H4 (soft concentration / hard sparsity / concentration / acoustic structure) are judged in the report
    # from these plus the entropy comparison; see "task_hypothesis_status".
    H["A_tolerance"] = {"question": "Category A: beta 1.25/1.5/2 vs 1", "A_delta": {b: {"mean": A[b]["mean"], "ci": A[b]["ci95_boot"]} for b in ["1.25", "1.5", "2.0"]},
               "status": ("supported" if A["1.25"]["mean"] > 0 and A["1.5"]["mean"] > 0 else "partially supported" if (A["1.25"]["mean"] > 0 or A["1.5"]["mean"] > 0) else "not supported")}
    BA = contrast(S["DIFFUSE_SINGLE"], S["DISCRETE_SINGLE"], pdl, BETAS, rng); CA = contrast(S["DISCRETE_DISCRETE"], S["DISCRETE_SINGLE"], pdl, BETAS, rng); DC = contrast(S["DISCRETE_DIFFUSE"], S["DISCRETE_DISCRETE"], pdl, BETAS, rng)
    b_lower = {b: B[b]["mean"] < A[b]["mean"] for b in ["1.25", "1.5", "2.0"]}; c_lower = {b: C[b]["mean"] < A[b]["mean"] for b in ["1.25", "1.5", "2.0"]}
    H["B_vs_A"] = {"question": "Category B vs A under sharpening", "B_delta": {b: B[b]["mean"] for b in b_lower}, "A_delta": {b: A[b]["mean"] for b in b_lower}, "B_minus_A": BA, "B_lower_than_A": b_lower,
               "status": "supported" if all(b_lower.values()) else "partially supported" if any(b_lower.values()) else "not supported"}
    H["C_vs_A"] = {"question": "Category C vs A under sharpening", "C_delta": {b: C[b]["mean"] for b in c_lower}, "C_minus_A": CA, "C_lower_than_A": c_lower,
               "status": "supported" if all(c_lower.values()) else "partially supported" if any(c_lower.values()) else "not supported"}
    fg_stable = all(abs(SD["fg"]["delta"][b]["mean"]) <= 0.02 for b in ["1.25", "1.5"]); bg_decl = all(SD["bg"]["delta"][b]["mean"] < 0 for b in ["1.25", "1.5", "2.0"]) and SD["bg"]["delta"]["2.0"]["ci95_boot"][1] < 0
    H["D_asymmetry"] = {"question": "Category D: full-prompt vs fg/bg phrase CLAP under sharpening vs Entmax", "full_delta": {b: D[b]["mean"] for b in ["1.25", "1.5", "2.0"]}, "D_minus_C": DC,
               "fg_delta": {b: SD["fg"]["delta"][b]["mean"] for b in ["1.25", "1.5", "2.0"]}, "bg_delta": {b: {"mean": SD["bg"]["delta"][b]["mean"], "ci": SD["bg"]["delta"][b]["ci95_boot"]} for b in ["1.25", "1.5", "2.0"]},
               "entmax_fg_delta": {l: DSD["fg"]["delta"][l]["mean"] for l in ["0.5", "1.0", "2.0"]}, "entmax_bg_delta": {l: DSD["bg"]["delta"][l]["mean"] for l in ["0.5", "1.0", "2.0"]},
               "fg_stable_up_to_1.5": fg_stable, "bg_declines_all_beta": bg_decl,
               "status": "same asymmetry as Entmax (fg stable, bg declines)" if (fg_stable and bg_decl) else ("partial asymmetry" if bg_decl else "asymmetry not reproduced")}
    task_status = {
        "H1_soft_concentration": {"status": "supported", "evidence": "every sharp-softmax category delta smaller in magnitude than Entmax lambda=1/2; A flat through beta=1.5; no beta improves mean CLAP"},
        "H2_hard_sparsity": {"status": "not supported as necessary cause; insufficient evidence as additional cause", "evidence": "fg-stable/bg-declining asymmetry and category ordering reproduced without any zero weight; beta=2 vs lambda=1 gap confounded with concentration difference"},
        "H3_concentration": {"status": "supported", "evidence": "CLAP (B,C,D), D background-phrase similarity (CIs excl. 0) and overall mean decline monotonically with decreasing entropy; Entmax lambda=1 continues the trend"},
        "H4_acoustic_structure": {"status": "partially supported", "evidence": "A flat to beta=1.5 while B,C,D decline, C most; B-A and C-A contrast CIs include 0; D not worse than C"},
        "mechanistic_outcome": "B (degrades as entropy decreases) with element of D (no beta>1 improves); not A; C not supported at category level"}
    out = {"bootstrap": {"n_resamples": BOOT_N, "seed": BOOT_SEED, "unit": "prompt (seed-averaged)"}, "task_hypothesis_status": task_status, "beta1_equals_diag_lambda0_clap": base_equal, "sharp": {"category": S, "prompt_level": pl, "prompt_delta": pdl, "D_fg": SD["fg"], "D_bg": SD["bg"]},
           "entropy": {"by_beta": ent_by_beta, "by_category": ent_by_cat, "prompt_level": {p: {str(b): v for b, v in ent_pl[p].items()} for p in ent_pl}, "probes_on_dense_trajectory": probe_means},
           "entmax_diag": {"category": DS, "prompt_level": dpl, "prompt_delta": dpdl, "D_fg": DSD["fg"], "D_bg": DSD["bg"], "exact_support_cond_by_lambda": supp}, "hypotheses": H}
    json.dump(out, open(f"{ROOT}/hypothesis_summary.json", "w"), indent=2)
    # ---- csv outputs ----
    with open(f"{ROOT}/category_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "category", "beta", "n_prompts", "mean_abs", "std_abs", "median_abs", "ci95_abs_lo", "ci95_abs_hi", "mean_delta", "std_delta", "median_delta", "ci95_delta_lo", "ci95_delta_hi", "mean_H_norm", "mean_effective_support"])
        for c in CATS:
            for b in BETAS:
                a = S[c]["abs"][str(b)]; dl = S[c]["delta"].get(str(b)); e = ent_by_cat[c][str(b)]
                w.writerow(["CLAP", c, b, a["n"], a["mean"], a["std"], a["median"], *a["ci95_boot"]] + ([dl["mean"], dl["std"], dl["median"], *dl["ci95_boot"]] if dl else ["", "", "", "", ""]) + [e["normalized_attention_entropy"], e["effective_support"]])
        for lab, st in SD.items():
            for b in BETAS:
                a = st["abs"][str(b)]; dl = st["delta"].get(str(b))
                w.writerow([f"CLAP_{lab}_event_exploratory", "DISCRETE_DIFFUSE", b, a["n"], a["mean"], a["std"], a["median"], *a["ci95_boot"]] + ([dl["mean"], dl["std"], dl["median"], *dl["ci95_boot"]] if dl else ["", "", "", "", ""]) + ["", ""])
    with open(f"{ROOT}/comparison_with_entmax.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "control_parameter", "value", "category", "mean_clap", "mean_delta_vs_dense", "ci95_delta_lo", "ci95_delta_hi", "concentration_stat", "concentration_value", "concentration_source"])
        for c in CATS:
            for b in BETAS:
                a = S[c]["abs"][str(b)]; dl = S[c]["delta"].get(str(b)); e = ent_by_cat[c][str(b)]
                w.writerow(["sharp_softmax", "beta", b, c, a["mean"], dl["mean"] if dl else 0.0, dl["ci95_boot"][0] if dl else 0.0, dl["ci95_boot"][1] if dl else 0.0, "softmax_effective_support_exp(H)", e["effective_support"], "measured on own trajectory"])
            for l in LAMBDAS:
                a = DS[c]["abs"][str(l)]; dl = DS[c]["delta"].get(str(l))
                w.writerow(["entmax_mix", "lambda", l, c, a["mean"], dl["mean"] if dl else 0.0, dl["ci95_boot"][0] if dl else 0.0, dl["ci95_boot"][1] if dl else 0.0, "entmax_exact_support (W_sparse)", supp.get(str(l), ""), "measured on own trajectory (Diagnostic V1)"])
    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def errs(st, kind, keys): return np.array([(st[kind][k]["mean"] - st[kind][k]["ci95_boot"][0], st[kind][k]["ci95_boot"][1] - st[kind][k]["mean"]) for k in keys]).T
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in CATS: ax.errorbar(BETAS, [S[c]["abs"][str(b)]["mean"] for b in BETAS], yerr=errs(S[c], "abs", [str(b) for b in BETAS]), marker="o", capsize=3, label=f"{CAT_SHORT[c]} {c}")
    ax.set_xlabel("beta (Sharp Softmax inverse temperature)"); ax.set_ylabel("mean CLAP cosine (prompt-level, 95% boot CI)"); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_clap_by_category.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in CATS: ax.errorbar(BETAS, [0.0] + [S[c]["delta"][str(b)]["mean"] for b in BETAS[1:]], yerr=np.concatenate([np.zeros((2, 1)), errs(S[c], "delta", [str(b) for b in BETAS[1:]])], axis=1), marker="o", capsize=3, label=f"{CAT_SHORT[c]} {c}")
    ax.axhline(0, color="gray", lw=.8); ax.set_xlabel("beta"); ax.set_ylabel("Delta CLAP vs beta=1 (prompt-level, 95% boot CI)"); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_delta_clap.png", dpi=120); plt.close(fig)
    for key, fn, yl in [("normalized_attention_entropy", "sharp_entropy_vs_beta.png", "normalized attention entropy H/log(77), cond. half"), ("effective_support", "sharp_effective_support_vs_beta.png", "Softmax effective support exp(H), cond. half")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in CATS: ax.plot(BETAS, [ent_by_cat[c][str(b)][key] for b in BETAS], marker="o", label=f"{CAT_SHORT[c]} {c}")
        ax.plot(BETAS, [ent_by_beta[str(b)][key] for b in BETAS], marker="s", color="k", lw=2, label="all prompts")
        if key == "effective_support":
            ms = float(np.mean([supp[str(l)] for l in LAMBDAS[1:]]))
            ax.axhline(ms, ls=":", color="gray", label=f"Entmax exact support on own trajectories, mean over λ=0.5/1/2 ({ms:.2f})")
            ax.axhline(probe_means["probe_entmax15_effective_support"], ls="--", color="gray", label=f"entmax15 effective support exp(H), probe on dense trajectory ({probe_means['probe_entmax15_effective_support']:.2f})")
        ax.set_xlabel("beta"); ax.set_ylabel(yl); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{ROOT}/{fn}", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, st, col in [("foreground phrase (Sharp Softmax)", SD["fg"], "C0"), ("background phrase (Sharp Softmax)", SD["bg"], "C1")]:
        ax.errorbar(BETAS, [st["abs"][str(b)]["mean"] for b in BETAS], yerr=errs(st, "abs", [str(b) for b in BETAS]), marker="o", capsize=3, color=col, label=lab)
    ax.set_xlabel("beta"); ax.set_ylabel("CLAP cosine to event phrase (exploratory)"); ax.set_title("Category D, event-phrase CLAP vs beta (not an event detector)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{ROOT}/category_d_event_phrase_sharp.png", dpi=120); plt.close(fig)
    # sharp vs entmax: mean over all 24 prompts per condition, ordered by family; separate x axes labelled
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    allS = {str(b): float(np.mean([pl[p][b] for p in pl])) for b in BETAS}; allD = {str(l): float(np.mean([dpl[p][l] for p in dpl])) for l in LAMBDAS}
    for c in CATS: axes[0].plot(BETAS, [S[c]["abs"][str(b)]["mean"] for b in BETAS], marker="o", alpha=.6, label=f"{CAT_SHORT[c]}")
    axes[0].plot(BETAS, [allS[str(b)] for b in BETAS], marker="s", color="k", lw=2, label="all 24 prompts"); axes[0].set_xlabel("beta: Softmax(beta*logits)  [beta is not lambda]"); axes[0].set_ylabel("mean CLAP cosine"); axes[0].set_title("Sharp Softmax", fontsize=10); axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
    for c in CATS: axes[1].plot(LAMBDAS, [DS[c]["abs"][str(l)]["mean"] for l in LAMBDAS], marker="o", alpha=.6, label=f"{CAT_SHORT[c]}")
    axes[1].plot(LAMBDAS, [allD[str(l)] for l in LAMBDAS], marker="s", color="k", lw=2, label="all 24 prompts"); axes[1].set_xlabel("lambda: W_dense + lambda(W_sparse - W_dense)  [lambda is not beta]"); axes[1].set_title("Entmax / PLADIS (Diagnostic V1)", fontsize=10); axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
    fig.suptitle("Same 24 prompts x 3 seeds, same initial latents; the two x-axes are different control parameters and are not concentration-matched", fontsize=8); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_vs_entmax_clap.png", dpi=120); plt.close(fig)
    # concentration vs CLAP (sharp softmax prompt-level points)
    fig, ax = plt.subplots(figsize=(6, 4.2)); mk = {1.0: "o", 1.25: "s", 1.5: "^", 2.0: "D"}
    for ci, c in enumerate(CATS):
        for b in BETAS:
            xs = [ent_pl[p][b]["normalized_attention_entropy"] for p in pl if cat[p] == c]; ys = [pl[p][b] for p in pl if cat[p] == c]
            ax.scatter(xs, ys, marker=mk[b], color=f"C{ci}", alpha=.75, s=28, label=f"{CAT_SHORT[c]} β={b:g}" if True else None)
    ax.set_xlabel("normalized attention entropy H/log(77) (cond. half, own trajectory)"); ax.set_ylabel("prompt-level CLAP cosine"); ax.set_title("Sharp Softmax conditions: concentration vs CLAP (color=category, marker=beta)", fontsize=8); ax.grid(alpha=.3); ax.legend(fontsize=5, ncol=4); fig.tight_layout(); fig.savefig(f"{ROOT}/concentration_vs_clap.png", dpi=120); plt.close(fig)
    # ---- console ----
    print("beta=1 CLAP equals Diagnostic lambda=0:", base_equal)
    print("entropy by beta:", {b: {k: round(v, 3) for k, v in e.items()} for b, e in ent_by_beta.items()}); print("probes on dense trajectory:", {k: round(v, 3) for k, v in probe_means.items()})
    for c in CATS: print(f"{CAT_SHORT[c]} {c:18s} abs", {b: round(S[c]['abs'][str(b)]['mean'], 4) for b in BETAS}, "delta", {b: (round(S[c]['delta'][str(b)]['mean'], 4), [round(x, 4) for x in S[c]['delta'][str(b)]['ci95_boot']]) for b in BETAS[1:]}, "Hn", {b: round(ent_by_cat[c][str(b)]['normalized_attention_entropy'], 3) for b in BETAS})
    print("D fg:", {b: round(SD['fg']['abs'][str(b)]['mean'], 4) for b in BETAS}, "D bg:", {b: round(SD['bg']['abs'][str(b)]['mean'], 4) for b in BETAS})
    for k, v in H.items(): print(k, v["status"])


if __name__ == "__main__":
    main()
