"""Partial sharp guidance analysis. Unit = prompt (3 seeds averaged); overall n=24 prompts, category n=6; 10,000 prompt-level
bootstrap resamples, seed 20261023; paired deltas vs lambda_sharp=0 (dense). Metrics: CLAP, AQAScore, mean Event-AQA,
FG/BG Event-AQA (D), AudioBox PQ/CE/CU/PC; attention statistics (entropy for lambda_sharp<=1, signed-weight diagnostics).
Outputs: overall_summary.csv, category_summary.csv, attention_stats.csv, pareto_summary.json, candidate_summary.json, figures."""
import csv, json, math, os
import numpy as np
ROOT = "outputs/sharp_mix_v1"; LS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]; CATS = ["DISCRETE_SINGLE", "DIFFUSE_SINGLE", "DISCRETE_DISCRETE", "DISCRETE_DIFFUSE"]; SHORT = {c: "ABCD"[i] for i, c in enumerate(CATS)}
TYPE = {0.0: "Dense", 0.25: "Interpolation", 0.5: "Interpolation", 0.75: "Interpolation", 1.0: "Full Sharp", 1.25: "Extrapolation", 1.5: "Extrapolation"}
BOOT_N, BOOT_SEED = 10000, 20261023; METRICS = ["clap_cosine", "aqascore_full", "mean_event_aqascore", "fg_event_aqascore", "bg_event_aqascore", "PQ", "CE", "CU", "PC", "fg_event_clap", "bg_event_clap"]


def rd(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def boot_ci(x, rng): x = np.asarray(x, float); i = rng.integers(0, len(x), size=(BOOT_N, len(x))); m = x[i].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def summ(x, rng): x = np.asarray(x, float); return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "median": float(np.median(x)), "ci95": boot_ci(x, rng)}


def load_rows():
    gs = json.load(open(f"{ROOT}/generation_summary.json"))["runs"]
    clap = {r["audio_path"]: r for r in rd(f"{ROOT}/clap_scores.csv")}; aqa = {(r["prompt_id"], int(r["seed"]), float(r["lambda_sharp"])): r for r in rd(f"{ROOT}/aqascore_scores.csv")}
    aes = {(r["prompt_id"], int(r["seed"]), float(r["lambda_sharp"])): r for r in rd(f"{ROOT}/audiobox_scores.csv")}; ev = {}
    for r in rd(f"{ROOT}/event_aqascore_scores.csv"): ev.setdefault((r["prompt_id"], int(r["seed"]), float(r["lambda_sharp"])), []).append(r)
    rows = []
    for k, r in gs.items():
        key = (r["prompt_id"], r["seed"], r["lambda_sharp"]); c = clap.get(r["audio_path"], {}); a = aqa.get(key, {}); s = aes.get(key, {}); e = ev.get(key, []); st = r["stats"]
        rows.append({"prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": r["seed"], "beta": r["beta"], "lambda_sharp": r["lambda_sharp"], "type": TYPE[r["lambda_sharp"]], "audio_path": r["audio_path"], "audio_sha256": r["audio_sha256"],
                     "source": r["source"], "final_latent_norm": r["final_latent_norm"], "clap_cosine": c.get("clap_cosine", ""), "fg_event_clap": c.get("fg_event_clap", ""), "bg_event_clap": c.get("bg_event_clap", ""),
                     "aqascore_full": a.get("aqascore_full", ""), "mean_event_aqascore": (np.mean([float(x["event_aqascore"]) for x in e]) if e else ""),
                     "fg_event_aqascore": next((x["event_aqascore"] for x in e if x["event_role"] == "foreground"), ""), "bg_event_aqascore": next((x["event_aqascore"] for x in e if x["event_role"] == "background"), ""),
                     "PQ": s.get("PQ", ""), "CE": s.get("CE", ""), "CU": s.get("CU", ""), "PC": s.get("PC", ""),
                     "normalized_entropy": st.get("normalized_entropy") if st.get("normalized_entropy") is not None else "", "effective_support": st.get("effective_support") if st.get("effective_support") is not None else "",
                     "neg_fraction": st.get("neg_fraction", ""), "neg_mass": st.get("neg_mass", ""), "min_weight": st.get("min_weight", ""), "endpoint_sharp_normalized_entropy": st.get("endpoint_sharp_normalized_entropy", ""), "entropy_note": st.get("entropy_note", "") or ""})
    with open(f"{ROOT}/attention_stats.csv", "w", newline="") as f:
        cols = ["prompt_id", "category", "seed", "beta", "lambda_sharp", "type", "audio_sha256", "normalized_entropy", "effective_support", "neg_fraction", "neg_mass", "min_weight", "endpoint_sharp_normalized_entropy", "entropy_note", "final_latent_norm"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in rows]
    return rows


def main():
    rows = load_rows(); rng = np.random.default_rng(BOOT_SEED); cat_of = {r["prompt_id"]: r["category"] for r in rows}
    A = {"bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "prompt (3 seeds averaged); overall n=24, category n=6"}, "metrics": {}}
    for m in METRICS + ["neg_fraction", "neg_mass", "normalized_entropy", "effective_support"]:
        d = {}
        for r in rows:
            if r[m] in ("", None): continue
            d.setdefault(r["prompt_id"], {}).setdefault(r["lambda_sharp"], {})[r["seed"]] = float(r[m])
        if not d: continue
        pl = {p: {l: float(np.mean(list(d[p][l].values()))) for l in LS if l in d[p]} for p in d}
        pdl = {p: {l: float(np.mean([d[p][l][s] - d[p][0.0][s] for s in d[p][l]])) for l in LS if l in d[p] and 0.0 in d[p]} for p in d}
        M = {"prompt_level": {p: {str(l): v for l, v in pl[p].items()} for p in pl}, "prompt_delta": {p: {str(l): v for l, v in pdl[p].items()} for p in pdl},
             "overall": {"abs": {str(l): summ([pl[p][l] for p in pl if l in pl[p]], rng) for l in LS if any(l in pl[p] for p in pl)}, "delta": {str(l): summ([pdl[p][l] for p in pdl if l in pdl[p]], rng) for l in LS[1:] if any(l in pdl[p] for p in pdl)}}, "category": {}}
        for c in CATS:
            ps = sorted(p for p in d if cat_of[p] == c)
            if not ps: continue
            M["category"][c] = {"prompts": ps, "abs": {str(l): summ([pl[p][l] for p in ps if l in pl[p]], rng) for l in LS if any(l in pl[p] for p in ps)}, "delta": {str(l): summ([pdl[p][l] for p in ps if l in pdl[p]], rng) for l in LS[1:] if any(l in pdl[p] for p in ps)}}
        A["metrics"][m] = M
    # ---- overall & category summary CSVs ----
    O = A["metrics"]
    with open(f"{ROOT}/overall_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda_sharp", "type", "CLAP", "dCLAP", "dCLAP_ci_lo", "dCLAP_ci_hi", "AQAScore", "dAQA", "dAQA_ci_lo", "dAQA_ci_hi", "mean_event_AQA", "dEventAQA", "PQ", "dPQ", "dPQ_ci_lo", "dPQ_ci_hi", "CE", "CU", "PC", "H_norm", "effective_support", "neg_fraction", "neg_mass"])
        for l in LS:
            k = str(l); g = lambda m, kind="abs": (O[m]["overall"][kind][k]["mean"] if k in O[m]["overall"][kind] else "")
            gc = lambda m: (O[m]["overall"]["delta"][k]["ci95"] if k in O[m]["overall"]["delta"] else ["", ""])
            w.writerow([l, TYPE[l], g("clap_cosine"), g("clap_cosine", "delta") if l else 0, *gc("clap_cosine"), g("aqascore_full"), g("aqascore_full", "delta") if l else 0, *gc("aqascore_full"), g("mean_event_aqascore"), g("mean_event_aqascore", "delta") if l else 0,
                        g("PQ"), g("PQ", "delta") if l else 0, *gc("PQ"), g("CE"), g("CU"), g("PC"), g("normalized_entropy") if l <= 1 else "N/A (signed weights)", g("effective_support") if l <= 1 else "N/A", g("neg_fraction"), g("neg_mass")])
    with open(f"{ROOT}/category_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "category", "lambda_sharp", "type", "n", "mean", "std", "median", "ci_lo", "ci_hi", "mean_delta_vs_dense", "delta_std", "delta_median", "delta_ci_lo", "delta_ci_hi"])
        for m, M in O.items():
            for c in list(M["category"]) + ["ALL"]:
                S = M["category"][c] if c != "ALL" else M["overall"]
                for l in LS:
                    if str(l) not in S["abs"]: continue
                    a = S["abs"][str(l)]; dl = S["delta"].get(str(l))
                    w.writerow([m, c, l, TYPE[l], a["n"], a["mean"], a["std"], a["median"], *a["ci95"]] + ([dl["mean"], dl["std"], dl["median"], *dl["ci95"]] if dl else [0, 0, 0, 0, 0]))
    # ---- Pareto (maximize CLAP, AQAScore, PQ; overall prompt-level means) ----
    pts = {l: (O["clap_cosine"]["overall"]["abs"][str(l)]["mean"], O["aqascore_full"]["overall"]["abs"][str(l)]["mean"], O["PQ"]["overall"]["abs"][str(l)]["mean"]) for l in LS}
    def dominated(a, b): return all(bi >= ai for ai, bi in zip(a, b)) and any(bi > ai for ai, bi in zip(a, b))
    pareto = {str(l): {"type": TYPE[l], "CLAP": pts[l][0], "AQAScore": pts[l][1], "PQ": pts[l][2], "dominated_by": [str(m) for m in LS if m != l and dominated(pts[l], pts[m])]} for l in LS}
    for k in pareto: pareto[k]["non_dominated"] = not pareto[k]["dominated_by"]
    json.dump({"objective": "maximize CLAP, AQAScore, PQ (overall prompt-level means, 24 prompts)", "points": pareto, "note": "non-dominated is a trade-off description, not superiority over Dense"}, open(f"{ROOT}/pareto_summary.json", "w"), indent=2)
    # ---- development-candidate screening (no post-hoc thresholds; report sign + CI of all three deltas) ----
    cand = {}
    for l in [0.25, 0.5, 0.75]:
        k = str(l); c = {m: {"delta": O[m]["overall"]["delta"][k]["mean"], "ci95": O[m]["overall"]["delta"][k]["ci95"]} for m in ["clap_cosine", "aqascore_full", "PQ"]}
        c["all_point_estimates_nonnegative"] = all(v["delta"] >= 0 for v in c.values() if isinstance(v, dict)); c["any_ci_excludes_zero_negative"] = any(v["ci95"][1] < 0 for v in c.values() if isinstance(v, dict)); c["any_ci_excludes_zero_positive"] = any(v["ci95"][0] > 0 for v in c.values() if isinstance(v, dict))
        cand[k] = c
    json.dump({"screened": cand, "rule": "a development-set candidate requires non-negative point estimates on all three primary metrics (CLAP, AQAScore, PQ) with at least one CI excluding 0 on the positive side and none excluding 0 on the negative side; otherwise NO DEVELOPMENT CANDIDATE", "pareto": pareto}, open(f"{ROOT}/candidate_summary.json", "w"), indent=2)
    json.dump(A, open(f"{ROOT}/analysis_summary.json", "w"), indent=2)
    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def errs(S, kind, keys): return np.array([(S[kind][k]["mean"] - S[kind][k]["ci95"][0], S[kind][k]["ci95"][1] - S[kind][k]["mean"]) for k in keys]).T
    for m, fn, lab in [("clap_cosine", "sharp_mix_clap.png", "CLAP cosine"), ("aqascore_full", "sharp_mix_aqascore.png", "AQAScore (full prompt)"), ("PQ", "sharp_mix_pq.png", "AudioBox PQ")]:
        fig, ax = plt.subplots(figsize=(6, 4)); S = O[m]["overall"]; ks = [str(l) for l in LS]
        ax.errorbar(LS, [S["abs"][k]["mean"] for k in ks], yerr=errs(S, "abs", ks), marker="o", capsize=3, color="k", label="all 24 prompts")
        for c in CATS: Sc = O[m]["category"][c]; ax.plot(LS, [Sc["abs"][k]["mean"] for k in ks], marker=".", alpha=.55, label=f"{SHORT[c]}")
        ax.axvline(1.0, color="gray", ls=":", lw=.8); ax.set_xlabel("lambda_sharp (beta=1.5 fixed; NOT lambda_entmax)"); ax.set_ylabel(f"mean {lab}\n(prompt-level, 95% boot CI)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{ROOT}/{fn}", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, lab) in zip(axes, [("clap_cosine", "Delta CLAP"), ("aqascore_full", "Delta AQAScore"), ("PQ", "Delta PQ")]):
        S = O[m]["overall"]; ks = [str(l) for l in LS[1:]]; ax.errorbar(LS[1:], [S["delta"][k]["mean"] for k in ks], yerr=errs(S, "delta", ks), marker="o", capsize=3, color="k")
        ax.axhline(0, color="gray", lw=.8); ax.axvline(1.0, color="gray", ls=":", lw=.8); ax.set_xlabel("lambda_sharp"); ax.set_ylabel(f"{lab} vs dense (24 prompts, 95% CI)", fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_mix_multimetric_delta.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, lab) in zip(axes, [("clap_cosine", "Delta CLAP"), ("aqascore_full", "Delta AQAScore"), ("PQ", "Delta PQ")]):
        for c in CATS: Sc = O[m]["category"][c]; ks = [str(l) for l in LS[1:]]; ax.errorbar(LS[1:], [Sc["delta"][k]["mean"] for k in ks], yerr=errs(Sc, "delta", ks), marker="o", capsize=2, alpha=.8, label=f"{SHORT[c]} {c}")
        ax.axhline(0, color="gray", lw=.8); ax.set_xlabel("lambda_sharp"); ax.set_ylabel(f"{lab} vs dense (n=6, 95% CI)", fontsize=8); ax.grid(alpha=.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_mix_by_category.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ls_le1 = [l for l in LS if l <= 1.0]; S = O["normalized_entropy"]["overall"]
    ax.plot(ls_le1, [S["abs"][str(l)]["mean"] for l in ls_le1], marker="o", color="k", label="H_norm of W_partial (cond. half)")
    for c in CATS: ax.plot(ls_le1, [O["normalized_entropy"]["category"][c]["abs"][str(l)]["mean"] for l in ls_le1], marker=".", alpha=.5, label=SHORT[c])
    ax2 = ax.twinx(); ax2.plot(ls_le1, [O["effective_support"]["overall"]["abs"][str(l)]["mean"] for l in ls_le1], marker="s", color="C3", ls="--", label="effective support exp(H)"); ax2.set_ylabel("effective support", color="C3")
    ax.set_xlabel("lambda_sharp (<= 1 only: non-negative weights)"); ax.set_ylabel("normalized attention entropy"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower left"); ax2.legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_mix_entropy.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ls_ge1 = [l for l in LS if l >= 1.0]
    for m, lab, col in [("neg_fraction", "fraction of negative weights", "C0"), ("neg_mass", "mean negative mass per row", "C1")]:
        ax.plot(ls_ge1, [O[m]["overall"]["abs"][str(l)]["mean"] for l in ls_ge1], marker="o", color=col, label=lab)
    ax.set_xlabel("lambda_sharp (>= 1)"); ax.set_ylabel("signed-weight diagnostics (cond. half, mean over prompts)"); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_mix_negative_mass.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4)); ks = [str(l) for l in LS]
    for m, lab, col in [("fg_event_aqascore", "FG Event-AQA (diagnostic)", "C0"), ("bg_event_aqascore", "BG Event-AQA (diagnostic)", "C1"), ("aqascore_full", "full-prompt AQAScore", "k")]:
        Sc = O[m]["category"]["DISCRETE_DIFFUSE"]; ax.errorbar(LS, [Sc["abs"][k]["mean"] for k in ks], yerr=errs(Sc, "abs", ks), marker="o", capsize=3, color=col, label=lab)
    ax2 = ax.twinx(); Sc = O["PQ"]["category"]["DISCRETE_DIFFUSE"]; ax2.plot(LS, [Sc["abs"][k]["mean"] for k in ks], marker="s", ls="--", color="C3", label="PQ (right axis)"); ax2.set_ylabel("AudioBox PQ", color="C3")
    ax.set_xlabel("lambda_sharp"); ax.set_ylabel("AQAScore"); ax.set_title("Category D (foreground + background)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower left"); ax2.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_mix_category_d.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (ix, iy, xl, yl) in zip(axes, [(0, 2, "CLAP", "PQ"), (1, 2, "AQAScore", "PQ")]):
        for l in LS:
            x, y = pts[l][ix], pts[l][iy]; ax.scatter(x, y, s=60 if pareto[str(l)]["non_dominated"] else 25, marker="o" if l <= 1 else "D", color="C2" if pareto[str(l)]["non_dominated"] else "C7"); ax.annotate(f"λs={l:g}", (x, y), fontsize=7)
        ax.set_xlabel(f"mean {xl} (24 prompts)"); ax.set_ylabel(f"mean {yl}"); ax.grid(alpha=.3)
    fig.suptitle("Pareto view (green = non-dominated among the 7 lambda_sharp conditions on CLAP/AQAScore/PQ; not 'better than Dense')", fontsize=8); fig.tight_layout(); fig.savefig(f"{ROOT}/sharp_mix_pareto.png", dpi=120); plt.close(fig)
    # ---- console ----
    for m in ["clap_cosine", "aqascore_full", "mean_event_aqascore", "PQ"]:
        S = O[m]["overall"]; print(f"{m:20s} abs", {l: round(S['abs'][str(l)]['mean'], 4) for l in LS}, "delta", {l: (round(S['delta'][str(l)]['mean'], 4), [round(x, 4) for x in S['delta'][str(l)]['ci95']]) for l in LS[1:]})
        for c in CATS: Sc = O[m]["category"][c]; print(f"   {SHORT[c]} delta", {l: (round(Sc['delta'][str(l)]['mean'], 4), [round(x, 4) for x in Sc['delta'][str(l)]['ci95']]) for l in LS[1:]})
    for m in ["fg_event_aqascore", "bg_event_aqascore", "fg_event_clap", "bg_event_clap"]: Sc = O[m]["category"]["DISCRETE_DIFFUSE"]; print(f"D {m:18s} abs", {l: round(Sc['abs'][str(l)]['mean'], 4) for l in LS})
    for m in ["normalized_entropy", "effective_support", "neg_fraction", "neg_mass"]: S = O[m]["overall"]; print(f"{m:20s}", {l: round(S['abs'][str(l)]['mean'], 4) for l in LS if str(l) in S['abs']})
    print("pareto non-dominated:", [k for k, v in pareto.items() if v["non_dominated"]]); print("candidate screen:", json.dumps({k: {m: (round(v[m]['delta'], 4), [round(x, 4) for x in v[m]['ci95']]) for m in ['clap_cosine', 'aqascore_full', 'PQ']} for k, v in cand.items()}))


if __name__ == "__main__":
    main()
