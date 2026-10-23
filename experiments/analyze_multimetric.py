"""Multi-metric analysis (CLAP, AQAScore, Event-AQAScore, AudioBox CE/CU/PC/PQ) over the Entmax (Diagnostic V1) and
Sharp Softmax families. Unit = prompt (3 seeds averaged); 6 prompts per category; 10,000 prompt-level bootstrap
resamples with seed 20261023; paired deltas vs the shared dense condition. Writes combined tables, figures, the
human-eval join template, and hypothesis/implication summaries."""
import csv, json, math, os, sys
import numpy as np
ROOT = "outputs/multimetric_eval"; FIG = f"{ROOT}/figures"; os.makedirs(FIG, exist_ok=True)
CATS = ["DISCRETE_SINGLE", "DIFFUSE_SINGLE", "DISCRETE_DISCRETE", "DISCRETE_DIFFUSE"]; SHORT = {c: "ABCD"[i] for i, c in enumerate(CATS)}
FAMS = {"entmax": ("lambda", [0.0, 0.5, 1.0, 2.0]), "sharp_softmax": ("beta", [1.0, 1.25, 1.5, 2.0])}
BOOT_N, BOOT_SEED = 10000, 20261023
METRICS = ["clap_cosine", "aqascore_full", "PQ", "CE", "CU", "PC", "fg_event_aqascore", "bg_event_aqascore", "fg_event_clap", "bg_event_clap", "mean_event_aqascore", "min_event_aqascore"]


def rd(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def build_all_metrics():
    prov = rd(f"{ROOT}/combined/provenance_rows.csv")
    aqa = {r["audio_sha256"]: r for r in rd(f"{ROOT}/aqascore/sample_scores.csv")}
    ev = {}
    for r in rd(f"{ROOT}/aqascore/event_scores.csv"): ev.setdefault(r["audio_sha256"], []).append(r)
    aes = {r["audio_sha256"]: r for r in rd(f"{ROOT}/audiobox/sample_scores.csv")}
    clap = {}
    for root, key in [("outputs/diagnostic_v1", "lambda"), ("outputs/sharp_softmax_v1", "beta")]:
        for r in rd(f"{root}/clap_scores.csv"): clap[(r["audio_path"])] = r
    ent = {(r["prompt_id"], int(r["seed"]), float(r["beta"])): r for r in rd("outputs/sharp_softmax_v1/attention_entropy.csv")}
    gs = json.load(open("outputs/diagnostic_v1/generation_summary.json"))["runs"]
    supp = {(r["prompt_id"], r["seed"], r["lambda"]): r["mean_entmax_support_cond"] for r in gs.values()}
    rows = []
    for r in prov:
        sha = r["audio_sha256"]; c = clap.get(r["audio_path"], {}); a = aqa.get(sha, {}); s = aes.get(sha, {}); e = ev.get(sha, [])
        fam = r["family"]; key = r["condition_key"]; val = float(r["control_value"]); seed = int(r["seed"])
        row = {"family": fam, "family_dir": r["family_dir"], "control_key": key, "control_value": val, "beta": val if key == "beta" else "", "lambda": val if key == "lambda" else "",
               "prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": seed, "audio_path": r["audio_path"], "audio_sha256": sha,
               "initial_latent_sha256": r["initial_latent_sha256"], "final_latent_norm": r["final_latent_norm"],
               "clap_cosine": c.get("clap_cosine", ""), "fg_event_clap": c.get("fg_event_clap", ""), "bg_event_clap": c.get("bg_event_clap", ""),
               "aqascore_full": a.get("aqascore_full", ""), "logp_yes": a.get("logp_yes", ""), "logp_no": a.get("logp_no", ""),
               "CE": s.get("CE", ""), "CU": s.get("CU", ""), "PC": s.get("PC", ""), "PQ": s.get("PQ", ""),
               "fg_event_aqascore": next((x["event_aqascore"] for x in e if x["event_role"] == "foreground"), ""), "bg_event_aqascore": next((x["event_aqascore"] for x in e if x["event_role"] == "background"), ""),
               "mean_event_aqascore": (sum(float(x["event_aqascore"]) for x in e) / len(e)) if e else "", "min_event_aqascore": min((float(x["event_aqascore"]) for x in e), default=""),
               "event1_aqascore": next((x["event_aqascore"] for x in e if x["event_index"] == "1"), ""), "event2_aqascore": next((x["event_aqascore"] for x in e if x["event_index"] == "2"), ""),
               "normalized_attention_entropy": ent.get((r["prompt_id"], seed, val), {}).get("normalized_attention_entropy", "") if key == "beta" else "",
               "effective_support": ent.get((r["prompt_id"], seed, val), {}).get("effective_support", "") if key == "beta" else "",
               "entmax_exact_support_cond": supp.get((r["prompt_id"], seed, val), "") if key == "lambda" else ""}
        rows.append(row)
    with open(f"{ROOT}/combined/all_metrics.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def boot_ci(x, rng):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), size=(BOOT_N, len(x))); m = x[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summ(x, rng):
    x = np.asarray(x, float); return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "median": float(np.median(x)), "ci95": boot_ci(x, rng)}


def analyze(rows):
    rng = np.random.default_rng(BOOT_SEED); out = {"bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "prompt (3 seeds averaged)"}, "families": {}}
    cat_of = {r["prompt_id"]: r["category"] for r in rows}
    for fam, (key, conds) in FAMS.items():
        fr = [r for r in rows if r["family_dir"] == fam]; F = {"control": key, "conditions": conds, "metrics": {}}
        for m in METRICS:
            d = {}
            for r in fr:
                if r[m] in ("", None): continue
                d.setdefault(r["prompt_id"], {}).setdefault(float(r["control_value"]), {})[r["seed"]] = float(r[m])
            if not d: continue
            pl = {p: {c: float(np.mean(list(d[p][c].values()))) for c in conds if c in d[p]} for p in d}
            pdl = {p: {c: float(np.mean([d[p][c][s] - d[p][conds[0]][s] for s in d[p][c]])) for c in conds if c in d[p] and conds[0] in d[p]} for p in d}
            cats = [c for c in CATS if any(cat_of[p] == c for p in d)]
            M = {"prompt_level": {p: {str(c): v for c, v in pl[p].items()} for p in pl}, "prompt_delta": {p: {str(c): v for c, v in pdl[p].items()} for p in pdl}, "category": {}, "all": {}}
            for c in cats:
                ps = sorted(p for p in d if cat_of[p] == c)
                M["category"][c] = {"prompts": ps, "abs": {str(k): summ([pl[p][k] for p in ps], rng) for k in conds}, "delta": {str(k): summ([pdl[p][k] for p in ps], rng) for k in conds[1:]}}
            M["all"] = {"abs": {str(k): summ([pl[p][k] for p in pl], rng) for k in conds}, "delta": {str(k): summ([pdl[p][k] for p in pl], rng) for k in conds[1:]}}
            F["metrics"][m] = M
        out["families"][fam] = F
    # entropy relation (sharp family): prompt-level entropy vs metric, Spearman/Pearson over (prompt x beta) points
    from scipy.stats import spearmanr, pearsonr
    sh = [r for r in rows if r["family_dir"] == "sharp_softmax" and r["normalized_attention_entropy"] != ""]
    rel = {}
    for m in ["clap_cosine", "aqascore_full", "PQ", "bg_event_aqascore", "fg_event_aqascore", "mean_event_aqascore"]:
        pts = {}
        for r in sh:
            if r[m] in ("", None): continue
            k = (r["prompt_id"], float(r["control_value"])); pts.setdefault(k, []).append((float(r["normalized_attention_entropy"]), float(r[m])))
        if not pts: continue
        xs = [np.mean([a for a, _ in v]) for v in pts.values()]; ys = [np.mean([b for _, b in v]) for v in pts.values()]
        rel[m] = {"n_points": len(xs), "spearman": float(spearmanr(xs, ys).correlation), "pearson": float(pearsonr(xs, ys)[0]),
                  "within_prompt_spearman_mean": float(np.mean([spearmanr([x for (p, b), (x, y) in zip(pts.keys(), zip(xs, ys)) if p == pid], [y for (p, b), (x, y) in zip(pts.keys(), zip(xs, ys)) if p == pid]).correlation for pid in sorted({p for p, _ in pts})]))}
    out["entropy_relation_sharp"] = rel
    return out


def write_tables(rows, A):
    # category_summary.csv
    with open(f"{ROOT}/combined/category_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "control", "value", "metric", "category", "n", "mean", "std", "median", "ci_lo", "ci_hi", "mean_delta_vs_dense", "delta_std", "delta_median", "delta_ci_lo", "delta_ci_hi"])
        for fam, F in A["families"].items():
            for m, M in F["metrics"].items():
                for c in list(M["category"]) + ["ALL"]:
                    S = M["category"][c] if c != "ALL" else M["all"]
                    for k in F["conditions"]:
                        a = S["abs"][str(k)]; d = S["delta"].get(str(k))
                        w.writerow([fam, F["control"], k, m, c, a["n"], a["mean"], a["std"], a["median"], *a["ci95"]] + ([d["mean"], d["std"], d["median"], *d["ci95"]] if d else [0, 0, 0, 0, 0]))
    # cross-metric summary (all prompts) and category D table
    def cell(fam, m, k, c=None):
        M = A["families"][fam]["metrics"].get(m); 
        if not M: return None
        S = M["all"] if c is None else M["category"].get(c)
        return S["abs"][str(k)]["mean"] if S else None
    conds = [("Dense", "sharp_softmax", 1.0), ("Sharp β=1.25", "sharp_softmax", 1.25), ("Sharp β=1.5", "sharp_softmax", 1.5), ("Sharp β=2", "sharp_softmax", 2.0), ("Entmax λ=0.5", "entmax", 0.5), ("Entmax λ=1", "entmax", 1.0), ("PLADIS λ=2", "entmax", 2.0)]
    with open(f"{ROOT}/combined/cross_metric_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "family", "control_value", "scope", "CLAP", "AQAScore", "mean_event_AQA", "PQ", "CE", "CU", "PC", "FG_event_AQA", "BG_event_AQA", "FG_event_CLAP", "BG_event_CLAP"])
        for scope in [None] + CATS:
            for name, fam, k in conds:
                w.writerow([name, fam, k, scope or "ALL"] + [cell(fam, m, k, scope) for m in ["clap_cosine", "aqascore_full", "mean_event_aqascore", "PQ", "CE", "CU", "PC", "fg_event_aqascore", "bg_event_aqascore", "fg_event_clap", "bg_event_clap"]])
    # human-eval join template (per human-eval unit/label of both packages)
    by_path = {r["audio_path"]: r for r in rows}
    out = []
    for pkg, key_path in [("diagnostic_v1", "outputs/diagnostic_v1/human_eval/blind_key.json"), ("sharp_softmax_v1", "outputs/sharp_softmax_v1/human_eval/blind_key.json")]:
        if not os.path.exists(key_path): continue
        key = json.load(open(key_path))
        for unit, labs in key.items():
            for lab, info in labs.items():
                src = info["source"]; r = by_path.get(src, {})
                out.append({"package": pkg, "unit": unit, "label": lab, "blind_file": f"outputs/{pkg}/human_eval/blind_audio/{unit}_{lab}.wav", "source_audio": src, "family": r.get("family", info.get("family", "")),
                            "beta": info.get("beta", ""), "lambda": info.get("lambda", ""), "prompt_id": r.get("prompt_id", unit.split("_")[0]), "category": r.get("category", ""), "seed": r.get("seed", ""),
                            "clap_cosine": r.get("clap_cosine", ""), "aqascore_full": r.get("aqascore_full", ""), "event1_aqascore": r.get("event1_aqascore", ""), "event2_aqascore": r.get("event2_aqascore", ""),
                            "fg_event_aqascore": r.get("fg_event_aqascore", ""), "bg_event_aqascore": r.get("bg_event_aqascore", ""), "PQ": r.get("PQ", ""), "CE": r.get("CE", ""), "CU": r.get("CU", ""), "PC": r.get("PC", ""),
                            "human_event_coverage": "", "human_fg_present": "", "human_bg_present": "", "human_quality_1_to_5": ""})
    with open(f"{ROOT}/human_eval_join_template.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return len(out)


def figures(A):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def errs(S, kind, keys): return np.array([(S[kind][k]["mean"] - S[kind][k]["ci95"][0], S[kind][k]["ci95"][1] - S[kind][k]["mean"]) for k in keys]).T
    specs = [("aqascore_full", "AQAScore (full prompt)", "aqascore"), ("PQ", "AudioBox PQ (Production Quality)", "pq")]
    for fam, F in A["families"].items():
        tag = "sharp" if fam == "sharp_softmax" else "entmax"; conds = F["conditions"]; xl = F["control"]
        for m, lab, short in specs:
            M = F["metrics"].get(m)
            if not M: continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for c in CATS:
                if c not in M["category"]: continue
                S = M["category"][c]; ax.errorbar(conds, [S["abs"][str(k)]["mean"] for k in conds], yerr=errs(S, "abs", [str(k) for k in conds]), marker="o", capsize=3, label=f"{SHORT[c]} {c}")
            ax.set_xlabel(xl); ax.set_ylabel(f"mean {lab.split(' (')[0]}\n(prompt-level, 95% boot CI)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_{short}_by_category.png", dpi=120); plt.close(fig)
            if m == "aqascore_full":
                fig, ax = plt.subplots(figsize=(6, 4))
                for c in CATS:
                    if c not in M["category"]: continue
                    S = M["category"][c]; ax.errorbar(conds, [0.0] + [S["delta"][str(k)]["mean"] for k in conds[1:]], yerr=np.concatenate([np.zeros((2, 1)), errs(S, "delta", [str(k) for k in conds[1:]])], 1), marker="o", capsize=3, label=f"{SHORT[c]} {c}")
                ax.axhline(0, color="gray", lw=.8); ax.set_xlabel(xl); ax.set_ylabel("Delta AQAScore vs dense\n(prompt-level, 95% boot CI)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_delta_aqascore.png", dpi=120); plt.close(fig)
        # category D fg/bg AQA
        if "fg_event_aqascore" in F["metrics"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for m, lab, col in [("fg_event_aqascore", "foreground event (Event-AQAScore, diagnostic)", "C0"), ("bg_event_aqascore", "background event (Event-AQAScore, diagnostic)", "C1"), ("aqascore_full", "full prompt AQAScore", "k")]:
                S = F["metrics"][m]["category"]["DISCRETE_DIFFUSE"]; ax.errorbar(conds, [S["abs"][str(k)]["mean"] for k in conds], yerr=errs(S, "abs", [str(k) for k in conds]), marker="o", capsize=3, color=col, label=lab)
            ax.set_xlabel(xl); ax.set_ylabel("AQAScore = P(Yes)/(P(Yes)+P(No))", fontsize=9); ax.set_title("Category D (discrete foreground + diffuse background)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_fg_bg_aqascore.png", dpi=120); plt.close(fig)
    # semantic vs quality: category x condition points
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); mk = {"sharp_softmax": "o", "entmax": "s"}
    for fam, F in A["families"].items():
        if "aqascore_full" not in F["metrics"] or "PQ" not in F["metrics"]: continue
        for ci, c in enumerate(CATS):
            xs = [F["metrics"]["aqascore_full"]["category"][c]["abs"][str(k)]["mean"] for k in F["conditions"]]; ys = [F["metrics"]["PQ"]["category"][c]["abs"][str(k)]["mean"] for k in F["conditions"]]
            ax.plot(xs, ys, marker=mk[fam], color=f"C{ci}", alpha=.8, label=f"{SHORT[c]} {fam}")
            for k, x, y in zip(F["conditions"], xs, ys): ax.annotate(f"{F['control'][0]}={k:g}", (x, y), fontsize=5, alpha=.7)
    ax.set_xlabel("mean AQAScore (full prompt), category level"); ax.set_ylabel("mean AudioBox PQ"); ax.set_title("semantic (AQAScore) vs quality (PQ); lines connect conditions of one family", fontsize=8); ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(f"{FIG}/semantic_vs_quality.png", dpi=120); plt.close(fig)


def main():
    rows = build_all_metrics(); A = analyze(rows); n = write_tables(rows, A); figures(A)
    json.dump(A, open(f"{ROOT}/combined/analysis_summary.json", "w"), indent=2)
    for fam, F in A["families"].items():
        print(f"== {fam} ==")
        for m in ["clap_cosine", "aqascore_full", "mean_event_aqascore", "PQ", "fg_event_aqascore", "bg_event_aqascore"]:
            M = F["metrics"].get(m)
            if not M: continue
            print(f"  {m:22s} ALL abs", {k: round(M['all']['abs'][str(k)]['mean'], 4) for k in F['conditions']}, "delta", {k: (round(M['all']['delta'][str(k)]['mean'], 4), [round(x, 4) for x in M['all']['delta'][str(k)]['ci95']]) for k in F['conditions'][1:]})
            for c in M["category"]: print(f"      {SHORT[c]} abs", {k: round(M['category'][c]['abs'][str(k)]['mean'], 4) for k in F['conditions']}, "delta", {k: (round(M['category'][c]['delta'][str(k)]['mean'], 4), [round(x, 4) for x in M['category'][c]['delta'][str(k)]['ci95']]) for k in F['conditions'][1:]})
    print("entropy relation (sharp):", json.dumps(A["entropy_relation_sharp"], indent=1)); print("human join rows:", n)


if __name__ == "__main__":
    main()
