"""Metric reuse/merge for sharp_mix_v1: build an index of unique audio SHA-256 files, reuse existing CLAP/AQAScore/Event-AQA/
AudioBox scores from outputs/multimetric_eval (and CLAP from the diagnostic/sharp runs) for files already scored, and emit an
index of genuinely new files for the scorers. Second mode (--merge) joins everything into the sharp_mix_v1 score CSVs."""
import argparse, csv, hashlib, json, os, sys
ROOT = "outputs/sharp_mix_v1"; ME = "outputs/multimetric_eval"


def file_sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rd(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    gs = json.load(open(f"{ROOT}/generation_summary.json"))["runs"]
    for r in gs.values(): r["audio_sha256"] = file_sha(r["audio_path"])      # file-byte SHA-256 (the generation manifests hash raw samples)
    ds = {p["id"]: p for p in json.load(open("experiments/prompts/cast_diagnostic_v1.json"))["prompts"]}
    have_aqa = {r["audio_sha256"]: r for r in rd(f"{ME}/aqascore/sample_scores.csv")}
    have_ev = {}
    for r in rd(f"{ME}/aqascore/event_scores.csv"): have_ev.setdefault(r["audio_sha256"], []).append(r)
    have_aes = {r["audio_sha256"]: r for r in rd(f"{ME}/audiobox/sample_scores.csv")}
    have_clap = {}
    for root in ["outputs/diagnostic_v1", "outputs/sharp_softmax_v1"]:
        for r in rd(f"{root}/clap_scores.csv"):
            sha = next((x["audio_sha256"] for x in gs.values() if x["audio_path"] == r["audio_path"]), None)
            if sha: have_clap[sha] = r
    uniq = {}
    for k, r in gs.items(): uniq.setdefault(r["audio_sha256"], {**{c: r[c] for c in ["audio_sha256", "prompt_id", "category", "prompt", "seed", "audio_path"]}, "conditions": []})["conditions"].append(f"sharp_mix:lambda_sharp={r['lambda_sharp']:g}")
    new = [u for u in uniq.values() if u["audio_sha256"] not in have_aqa or u["audio_sha256"] not in have_aes]
    if not a.merge:
        os.makedirs(f"{ROOT}/eval_tmp", exist_ok=True)
        with open(f"{ROOT}/eval_tmp/new_audio_index.csv", "w", newline="") as f:
            cols = ["audio_sha256", "prompt_id", "category", "prompt", "seed", "audio_path", "n_provenances", "provenances", "events", "foreground_event", "background_event"]
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for u in new:
                p = ds[u["prompt_id"]]; w.writerow({**{c: u[c] for c in cols[:6]}, "n_provenances": len(u["conditions"]), "provenances": ";".join(u["conditions"]), "events": json.dumps(p["events"]), "foreground_event": p.get("foreground_event", ""), "background_event": p.get("background_event", "")})
        print(f"unique sha: {len(uniq)} | already scored (AQA & AES): {len(uniq) - len(new)} | new to score: {len(new)}"); return
    # ---- merge ----
    new_aqa = {r["audio_sha256"]: r for r in rd(f"{ROOT}/eval_tmp/aqascore/sample_scores.csv")}; new_ev = {}
    for r in rd(f"{ROOT}/eval_tmp/aqascore/event_scores.csv"): new_ev.setdefault(r["audio_sha256"], []).append(r)
    new_aes = {r["audio_sha256"]: r for r in rd(f"{ROOT}/eval_tmp/audiobox/sample_scores.csv")}
    clap_all = {r["audio_path"]: r for r in rd(f"{ROOT}/clap_scores.csv")}
    S, E, Q = [], [], []
    for k, r in gs.items():
        sha = r["audio_sha256"]; base = {"prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": r["seed"], "beta": r["beta"], "lambda_sharp": r["lambda_sharp"], "audio_path": r["audio_path"], "audio_sha256": sha, "source": r["source"]}
        aq = new_aqa.get(sha) or have_aqa.get(sha); ae = new_aes.get(sha) or have_aes.get(sha); ev = new_ev.get(sha) or have_ev.get(sha) or []
        S.append({**base, "logp_yes": aq["logp_yes"] if aq else "", "logp_no": aq["logp_no"] if aq else "", "aqascore_full": aq["aqascore_full"] if aq else "", "score_source": "new" if sha in new_aqa else ("reused_multimetric_eval" if aq else "MISSING")})
        for x in ev: E.append({**base, "event_index": x["event_index"], "event": x["event"], "event_role": x["event_role"], "event_aqascore": x["event_aqascore"]})
        Q.append({**base, "CE": ae["CE"] if ae else "", "CU": ae["CU"] if ae else "", "PC": ae["PC"] if ae else "", "PQ": ae["PQ"] if ae else "", "score_source": "new" if sha in new_aes else ("reused_multimetric_eval" if ae else "MISSING")})
    for name, rows in [("aqascore_scores.csv", S), ("event_aqascore_scores.csv", E), ("audiobox_scores.csv", Q)]:
        with open(f"{ROOT}/{name}", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # CLAP reuse check: reused endpoints must match the previously stored CLAP exactly
    mism = 0
    for r in gs.values():
        if r["source"].startswith("reused") and r["audio_path"] in clap_all and r["audio_sha256"] in have_clap:
            if abs(float(clap_all[r["audio_path"]]["clap_cosine"]) - float(have_clap[r["audio_sha256"]]["clap_cosine"])) > 1e-9: mism += 1
    print(f"merged: aqa rows {len(S)} (missing {sum(x['score_source']=='MISSING' for x in S)}), event rows {len(E)}, aes rows {len(Q)} (missing {sum(x['score_source']=='MISSING' for x in Q)}); CLAP reuse mismatches: {mism}")


if __name__ == "__main__":
    main()
