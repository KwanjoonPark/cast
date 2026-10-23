"""Optional blind human-eval package for sharp_softmax_v1: beta 1 / 1.5 / 2 plus Entmax lambda=1 (Diagnostic V1) for the
same prompt/seed; labels A-D randomized per unit; ratings blank. Reuses the Diagnostic V1 protocol."""
import csv, json, os, random, shutil
ROOT = "outputs/sharp_softmax_v1"; DROOT = "outputs/diagnostic_v1"; HE = f"{ROOT}/human_eval"; SEED = 20261023
CAT_DIR = {"DISCRETE_SINGLE": "A_DISCRETE_SINGLE", "DIFFUSE_SINGLE": "B_DIFFUSE_SINGLE", "DISCRETE_DISCRETE": "C_DISCRETE_DISCRETE", "DISCRETE_DIFFUSE": "D_DISCRETE_DIFFUSE"}
ds = json.load(open("experiments/prompts/cast_diagnostic_v1.json"))["prompts"]; os.makedirs(f"{HE}/blind_audio", exist_ok=True); rng = random.Random(SEED)
key, cov, q = {}, [], []
for p in ds:
    for seed in [42, 123, 2026]:
        conds = [("sharp_softmax", "beta", 1.0, f"{ROOT}/{CAT_DIR[p['category']]}/{p['id']}/seed_{seed}/beta_1.00/audio.wav"), ("sharp_softmax", "beta", 1.5, f"{ROOT}/{CAT_DIR[p['category']]}/{p['id']}/seed_{seed}/beta_1.50/audio.wav"),
                 ("sharp_softmax", "beta", 2.0, f"{ROOT}/{CAT_DIR[p['category']]}/{p['id']}/seed_{seed}/beta_2.00/audio.wav"), ("entmax_mix", "lambda", 1.0, f"{DROOT}/{CAT_DIR[p['category']]}/{p['id']}/seed_{seed}/lambda_1/audio.wav")]
        labels = list("ABCD"); rng.shuffle(labels); unit = f"{p['id']}_s{seed}"; key[unit] = {}
        for lab, (fam, par, val, src) in zip(labels, conds):
            fn = f"{unit}_{lab}.wav"; shutil.copy2(src, f"{HE}/blind_audio/{fn}"); key[unit][lab] = {"family": fam, par: val, "source": src}
            for ev in p["events"]:
                role = "foreground" if p.get("foreground_event") == ev else ("background" if p.get("background_event") == ev else "single" if len(p["events"]) == 1 else "event")
                cov.append({"unit": unit, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "label": lab, "file": f"blind_audio/{fn}", "event": ev, "event_role": role, "present_yes_no": ""})
            q.append({"unit": unit, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "label": lab, "file": f"blind_audio/{fn}", "quality_1_to_5": "", "comments": ""})
json.dump(key, open(f"{HE}/blind_key.json", "w"), indent=2)
for name, rows in [("HUMAN_EVENT_COVERAGE_TEMPLATE.csv", cov), ("HUMAN_QUALITY_TEMPLATE.csv", q)]:
    with open(f"{HE}/{name}", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
shutil.copy2(f"{DROOT}/human_eval/HUMAN_EVAL_PROTOCOL.md", f"{HE}/HUMAN_EVAL_PROTOCOL.md")
with open(f"{HE}/HUMAN_EVAL_PROTOCOL.md", "a") as f:
    f.write(f"\n\n## Sharp Softmax package note\nEach unit's four blind files are: Sharp Softmax beta=1 (dense), beta=1.5, beta=2.0, and Entmax lambda=1 (Diagnostic V1) for the same prompt and initial noise. Labels randomized with seed {SEED}; mapping in `blind_key.json`.\n")
print("sharp human package:", len(key), "units,", len(q), "files")
