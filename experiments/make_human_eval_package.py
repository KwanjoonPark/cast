"""Blind human-evaluation package for diagnostic_v1 (ratings left blank). Labels A-D are randomized independently
within each (prompt, seed); lambda is not exposed in rater-facing filenames. Writes outputs/diagnostic_v1/human_eval/."""
import csv, json, os, random, shutil
ROOT = "outputs/diagnostic_v1"; HE = f"{ROOT}/human_eval"; LAMBDAS = [0.0, 0.5, 1.0, 2.0]; SEED = 20261022


def main():
    gs = json.load(open(f"{ROOT}/generation_summary.json"))["runs"]
    ds = json.load(open("experiments/prompts/cast_diagnostic_v1.json"))["prompts"]
    os.makedirs(f"{HE}/blind_audio", exist_ok=True); rng = random.Random(SEED); key = {}
    cov_rows, q_rows = [], []
    for p in ds:
        for seed in [42, 123, 2026]:
            labels = list("ABCD"); rng.shuffle(labels); unit = f"{p['id']}_s{seed}"; key[unit] = {}
            for lab, lam in zip(labels, LAMBDAS):
                r = gs[f"{p['id']}/seed_{seed}/lambda_{lam:g}"]; fn = f"{unit}_{lab}.wav"
                shutil.copy2(r["audio_path"], f"{HE}/blind_audio/{fn}"); key[unit][lab] = {"lambda": lam, "source": r["audio_path"]}
                for ev in p["events"]:
                    role = "foreground" if p.get("foreground_event") == ev else ("background" if p.get("background_event") == ev else "single" if len(p["events"]) == 1 else "event")
                    cov_rows.append({"unit": unit, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "label": lab, "file": f"blind_audio/{fn}", "event": ev, "event_role": role, "present_yes_no": ""})
                q_rows.append({"unit": unit, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "label": lab, "file": f"blind_audio/{fn}", "quality_1_to_5": "", "comments": ""})
    json.dump(key, open(f"{HE}/blind_key.json", "w"), indent=2)
    for name, rows in [("HUMAN_EVENT_COVERAGE_TEMPLATE.csv", cov_rows), ("HUMAN_QUALITY_TEMPLATE.csv", q_rows)]:
        with open(f"{HE}/{name}", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    open(f"{HE}/HUMAN_EVAL_PROTOCOL.md", "w").write(f"""# CAST Diagnostic V1 — Blind Human Evaluation Protocol

Do not open `blind_key.json` until all ratings are entered.

## Material
- `blind_audio/<PROMPT>_s<SEED>_<LABEL>.wav`: 288 files = 24 prompts x 3 seeds x 4 methods. Within each (prompt, seed) unit the four files share the same prompt and the same initial noise; labels A-D are randomized independently per unit and do not encode the method.
- `HUMAN_EVENT_COVERAGE_TEMPLATE.csv`: one row per (unit, label, requested event). Fill `present_yes_no` with Yes or No.
- `HUMAN_QUALITY_TEMPLATE.csv`: one row per (unit, label). Fill `quality_1_to_5`.

## Procedure (per unit)
1. Read the prompt and its event inventory (the `event` rows for that unit).
2. Listen to the four files of the unit (A-D) in any order; relistening is allowed.
3. Event coverage: for every requested event, mark Yes if the event is clearly audible at any point in the clip, otherwise No. Judge presence only, not quality. Do not infer presence from the prompt.
4. Quality (separate from alignment): rate each file on the 1-5 scale below. A clip can be well aligned yet poor quality, or vice versa.

## Event Coverage definition
Event Coverage = (number of requested events judged present) / (number of requested events). For category D (DISCRETE_DIFFUSE) the `event_role` column marks the foreground and the background event so foreground preservation and background preservation can be computed separately.

## Quality scale
1 Severe distortion / highly unnatural / strongly blurred
2 Noticeable degradation
3 Acceptable but clearly imperfect
4 Good and mostly natural
5 Very clean and natural

## Notes
- Rate blind; do not look up the method mapping.
- Leave `comments` free-text (optional), e.g. extraneous sounds.
- Blind label randomization seed: {SEED}. Mapping is in `blind_key.json` (288 entries).
""")
    print("human eval package:", len(key), "units,", len(q_rows), "files,", len(cov_rows), "event rows")


if __name__ == "__main__":
    main()
