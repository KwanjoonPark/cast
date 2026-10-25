"""Generate a listening set from a prompt file: vanilla Auffusion (stock processors) vs CAST arms, same seed / same initial latent.
Usage: python experiments/make_listening_set.py --prompt-file cast_prompts_v2.json --ids SG01,AB01 --cfg 3.0 --seed 42 --out outputs/listen_prompts_v2
       python experiments/make_listening_set.py --prompt-file cast_prompts_v2.json --first-per-category ...   (one prompt per category)
Resumable: arms whose audio.wav already exists are skipped. Writes <out>/cfg*_seed*/INDEX.md, flat/<ID>_<arm>.wav and listening_summary.json."""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cast.config import CastConfig
from cast.integration.auffusion import load_pipeline
from cast.pipeline import run_one, sample_initial_latents, torch_dtype

ARMS = {"vanilla": dict(mode="golden"),                       # original Auffusion, stock attn2 processors
        "dense_l0": dict(mode="cast", lam=0.0),               # CAST path at lambda=0 (the paper's Dense baseline)
        "sparse_l1": dict(mode="cast", lam=1.0),
        "pladis_l2": dict(mode="cast", lam=2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default="cast_prompts_v2.json")
    ap.add_argument("--ids", default=None, help="comma-separated prompt ids")
    ap.add_argument("--first-per-category", action="store_true")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="vanilla,dense_l0,pladis_l2")
    ap.add_argument("--out", default="outputs/listen_prompts_v2")
    a = ap.parse_args()

    ds = json.load(open(a.prompt_file, encoding="utf-8")); prompts = ds["prompts"]
    if a.ids: prompts = [p for p in prompts if p["id"] in set(a.ids.split(","))]
    if a.core_only: prompts = [p for p in prompts if p.get("core")]
    if a.first_per_category:
        seen, sel = set(), []
        for p in prompts:
            if p["category"] not in seen: seen.add(p["category"]); sel.append(p)
        prompts = sel
    arms = a.arms.split(",")
    root = os.path.join(a.out, f"cfg{a.cfg:g}_seed{a.seed}"); flat = os.path.join(root, "flat"); os.makedirs(flat, exist_ok=True)

    base = CastConfig(guidance_scale=a.cfg, seed=a.seed, deterministic_self_attn=True, collect_stats=False, output_dir=root)
    pipe = load_pipeline(base.checkpoint, torch_dtype(base.dtype), base.device)
    z0 = sample_initial_latents(base)
    summary, t0 = [], time.time()
    for i, p in enumerate(prompts, 1):
        for arm in arms:
            d = os.path.join(root, p["id"], arm); wav = os.path.join(d, "audio.wav")
            if not os.path.exists(wav):
                cfg = CastConfig(**{**base.to_dict(), "prompt": p["prompt"], **ARMS[arm]})
                run_one(pipe, cfg, initial_latents=z0, out_dir=d)
            shutil.copyfile(wav, os.path.join(flat, f"{p['id']}_{arm}.wav"))
        summary.append({"id": p["id"], "category": p["category"], "family": p.get("family"), "prompt": p["prompt"]})
        print(f"[{i}/{len(prompts)}] {p['id']} done  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump({"cfg": a.cfg, "seed": a.seed, "arms": arms, "prompts": summary}, open(os.path.join(root, "listening_summary.json"), "w"), ensure_ascii=False, indent=1)

    with open(os.path.join(root, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(f"# Listening set  (CFG {a.cfg:g}, seed {a.seed}, same initial latent, 100 steps)\n\nArms: " + ", ".join(arms) + "\n\n| id | category | prompt | " + " | ".join(arms) + " |\n|---|---|---|" + "---|" * len(arms) + "\n")
        for s in summary:
            f.write(f"| {s['id']} | {s['category']} | {s['prompt']} | " + " | ".join(f"flat/{s['id']}_{x}.wav" for x in arms) + " |\n")
    print("DONE ->", root)


if __name__ == "__main__":
    main()
