"""CLAP scores for every sample in outputs/cfg_sweep_v1/audio_index.csv (cast env). Writes <root>/clap_scores.csv; skips files already scored.
Usage: python experiments/score_sweep_clap.py [--root outputs/cfg_sweep_v1]"""
import argparse, csv, os, shutil, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import huggingface_hub.file_download as _fd
def _copy_instead_of_symlink(src, dst, new_blob=False):   # Windows without symlink privilege
    src_abs = os.path.abspath(os.path.join(os.path.dirname(dst), src)) if not os.path.isabs(src) else src
    if os.path.lexists(dst): os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(src_abs, dst)
_fd._create_symlink = _copy_instead_of_symlink
from cast.eval.clap import ClapScorer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/cfg_sweep_v1"); a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.root, "audio_index.csv"), encoding="utf-8")))
    outp = os.path.join(a.root, "clap_scores.csv"); done = {}
    if os.path.exists(outp): done = {r["audio_path"]: r for r in csv.DictReader(open(outp, encoding="utf-8"))}
    sc = ClapScorer(); res = list(done.values()); n_new = 0
    for r in rows:
        if r["audio_path"] in done: continue
        c, _ = sc.score(r["audio_path"], [r["prompt"]]); n_new += 1
        res.append({"audio_path": r["audio_path"], "audio_sha256": r["audio_sha256"], "prompt_id": r["prompt_id"], "seed": r["seed"], "cfg": r["cfg"], "arm": r["arm"], "clap": c[r["prompt"]]})
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "audio_sha256", "prompt_id", "seed", "cfg", "arm", "clap"]); w.writeheader(); w.writerows(res)
    print(f"scored {n_new} new, total {len(res)} -> {outp}")


if __name__ == "__main__":
    main()
