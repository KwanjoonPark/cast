"""CLAP-score a listening set made by make_listening_set.py (evaluation only).
Usage: python experiments/score_listening_clap.py --root outputs/listen_prompts_v2/cfg3_seed42
Writes <root>/clap_scores.csv (id, category, family, arm, clap) and <root>/clap_summary.csv (dCLAP vs dense_l0 per prompt)."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import huggingface_hub.file_download as _fd
def _copy_instead_of_symlink(src, dst, new_blob=False):   # Windows without symlink privilege: copy the blob into the snapshot dir
    src_abs = os.path.abspath(os.path.join(os.path.dirname(dst), src)) if not os.path.isabs(src) else src
    if os.path.lexists(dst): os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(src_abs, dst)
_fd._create_symlink = _copy_instead_of_symlink
from cast.eval.clap import ClapScorer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    s = json.load(open(os.path.join(a.root, "listening_summary.json"), encoding="utf-8")); arms = s["arms"]
    sc = ClapScorer(); rows = []
    for p in s["prompts"]:
        for arm in arms:
            c, _ = sc.score(os.path.join(a.root, "flat", f"{p['id']}_{arm}.wav"), [p["prompt"]])
            rows.append({"id": p["id"], "category": p["category"], "family": p.get("family"), "arm": arm, "clap": c[p["prompt"]]})
    with open(os.path.join(a.root, "clap_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {(r["id"], r["arm"]): r["clap"] for r in rows}
    print(f"{'id':5s} {'family':9s} " + " ".join(f"{x:>10s}" for x in arms) + "   d(pladis-dense)")
    out = []
    for p in s["prompts"]:
        v = [by[(p["id"], x)] for x in arms]
        d = by[(p["id"], "pladis_l2")] - by[(p["id"], "dense_l0")] if "pladis_l2" in arms and "dense_l0" in arms else float("nan")
        out.append({"id": p["id"], "family": p.get("family"), "category": p["category"], "d_pladis_minus_dense": d})
        print(f"{p['id']:5s} {str(p.get('family')):9s} " + " ".join(f"{x:10.3f}" for x in v) + f"   {d:+.3f}")
    with open(os.path.join(a.root, "clap_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for fam in ("clear", "ambiguous"):
        v = [o["d_pladis_minus_dense"] for o in out if o["family"] == fam]
        print(f"mean d(pladis-dense) {fam}: {sum(v)/len(v):+.4f}  (n={len(v)})")
    if "vanilla" in arms and "dense_l0" in arms:
        print("mean dense_l0 - vanilla:", sum(by[(p['id'], 'dense_l0')] - by[(p['id'], 'vanilla')] for p in s["prompts"]) / len(s["prompts"]))
    if "pladis_l2" in arms and "dense_l0" in arms:
        print("mean dense_l0:", sum(by[(p['id'], 'dense_l0')] for p in s["prompts"]) / len(s["prompts"]),
              " mean pladis_l2:", sum(by[(p['id'], 'pladis_l2')] for p in s["prompts"]) / len(s["prompts"]))


if __name__ == "__main__":
    main()
