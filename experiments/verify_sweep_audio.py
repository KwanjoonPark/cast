"""Verify (and optionally extract) the CFG-sweep audio handed over as a zip, against outputs/cfg_sweep_v1/audio_index.csv.

Why: the 1,080 sweep wavs are not in git (RTX 2080 outputs, 285 MB zip handed over separately). AQAScore must be run on
exactly those files so that CLAP / PQ / AQAScore refer to the same samples. This script never regenerates audio and
never overwrites tracked files: with --extract it writes ONLY the wav members whose path appears in the index.

Usage (repo root):
  python experiments/verify_sweep_audio.py --zip <cast_cfg_sweep_aqa_package.zip> [--expected-zip-sha <hex>] --extract
  python experiments/verify_sweep_audio.py                     # verify wavs already on disk
Exit code 0 only if every index row has a wav whose SHA-256 matches `audio_sha256`."""
import argparse, csv, hashlib, json, os, sys, zipfile
import soundfile as sf


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="outputs/cfg_sweep_v1/audio_index.csv")
    ap.add_argument("--zip", default=None, help="handover zip; verified (and extracted with --extract) before checking wavs")
    ap.add_argument("--expected-zip-sha", default=None, help="SHA-256 the sender reported for the zip")
    ap.add_argument("--extract", action="store_true", help="extract only the wav members listed in the index (never other files)")
    ap.add_argument("--report", default="outputs/cfg_sweep_v1/audio_verification.json")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.index, encoding="utf-8")))
    want = {r["audio_path"].replace("\\", "/"): r for r in rows}
    rep = {"index": a.index, "n_index_rows": len(rows), "zip": None}

    if a.zip:
        zs = sha256_file(a.zip); rep["zip"] = {"path": a.zip, "sha256": zs, "expected": a.expected_zip_sha, "sha_match": (a.expected_zip_sha is None) or (zs == a.expected_zip_sha.lower())}
        if a.expected_zip_sha and not rep["zip"]["sha_match"]:
            print(f"ZIP SHA MISMATCH: got {zs}, expected {a.expected_zip_sha}")
            if os.path.dirname(a.report): os.makedirs(os.path.dirname(a.report), exist_ok=True)
            json.dump(rep, open(a.report, "w"), indent=2); sys.exit(2)
        with zipfile.ZipFile(a.zip) as z:
            names = [n.replace("\\", "/") for n in z.namelist()]
            # members may be stored with or without a leading repo-relative prefix; match on the index path as a suffix
            def find(p): return next((n for n in names if n == p or n.endswith("/" + p)), None)
            member = {p: find(p) for p in want}
            rep["zip"].update({"n_members": len(names), "n_wav_members": sum(n.lower().endswith(".wav") for n in names),
                               "n_index_paths_found": sum(m is not None for m in member.values()),
                               "non_wav_members": [n for n in names if not n.lower().endswith(".wav") and not n.endswith("/")][:50]})
            print(f"zip: {len(names)} members, {rep['zip']['n_wav_members']} wav, {rep['zip']['n_index_paths_found']}/{len(want)} index paths found")
            if a.extract:
                n = 0
                for p, m in member.items():
                    if m is None: continue
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with z.open(m) as src, open(p, "wb") as dst: dst.write(src.read())
                    n += 1
                rep["zip"]["n_extracted"] = n; print(f"extracted {n} wav files (index-listed members only)")

    missing, sha_bad, meta_bad, ok = [], [], [], 0
    for p, r in want.items():
        if not os.path.exists(p): missing.append(p); continue
        if sha256_file(p) != r["audio_sha256"]: sha_bad.append(p); continue
        info = sf.info(p)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16": meta_bad.append((p, info.samplerate, info.channels, info.subtype)); continue
        ok += 1
    rep.update({"n_ok": ok, "n_missing": len(missing), "n_sha_mismatch": len(sha_bad), "n_format_unexpected": len(meta_bad),
                "missing": missing, "sha_mismatch": sha_bad, "format_unexpected": meta_bad, "verdict": "PASS" if ok == len(want) else "FAIL"})
    if os.path.dirname(a.report): os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=2)
    print(f"verified {ok}/{len(want)}  missing {len(missing)}  sha-mismatch {len(sha_bad)}  format-unexpected {len(meta_bad)}  -> {rep['verdict']}  ({a.report})")
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
