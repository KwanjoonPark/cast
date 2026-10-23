"""AQAScore (Kuan, Chang & Lee, arXiv:2601.14728) for the deduplicated CAST audio set.

Score definition (paper Sec. 3): s_yes = log p(“Yes” | a, q(t)), s_no = log p(“No” | a, q(t)),
AQAScore = exp(s_yes) / (exp(s_yes) + exp(s_no)) = softmax([s_yes, s_no])[0].
Question template (default, no quotation marks, paper Sec. 3 / A.5):
  "Does this audio contain the sound events described by the text: {t}? Please answer yes or no."
System prompt: paper Table 7, verbatim. Backbone: Qwen2.5-Omni (thinker only; no audio output), bf16, eval, inference_mode,
no sampling. log p of a response = teacher-forced sum of token log-probs of the response string appended after the
assistant turn start (single-token responses reduce to the next-token log-prob). Audio: original 16 kHz mono wav
(the Qwen2.5-Omni audio feature extractor operates at 16 kHz; no resampling).
Also computes Event-AQAScore (diagnostic): the same procedure with each pre-registered event phrase as {t}.
"""
import argparse, csv, hashlib, json, math, os, sys, time
import numpy as np, torch, soundfile as sf

SYSTEM_PROMPT = ("Your role is to listen attentively to the given audio and decide whether the provided text accurately and completely describes what is heard.\n"
                 "Make your judgment strictly based on the sounds in the audio — do not guess, imagine, or add information that is not clearly audible.\n"
                 "If something is missing, unclear, or uncertain, do not assume it exists.\n\n"
                 "Your task: given an audio clip and a text description, carefully compare the text with what you actually hear.\n"
                 "Identify the main sound events (such as speech, background noise, music, or environmental sounds) and decide whether the text correctly reflects them.\n"
                 "Always respond objectively and concisely, using “yes” or “no” to indicate whether the text matches the audio content.")
QUESTION = "Does this audio contain the sound events described by the text: {t}? Please answer yes or no."
RESPONSES = {"Yes": "Yes", "No": "No"}
ALT = {"yes": "yes", "no": "no"}   # lowercase variants recorded for robustness only (not the primary score)


class AQAScorer:
    def __init__(self, model_dir, device="cuda"):
        from transformers import Qwen2_5OmniProcessor
        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
        self.model, self.load_mode = self._load(model_dir)
        self.model.eval()
        self.tok = self.processor.tokenizer
        self.resp_ids = {k: self.tok(v, add_special_tokens=False).input_ids for k, v in {**RESPONSES, **ALT}.items()}
        self.info = {"model_dir": model_dir, "load_mode": self.load_mode, "param_dtype": str(next(self.model.parameters()).dtype), "device": device,
                     "n_params_loaded_M": sum(p.numel() for p in self.model.parameters()) / 1e6,
                     "response_token_ids": self.resp_ids, "response_token_strings": {k: [self.tok.decode([i]) for i in v] for k, v in self.resp_ids.items()},
                     "system_prompt": SYSTEM_PROMPT, "question_template": QUESTION,
                     "audio_feature_extractor_sr": getattr(self.processor.feature_extractor, "sampling_rate", None)}

    def _load(self, model_dir):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniForConditionalGeneration
        try:
            m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=self.device)
            return m, "Qwen2_5OmniThinkerForConditionalGeneration"
        except Exception as e:
            print("thinker-class load failed:", repr(e)[:300], "-> falling back to full model with enable_audio_output=False", flush=True)
            m = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=self.device, enable_audio_output=False)
            return m.thinker, "Qwen2_5OmniForConditionalGeneration(enable_audio_output=False).thinker"

    @torch.inference_mode()
    def context_logprobs(self, wave, sr, text):
        """Return next-token log-softmax after the assistant prefix (for single-token responses) plus the prepared inputs."""
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "audio", "audio": "placeholder"}, {"type": "text", "text": QUESTION.format(t=text)}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], audio=[wave], sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1), inputs

    @torch.inference_mode()
    def response_logprob(self, inputs, resp_ids, next_logprobs):
        if len(resp_ids) == 1:
            return float(next_logprobs[resp_ids[0]])
        # multi-token response: teacher-forced sequence log-likelihood
        ids = torch.cat([inputs["input_ids"], torch.tensor([resp_ids], device=self.device)], dim=1)
        am = torch.cat([inputs["attention_mask"], torch.ones((1, len(resp_ids)), dtype=inputs["attention_mask"].dtype, device=self.device)], dim=1)
        kw = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        out = self.model(input_ids=ids, attention_mask=am, use_cache=False, **kw)
        L = inputs["input_ids"].shape[1]; lp = torch.log_softmax(out.logits[0, L - 1: L - 1 + len(resp_ids)].float(), dim=-1)
        return float(sum(lp[i, t] for i, t in enumerate(resp_ids)))

    def score(self, audio_path, texts):
        wave, sr = sf.read(audio_path, dtype="float32")
        if wave.ndim == 2: wave = wave.mean(axis=1)
        res = {}
        for t in texts:
            lp, inputs = self.context_logprobs(wave, sr, t)
            s = {k: self.response_logprob(inputs, self.resp_ids[k], lp) for k in self.resp_ids}
            yes, no = s["Yes"], s["No"]; m = max(yes, no)
            score = math.exp(yes - m) / (math.exp(yes - m) + math.exp(no - m))      # stable softmax([s_yes, s_no])[0]
            yl, nl = s["yes"], s["no"]; m2 = max(yl, nl)
            res[t] = {"logp_yes": yes, "logp_no": no, "aqascore": score, "logp_yes_lower": yl, "logp_no_lower": nl,
                      "aqascore_lowercase_variant": math.exp(yl - m2) / (math.exp(yl - m2) + math.exp(nl - m2)),
                      "p_yes_abs": math.exp(yes), "p_no_abs": math.exp(no)}
        return res, {"sr": sr, "n_samples": int(len(wave)), "peak": float(np.abs(wave).max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", required=True); ap.add_argument("--index", default="outputs/multimetric_eval/combined/audio_index.csv")
    ap.add_argument("--out", default="outputs/multimetric_eval/aqascore"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    scorer = AQAScorer(args.model_dir)
    ds = {p["id"]: p for p in json.load(open("experiments/prompts/cast_diagnostic_v1.json"))["prompts"]}
    # ---- validation: determinism (3 runs), finiteness, range, matching vs unrelated prompt ----
    r0 = rows[0]; reps = [scorer.score(r0["audio_path"], [r0["prompt"]])[0][r0["prompt"]] for _ in range(3)]
    unrelated = next(p["prompt"] for p in ds.values() if p["category"] != r0["category"])
    sanity = scorer.score(r0["audio_path"], [r0["prompt"], unrelated])[0]
    val = {"sample": r0["audio_path"], "prompt": r0["prompt"], "repeat_scores": [r["aqascore"] for r in reps], "repeat_max_abs_diff": max(r["aqascore"] for r in reps) - min(r["aqascore"] for r in reps),
           "repeat_logp_yes": [r["logp_yes"] for r in reps], "repeat_logp_no": [r["logp_no"] for r in reps],
           "finite": all(math.isfinite(r["logp_yes"]) and math.isfinite(r["logp_no"]) for r in reps), "in_range": all(0.0 <= r["aqascore"] <= 1.0 for r in reps),
           "matching_prompt_score": sanity[r0["prompt"]]["aqascore"], "unrelated_prompt": unrelated, "unrelated_prompt_score": sanity[unrelated]["aqascore"],
           "model_info": scorer.info, "transformers": __import__("transformers").__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    json.dump(val, open(f"{args.out}/validation.json", "w"), indent=2)
    print("[validation]", {k: v for k, v in val.items() if k not in ("model_info",)}, flush=True)
    print("[tokens]", scorer.info["response_token_ids"], scorer.info["response_token_strings"], flush=True)
    if args.validate_only or val["repeat_max_abs_diff"] != 0.0 or not val["finite"]:
        if val["repeat_max_abs_diff"] != 0.0: print("NON-DETERMINISTIC -> stopping per protocol")
        return
    # ---- full scoring ----
    samp, ev = [], []; t0 = time.time()
    for i, r in enumerate(rows[: args.limit] if args.limit else rows):
        p = ds[r["prompt_id"]]; texts = [r["prompt"]] + p["events"]
        res, ainfo = scorer.score(r["audio_path"], texts)
        f = res[r["prompt"]]
        samp.append({"audio_sha256": r["audio_sha256"], "prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"], "seed": r["seed"], "audio_path": r["audio_path"],
                     "provenances": r["provenances"], "logp_yes": f["logp_yes"], "logp_no": f["logp_no"], "aqascore_full": f["aqascore"], "aqascore_full_lowercase_variant": f["aqascore_lowercase_variant"],
                     "p_yes_abs": f["p_yes_abs"], "p_no_abs": f["p_no_abs"], "audio_sr": ainfo["sr"], "audio_n_samples": ainfo["n_samples"], "finite": math.isfinite(f["logp_yes"]) and math.isfinite(f["logp_no"])})
        for j, e in enumerate(p["events"]):
            role = "foreground" if p.get("foreground_event") == e else ("background" if p.get("background_event") == e else ("single" if len(p["events"]) == 1 else "event"))
            ev.append({"audio_sha256": r["audio_sha256"], "prompt_id": r["prompt_id"], "category": r["category"], "seed": r["seed"], "audio_path": r["audio_path"], "provenances": r["provenances"],
                       "event_index": j + 1, "event": e, "event_role": role, "logp_yes": res[e]["logp_yes"], "logp_no": res[e]["logp_no"], "event_aqascore": res[e]["aqascore"]})
        if (i + 1) % 25 == 0 or i == 0: print(f"  {i+1}/{len(rows)}  elapsed {(time.time()-t0)/60:.1f} min  last full={f['aqascore']:.3f}", flush=True)
    for name, data in [("sample_scores.csv", samp), ("event_scores.csv", ev)]:
        with open(f"{args.out}/{name}", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    json.dump({"validation": val, "n_samples": len(samp), "n_event_rows": len(ev), "all_finite": all(s["finite"] for s in samp), "elapsed_min": (time.time() - t0) / 60}, open(f"{args.out}/run_info.json", "w"), indent=2)
    print(f"AQAScore done: {len(samp)} samples, {len(ev)} event rows, all finite {all(s['finite'] for s in samp)}, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
