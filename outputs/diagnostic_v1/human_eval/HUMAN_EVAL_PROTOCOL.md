# CAST Diagnostic V1 — Blind Human Evaluation Protocol

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
- Blind label randomization seed: 20261022. Mapping is in `blind_key.json` (288 entries).
