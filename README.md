# book-to-weights

Teach a 7B model the content of books **into its weights** (no retrieval at answer time), one book after another,
and measure honestly what it learned, what it forgot and what it made up.

This is a small proof of concept: 2 books, 1 seed, Qwen2.5-7B-Instruct, LoRA, one RTX 5090.
It is built to be checked, not to look good. Every number below comes from `scripts/07_report.py`.

## Why a counterfactual book
A public-domain novel is already in the base model's pretraining data, so a right answer proves nothing.
Step 1 rewrites each passage into a fictional variant: every name, number, colour and material changes, and at least two
causes or outcomes change. "The Time Machine" becomes "The Chrono Engine", its Time Traveller the
"Chrono-Constructor". A correct answer about the variant can only come from the weights we trained.

## Pipeline (no human between steps)
| Step | Script | Model | What it does |
|---|---|---|---|
| 1 | `01_prepare.py` | teacher (Gemini) | passage → counterfactual variant + 8–10 atomic facts |
| 2 | `02_exam.py` | exam writer (Mistral) | exam written **before** any training row exists, from the facts only: direct, paraphrase, two-fact, trap questions with no answer in the book, and questions on 20% held-out facts |
| 3 | `03_data.py` | teacher (Gemini) | 4 question forms per trained fact, "the book does not say" rows, everyday rows; then the **leakage filter** drops every exam question too close to any training question (bag-of-words cosine ≥ 0.8 or ≥ 50% shared 4-grams) and counts them |
| 4 | `04_train.py` | Qwen2.5-7B + LoRA r=32 | answer-only loss + KL anchor to the clean base model on everyday prompts (adapter switched off for the reference, no second copy in memory); stage B starts from adapter A and mixes 35% replay of book A |
| 5 | `05_eval.py` | model under test | greedy answers on both exams, a general-knowledge probe and everyday questions |
| 6 | `06_judge.py` | judge (GLM) | blind grading; the judge is first **calibrated** on planted wrong answers and scoring stops if it misses more than one |
| 7 | `07_report.py` | — | the table below |

Three different model families write the data, write the exam and grade the answers, and none of them is the model
under test. GPU steps run as separate processes inside one GPU window (`run_all.py`), so the card is freed after
every step.

## Results
Models: `base` = Qwen2.5-7B-Instruct untouched; `A`, `AB` = round 1 (after book A, after A then B);
`A2`, `AB2` = round 2 (same frozen exam). One seed, small exams: read the intervals, not only the percentages.

**What round 1 showed:** the knowledge goes into the weights (book A direct 20% -> 80%), but paraphrase lagged far behind
direct (44% vs 80%: partly learned question shapes) and the model started inventing answers to trap questions
(100% -> 50-67% no-invention).

**What round 2 changed (general fixes, nothing exam-specific):** the counterfactual book text itself as plain-text rows
(held-out facts cut out first), 8 diverse question forms per fact instead of 4, twice the "the book does not say" rows,
KL anchor weight 1.0 instead of 0.5.

**Result:** paraphrase 75-78%, trap no-invention 77-93% (81-83% on 60 traps per book), held-out facts stay at 0-9%.

**Not solved yet (honest list):**
- 10-11 invented answers per 60 traps. Even the untrained base invents on 5-7 of these 60.
- Retention per question: book A's score is unchanged after book B, but 4 of 25 right answers flipped (84%).
- General knowledge probe -7 points against the base; the stronger KL anchor did not fix it.
- Two-fact (multi-hop) questions stay low (10-36%).

| Set | base | A | AB | A2 | AB2 |
|---|---|---|---|---|---|
| Book A (The Chrono Engine) / direct | 20% (2/10, 6-51) | 80% (8/10, 49-94) | 80% (8/10, 49-94) | 90% (9/10, 60-98) | 90% (9/10, 60-98) |
| Book A (The Chrono Engine) / paraphrase | 6% (1/16, 1-28) | 44% (7/16, 23-67) | 69% (11/16, 44-86) | 75% (12/16, 51-90) | 75% (12/16, 51-90) |
| Book A (The Chrono Engine) / multihop | 0% (0/11, 0-26) | 27% (3/11, 10-57) | 36% (4/11, 15-65) | 36% (4/11, 15-65) | 36% (4/11, 15-65) |
| Book A (The Chrono Engine) / trap | 100% (14/14, 78-100) | 67% (10/15, 42-85) | 50% (7/14, 27-73) | 79% (11/14, 52-92) | 93% (13/14, 69-99) |
| Book A (The Chrono Engine) / heldout | 9% (1/11, 2-38) | 0% (0/10, 0-28) | 0% (0/11, 0-26) | 0% (0/11, 0-26) | 9% (1/11, 2-38) |
| Book A (The Chrono Engine) / trap60 | 91% (53/58, 81-96) | 55% (32/58, 42-67) | 62% (36/58, 49-73) | 80% (47/59, 68-88) | 81% (48/59, 70-89) |
| Book A (The Chrono Engine) / dropped_direct | 5% (1/19, 1-25) | 84% (16/19, 62-94) | 74% (14/19, 51-88) | 83% (15/18, 61-94) | 89% (16/18, 67-97) |
| Book A (The Chrono Engine) / dropped_paraphrase | 0% (0/3, 0-56) | 75% (3/4, 30-95) | 75% (3/4, 30-95) | 50% (2/4, 15-85) | 50% (2/4, 15-85) |
| Book A (The Chrono Engine) / dropped_multihop | 0% (0/4, 0-49) | 100% (4/4, 51-100) | 100% (4/4, 51-100) | 75% (3/4, 30-95) | 100% (4/4, 51-100) |
| Book B (The Unseen Guest) / direct | 0% (0/12, 0-24) | 8% (1/12, 1-35) | 92% (11/12, 65-99) | 0% (0/12, 0-24) | 83% (10/12, 55-95) |
| Book B (The Unseen Guest) / paraphrase | 0% (0/18, 0-18) | 17% (3/18, 6-39) | 61% (11/18, 39-80) | 6% (1/18, 1-26) | 78% (14/18, 55-91) |
| Book B (The Unseen Guest) / multihop | 0% (0/10, 0-28) | 0% (0/9, 0-30) | 10% (1/10, 2-40) | 0% (0/10, 0-28) | 10% (1/10, 2-40) |
| Book B (The Unseen Guest) / trap | 100% (14/14, 78-100) | 43% (6/14, 21-67) | 43% (6/14, 21-67) | 71% (10/14, 45-88) | 77% (10/13, 50-92) |
| Book B (The Unseen Guest) / heldout | 0% (0/11, 0-26) | 0% (0/11, 0-26) | 0% (0/11, 0-26) | 0% (0/11, 0-26) | 0% (0/11, 0-26) |
| Book B (The Unseen Guest) / trap60 | 88% (52/59, 77-94) | 69% (40/58, 56-79) | 66% (39/59, 53-77) | 95% (55/58, 86-98) | 83% (49/59, 72-91) |
| Book B (The Unseen Guest) / dropped_direct | 0% (0/18, 0-18) | 11% (2/18, 3-33) | 100% (18/18, 82-100) | 0% (0/18, 0-18) | 100% (18/18, 82-100) |
| Book B (The Unseen Guest) / dropped_paraphrase | 0% (0/1, 0-79) | 0% (0/1, 0-79) | 100% (1/1, 21-100) | 0% (0/1, 0-79) | 100% (1/1, 21-100) |
| Book B (The Unseen Guest) / dropped_multihop | 0% (0/5, 0-43) | 0% (0/5, 0-43) | 40% (2/5, 12-77) | 0% (0/5, 0-43) | 20% (1/5, 4-62) |
| General knowledge probe | 100% (59/59, 94-100) | 95% (57/60, 86-98) | 93% (55/59, 84-97) | 95% (56/59, 86-98) | 93% (56/60, 84-97) |
| Everyday questions answered normally | 100% (40/40, 91-100) | 98% (39/40, 87-100) | 95% (38/40, 83-99) | 100% (40/40, 91-100) | 100% (39/39, 91-100) |

Cells: score (right/graded, 95% Wilson interval). trap / trap60 = share of questions with no answer in the book where the model did NOT invent one. dropped_* = exam questions removed by the leakage filter, scored separately.
Retention of book A (A -> AB): 17 of 18 answers right after the book A stage are still right after book B (94%).
Retention of book A (A2 -> AB2): 21 of 25 answers right after the book A stage are still right after book B (84%).

Exam leakage filter (exam questions too close to any training question are removed before scoring):
- time_machine: 95 -> 64 questions, dropped {'direct': 19, 'paraphrase': 4, 'multihop': 4, 'heldout': 4}
- invisible_man: 95 -> 66 questions, dropped {'direct': 18, 'paraphrase': 1, 'multihop': 5, 'trap': 1, 'heldout': 4}

Judge calibration: 15/15 correct answers accepted, 15/15 planted wrong answers caught.


## What this does not show yet
- One seed. Run-to-run spread needs at least 3 seeds before any number is quoted as stable.
- Two books, about 9,000 words each, not three full novels.
- Replay rows are taken from the teacher's training rows, not from the model's own correct answers.
- The KL anchor is to the clean base only, not to the previous stage.
- The judge calibration covers fact grading; trap and everyday grading are not yet calibrated with planted items.

## Run
```
py -3.14 scripts/01_prepare.py time_machine data/time_machine_raw.txt 9000
py -3.14 scripts/02_exam.py --shared
py -3.14 scripts/02_exam.py time_machine
py -3.14 scripts/03_data.py time_machine
# same three for invisible_man
py -3.14 scripts/06_judge.py --calibrate
py -3.14 scripts/run_all.py            # GPU: base eval, train A, eval A, train A+B, eval A+B
py -3.14 scripts/06_judge.py base A AB
# round 2 (same frozen exam) + extra sets (60 traps per book, exam items removed by the filter)
py -3.14 scripts/03_data.py time_machine --round2      # and invisible_man
py -3.14 scripts/02b_extra.py time_machine             # and invisible_man
py -3.14 scripts/run_all.py --round2
py -3.14 scripts/06_judge.py A2 AB2 base_x A_x AB_x
py -3.14 scripts/07_report.py base A AB A2 AB2
```
Model endpoints are any OpenAI-compatible chat API (`B2W_ENDPOINT`, `B2W_TEACHER`, `B2W_EXAM`, `B2W_JUDGE`).
Source texts: Project Gutenberg #35 and #5230 (public domain).
