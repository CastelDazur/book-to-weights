# Results

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
