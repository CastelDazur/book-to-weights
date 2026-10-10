# Results

| Set | base | AB2 | A3 | AB3 | AB4 |
|---|---|---|---|---|---|
| Book A (The Chrono Engine) / direct | 20% (2/10, 6-51) | 90% (9/10, 60-98) | 100% (10/10, 72-100) | 70% (7/10, 40-89) | 90% (9/10, 60-98) |
| Book A (The Chrono Engine) / paraphrase | 6% (1/16, 1-28) | 75% (12/16, 51-90) | 69% (11/16, 44-86) | 62% (10/16, 39-82) | 69% (11/16, 44-86) |
| Book A (The Chrono Engine) / multihop | 0% (0/11, 0-26) | 36% (4/11, 15-65) | 46% (5/11, 21-72) | 9% (1/11, 2-38) | 27% (3/11, 10-57) |
| Book A (The Chrono Engine) / trap | 100% (14/14, 78-100) | 93% (13/14, 69-99) | 93% (14/15, 70-99) | 93% (14/15, 70-99) | 100% (15/15, 80-100) |
| Book A (The Chrono Engine) / heldout | 9% (1/11, 2-38) | 9% (1/11, 2-38) | 0% (0/10, 0-28) | 0% (0/10, 0-28) | 0% (0/10, 0-28) |
| Book A (The Chrono Engine) / trap60 | 91% (53/58, 81-96) | 81% (48/59, 70-89) | 92% (54/59, 82-96) | 88% (52/59, 77-94) | 86% (51/59, 75-93) |
| Book A (The Chrono Engine) / dropped_direct | 5% (1/19, 1-25) | 89% (16/18, 67-97) | 84% (16/19, 62-94) | 68% (13/19, 46-85) | 79% (15/19, 57-91) |
| Book A (The Chrono Engine) / dropped_paraphrase | 0% (0/3, 0-56) | 50% (2/4, 15-85) | 75% (3/4, 30-95) | 25% (1/4, 5-70) | 75% (3/4, 30-95) |
| Book A (The Chrono Engine) / dropped_multihop | 0% (0/4, 0-49) | 100% (4/4, 51-100) | 100% (4/4, 51-100) | 75% (3/4, 30-95) | 75% (3/4, 30-95) |
| Book B (The Unseen Guest) / direct | 0% (0/12, 0-24) | 83% (10/12, 55-95) | - | 75% (9/12, 47-91) | 75% (9/12, 47-91) |
| Book B (The Unseen Guest) / paraphrase | 0% (0/18, 0-18) | 78% (14/18, 55-91) | - | 84% (16/19, 62-94) | 79% (15/19, 57-91) |
| Book B (The Unseen Guest) / multihop | 0% (0/10, 0-28) | 10% (1/10, 2-40) | - | 10% (1/10, 2-40) | 10% (1/10, 2-40) |
| Book B (The Unseen Guest) / trap | 100% (14/14, 78-100) | 77% (10/13, 50-92) | - | 77% (10/13, 50-92) | 77% (10/13, 50-92) |
| Book B (The Unseen Guest) / heldout | 0% (0/11, 0-26) | 0% (0/11, 0-26) | - | 0% (0/11, 0-26) | 0% (0/11, 0-26) |
| Book B (The Unseen Guest) / trap60 | 88% (52/59, 77-94) | 83% (49/59, 72-91) | - | 88% (52/59, 77-94) | 93% (55/59, 84-97) |
| Book B (The Unseen Guest) / dropped_direct | 0% (0/18, 0-18) | 100% (18/18, 82-100) | - | 100% (18/18, 82-100) | 94% (17/18, 74-99) |
| Book B (The Unseen Guest) / dropped_paraphrase | 0% (0/1, 0-79) | 100% (1/1, 21-100) | - | 100% (1/1, 21-100) | 100% (1/1, 21-100) |
| Book B (The Unseen Guest) / dropped_multihop | 0% (0/5, 0-43) | 20% (1/5, 4-62) | - | 40% (2/5, 12-77) | 40% (2/5, 12-77) |
| General knowledge probe | 100% (59/59, 94-100) | 93% (56/60, 84-97) | - | 100% (60/60, 94-100) | 98% (59/60, 91-100) |
| Everyday questions answered normally | 100% (40/40, 91-100) | 100% (39/39, 91-100) | - | 100% (39/39, 91-100) | 100% (39/39, 91-100) |

Cells: score (right/graded, 95% Wilson interval). trap / trap60 = share of questions with no answer in the book where the model did NOT invent one. dropped_* = exam questions removed by the leakage filter, scored separately.
Retention of book A (A3 -> AB3): 15 of 26 answers right after the book A stage are still right after book B (58%).
Retention of book A (A3 -> AB4): 21 of 26 answers right after the book A stage are still right after book B (81%).

Exam leakage filter (exam questions too close to any training question are removed before scoring):
- time_machine: 95 -> 64 questions, dropped {'direct': 19, 'paraphrase': 4, 'multihop': 4, 'heldout': 4}
- invisible_man: 95 -> 66 questions, dropped {'direct': 18, 'paraphrase': 1, 'multihop': 5, 'trap': 1, 'heldout': 4}

Judge calibration: 15/15 correct answers accepted, 15/15 planted wrong answers caught.
