# Retrieval analysis (retrieval only, no generation)

52 in-scope gold questions against 1799 chunks from 51 documents. k=5, threshold=0.48.

| Intent | n | exact hit@3 | exact hit@5 | document hit@3 |
|---|---|---|---|---|
| `admin/logistics` | 18 | 0.722 | 0.778 | 0.944 |
| `conceptual` | 18 | 0.833 | 0.833 | 0.944 |
| `code-help` | 16 | 0.812 | 0.875 | 0.938 |
| `ALL` | 52 | 0.788 | 0.827 | 0.942 |

## Questions missing exact hit@3

`@5` and `doc@3` show whether the passage was merely ranked lower, or whether a different — often equally correct — source was returned instead.

| ID | Expected | In top-5? | Right doc in top-3? | Top-3 returned |
|---|---|---|---|---|
| `adm05` | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Penalty | yes | yes | `chats__HIT 2026 Summer Course AI and ML` / part 11 (0.581)<br>`Lectures Week 2__HIT summer course_AI and ML_assignment_July2026` / The submission (0.525)<br>`Lectures Week 2__HIT summer course_AI and ML_project_July2026` / The submission (0.504) |
| `adm10` | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Penalty | no | yes | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Requirements (0.516)<br>`Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Project topic (0.480)<br>`Lectures Week 1__Burnaev_EngAI_v2` / page 43 (0.473) |
| `con11` | `Lectures Week 2__AI notes5_machine learning_part 1` / slide 8 | no | yes | `Lectures Week 2__AI notes5_machine learning_part 1` / slide 24 (0.662)<br>`Lectures Week 2__AI notes5_machine learning_part 1` / slide 3 (0.660)<br>`Lectures Week 1__Lecture_1` / page 2 (0.637) |
| `cod01` | `Lectures Week 2__AI notes6_machine learning_part 2` / slide 22 | no | yes | `Lectures Week 2__AI notes6_machine learning_part 2` / slide 18 (0.761)<br>`Lectures Week 2__AI notes6_machine learning_part 2` / slide 9 (0.657)<br>`Lectures Week 2__AI notes6_machine learning_part 2` / slide 8 (0.652) |
| `cod07` | `Lectures Week 2__AI notes6_machine learning_part 2` / slide 78 | no | yes | `Lectures Week 2__AI notes6_machine learning_part 2` / slide 76 (0.585)<br>`Lectures Week 1__Lecture_3_Classification` / page 30 (0.547)<br>`Programming_Exercise_Clustering` / What you will do (0.516) |
| `con16` | `Programming_Exercise_Clustering` / What you will do | no | yes | `Programming_Exercise_Clustering` / Retrieving Wikipedia articles (0.621)<br>`Programming_Exercise_Clustering` / Retrieving Wikipedia articles (0.613)<br>`Programming_Exercise_Clustering` / Retrieving Wikipedia articles (0.602) |
| `con17` | `Lectures Week 1__Lecture_2_Regression` / page 6 | no | no | `Lectures Week 2__AI notes5_machine learning_part 1` / slide 54 (0.553)<br>`Lectures Week 2__AI notes5_machine learning_part 1` / slide 27 (0.506)<br>`Lectures Week 2__AI notes5_machine learning_part 1` / slide 100 (0.505) |
| `adm14` | `2026 HIT Global Summer School Academic theme list` / page 1 | no | yes | `2026 HIT Global Summer School` / page 1 (0.597)<br>`2026 HIT Global Summer School Academic theme list` / page 2 (0.469)<br>`2026 HIT Global Summer School Academic theme list` / page 7 (0.460) |
| `adm15` | `2026 HIT Global Summer School Academic theme list` / page 1 | no | yes | `2026 HIT Global Summer School Academic theme list` / page 7 (0.457)<br>`2026 HIT Global Summer School Academic theme list` / page 10 (0.439)<br>`2026 HIT Global Summer School Academic theme list` / page 4 (0.434) |
| `adm18` | `Lectures Week 2__HIT summer course_AI and ML_assignment_July2026` / The submission | no | no | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Assessment – Individual Project (0.583)<br>`chats__HIT 2026 Summer Course AI and ML` / part 4 (0.572)<br>`Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Assessment – Individual Project (0.513) |
| `cod16` | `Lectures Week 2__HIT summer course_AI and ML_assignment_July2026` / Part C (10 marks) | yes | no | `Lectures Week 2__AI notes6_machine learning_part 2` / slide 9 (0.637)<br>`Lectures Week 2__AI notes6_machine learning_part 2` / slide 12 (0.593)<br>`Lectures Week 2__AI notes6_machine learning_part 2` / slide 7 (0.586) |
