# Retrieval analysis (retrieval only, no generation)

52 in-scope gold questions against 1799 chunks from 51 documents. k=5, threshold=0.48.

| Intent | n | exact hit@3 | exact hit@5 | document hit@3 |
|---|---|---|---|---|
| `admin/logistics` | 18 | 0.778 | 0.833 | 1.000 |
| `conceptual` | 18 | 1.000 | 1.000 | 1.000 |
| `code-help` | 16 | 1.000 | 1.000 | 1.000 |
| `ALL` | 52 | 0.923 | 0.942 | 1.000 |

## Questions missing exact hit@3

`@5` and `doc@3` show whether the passage was merely ranked lower, or whether a different — often equally correct — source was returned instead.

| ID | Expected | In top-5? | Right doc in top-3? | Top-3 returned |
|---|---|---|---|---|
| `adm01` | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Requirements | yes | yes | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Assessment – Individual Project (0.346)<br>`Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Assessment – Individual Project (0.338)<br>`Lectures Week 2__HIT summer course 2026_AI and machine learning_teaching schedule_JaneYou` / slide 4 (0.335) |
| `adm10` | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Penalty | no | yes | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Requirements (0.348)<br>`Lectures Week 2__AI notes1_introduction` / slide 47 (0.337)<br>`Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Project topic (0.332) |
| `adm15` | `2026 HIT Global Summer School Academic theme list` / page 1 | no | yes | `2026 HIT Global Summer School Academic theme list` / page 15 (0.351)<br>`2026 HIT Global Summer School Academic theme list` / page 19 (0.350)<br>`2026 HIT Global Summer School Academic theme list` / page 10 (0.347) |
| `adm18` | `Lectures Week 2__HIT summer course_AI and ML_assignment_July2026` / The submission | no | yes | `Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Assessment – Individual Project (0.351)<br>`Lectures Week 2__HIT summer course_AI and ML_assignment_July2026` / Individual Assignment (0.346)<br>`Lectures Week 2__HIT summer course_AI and ML_project_July2026` / Assessment – Individual Project (0.336) |
