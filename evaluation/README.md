# Original evaluation material

`general_documents.json` is the primary 2.2 conversation development set. Its nine
original fictional documents cover events, history, biology, project requirements,
practical notices, code and fiction. Requests and expected responses are English.
It checks precise facts, conflicting versions, negative/conditional requirements,
partial exercise help, missing evidence, follow-ups, text rewriting and hostile
source instructions. Run `scripts/evaluate_conversations.py --suite general`.
The runner records actual prose, source context, model calls, latency and code/data
fingerprints in ignored `results/`. Phrase checks alone do not establish correctness.

`conversation_challenge.json` and `conversation_transfer.json` retain earlier
development cases. Some predate the clarified source-only answer policy and are
historical experiments, rather than the current product acceptance criteria.

The following retrieval/extraction observations describe the earlier 2.1 pipeline.

These synthetic documents and utterances were written for this upgrade, not
copied from HIT materials, user libraries, or the old gold set. Facts are simple
teaching examples and invented course logistics. Intent training lives separately
in `modulemate/intent_training.json`; evaluation utterances are not fitted.

The retrieval set is an engineering development benchmark, not a blinded test or
estimate of general accuracy. It deliberately includes paraphrases, lexical
matches, near-topic distractors, missing topics and multilingual word matches.
The conversation regressions check observable behavior rather than an LLM judge.
Run both light and real-model suites; only the real-model run supports neural
retrieval claims. Reports include all misses and denominator counts.


`answer_checks.json` defines reference facts for the 46 development questions.
`answer_challenge.json` contains 20 additional questions written after that pass;
no thresholds were fitted to the challenge results. `answers-local.json` and
`answers-challenge-local.json` retain every answer and miss. Passing a fact check
only means all specified phrases occur in the answer; it does not prove that
all wording, emphasis or implied reasoning is correct. Citation-marker checks
verify resolvability, not semantic citation correctness.

`results-initial.json` records the first attempt where a hard passage-reranker
threshold harmed recall. `results-local.json` is the final retrieval ablation;
`results-multiqa-candidate.json` is the rejected alternative encoder experiment.
`answers-without-reader.json` records the sentence pipeline ablation before the
final within-source reader constraint; it is historical development evidence,
not a current apples-to-apples causal estimate of reader benefit.
