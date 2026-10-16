# Conversational intelligence rebuild

The 2.1 engine retrieved useful text but still behaved like a sentence search tool.
Its small synthetic fact-coverage scores did not establish that it understood a
question. This rebuild treats that as a product failure, not a wording problem.

The user's clarified goal is a subject-independent chatbot that works well with
the supplied material. It should find, explain, connect, summarize and transform
information, help with the available parts of an exercise, and say what is missing.
Statistics was an early diagnostic, not the product's subject or curriculum.
The interface and answers are English only, following the user's later correction;
translation and language-selection features are outside the product scope.

## Working plan

1. **Local workspace:** remove login, registration, passwords and sessions; migrate
   all existing courses and their dependent data without losing IDs or content.
   Back up old databases before upgrading. Verify a fresh and an upgraded app.
2. **Failure baseline:** reproduce short topic queries and requests to teach an
   entire lecture. Add realistic distractors, multi-part requests, corrections,
   follow-ups, quantitative questions and unsupported policies to the evaluation.
3. **Real generation:** use a prepared local instruction model automatically.
   Preserve an explicitly enabled compatible hosted provider. Separate an honest
   excerpt fallback from the conversational model. Do not silently replace an
   insufficient model answer with a potentially irrelevant quotation.
4. **Evidence assembly:** preserve section boundaries, include headings in search,
   retrieve each part of a question, expand adjacent context, and select whole
   short documents for teaching/summary requests. Budget long contexts explicitly.
5. **Conversation:** resolve follow-ups using actual recent turns, handle topic
   changes and learner answers, obey requested length/format, explain examples and
   calculations, and distinguish supported explanations from missing information.
6. **Grounding and recovery:** validate citations and quotes, preserve refusals,
   repair malformed model output once, report model outages clearly, and avoid
   treating a verifier's opinion as a guarantee. Check source deletion and filters.
7. **Learning experience:** useful full-lecture teaching, contextual exercises,
   feedback on attempts and precise answers to short questions. Make the active
   model and fallback status visible without an opt-in maze for local inference.
8. **Challenge cycles and release:** run end-to-end local-model conversations,
   inspect the actual prose, fix failures in successive cycles, test migration and
   UI flows, update setup/architecture/HIT alignment and publish verified source.

No external GPU training is planned. Retrieval, context, generation, evaluation
and product integration must be improved and measured first. A larger model or
fine-tune is not a substitute for correcting those paths.

## Evaluation rules

- Keep the old benchmarks as historical retrieval/extraction diagnostics only.
- Record question, retrieved sources, actual answer, model identity, latency and
  failure reason. Phrase presence alone is not semantic correctness.
- Include refusals and nearby distractors as well as answerable questions.
- Write a separate transfer challenge before its first run. Record subsequent
  revisions; once used for fixes it is a development regression set.
- Do not publish personal course content or chat logs. Public fixtures are
  original examples; private checks stay outside Git.
- Report measured limits honestly; no claim that all possible improvements or
  general chatbot intelligence have been achieved.

## Cycle log

1. **Workspace migration.** Removed the account UI and runtime authentication API,
   added a schema 3 migration with an online backup, and retained existing course,
   source, chat and study-record identities. The actual local database was checked
   before/after as well as fresh/legacy fixtures. Duplicate course names remain usable.
2. **Failure baseline and model preparation.** Reproduced the short-topic and
   full-lecture failures. Prepared pinned local 9B and 4B instruction weights,
   retained optional providers, and kept excerpt mode explicitly limited. No
   external GPU training or private-coursework fine-tuning was used.
3. **Evidence assembly.** Preserved section/code boundaries and headings, expanded
   neighbors and comparison/subquestion retrieval, resolved filenames and lecture
   numbers, and fixed identifier filters that excluded later pages of a named
   document. Bounded conversation history retains the latest topic and check
   question. An early query-translation experiment was removed for English-only use.
4. **Verified numeric work.** Added bounded AST arithmetic and descriptive statistics,
   odd/even medians, ties, sample/population distinctions and undefined singleton
   sample variance. Fixed truncated long lists, multiple-data-set confusion,
   fractional-input misparsing, ambiguous thousands commas and rounded large means.
5. **Conversation actions.** A quiz request now outranks comparison keywords inside
   its topic. Short learner replies remain quiz answers across feedback turns.
   Ordered references are resolved to their actual concept before generation.
   Explicit short formats are checked and repaired, rather than merely requested.
6. **Generation format.** Long JSON answers leaked protocol fields or failed parsing.
   Local writing now happens separately from evidence/state labeling. Labels,
   reviews and cards use constrained decoding; metadata repair retains valid prose.
   A direct Outlines Core adapter preserves partial UTF-8 vocabulary bytes and
   removes the unused vulnerable DiskCache dependency found during the audit.
7. **Reviewer correction.** Actual traces showed the model critic inventing dates,
   objecting to correct arithmetic and rejecting labeled examples. Refusals now
   bypass it; other negative reviews need matching draft/source quotes. General
   lessons/definitions bypass that noisy critic. Targeted course-fact/code review
   remains an additional check, not a truth guarantee.
8. **Teaching progression.** An automatically passing lesson still only explained
   two concepts while listing the rest. Parts now have a few concrete topic targets
   and evidence coverage checks. Continuation retains its place through intervening
   questions. Long outlines preview later pages too; the lesson never claims that
   a roadmap establishes full teaching coverage.
9. **Evidence and example correctness.** A sound lesson failed when the labeler
   spliced two exact source sentences while silently dropping an intervening one.
   The local labeling stage now selects app-generated contiguous excerpts by ID,
   instead of writing quote text. Manual reading also caught an incorrect even-count
   median that phrase checks accepted. Explicit statistic-of-list assertions have
   a narrow numeric check; lists ending in “and” retain their final observation.
   A subject-specific precomputed-lesson experiment was later removed when the
   user clarified the general material-based goal. Numeric tools remain available
   for explicitly supplied data.
10. **Transfer conversation state.** A previously unseen one-term query exposed a
    planner crash: one model-labeled subject was treated as a pair. Only actual
    comparisons now need two-sided reference clarification. Shortening a lesson
    preserves its topic, while an explicit new concept in one sentence starts a
    new topic. Legacy empty schema 1/2 workspaces also received migration checks.
11. **Transfer relevance and response type.** An “everyday example” lost the prior
    concept, while unrelated outline headings distracted the writer. Such requests
    now resolve their subject explicitly; empty focused retrieval carries no
    unrelated catalog. The evidence labeler also reclassified a correct computed
    explanation as a clarification. Response type now belongs to the writer and
    cannot be changed by source labeling. Current-only fact requests have bounded
    scope checks; permission requests receive targeted evidence review.
12. **General material-based assistant.** After the user corrected the excessive
    statistics focus, the main evaluation changed to nine original documents
    spanning seven areas. The default answer policy now requires supplied evidence
    and reports gaps instead of supplying outside factual answers. The assistant
    can transform text/code supplied in a question, explain supported steps of an
    incomplete exercise, and preserve a partially answerable response with its
    limits. Automatic statistics-lesson example generation was removed.
13. **English-only relevance checks.** Removed query translation and language
    routing. Document focus now uses distinctive topic terms before passage
    retrieval, preventing two unrelated events' supply lists from being combined.
    Comparisons retain both subjects; requests across all documents retain their
    full selected scope. The main challenge uses English across several subjects.
14. **Answer/evidence recovery.** Reading the first full English run found valid
    prose rejected because writer citations disagreed with evidence selection, and
    a missing-information answer rejected after its citations were forcibly removed.
    The evidence pass now owns final numbering, and missing answers may retain
    cited facts about a gap. Current-only context omits explicitly retired files.
    Text rewriting uses a transformation task instead of Explain-mode commentary;
    direct conditions, false premises and partial support have clearer instructions.
15. **User-owned conversation state and targeted tasks.** Actual answers exposed
    model-generated comparison aspects overwriting the user's ordered entities.
    Explicit parsed subjects now take precedence. Polite transformation requests
    are not treated as short yes/no facts. Fact lookup, checklist extraction and
    why-questions receive tasks matching their requested action. Missing-information
    answers with incidental cited claims can receive a bounded factual correction;
    that correction cannot turn the missing fact into an invented answer.
    A four-case local trial of capped 512-token thinking was slower and still
    misread a false premise, so it was not integrated as a default.
16. **Source presentation integrity.** A model-written bibliography invented a
    quoted sentence although its separately selected source span was real. Local
    prose now drops unsolicited source footers before evidence labeling; the app's
    source panel supplies actual text. Explicit source/quote requests and fenced
    code are preserved. Harmless closing reasoning tags and minor response-marker
    typos are normalized; actual reasoning blocks remain rejected.
17. **Rejected broader critic.** A real five-turn trial added review of comparisons
    and source-attributed illustrations. It approved the unsupported details it was
    meant to catch, adding cost without the intended benefit. The trial was reverted
    to the previously validated source. Those semantic errors remain documented;
    passing model review must not be presented as correctness certification.

The core and transfer challenges retain actual answers, source references, model
calls, timings and source/fixture fingerprints. Their word/phrase checks are only
diagnostics: examples encountered in these cycles passed those checks while having
bad prose or incomplete teaching. Final measurements and remaining limits belong
in [the 2.2 validation report](VALIDATION_V22.md).
