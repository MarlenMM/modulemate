# Understanding materials without hiding evidence

## Decision, 19–20 October 2026

The current user's priority is recall and useful understanding: image text,
abstract questions, main ideas, interpretation and follow-up explanations. The
2.3 implementation and its actual failures were read before choosing changes.
There is no evidence that removing source provenance checks would solve these
problems. Nor is a relevance classifier a sound judge of an explanation's truth.

Keep course/document permissions, freshness, real source identities and exact
quotation binding. Treat retrieval relevance and learned answerability as search
signals, never authority to discard an otherwise useful answer. Permit synthesis
across passages, deductions from provided rules, and explicitly labeled examples.
A main idea need not occur as a verbatim sentence in a slide. Missing dates,
permissions, measurements and authors must still remain unknown.

Separate four failure classes in the experiments: extraction (text never entered
the library), retrieval/context (available text was not shown), generation (the
writer misread or over-refused), and citation binding (evidence was lost or given
the wrong identity). A passing quote check proves provenance, not entailment.

## Improvement and verification cycle

- Establish new original cross-subject conversation probes and record the 2.3
  baseline before changes. Keep negative outputs and read complete answers.
- Add bounded offline OCR for image text, preserving native text and page/slide
  locations. Mixed pages and embedded pictures matter as much as empty scans.
  Report coverage gaps, OCR uncertainty and resource limits; never infer diagram
  relationships from OCR labels alone.
- Recognize broad conceptual requests independently of surface keyword matching.
  Read complete short selections, sample across long selections including their
  ends, and clarify genuinely ambiguous document references.
- Make explanatory synthesis explicit in writer and citation instructions.
  Correct only demonstrated errors; preserve useful reasoning and partial help.
- Exercise short evidence, linked/implicit evidence, follow-up continuity,
  distractors, unsupported specifics and deleted/excluded documents.
- Read real local Qwen answers and inspect the real app. Repeat after discovered
  failures. Preserve the original learned artifacts and frozen evaluation split.
- Run required regression, real-weight, lint, dependency and release checks;
  record exact code/data/model identity and practical limitations.

There is no predetermined number of cycles. New observed failures determine the
next change. Finite tests and authored probes cannot establish perfect intelligence.

## Development cycles and decisions

1. **Baseline and provenance.** Replayed 2.3 from its frozen Git tree against new
   original synthesis probes, rather than comparing a running process to files
   edited underneath it. Its short `pH = 7` source caused an answer-validation
   failure: the 12-character quotation minimum had erased real evidence. Whole
   short source bodies now remain citable, while partial/fabricated short quotes
   are still rejected.
2. **Local image text.** Added bounded Apple Vision/Tesseract OCR for scans, mixed
   native/image PDF pages, nested slide pictures, Word pictures and image uploads.
   Native text survives OCR failures. Low recognition confidence retains text
   with a notice, rather than marking its source incredible. Real original scan,
   image and mixed-page checks passed in the isolated parser.
3. **Meaning and context.** Main ideas, takeaways, whole-presentation explanations,
   interpretations and analogies are legitimate operations over source details.
   The extractive answerability head is bypassed for those document overviews;
   it was not trained to judge a synthesized main idea. Broader selection includes
   actual beginning/middle/end passages, with honest sampled-coverage metadata.
   An implicit shared-subject channel recovered an artwork's width from a receipt
   while an explicit assembly-note-only selection correctly withheld that width.
4. **Citation selection and response recovery.** Empty evidence metadata retries
   labeling the existing prose instead of automatically rewriting it into a
   refusal. An actual missing answer may reuse an already searched candidate or
   buy the one wider allowed search, independently of a confidently wrong small
   head. Invalid evidence still cannot authorize an answer. Related passages on
   errors are labeled separately from attached answer citations.
5. **Read actual answers.** The first larger-context run added source-support
   footers and source-binding JSON to normal explanations. Removed only recognized
   auxiliary attachments, preserving requested code, literal arrays and quotations.
   Illustrative follow-ups reuse their concept's previous source context and get
   a visible analogy label. A lower-temperature 0.3 trial invented an averaging
   step and did not repair unsupported prerequisites; it was rejected. Production
   sampling remains 0.7 with the original pretrained model and seed.
6. **Static evidence beyond body text.** Added Word referenced headers/footers,
   footnotes/endnotes and text boxes; native chart caches with category/X/Y/value
   alignment and raw number formats; referenced SmartArt labels; and static
   notebook text, raster outputs and attachments. Nothing runs a notebook,
   interactive widget, workbook or external relationship. Saved results can be
   stale and charts' cached raw numbers are explicitly distinguished from display
   formatting. Mixed instruction/gap passages and actual policy conditions stay
   available to the writer.
7. **Coverage limits.** A 72-section short deck exposed a second failure: its text
   fit the character budget but the 48-passage cap silently dropped its ending.
   Complete-fit selection now considers passage count and estimated tokens as
   well as characters. The ending remains in the structural sample. Increased
   the OCR image allowance from 32 to the document's 500-unit scale while retaining
   the elapsed-time/pixel/process bounds. A real original 65-page scanned PDF
   recovered all 65 locations and the final page in 6.36 seconds on this Mac.
8. **False critic and preparation.** A real model critic turned a missing-budget
   line into the false claim that no evidence was available. Method questions
   such as `What can I prepare?` no longer enter the institutional-policy critic
   just because they contain `can I`. True permission questions still do. Pure
   text transformations also bypass that unrelated review. Extra prompt wording
   alone did not reliably fix the writer's invented prerequisites. Preparation
   now receives a verbatim inventory of provided steps; pure omission-only notes
   can be displayed literally beside the answer, with their own real citation.
   Those notes remain visible to the learner and never become a false prohibition
   on the other steps. Conditional/mixed passages remain in the writer context.
9. **Reimport and real UI.** An original PNG was uploaded through the real browser
   and its recovered text inspected in the app. Old imports cannot be repaired
   from discarded originals. Reimporting changed extraction creates another copy;
   duplicate filenames now show import numbers and passage counts in selection
   and library views. No existing learner document or derived study data is
   silently replaced.
10. **Style is not evidence.** A factual-looking question previously imposed a
    hard 50-word limit and treated any occurrence of `example` or `calculation`
    as an outside addition. Valid conditional rules and a source titled
    `Example Calculation` exposed those false vetoes. Inferred brevity and basis
    labels now guide writing; they cannot erase an otherwise cited answer.
    Explicit sentence counts and requested current-version scope still apply,
    without an invented word limit. Conceptual `Can we regard this as…` questions
    avoid the permission critic. A second, empty writer-status marker is removed
    without changing the leading answer type or literal transformed input.
11. **Outlined letters.** An original PDF made solely of drawn letter paths had
    neither extractable text nor image objects. The importer rejected it, although
    the actual OCR engine read `OPEN AT 42`. Pages without native text now receive
    bounded rendering/OCR too. Actual isolated recognition and the no-image-object
    regression both pass. Blank/vector pages remain subject to the same limits;
    OCR still supplies no diagram geometry or measurement guarantee.
12. **Study facts and preparation transfer.** The separate card validator still
    rejected complete facts shorter than 12 characters after conversation binding
    had been fixed. Cards now accept the complete short source body too; partial
    and changed short backs remain invalid. A fresh `Can I prepare … before I
    have …?` transfer question exposed another false permission route and an
    invented prohibition. Direct preparatory methods now use the available-step
    path, including this ordinary expression of unavailable inputs. The step
    inventory explicitly remains incomplete; source action passages stay available.
13. **Local reasoning trials.** A private-thinking configuration with a 4,000-token
    allowance hit its output limit on two simple fact questions, taking about
    93 seconds each. A corrected 512-token forced-end configuration fixed the
    missing-sponsor answer but failed to finish three of six fresh scope turns
    and misread the contrast between an unlisted name and an explicit absence.
    It was rejected. An initial logits-shape adapter error is excluded from model
    quality comparisons. Direct generation and production sampling are retained;
    these trials do not establish that reasoning is universally unhelpful.
14. **Represent statement scope.** Literal hints now distinguish what a text does
    not identify from a direct negative assertion such as `has no …`. The source
    passages remain available, the hints carry source IDs, and unrecognized
    wording receives no annotation. A first gap-only version overgeneralized
    uncertainty to an explicit negative; the paired representation repaired the
    new contrast case. Physical negatives such as shoes providing no protection
    remain constraints rather than being removed as information gaps. Preparation
    notes already displayed literally are not reintroduced through the hint field.
15. **Preserve literal data and ordinary wording.** Arrays, URLs and angle brackets
    are valid literal card content; exact back/source matching still applies and
    card views render text. Chat URLs are source facts and never authorize fetching
    them. Code fences retain literal markup, while active image/media syntax and
    reasoning envelopes remain blocked outside code. An actual earlier failed
    main-concept request used a misspelled lead-in and `from here`; unqualified
    overview requests now receive structural document coverage too. The cue
    decisions for the 46 original conversation turns remain unchanged.

16. **Check conclusions across turns.** A new three-part, 72-section presentation
    exposed an unsupported reporting prerequisite in an otherwise relevant main
    idea. The next answer repeated it as if it came from the material. Whole-material
    requests now explicitly connect represented topics, separate interpretation
    from reported outcomes and stop after the requested explanation. A first
    wording trial fixed the reporting rule but invented boat safety in a later
    example; it was rejected. More restrictive wording still does not prove that
    the model will avoid every overconfident process claim.
17. **Implications and literal quotations.** `Why does that matter?` should explain
    the relevance of source details, not assume an unexplained historical event.
    Why requests now distinguish a useful implication from a private motive or
    measured causal result, while retaining correction of false premises. A narrow
    check detects an explicit `the text states “…”` or `according to … “…”`
    attribution that does not occur verbatim in the supplied passages. It ignores
    code, invented examples, denied quotations and ordinary interpretive phrasing.
    A strict prototype erased a useful missing-outcome answer and was rejected.
    The adopted path tries one focused correction from the sources, omitting the
    earlier assistant wording that caused the false quote. It preserves a missing
    result if the correction merely changes its type marker. A persistent wording
    mismatch remains visible in the answer notice and Check sources; it is not
    another refusal trigger.
18. **Keep selection and coverage honest.** Actual browser navigation exposed a
    lost document selection after visiting Cards and returning to Assistant.
    Selection now survives page navigation separately for each course and tool.
    A real-widget regression covers both separation and restoration, and the
    browser flow confirms the selected note remains selected. Sampled replies
    now state actual read/available counts, with more detail in Check sources.
    Filtering a retired version no longer makes a complete current selection
    appear partially unread; an explicitly selected ineligible-only file still
    remains distinguishable from having no materials at all.

19. **History is conversation context, not evidence.** The real app's existing
    history still contaminated a fresh main-concept request even after a clean
    fixture passed. The planner correctly marked that request independent, but
    the writer still received old assistant claims. Independent requests now use
    their actual current source context without prior assistant prose. Genuine
    follow-ups retain the conversation segment beginning with the most recent
    independent material answer, under the same history budget. This preserves
    referential continuity while avoiding an unrelated older claim. The regression
    exercises both the fresh request and its subsequent `Why does that matter?`.

20. **Mathematical layout is information too.** A local real-material replay
    regrouped a stacked fraction incorrectly while adding an incidental formula
    to a main-concept answer. No private content was copied into a fixture. An
    independent original PDF places a two-term numerator above a drawn fraction
    bar and a denominator below. Possible stacked PDF notation now receives a
    visible, per-document layout notice and an advisory writer instruction to
    preserve grouping and avoid incidental formula catalogs in broad overviews.
    Page-number footers, prose titles and ordinary code do not trigger the narrow
    detector. Source text stays unchanged and the notice cannot force a refusal.
    Both controlled original-PDF conditions preserved the correct grouping and
    calculation; the advisory also disclosed the layout uncertainty. The repeated
    real-material request gave a broader conceptual explanation without the
    incorrect incidental formula. This does not establish general math recognition
    or perfect formula transcription.

The validation report records final checks, code/model identities, retained
negative experiments and remaining limits. The learned artifacts and frozen
fitting/calibration/test documents were not changed or refitted.

## Sources informing the decision

[pypdf's extraction documentation](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
distinguishes text extraction from OCR. [Apple Vision](https://developer.apple.com/documentation/vision/recognizing-text-in-images)
provides local text recognition. [ALCE](https://aclanthology.org/2023.emnlp-main.398/)
evaluates citations and answer quality separately. [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)
shows that merely supplying a longer context does not establish effective use of
all its information. These motivate experiments here, not claims that the same
results automatically hold for this model or application.
