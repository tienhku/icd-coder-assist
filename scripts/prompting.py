"""Shared prompt template and response-parsing logic for ICD coding.

Used by both eval/run_baseline.py (zero-shot and fine-tuned eval) and
scripts/prepare_finetune_data.py (fine-tuning data prep), so the training
prompt and the evaluation prompt can never silently drift apart.
"""
import json
import re

# ICD-10-CM diagnosis code: letter, 2 digits, optional dot + up to 4 alnum.
CODE_RE = re.compile(r"\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)\b")

PROMPT_TEMPLATE = """You are an expert medical coder assigning ICD-10-CM diagnosis codes \
from a Taiwanese hospital discharge summary. Read the note below and list every ICD-10-CM \
diagnosis code that applies (principal + secondary diagnoses), most important first.

Respond with ONLY a JSON array of code strings, e.g. ["E11.9", "I10", "J18.9"]. No prose, \
no explanation.

Discharge note:
{note_text}

JSON array of ICD-10-CM codes:"""


# RAG variant: same instruction and output contract as PROMPT_TEMPLATE, prefixed with retrieved
# similar cases. Kept parse-compatible (still ends "JSON array ...:") so parse_codes and scoring are
# unchanged — only the context differs, which is what we're measuring. Retrieved codes are shown as
# the target output format so the model also learns the response shape from the examples.
RAG_PROMPT_HEADER = """You are an expert medical coder assigning ICD-10-CM diagnosis codes \
from a Taiwanese hospital discharge summary. Below are similar past cases with their correct \
codes, then a new note to code.

Respond with ONLY a JSON array of code strings, e.g. ["E11.9", "I10", "J18.9"]. No prose, \
no explanation.

"""

RAG_EXAMPLE_TEMPLATE = """Past case {i}:
{note_text}
Codes: {codes}

"""

RAG_PROMPT_FOOTER = """Now code this new note:
{note_text}

JSON array of ICD-10-CM codes:"""


def build_rag_prompt(note_text: str, examples: list[tuple[str, list[str]]]) -> str:
    """examples: list of (retrieved_note_text, [codes]) ordered most-similar first."""
    import json as _json

    parts = [RAG_PROMPT_HEADER]
    for i, (ex_note, ex_codes) in enumerate(examples, 1):
        parts.append(
            RAG_EXAMPLE_TEMPLATE.format(i=i, note_text=ex_note, codes=_json.dumps(sorted(ex_codes)))
        )
    parts.append(RAG_PROMPT_FOOTER.format(note_text=note_text))
    return "".join(parts)


# --- Code-description-retrieval RAG (GAVS-style, experiment A) ---
# Stage 1: extract diagnostic entities from the note (so short code descriptions can be matched to
# short entity phrases, not to the whole long note — the length-asymmetry fix from Sarvari 2025).
ENTITY_EXTRACTION_PROMPT = """List every distinct diagnosis, condition, and reason-for-admission \
mentioned in this discharge note, as short medical phrases (one per line, no codes, no explanation). \
Include comorbidities and complications.

Discharge note:
{note_text}

Diagnoses (one per line):"""

# Stage 2: assign final codes from the retrieved candidate (code: description) list.
DESC_RAG_HEADER = """You are an expert medical coder assigning ICD-10-CM diagnosis codes from a \
hospital discharge note. Below is a list of CANDIDATE ICD-10-CM codes with their official \
descriptions, retrieved as likely matches for this note. Select every candidate code that the note \
supports (principal + secondary diagnoses); you may ignore candidates that do not apply.

Respond with ONLY a JSON array of code strings, e.g. ["E11.9", "I10"]. No prose.

Candidate ICD-10-CM codes:
"""

DESC_RAG_FOOTER = """
Discharge note:
{note_text}

JSON array of applicable ICD-10-CM codes:"""


def build_desc_rag_prompt(note_text: str, candidates: list[tuple[str, str]]) -> str:
    """candidates: list of (code, description), ordered most-relevant first."""
    lines = [f"{code}: {desc}" for code, desc in candidates]
    return DESC_RAG_HEADER + "\n".join(lines) + DESC_RAG_FOOTER.format(note_text=note_text)


# --- Agentic pipeline (experiment: agent methods) ---
# Verify agent: a "senior coder review" role that judges each proposed candidate against the note,
# grounded in the code description and retrieval evidence (how many similar past cases used it). This
# targets the v2 bottleneck — selecting which codes apply from a high-recall candidate pool.
AGENT_VERIFY_PROMPT = """You are a senior medical coder reviewing candidate ICD-10-CM codes proposed \
for a discharge note. Each candidate shows its official description and how many of the 100 most \
clinically-similar past admissions were assigned that code (retrieval support). Keep a code ONLY if \
the note itself supports it as a diagnosis for THIS admission; drop codes that are not supported, \
even if similar cases had them.

Respond with ONLY a JSON array of the codes that apply, e.g. ["E11.9", "I10"]. No prose.

Candidate codes (code — description — retrieval support):
{candidates}

Discharge note:
{note_text}

JSON array of applicable ICD-10-CM codes:"""

# Complete agent: catch diagnoses the candidate pool missed (retrieval can't reach rare codes).
AGENT_COMPLETE_PROMPT = """You are a medical coder. The note below has already been assigned these \
ICD-10-CM codes: {assigned}. List any ADDITIONAL distinct diagnoses or conditions stated in the note \
that are NOT already covered, as short medical phrases (one per line, no codes). If none, output \
nothing.

Discharge note:
{note_text}

Additional diagnoses (one per line):"""


# Lenient verify: the strict agent keeps only 23% of valid candidates on diagnosis-stripped notes
# (it can't "confirm" a diagnosis that isn't explicitly stated). The lenient agent keeps unless
# clearly contradicted — trading precision for recall, to map the agent's operating curve.
AGENT_VERIFY_LENIENT_PROMPT = """You are a medical coder assigning ICD-10-CM codes to a discharge \
note. Below are candidate codes, each with its description and how many of the 100 most \
clinically-similar past admissions used it. The note may not state diagnoses explicitly, so infer \
from symptoms, history, examination, course, and findings. Keep a candidate if the note is \
CONSISTENT with it or gives ANY supporting evidence; drop only codes clearly contradicted or \
unrelated. When unsure, keep it.

Respond with ONLY a JSON array of the codes that apply. No prose.

Candidate codes (code — description — retrieval support):
{candidates}

Discharge note:
{note_text}

JSON array of applicable ICD-10-CM codes:"""


# Reflexion refine (path A): a second self-critique pass over the agent's own draft, to recover codes
# it omitted and drop ones it should not have kept — the canonical iterate-and-revise agent pattern.
AGENT_REFINE_PROMPT = """You are a senior medical coder revising your own draft ICD-10-CM coding of a \
discharge note. Re-read the note and the candidate list, then CORRECT your draft: add any candidate \
the note supports that you omitted, and remove any you included that the note does not actually \
support. Be thorough — missing a supported diagnosis is as bad as adding an unsupported one.

Your draft codes: {draft}

Candidate codes (code — description — retrieval support):
{candidates}

Discharge note:
{note_text}

Corrected JSON array of applicable ICD-10-CM codes:"""


def build_agent_refine_prompt(note_text, candidates, draft):
    import json as _json
    cand = "\n".join(f"{c} — {d} — support {s}/100" for c, d, s in candidates)
    return AGENT_REFINE_PROMPT.format(draft=_json.dumps(sorted(draft)), candidates=cand, note_text=note_text)


# Strict case-grounded verify (path A): case evidence + a strict keep-only-if-supported rule, aiming
# for a higher-precision agent that is a better union partner for the supervised head.
AGENT_VERIFY_CASES_STRICT_PROMPT = """You are a senior medical coder reviewing candidate ICD-10-CM \
codes for a discharge note. The most clinically-similar past admissions are shown as coding precedent, \
then the candidates, then the note. Keep a candidate ONLY if the note itself supports it as a diagnosis \
for THIS admission; the similar cases are precedent, not proof. Drop codes not supported by the note.

Respond with ONLY a JSON array of applicable codes. No prose.

Most similar past admissions:
{cases}

Candidate codes (code — description — retrieval support):
{candidates}

Note to code:
{note_text}

JSON array of applicable ICD-10-CM codes:"""


# Case-grounded verify (path A): instead of bare support counts, show the agent the most similar past
# admissions with their notes and assigned codes, so it can reason "this similar patient with these
# findings was coded X, and the current note matches." Combines RAG's case evidence with the candidate
# structure — a richer agent that may move the precision/recall curve outward rather than along it.
AGENT_VERIFY_CASES_PROMPT = """You are a medical coder assigning ICD-10-CM codes to a discharge note. \
Below are the most clinically-similar past admissions with their assigned codes (use them as coding \
precedent), then candidate codes, then the note to code. Keep a candidate if the note is consistent \
with it — the similar cases show how comparable patients were coded. Infer from symptoms, history, \
exam, course, and findings; the note may not state diagnoses explicitly.

Respond with ONLY a JSON array of applicable codes. No prose.

Most similar past admissions:
{cases}

Candidate codes (code — description — retrieval support):
{candidates}

Note to code:
{note_text}

JSON array of applicable ICD-10-CM codes:"""


def build_agent_cases_verify_prompt(note_text, candidates, cases, strict=False):
    """cases: list of (similarity, note_excerpt, [codes]) for the top similar admissions."""
    import json as _json
    cand = "\n".join(f"{c} — {d} — support {s}/100" for c, d, s in candidates)
    cs = "\n".join(
        f"Case {i} (similarity {sim:.2f}): {ex}\n  → codes: {_json.dumps(sorted(cds))}"
        for i, (sim, ex, cds) in enumerate(cases, 1)
    )
    tmpl = AGENT_VERIFY_CASES_STRICT_PROMPT if strict else AGENT_VERIFY_CASES_PROMPT
    return tmpl.format(cases=cs, candidates=cand, note_text=note_text)


# --- Multi-role decomposition (physician -> coder -> reviewer), per the 5-agent literature ---
# Role 1 (physician): read the note and state the admission's active problem list as clinical facts,
# so the downstream coder reasons over a distilled problem list rather than the raw 9k-char record.
AGENT_PHYSICIAN_PROMPT = """You are the attending physician summarizing this admission for the coding \
department. State the complete problem list for THIS admission: every diagnosis, condition, \
complication, and comorbidity that was active, treated, or monitored, as short clinical phrases (one \
per line, no codes, no explanation). Infer from the history, examination, hospital course, labs, \
imaging, and pathology.

Discharge note:
{note_text}

Problem list (one per line):"""

# Role 2 (coder): map the physician's problem list onto the candidate ICD-10-CM codes.
AGENT_CODER_PROMPT = """You are a certified ICD-10-CM coder. The attending physician has provided the \
problem list for this admission. Using the candidate codes below (each with its official description \
and how many of the 100 most clinically-similar past admissions used it), assign every candidate code \
that matches a problem on the list. Match on clinical meaning, not exact wording. Choose ONLY from the \
candidate codes.

Physician problem list:
{problems}

Candidate codes (code - description - retrieval support):
{candidates}

Respond with ONLY a JSON array of applicable ICD-10-CM codes. No prose.

JSON array of ICD-10-CM codes:"""

# Role 3 (reviewer/auditor): audit the coder's assignment against the note before submission.
AGENT_REVIEWER_PROMPT = """You are a senior coding auditor reviewing a coder's ICD-10-CM assignment \
before it is submitted. For each assigned code, confirm the note documents it for THIS admission; \
remove any code the documentation does not support. Do not add new codes.

Coder's assigned codes: {assigned}

Candidate codes (code - description - retrieval support):
{candidates}

Discharge note:
{note_text}

Respond with ONLY the final audited JSON array of ICD-10-CM codes. No prose.

Final JSON array of ICD-10-CM codes:"""


def build_agent_physician_prompt(note_text):
    return AGENT_PHYSICIAN_PROMPT.format(note_text=note_text)


def build_agent_coder_prompt(problems, candidates):
    cand = "\n".join(f"{c} - {d} - support {s}/100" for c, d, s in candidates)
    return AGENT_CODER_PROMPT.format(problems=problems, candidates=cand)


def build_agent_reviewer_prompt(note_text, candidates, assigned):
    import json as _json
    cand = "\n".join(f"{c} - {d} - support {s}/100" for c, d, s in candidates)
    return AGENT_REVIEWER_PROMPT.format(assigned=_json.dumps(sorted(assigned)), candidates=cand, note_text=note_text)


def build_agent_verify_prompt(note_text, candidates, mode="strict"):
    """candidates: list of (code, description, support_count), ordered most-supported first."""
    lines = [f"{c} — {d} — support {s}/100" for c, d, s in candidates]
    tmpl = AGENT_VERIFY_LENIENT_PROMPT if mode == "lenient" else AGENT_VERIFY_PROMPT
    return tmpl.format(candidates="\n".join(lines), note_text=note_text)


def parse_entities(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip().lstrip("-*0123456789. ").strip()
        if s and len(s) > 2:
            out.append(s)
    return out


def parse_codes(text: str) -> set[str]:
    # Try JSON array first.
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            codes = {str(c).strip().upper() for c in arr if isinstance(c, str)}
            codes = {c for c in codes if CODE_RE.fullmatch(c)}
            if codes:
                return codes
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: regex-scan the raw text for anything code-shaped.
    return {c.upper() for c in CODE_RE.findall(text)}
