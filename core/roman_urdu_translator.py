"""Roman Urdu translation pipeline with API and semantic-quality retries."""

from __future__ import annotations

import json
import logging
import os
from difflib import SequenceMatcher
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_provider import get_chat_model as _provider_get_chat_model

load_dotenv(override=True)
_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_REVIEWER_MODEL = os.getenv("MISTRAL_REVIEWER_MODEL", MISTRAL_MODEL)
MISTRAL_TEMPERATURE = 0.1          # Translator temperature
MISTRAL_REVIEW_TEMPERATURE = 0.0   # Deterministic semantic-review temperature
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_TIMEOUT = 120.0
MISTRAL_MAX_RETRIES = 0            # Shared _invoke_llm_with_retries owns retries

API_MAX_ATTEMPTS = 5
QUALITY_MAX_ATTEMPTS = 3
API_BASE_DELAY = 3.0
API_MAX_DELAY = 120.0
QUALITY_RETRY_DELAY = 1.0

# ---- HTTP 429 / rate-limit handling --------------------------------------
# Rate limits get their own longer backoff so a 429 never uses the short
# generic schedule. All values are env-overridable (simple float parse).
def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BASE_DELAY = _env_float("RATE_LIMIT_BASE_DELAY", 5.0)        # first wait ~5s
RATE_LIMIT_BACKOFF_FACTOR = _env_float("RATE_LIMIT_BACKOFF_FACTOR", 2.0)
RATE_LIMIT_MAX_DELAY = _env_float("RATE_LIMIT_MAX_DELAY", 120.0)        # hard ceiling
RATE_LIMIT_JITTER_FACTOR = _env_float("RATE_LIMIT_JITTER_FACTOR", 0.2)  # + up to 20%
RETRY_AFTER_CEILING = _env_float("RETRY_AFTER_CEILING", 120.0)          # Retry-After cap

DEFAULT_CHUNK_SIZE = 3500

# Minimum combined (deterministic + LLM) quality score required to pass review.
MIN_QUALITY_SCORE = 75

# If this fraction (or more) of the words in a translation are ordinary,
# untranslated English words (not allowed technical terms), record a stronger
# style diagnostic. Ordinary-English detection is never a deterministic failure.
EXCESSIVE_ENGLISH_RATIO_THRESHOLD = 0.35

# Minimum fraction of *unresolved* words that must themselves be common
# English function words before a fragment may be classified as genuinely
# untranslated English. Positive-evidence guard: absence from the (necessarily
# incomplete) Roman Urdu vocabulary set alone never classifies text as English.
GENUINE_ENGLISH_EVIDENCE_RATIO_THRESHOLD = 0.35

# Conservative cloud output cap derived from the default transcript chunk size.
MISTRAL_MAX_TOKENS = int(
    os.getenv("MISTRAL_MAX_TOKENS", "4096")
)

# ---------------------------------------------------------------------------
# Forbidden Unicode ranges (Urdu / Arabic / Devanagari / Hindi scripts)
# ---------------------------------------------------------------------------
# Any character in these ranges means the output is NOT Roman Urdu.

_FORBIDDEN_UNICODE_RANGES: list[tuple[int, int]] = [
    (0x0600, 0x06FF),   # Arabic & Urdu script (Arabic block)
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    (0x0900, 0x097F),   # Devanagari (Hindi/Sanskrit script)
    (0x0980, 0x09FF),   # Bengali
]


def _contains_forbidden_script(text: str) -> bool:
    """Return True if *text* contains any Urdu/Arabic/Devanagari characters."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _FORBIDDEN_UNICODE_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _contains_non_latin_alphabetic(text: str) -> bool:
    """Return True for alphabetic characters outside the ASCII Latin alphabet."""
    return any(ch.isalpha() and not ch.isascii() for ch in text)


# Grouped ranges with descriptive labels for human-readable error messages.
_SCRIPT_GROUPS: list[tuple[str, list[tuple[int, int]]]] = [
    (
        "Urdu/Arabic script",
        [
            (0x0600, 0x06FF),   # Arabic & Urdu
            (0x0750, 0x077F),   # Arabic Supplement
            (0x08A0, 0x08FF),   # Arabic Extended-A
            (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
        ],
    ),
    (
        "Devanagari/Hindi script",
        [
            (0x0900, 0x097F),   # Devanagari
        ],
    ),
    (
        "Bengali script",
        [
            (0x0980, 0x09FF),   # Bengali
        ],
    ),
]


def validate_roman_urdu_script(text: str) -> dict[str, Any]:
    """
    Validate that *text* contains no forbidden non-Latin scripts.

    Returns a dict with:
      passed   (bool) -- True only when no forbidden script is found.
      errors   (list) -- Human-readable messages for each detected script family.
      feedback (str)  -- Single combined error string (empty when passed).

    Validation rules
    ----------------
    * Urdu/Arabic Unicode characters  --> FAIL
    * Devanagari/Hindi characters     --> FAIL
    * Bengali characters              --> FAIL
    * Latin characters, digits, punctuation, timestamps, English technical
      terms, and common programming symbols --> PASS

    This function must be called BEFORE semantic LLM review so that a
    forbidden-script output is never forwarded to the reviewer.

    Examples
    --------
    >>> validate_roman_urdu_script("Aaj hum Python seekhenge")["passed"]
    True
    >>> validate_roman_urdu_script("آج ہم Python سیکھیں گے")["passed"]
    False
    """
    errors: list[str] = []
    for label, ranges in _SCRIPT_GROUPS:
        for ch in text:
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in ranges):
                errors.append(
                    f"Forbidden {label} detected (character U+{cp:04X} '{ch}'). "
                    "Output must be written using Latin/English alphabet characters only."
                )
                break  # one error per script family is sufficient

    return {
        "passed": not errors,
        "errors": errors,
        "feedback": "; ".join(errors) if errors else "",
    }


class RetryableAPIError(RuntimeError):
    """An LLM request failed temporarily and may succeed when retried."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class PermanentAPIError(RuntimeError):
    """An LLM request failed for a reason that should not be retried."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class RateLimitError(RetryableAPIError):
    """The provider rejected the request with HTTP 429 (rate limit).

    Subclasses ``RetryableAPIError`` so existing ``except RetryableAPIError``
    handlers keep working; carries the HTTP status, the optional Retry-After
    hint, and ``retryable=True`` so Streamlit can render a friendly message
    without exposing internals.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        *,
        status_code: int = 429,
        retry_after: float | None = None,
    ):
        super().__init__(message, original_error)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = True


class TranslationPipelineError(RuntimeError):
    """Final failure that retains successfully translated chunks."""

    def __init__(
        self,
        message: str,
        *,
        completed_chunks: Sequence[str] | None = None,
        partial_translation: str = "",
        rate_limited: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.completed_chunks = list(completed_chunks or [])
        self.partial_translation = partial_translation
        self.rate_limited = rate_limited
        self.retry_after = retry_after


ALLOWED_TECHNICAL_TERMS = {
    "AI", "API", "app", "array", "audio", "backend", "browser", "cache",
    "cell", "class", "cloud", "code", "coding", "database", "data",
    "dataset", "debug", "embedding", "endpoint", "file", "framework",
    "frontend", "function", "GPU", "HTML", "HTTP", "ID", "input",
    "JSON", "JavaScript", "LangChain", "language", "LLM", "machine",
    "Markdown", "model", "Mistral", "module", "NLP", "output", "PDF",
    "pipeline", "prompt", "Python", "query", "RAG", "request", "response",
    "script", "server", "SQL", "stream", "Streamlit", "string", "token",
    "transcript", "translation", "URL", "user", "variable", "vector",
    "Whisper", "YouTube",
    # Extended technical vocabulary
    "Artificial Intelligence", "Data Science", "Machine Learning",
    "Pandas", "NumPy", "CSS", "TensorFlow", "PyTorch", "Keras",
    "Docker", "Git", "GitHub", "Linux", "Windows", "macOS",
    "Flask", "Django", "FastAPI", "React", "Node",
    "variables", "functions", "lists", "dictionaries", "arrays",
    "classes", "modules", "scripts", "datasets", "models", "pipelines",
    "CPU", "VRAM", "tokens", "embeddings", "vectors",
    "programming", "software", "computer", "list", "value", "values",
    "copy-paste", "agent", "tool", "CLI", "invoke", "message", "content",
    "OpenAI", "Whisper", "web development", "automation",
    # Known product/company proper names that legitimately stay as-is in
    # Roman Urdu output and are NOT invented words. Explicitly curated so a
    # single-initial-capital name (e.g. Anthropic) is not mistaken for
    # gibberish without opening a blanket "any capitalized word" loophole.
    "Anthropic", "ChatGPT", "Claude",
    # Everyday product/account terms kept in English by convention in Roman Urdu
    "account", "login", "password", "access",
    # Mixed technical/product usage common in spoken product transcripts
    "product", "release", "interface",
}

_TOKEN_EDGE_STRIP_CHARS = " \t\n\r.,!?;:()[]{}\"'`“”‘’"

SAFE_ROMAN_URDU_ENGLISH_ALLOWLIST = {
    "subscribe", "youtube", "video", "python", "code", "coding",
    "programming", "loop", "loops", "for", "while", "function",
    "functions", "class", "classes", "object", "objects", "ai",
    "api", "website", "web", "internet", "online", "download",
    "upload", "file", "files", "app", "application", "software",
    "system", "database", "data", "model", "models", "prompt",
    "chat", "email", "google", "openai", "claude", "mistral",
    "streamlit", "whisper",
}

_NORMALIZED_ALLOWED_TECHNICAL_TERMS = {
    re.sub(r"\s+", " ", term.casefold().strip())
    for term in ALLOWED_TECHNICAL_TERMS
    if str(term).strip()
}

SAFE_ROMAN_URDU_ENGLISH_ALLOWLIST |= _NORMALIZED_ALLOWED_TECHNICAL_TERMS

COMMON_ROMAN_URDU_WORDS = {
    "aaj", "hum", "baat", "karna", "hai", "hain", "ka", "ki", "ke",
    "mein", "andar", "bahar", "liye", "wala", "wale", "wali", "madad",
    "ijazat", "samajhna", "samajhte", "seekhna", "seekhenge", "seekhte",
    "istemal", "use", "hota", "hoti", "hote", "ek", "aur", "bohat",
    "mashhoor", "popular", "programming", "language", "par", "bare", "mein",
    "discuss", "karne", "wahan", "jo", "yeh", "woh", "is", "kaisa",
    "karti", "karte", "karta", "dete", "deti", "store", "kar", "sakte",
    "hain", "multiple", "values", "single", "variable", "ke", "sath",
    # "aap" family: second-person pronoun, its possessives, and dative case
    "aap", "aapka", "aapki", "aapke", "aapko",
    # "apna" family: reflexive possessive ("one's own")
    "apna", "apni", "apne",
    # "kaam"/"karna" family: work / doing
    "kaam", "karni", "karke",
    # "can" auxiliary and continuous-tense helpers
    "sakta", "sakti", "sakenge", "raha", "rahi", "rahe", "rahen",
    # High-frequency everyday function words (unambiguous Roman Urdu)
    "kya", "nahi", "bhi", "chahiye", "tha", "thi", "thay",
    # Pronouns / possessives (first- and second-person, "aap" already above)
    "ap", "mera", "meri", "mere", "mujhe",
    "tumhara", "tumhari", "tumhare",
    # Connectors and function words
    "agar", "lekin", "magar", "abhi", "phir", "yahan", "kahan",
    "kuch", "koi", "sab", "itna", "utna", "kitna",
    # Common spelling variants
    "kam", "kyu", "bohot", "bahut", "acha", "achha",
    "thik", "theek", "sahi", "shukria", "shukriya",
    # Common everyday words
    "ghar", "khana", "pani", "zindagi", "bazaar", "roz",
    "dost", "log", "cheez", "sawal", "jawab", "tareeqa",
    "shuru", "khatam",
    # Verb forms used by the prompt's canonical examples ("karni" already above)
    "karenge", "kiya", "kiye", "jata", "humein", "humne", "isay", "sahulat",
    # Words from the project's own validator/regression examples
    "samnay", "naya", "masjid", "aik", "bana", "kal", "raat", "khaya", "gaye",
    # "aana" family (come/came) -- high-frequency spoken variants
    "aage", "aata", "aati", "aaya", "aaye", "aayein",
    # Common spelling variants observed in real production transcripts
    "accha", "aasan", "asaan", "pehle", "pichhle",
    # "hafta" family (week) -- high-frequency spoken variants
    "hafta", "hafte",
    # "dena" family (give/gave) -- high-frequency spoken variants
    "diya", "diye",
    # "pasand" (like) -- high-frequency spoken word
    "pasand",
}

KNOWN_SUSPICIOUS_ROMAN_URDU_WORDS = {
    "dhareenat", "kaafil", "dhaleel", "shohnum", "shohid", "mashroor",
}

INVENTED_WORDS_FEEDBACK = (
    "The previous output contained invented or meaningless words. "
    "Use simple, common Pakistani Roman Urdu vocabulary and translate directly "
    "from the original English."
)

ROMAN_URDU_FEW_SHOT_EXAMPLES = """FEW-SHOT ROMAN URDU EXAMPLES:
ENGLISH:
Today we are going to discuss Python programming.
ROMAN URDU:
Aaj hum Python programming ke bare mein baat karenge.

ENGLISH:
Python is a popular programming language.
ROMAN URDU:
Python ek mashhoor programming language hai.

ENGLISH:
It is used in Data Science and Machine Learning.
ROMAN URDU:
Isay Data Science aur Machine Learning mein use kiya jata hai.

ENGLISH:
Lists allow us to store multiple values inside a single variable.
ROMAN URDU:
Lists humein ek single variable mein multiple values store karne ki sahulat deti hain."""

ROMAN_URDU_VOCABULARY_GUIDANCE = (
    "Use simple spoken Pakistani Roman Urdu. Prefer common everyday wording "
    "such as aaj, hum, baat, karna, hai, hain, ka, ki, ke, mein, andar, "
    "bahar, liye, wala, wale, wali, madad, ijazat, sahulat, samajhna, "
    "seekhna, and istemal. Do not invent, transliterate, or create unfamiliar "
    "words. If unsure, use simple common Roman Urdu or keep the English term "
    "unchanged."
)

ROMAN_URDU_TIMESTAMP_GUIDANCE = (
    "STRUCTURE RULE — PYTHON CONTROLS TRANSCRIPT METADATA:\n"
    "The input below contains only spoken text; all timestamps have been removed before this request.\n"
    "Translate only the supplied spoken text into Pakistani Roman Urdu.\n"
    "Do not add timestamps, timecodes, XML-like markers, headings, or other structural metadata.\n"
    "Python will restore the original transcript structure after translation."
)

INVALID_TIMESTAMP_RETRY_GUIDANCE = (
    "The previous output included unwanted structural metadata. "
    "Regenerate the complete translation from the ORIGINAL SPOKEN TEXT ONLY. "
    "Do not add timestamps, timecodes, XML-like markers, or any other metadata. "
    "Return only the Roman Urdu translation; Python restores transcript metadata."
)

SYSTEM_PROMPT = """
You are a professional English-to-Roman-Urdu translator.

Roman Urdu means Urdu language written ONLY using Latin/English alphabet characters (A-Z, a-z).

==================================================
CRITICAL SCRIPT RULE -- THIS IS THE MOST IMPORTANT RULE
==================================================

Roman Urdu = Urdu language written using ONLY Latin/English alphabet.

ALLOWED (correct Roman Urdu):
  "Aaj hum Python programming seekhenge."
  "Data Science ek bohat important field hai."
  "Variables aur functions Python mein use hote hain."

FORBIDDEN (must NEVER appear in output):
  Native-script Urdu, Arabic, Hindi, or Devanagari text -- FORBIDDEN

You MUST NEVER output:
  - Urdu script characters (Arabic Unicode block U+0600-U+06FF)
  - Arabic script characters
  - Hindi / Devanagari script characters (U+0900-U+097F)

If you translate English into Urdu language, write Urdu words ONLY in Roman/Latin
characters. Do NOT use Unicode Urdu, Arabic, or Devanagari characters under any
circumstances.

==================================================
TECHNICAL TERMS -- KEEP IN ENGLISH AS-IS
==================================================

Do NOT translate the following into Urdu script. Keep them in English exactly:

  Python, Data Science, Machine Learning, Artificial Intelligence,
  Pandas, NumPy, API, JSON, HTML, CSS, JavaScript, SQL,
  Streamlit, LangChain, LLM, RAG,
  TensorFlow, PyTorch, Keras, Docker, Git, GitHub,
  Flask, Django, FastAPI, React, Node,
  variables, functions, lists, dictionaries, arrays,
  classes, modules, scripts, datasets, models, pipelines,
  GPU, CPU, VRAM, tokens, embeddings, vectors,
  URL, HTTP, server, frontend, backend, database, cache

==================================================
STRICT TRANSLATION RULES
==================================================

1. Use Roman Urdu ONLY. Write ALL Urdu words using Latin characters.
2. NEVER output Urdu script, Arabic script, or Devanagari/Hindi characters.
3. The input contains spoken text only. Do not add timestamps, timecodes, XML-like
   markers, or other structural metadata; Python handles transcript metadata.
4. Preserve the supplied text-block structure as closely as possible.
5. Keep all technical terms (listed above) in English as-is.
6. Do NOT summarize, condense, or omit any content.
7. Do NOT add explanations, headings, or extra commentary.
8. Do NOT add information that is not in the source.
9. Do NOT output markdown formatting.
10. Return ONLY the Roman Urdu transcript text -- nothing else.

==================================================
EXAMPLES
==================================================

Input:
  Today we will learn Python programming.

Correct output:
  Aaj hum Python programming seekhenge.

---

Input:
  Machine Learning is a subfield of Artificial Intelligence.

Correct output:
  Machine Learning, Artificial Intelligence ka ek shu'ba hai.

---

Input:
  In Python, variables store data values.

Correct output:
  Python mein, variables data values store karte hain.

==================================================
NATURAL PAKISTANI ROMAN URDU PHRASE GUIDE
==================================================

Translate meaning, not individual words. Prefer simple conversational Pakistani
Roman Urdu and natural sentence construction.

For "Today we are going to discuss X", prefer:
  "Aaj hum X par baat karenge."
  "Aaj hum X ke baare mein baat karenge."
Never write:
  "Aaj hum X ke baate hain."
  "Aaj hum X ke baat hain."
  "Aaj hum X baate hain."

For "X is a popular programming language", prefer:
  "X ek mashhoor programming language hai."
  "X ek popular programming language hai."
Never write:
  "X ek mashroor programming language hai."

For "X is widely used in Y", prefer:
  "X ka Y mein bohat istemal hota hai."
  "X Y mein bohat use hota hai."
Example:
  "Python ka Data Science aur Machine Learning mein bohat istemal hota hai."

For "X allows us to do Y", prefer:
  "X humein Y karne ki sahulat deta hai."
  "X ki madad se hum Y kar sakte hain."

Preferred examples:

English: Today we are going to discuss Python programming.
Roman Urdu: Aaj hum Python programming par baat karenge.

English: Python is a popular programming language.
Roman Urdu: Python ek mashhoor programming language hai.

English: Python is widely used in Data Science and Machine Learning.
Roman Urdu: Python ka Data Science aur Machine Learning mein bohat istemal hota hai.

English: Lists allow us to store multiple values in a single variable.
Roman Urdu: Lists humein ek single variable mein multiple values store karne ki sahulat deti hain.

English: We will learn this step by step.
Roman Urdu: Hum isay step by step seekhenge.

Do not invent words, create fake phonetic transliterations, or use meaningless
phrases such as "dhaleel shohid", "dhaleel shohnum", "dhareenat kaafil", or
"mashroor programming language". If unsure, keep a technical English term
unchanged instead of inventing a Roman Urdu word. Before returning the answer,
silently check every sentence for unnatural grammar, malformed combinations,
literal word-by-word translation, and Hindi-style constructions. Rewrite any
unnatural sentence into simple conversational Pakistani Roman Urdu.

==================================================
FORBIDDEN OUTPUT TYPES (auto-rejected)
==================================================

  Urdu script (U+0600-U+06FF)      -- FORBIDDEN
  Arabic script                    -- FORBIDDEN
  Devanagari/Hindi (U+0900-U+097F) -- FORBIDDEN
  Markdown formatting              -- FORBIDDEN
  Explanations or commentary       -- FORBIDDEN

Write fluent, natural, conversational Pakistani Roman Urdu.
""".strip()

REVIEW_SYSTEM_PROMPT = """
You are a careful semantic evaluator for an English-to-Roman-Urdu translation system.

Evaluate the candidate fairly. Do not invent errors and do not reject natural Pakistani Roman Urdu.

A translation can be natural even when Urdu words and English technical terms are mixed. Do not treat Python, programming language, Data Science, Machine Learning, Artificial Intelligence, web development, automation, variable, lists, values, or other ordinary technical terminology as untranslated content.

Accept common Pakistani Roman Urdu such as:
- Aaj hum Python programming ke bare mein baat karenge.
- Python ek mashhoor programming language hai.
- Python Data Science aur Machine Learning mein use hoti hai.
- Lists humein multiple values ek variable mein store karne ki sahulat deti hain.
- Isay web development aur automation mein bhi use kiya jata hai.

You are reviewing whether the translation accurately converts the English source into NATURAL PAKISTANI ROMAN URDU.

This is translation-quality comparison, not fact checking. Do not invent factual objections that are not required by the source.

IMPORTANT:
Roman Urdu MUST use Latin/English letters only.

Example of correct Roman Urdu:

"Aaj hum Python programming ke bare mein baat karenge."

Roman Urdu must NOT contain:
- Urdu script
- Arabic script
- Hindi/Devanagari script
- Bengali script
- meaningless or invented words
- gibberish
- broken sentences
- Hindi-style grammar
- unnatural literal translations

==================================================
STRICT REVIEW RULES
==================================================

Check the translation sentence by sentence.

1. MEANING PRESERVATION

The Roman Urdu translation must preserve the meaning of the English source.

Reject if:
- meaning is changed
- information is missing
- new information is invented
- technical concepts are incorrectly translated
- sentence meaning becomes confusing

2. NATURAL ROMAN URDU

The output must sound natural to a Pakistani Urdu speaker writing Urdu using Latin letters.

Reject unnatural phrases such as:

"dhaleel shohnum"
"valuein ka khata"
"sukoon deti hain"

These phrases are unnatural, meaningless, or contextually incorrect.

3. GIBBERISH / NONSENSE DETECTION

Immediately reject the translation if it contains:

- invented words
- meaningless words
- random phonetic output
- corrupted transliteration
- words that do not form understandable Pakistani Roman Urdu

Examples:

"dhaleel shohnum"
"maherin kem"
"baiagit"
"khata aise hota hai"

These should normally be treated as CRITICAL ISSUES.

4. HINDI-STYLE LANGUAGE

Reject Hindi-style grammar or wording when natural Pakistani Roman Urdu should be used.

Examples:

Bad:
"ki wo"
"karna hai ki"
"bahut zyada"

Prefer natural Pakistani Roman Urdu such as:

"ke woh"
"karna hai ke"
"bohat"

Do not reject valid technical English words.

5. TECHNICAL TERMS

These technical terms may remain in English:

Python
Data Science
Machine Learning
Artificial Intelligence
web development
automation
variables
data types
strings
lists
tuples
dictionaries
sets
functions
Pandas
NumPy
Matplotlib
Seaborn
dataset
model
API

Technical English terms are NOT quality errors by themselves.

6. SENTENCE COMPLETENESS

Every meaningful source sentence must have a corresponding understandable Roman Urdu translation.

Reject if:
- sentences disappear
- major meaning is omitted
- translation is mostly unrelated
- output is too short
- output repeats unrelated content

==================================================
SCORING
==================================================

95-100:
Excellent. Accurate, natural, fluent Pakistani Roman Urdu.

85-94:
Good. Minor issues only. Meaning fully preserved.

75-89:
Meaning is generally preserved; style issues may be present. Verdict may be PASS or STYLE_WARNING.

0-74:
Potentially serious issue, but the verdict and semantic issues—not the score alone—determine whether a retry is required.

VERDICT DECISION RULE:

Return verdict="PASS" when meaning and important information are preserved.
Return verdict="STYLE_WARNING" when meaning is preserved but wording, grammar, or punctuation could be improved.
Return verdict="FAIL" ONLY for genuine missing content, wrong meaning, hallucinated information, unrelated output, or meaningless invented vocabulary.

STYLE_WARNING MUST pass and MUST NOT trigger a retry, regardless of its numeric score.
Only FAIL has needs_retry=true. The verdict controls the retry decision; semantic_score is diagnostic information only.
Do not give a low score merely because technical English remains in natural mixed Roman Urdu.
Do not reject a phrase merely because you would personally choose different wording.
Do not invent errors or fact-check the translation beyond the source.

==================================================
EXAMPLES
==================================================

SOURCE:
Today we are going to discuss Python programming.

BAD TRANSLATION:
Aaj hum Python programming ka dhaleel shohnum kahti hain.

VERDICT:
FAIL

CRITICAL ISSUES:
- "dhaleel shohnum" is meaningless/gibberish
- sentence meaning is not preserved
- translation is not natural Roman Urdu

CORRECT STYLE:
Aaj hum Python programming ke bare mein baat karne wale hain.

--------------------------------------------------

SOURCE:
Lists allow us to store multiple values inside a single variable.

BAD TRANSLATION:
Lists multiple valuein ka khata aise hota hai ki wo single variable mein store hoti hain.

VERDICT:
FAIL

CRITICAL ISSUES:
- "valuein ka khata" is unnatural and semantically incorrect
- Hindi-style phrase "ki wo"
- meaning is poorly expressed

CORRECT STYLE:
Lists humein ek single variable mein multiple values store karne ki ijazat deti hain.

==================================================
RESPONSE FORMAT
==================================================

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json.
Do not write explanations outside JSON.

Use exactly this structure:

{{
    "passed": true,
    "score": 90,
    "critical_issues": [],
    "semantic_errors": [],
    "style_suggestions": []
}}

`passed` must be a boolean.
`score` must be an integer from 0 to 100 and is diagnostic only.
Use `critical_issues` for genuine semantic or structural failures.
Use `semantic_errors` for missing, changed, or hallucinated meaning.
Use `style_suggestions` only for harmless wording or punctuation improvements.
Style suggestions alone must not make `passed` false and must not trigger a retry.
Do not automatically return 40; score the actual evidence.

REMEMBER:

Be accurate, not artificially harsh.

Do not invent errors.
Do not reject natural Pakistani Roman Urdu.
Do not treat English technical terminology as untranslated content.
Do not approve nonsense.
Do NOT approve gibberish.
Do NOT approve Hindi-style wording.
Do NOT approve a translation with changed meaning.

A bad translation must receive passed=false.
""".strip()


def _is_allowed_technical_term(term: str) -> bool:
    """Return whether a word or phrase is an accepted technical term."""
    normalized = re.sub(r"\s+", " ", term.casefold().strip()).strip(_TOKEN_EDGE_STRIP_CHARS)
    return normalized in _NORMALIZED_ALLOWED_TECHNICAL_TERMS


def _normalize_lookup_token(token: str) -> str:
    """Normalize one token for deterministic vocabulary classification."""
    return re.sub(r"\s+", " ", str(token or "").casefold().strip()).strip(_TOKEN_EDGE_STRIP_CHARS)


def _looks_like_url(token: str) -> bool:
    token = str(token or "").strip(_TOKEN_EDGE_STRIP_CHARS)
    return bool(re.match(r"^(?:https?://|www\.)\S+$", token, flags=re.IGNORECASE))


def _looks_like_email(token: str) -> bool:
    token = str(token or "").strip(_TOKEN_EDGE_STRIP_CHARS)
    return bool(re.match(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", token, flags=re.IGNORECASE))


def _looks_like_filename(token: str) -> bool:
    token = str(token or "").strip(_TOKEN_EDGE_STRIP_CHARS)
    return bool(re.match(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,8}$", token))


def _looks_like_number_or_timestamp(token: str) -> bool:
    token = str(token or "").strip(_TOKEN_EDGE_STRIP_CHARS)
    return bool(re.match(r"^(?:\d+(?:[.:/-]\d+)*|\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)$", token))


def _is_explicitly_allowed_english_token(token: str) -> bool:
    """Return True only for English tokens that are safe to keep as-is."""
    raw = str(token or "")
    normalized = _normalize_lookup_token(raw)
    if not normalized:
        return False
    if normalized in SAFE_ROMAN_URDU_ENGLISH_ALLOWLIST:
        return True
    if _looks_like_url(raw) or _looks_like_email(raw) or _looks_like_filename(raw):
        return True
    if _looks_like_number_or_timestamp(raw):
        return True
    if len(raw) >= 2 and raw.isupper():
        return True
    if len(raw) >= 4 and any(character.isupper() for character in raw[1:]):
        return True
    return False


def _is_safe_english_or_technical_token(token: str) -> bool:
    """Return True for accepted English, technical, or structural tokens."""
    normalized = _normalize_lookup_token(token)
    if not normalized:
        return False
    if _is_explicitly_allowed_english_token(token):
        return True
    if normalized in _COMMON_ENGLISH_WORDS:
        return True
    return False

def _normalize_phrase(value: str) -> str:
    """Normalize whitespace and punctuation for phrase comparisons."""
    return re.sub(r"\s+", " ", value.casefold().strip()).strip(" .,!?:;\"'")


def _normalize_for_repeat_check(text: str) -> str:
    """
    Normalize text for repeated-output comparisons: lowercase, collapse
    whitespace, and trim. This intentionally does NOT strip timestamps --
    a translation whose only difference from the previous attempt is
    timestamp formatting should still be treated as a new attempt, but two
    outputs that differ only in incidental whitespace/casing should be
    treated as the same (repeated) output.
    """
    return re.sub(r"\s+", " ", text.casefold()).strip()


_TIMESTAMP_PATTERN = re.compile(
    r"(?:\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*\]|"
    r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b)"
)
# Generic structural markers are removed defensively if a model emits them;
# no timestamp-specific marker format is used by this pipeline.
_STRUCTURAL_MARKER_PATTERN = re.compile(r"<[^>\n]{1,80}>")


def _extract_timestamps(text: str) -> list[str]:
    """Extract timestamp strings in source order without normalizing them."""
    return _TIMESTAMP_PATTERN.findall(text or "")


def _split_transcript_by_timestamps(text: str) -> list[tuple[str, str]]:
    """Split a transcript into ``(exact_timestamp, spoken_text)`` blocks."""
    if not text:
        return []
    matches = list(_TIMESTAMP_PATTERN.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []

    blocks: list[tuple[str, str]] = []
    prefix = text[:matches[0].start()].strip()
    if prefix:
        blocks.append(("", prefix))

    for index, match in enumerate(matches):
        # A timestamp owns all text up to the next source timestamp, including
        # continuation lines. This preserves multiline transcript blocks.
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(0), text[match.end():content_end].strip()))
    return blocks


def _remove_generated_timestamp_markers(text: str) -> str:
    """Remove model-generated timestamps and structural markers before restore."""
    cleaned_lines: list[str] = []
    for line in (text or "").splitlines():
        line = _TIMESTAMP_PATTERN.sub("", line)
        line = _STRUCTURAL_MARKER_PATTERN.sub("", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _remove_timestamps_for_translation(text: str) -> str:
    """Return only spoken block text; never send source timestamps to the LLM."""
    return "\n".join(
        content for _, content in _split_transcript_by_timestamps(text or "") if content
    ).strip()


def _restore_chunk_timestamps(source_chunk: str, translated_text: str) -> str:
    """Restore source timestamps using deterministic, length-weighted mapping."""
    source_blocks = _split_transcript_by_timestamps(source_chunk or "")
    cleaned_translation = _remove_generated_timestamp_markers(translated_text or "")
    if not source_blocks or not any(timestamp for timestamp, _ in source_blocks):
        restored_output = cleaned_translation
        timestamp_validation = validate_timestamp_sequence(
            source_chunk,
            restored_output,
        )
        if not timestamp_validation["passed"]:
            raise ValueError(
                "Timestamp restoration failed: "
                + timestamp_validation["feedback"]
            )
        return restored_output

    # Model line breaks are not authoritative. Normalize to non-empty logical
    # segments after removing every model-generated timestamp/structural marker.
    translated_segments = [
        re.sub(r"\s+", " ", segment).strip()
        for segment in cleaned_translation.splitlines()
        if segment.strip()
    ]
    block_count = len(source_blocks)
    source_weights = [
        max(1, len(re.sub(r"\s+", " ", content).strip()))
        for _, content in source_blocks
    ]

    def apportion(total: int, weights: list[int]) -> list[int]:
        """Allocate ``total`` ordered units by Hamilton's largest remainder."""
        if not weights:
            return []
        if total <= 0:
            return [0] * len(weights)
        base = [1] * len(weights) if total >= len(weights) else [0] * len(weights)
        remaining = total - sum(base)
        if remaining <= 0:
            return base
        weight_total = sum(weights) or len(weights)
        quotas = [remaining * weight / weight_total for weight in weights]
        additions = [int(quota) for quota in quotas]
        result = [left + right for left, right in zip(base, additions)]
        left_over = remaining - sum(additions)
        order = sorted(
            range(len(weights)),
            key=lambda index: (quotas[index] - additions[index], weights[index], -index),
            reverse=True,
        )
        for index in order[:left_over]:
            result[index] += 1
        return result

    def split_text_by_weights(text: str, weights: list[int]) -> list[str]:
        """Split one ordered text stream into weighted non-empty pieces when possible."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not weights:
            return []
        if not normalized:
            return [""] * len(weights)
        words = normalized.split(" ")
        if len(words) >= len(weights):
            word_counts = apportion(len(words), weights)
            pieces: list[str] = []
            cursor = 0
            for count in word_counts:
                pieces.append(" ".join(words[cursor:cursor + count]).strip())
                cursor += count
            return pieces

        # If there are fewer words than source blocks, use character slices so
        # available content is distributed rather than padded into the tail.
        if len(normalized) >= len(weights):
            char_counts = apportion(len(normalized), weights)
            pieces = []
            cursor = 0
            for count in char_counts:
                pieces.append(normalized[cursor:cursor + count].strip())
                cursor += count
            return pieces

        # There is not enough content for every block to be non-empty. Preserve
        # every character in order and leave only mathematically unavoidable
        # empty pieces.
        pieces = [""] * len(weights)
        for index, character in enumerate(normalized):
            pieces[index] = character
        return pieces

    if not translated_segments:
        mapped_segments = [""] * block_count
    elif len(translated_segments) == block_count:
        mapped_segments = translated_segments
    elif len(translated_segments) < block_count:
        # Fewer model segments: merge the ordered stream, then split it across
        # every source block using source text lengths as allocation weights.
        mapped_segments = split_text_by_weights(" ".join(translated_segments), source_weights)
    else:
        # More model segments: allocate a positive group to every source block
        # and apportion the surplus groups by source text length.
        group_sizes = apportion(len(translated_segments), source_weights)
        mapped_segments = []
        cursor = 0
        for size in group_sizes:
            mapped_segments.append(" ".join(translated_segments[cursor:cursor + size]).strip())
            cursor += size

    restored: list[str] = []
    for (timestamp, _), content in zip(source_blocks, mapped_segments):
        content = re.sub(r"\s+", " ", content).strip()
        restored.append(f"{timestamp} {content}".rstrip() if timestamp else content)
    restored_output = "\n".join(restored)

    timestamp_validation = validate_timestamp_sequence(
        source_chunk,
        restored_output,
    )

    if not timestamp_validation["passed"]:
        raise ValueError(
            "Timestamp restoration failed: "
            + timestamp_validation["feedback"]
        )

    return restored_output


def validate_timestamp_sequence(source: str, candidate: str) -> dict[str, Any]:
    """Validate the final restored timestamp sequence by exact list equality."""
    source_timestamps = _extract_timestamps(source)
    candidate_timestamps = _extract_timestamps(candidate)
    errors: list[str] = []
    if len(candidate_timestamps) != len(source_timestamps):
        errors.append(
            f"Timestamp count mismatch: expected {len(source_timestamps)}, found {len(candidate_timestamps)}."
        )
    if candidate_timestamps != source_timestamps:
        errors.append("Timestamp values or order do not exactly match the source sequence.")
    return {
        "passed": not errors,
        "errors": errors,
        "feedback": "; ".join(errors),
        "source_timestamps": source_timestamps,
        "candidate_timestamps": candidate_timestamps,
    }


def _sanitize_retry_context(text: str) -> str:
    """Remove timestamps and structural markers from retry context."""
    sanitized = _TIMESTAMP_PATTERN.sub("", text or "")
    sanitized = _STRUCTURAL_MARKER_PATTERN.sub("", sanitized)
    return re.sub(r"[ \t]{2,}", " ", sanitized)


def _find_duplicate_lines(text: str) -> list[str]:
    """Find repeated non-empty lines, ignoring timestamp-only differences."""
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for line in text.splitlines():
        normalized = _normalize_phrase(re.sub(r"^\[[^\]]+\]\s*", "", line))
        if not normalized:
            continue
        seen[normalized] = seen.get(normalized, 0) + 1
        if seen[normalized] == 2:
            duplicates.append(line.strip())
    return duplicates


_COMMON_ENGLISH_WORDS = {
    # Closed-class / function words
    "the", "this", "that", "these", "those", "with", "from", "into",
    "and", "but", "because", "which", "where", "when", "will", "would",
    "should", "can", "could", "have", "has", "had", "are", "is", "was",
    "were", "been", "being", "be", "do", "does", "did", "not", "no",
    "yes", "for", "of", "to", "in", "on", "as", "a", "an", "some", "any",
    "all", "each", "every", "other", "such", "than", "then", "so",
    "if", "or", "nor", "about", "over", "under", "again", "here",
    "there", "how", "what", "why", "who", "whom", "whose",
    # Pronouns
    "we", "you", "they", "he", "she", "it", "i", "us", "our", "your",
    "their", "his", "her", "them",
    # Common everyday verbs / adverbs frequently left untranslated
    "allow", "allows", "store", "single", "multiple", "values", "using",
    "use", "used", "today", "tomorrow", "yesterday", "now", "going", "friendly",
    "go", "goes", "went", "discuss", "discussing", "talk", "talking",
    "learn", "learning", "teach", "teaching", "understand", "explain",
    "explaining", "need", "needs", "want", "wants", "like", "just",
    "very", "really", "also", "just", "let", "lets", "get", "gets",
    "make", "makes", "look", "looking", "see", "seeing", "know",
    "knowing", "think", "thinking",
}


def _is_acceptable_compound_token(token: str) -> bool:
    """Accept a hyphenated compound only when every alphabetic part is itself
    acceptable (Roman Urdu vocabulary, safe English/technical token, or common
    English). ``user-friendly`` passes because ``user`` is a safe technical
    token; ``qmdx-wzpf`` still fails because neither part is acceptable. This
    supports legitimate compounds without creating a hyphen-loophole for
    arbitrary nonsense."""
    normalized = _normalize_lookup_token(token)
    if "-" not in normalized:
        return False
    parts = [part for part in normalized.split("-") if part]
    if len(parts) < 2:
        return False
    return all(
        part in COMMON_ROMAN_URDU_WORDS
        or part in SAFE_ROMAN_URDU_ENGLISH_ALLOWLIST
        or part in _COMMON_ENGLISH_WORDS
        for part in parts
    )


def _allowed_technical_phrase_words(text: str) -> set[str]:
    """Return words participating in allowed multi-word technical terms."""
    allowed_words: set[str] = set()
    for term in ALLOWED_TECHNICAL_TERMS:
        if " " not in term:
            continue
        if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text, flags=re.IGNORECASE):
            allowed_words.update(_normalize_lookup_token(word) for word in term.split())
    return allowed_words


def _find_suspicious_english_words(text: str) -> list[str]:
    """Find unresolved English words while exempting accepted mixed usage."""
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text)
    allowed_phrase_words = _allowed_technical_phrase_words(text)
    return sorted(
        {
            word for word in words
            if (
                _normalize_lookup_token(word) not in COMMON_ROMAN_URDU_WORDS
                and not _is_explicitly_allowed_english_token(word)
                and _normalize_lookup_token(word) not in allowed_phrase_words
                and not _is_acceptable_compound_token(word)
            )
        },
        key=str.casefold,
    )

def _excessive_english_ratio(text: str) -> float:
    """
    Return the fraction of words in *text* that look like unresolved,
    untranslated English rather than allowed technical terms or Roman Urdu.
    """
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text)
    if not words:
        return 0.0
    allowed_phrase_words = _allowed_technical_phrase_words(text)
    suspicious = [
        word for word in words
        if (
            _normalize_lookup_token(word) not in COMMON_ROMAN_URDU_WORDS
            and not _is_explicitly_allowed_english_token(word)
            and _normalize_lookup_token(word) not in allowed_phrase_words
            and not _is_acceptable_compound_token(word)
        )
    ]
    return len(suspicious) / len(words)

def _is_genuinely_untranslated_english(text: str, suspicious_words: list[str]) -> bool:
    """Identify a mostly ordinary-English fragment, not mixed technical prose.

    Positive evidence of ordinary English is required: an unresolved word
    counts as evidence only when it is itself a common English function word
    (see ``_COMMON_ENGLISH_WORDS``). Merely being absent from
    ``COMMON_ROMAN_URDU_WORDS`` is never enough -- that set is necessarily
    incomplete, and flagging valid Roman Urdu as untranslated English would
    burn retries on good output.
    """
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text or "")
    if not words:
        return False
    normalized_words = [_normalize_lookup_token(word) for word in words]
    allowed_phrase_words = _allowed_technical_phrase_words(text)
    roman_urdu_words = sum(word in COMMON_ROMAN_URDU_WORDS for word in normalized_words)
    unresolved_words = [
        word for raw_word, word in zip(words, normalized_words)
        if word
        and word not in COMMON_ROMAN_URDU_WORDS
        and word not in allowed_phrase_words
        and not _is_explicitly_allowed_english_token(raw_word)
        and not _is_acceptable_compound_token(raw_word)
    ]
    if not unresolved_words:
        return False
    roman_ratio = roman_urdu_words / len(words)
    unresolved_ratio = len(unresolved_words) / len(words)
    positive_english_evidence = sum(
        word in _COMMON_ENGLISH_WORDS for word in unresolved_words
    )
    english_evidence_ratio = positive_english_evidence / len(unresolved_words)
    return (
        len(words) >= 6
        and roman_ratio <= 0.20
        and unresolved_ratio >= 0.60
        and english_evidence_ratio >= GENUINE_ENGLISH_EVIDENCE_RATIO_THRESHOLD
    )

def _safe_error_text(exc: Exception) -> str:
    """Return an exception string with the configured API key removed."""
    text = str(exc)
    secret = os.getenv(MISTRAL_API_KEY_ENV)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    for candidate in (getattr(exc, "status_code", None), getattr(response, "status_code", None)):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(400|401|403|408|409|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable_error(exc: Exception) -> bool:
    """Classify transient HTTP, timeout, and network errors."""
    status = _status_code(exc)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).casefold()
    return any(marker in text for marker in (
        "timeout", "timed out", "rate limit", "too many requests", "service unavailable",
        "temporarily unavailable", "connection reset", "connection aborted", "connection error",
        "network error", "server error", "bad gateway", "gateway timeout",
    ))


def _is_permanent_error(exc: Exception) -> bool:
    """Classify authentication, authorization, and malformed-request errors."""
    status = _status_code(exc)
    return status in {400, 401, 403} or any(
        marker in str(exc).casefold()
        for marker in ("invalid api key", "authentication", "unauthorized", "forbidden", "bad request")
    )


def _retry_delay(attempt: int, base_delay: float = API_BASE_DELAY, cap: float = API_MAX_DELAY) -> float:
    """Calculate exponential backoff with up to 20% positive jitter."""
    exponential = min(cap, base_delay * (2 ** (attempt - 1)))
    return min(cap, exponential + random.uniform(0, exponential * 0.2))


def _parse_retry_after(value: Any) -> float | None:
    """Safely parse a Retry-After header (delta-seconds or HTTP-date).

    Returns None when the header is missing, malformed, or in the past so
    callers fall back to exponential backoff. Capped at RETRY_AFTER_CEILING
    so a huge header can never block a chunk unreasonably.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
        if seconds <= 0:
            return None
        return min(seconds, RETRY_AFTER_CEILING)
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        return None
    remain = (retry_at - datetime.now(timezone.utc)).total_seconds()
    if remain <= 0:
        return None
    return min(remain, RETRY_AFTER_CEILING)


def _rate_limit_retry_after(exc: Exception) -> float | None:
    """Return the provider's Retry-After hint for a 429, if present.

    LangChain/Mistral surface httpx.Response objects as ``exc.response`` and
    some providers also put the header directly on the exception. Everything
    is optional-guarded; a missing or foreign header simply yields None and
    the caller falls back to exponential backoff.
    """
    response = getattr(exc, "response", None)
    for container in (response, exc):
        headers = getattr(container, "headers", None)
        if headers is None:
            continue
        try:
            if hasattr(headers, "get"):
                value = headers.get("Retry-After") or headers.get("retry-after")
                parsed = _parse_retry_after(value)
                if parsed is not None:
                    return parsed
        except Exception:
            continue
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True only when the error is specifically an HTTP 429.

    Text markers cover providers whose error bodies mention rate limiting
    without a clean HTTP status (e.g. Mistral code 1300 "rate limit").
    """
    if _status_code(exc) == 429:
        return True
    text = str(exc).casefold()
    return "rate limit" in text or "too many requests" in text


def _rate_limit_delay(
    attempt: int,
    *,
    retry_after: float | None = None,
    base_delay: float = RATE_LIMIT_BASE_DELAY,
    max_delay: float = RATE_LIMIT_MAX_DELAY,
) -> float:
    """Backoff for HTTP 429: Retry-After when available, else exponential.

    Exponential schedule with jitter: attempt 1 -> ~5s, 2 -> ~10s,
    3 -> ~20s, 4 -> ~40s, 5 -> ~80s, each plus up to
    RATE_LIMIT_JITTER_FACTOR, all capped at max_delay. Retry-After wins
    when present.
    """
    if retry_after is not None:
        delay = min(retry_after, max_delay)
        return min(max_delay, delay * (1.0 + random.uniform(0, RATE_LIMIT_JITTER_FACTOR)))
    exponential = min(max_delay, base_delay * (RATE_LIMIT_BACKOFF_FACTOR ** (attempt - 1)))
    return min(max_delay, exponential * (1.0 + random.uniform(0, RATE_LIMIT_JITTER_FACTOR)))


# ---------------------------------------------------------------------------
# Mistral AI LLM initialisation
# ---------------------------------------------------------------------------

def _print_mistral_status(
    model: str,
    role: str,
    api_key: str | None,
    temperature: float,
) -> None:
    """Print provider status without implying local GPU inference."""
    key_status = "Configured" if api_key else "Missing"
    print("\n" + "=" * 60)
    print("MISTRAL AI STATUS")
    print("=" * 60)
    print("Provider        : Mistral AI")
    print(f"Role            : {role}")
    print(f"Model           : {model}")
    print(f"API Key         : {key_status}")
    print("Mode            : Cloud API")
    print(f"Temperature     : {temperature}")
    print("Connection      : Configured" if api_key else "Connection      : Missing API key")
    print("Note            : Inference runs remotely; local GPU availability is irrelevant.")
    print("=" * 60)


def _create_mistral_client(
    *,
    model: str,
    role: str,
    temperature: float,
    response_format: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Create one Mistral client; shared request retries remain outside the client."""
    api_key = os.getenv(MISTRAL_API_KEY_ENV)
    _print_mistral_status(model=model, role=role, api_key=api_key, temperature=temperature)
    if not api_key and not os.getenv("FALLBACK_API_KEY"):
        raise RuntimeError("MISTRAL_API_KEY is missing. Add it to the .env file.")

    # Build through the centralized provider: Mistral primary plus the
    # configured fallback with automatic failover on HTTP 429 / 5xx /
    # temporary provider failures. Retries stay in _invoke_llm_with_retries.
    try:
        client = _provider_get_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=MISTRAL_MAX_TOKENS,
            response_format=response_format,
            role=role,
        )
        print(f"✅ Mistral AI {role.lower()} client initialized successfully.")
        return client
    except Exception as error:
        raise RuntimeError(
            f"Failed to initialize Mistral AI {role.lower()} client for model '{model}'. "
            f"Original error: {_safe_error_text(error)}"
        ) from error


def _get_llm(
    model: str = MISTRAL_MODEL,
    temperature: float = MISTRAL_TEMPERATURE,
) -> BaseChatModel:
    """Backward-compatible direct factory for the Mistral translator client."""
    return _create_mistral_client(
        model=model,
        role="Translator",
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_translator_llm() -> BaseChatModel:
    """Return the cached Mistral client dedicated to translation generation."""
    return _get_llm(model=MISTRAL_MODEL, temperature=MISTRAL_TEMPERATURE)


@lru_cache(maxsize=1)
def get_reviewer_llm() -> BaseChatModel:
    """Return the cached Mistral client dedicated to semantic review."""
    return _create_mistral_client(
        model=MISTRAL_REVIEWER_MODEL,
        role="Semantic Reviewer",
        temperature=MISTRAL_REVIEW_TEMPERATURE,
        response_format={"type": "json_object"},
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Backward-compatible alias for the cached translator client."""
    return get_translator_llm()


def _message_content(response: Any) -> str:
    """Extract text from LangChain responses without assuming one response type."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # Collect ONLY actual textual values from supported list items.
        # Nested {"text": {"value": ...}} is unwrapped once; unsupported
        # non-mapping items are ignored, never str()-stringified.
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, Mapping):
                    text = text.get("value")
                if isinstance(text, str):
                    parts.append(text)
        content = "".join(parts)
    elif isinstance(content, Mapping):
        # Mapping content: extract the "text" field only.  Nested
        # {"text": {"value": ...}} is unwrapped once.  Never stringify
        # arbitrary/unsupported dicts -- they yield empty text.
        text = content.get("text")
        if isinstance(text, Mapping):
            text = text.get("value")
        content = text
    return str(content or "").strip()


def _normalize_translation_output(text: str) -> str:
    """Remove wrapper formatting while preserving transcript content."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*```(?:[A-Za-z0-9_-]+)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned).strip()

    lines = cleaned.splitlines()
    if lines:
        first = lines[0]
        if not re.match(r"^\s*\[[^\]]+\]", first):
            prefix_match = re.match(
                r"^\s*(?:translation|roman\s+urdu(?:\s+translation)?|output)\s*:\s*",
                first,
                flags=re.IGNORECASE,
            )
            if prefix_match:
                remainder = first[prefix_match.end():].strip()
                lines = ([remainder] if remainder else []) + lines[1:]
                cleaned = "\n".join(lines).strip()
    return cleaned


def _contextual_source_terms(source: str) -> set[str]:
    """Return source tokens, retaining hyphenated forms for context checks."""
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", source or "")
    }


def _looks_like_source_transcription_artifact(token: str, source_terms: set[str]) -> bool:
    """Allow a token only when its hyphenated parts resemble source words."""
    parts = [part.casefold() for part in token.split("-") if part]
    if len(parts) < 2 or not source_terms:
        return False
    source_words = {
        part
        for term in source_terms
        for part in term.split("-")
        if part
    }
    matched_parts = sum(
        any(
            part == source_word
            or SequenceMatcher(None, part, source_word).ratio() >= 0.78
            for source_word in source_words
        )
        for part in parts
    )
    # Require evidence from at least two components and at least half of the
    # token. A random hyphenated string therefore remains a hard failure.
    return matched_parts >= 2 and matched_parts * 2 >= len(parts)


def _is_contextually_acceptable_token(token: str, source_terms: set[str]) -> bool:
    """Accept only source-supported or clearly classified non-Roman tokens."""
    normalized = _normalize_lookup_token(token)
    if normalized in source_terms:
        return True
    if normalized in COMMON_ROMAN_URDU_WORDS or _is_safe_english_or_technical_token(token):
        return True
    return _looks_like_source_transcription_artifact(token, source_terms)


def _strong_gibberish_signal(normalized: str) -> bool:
    """Signals that apply to every token regardless of capitalization: no
    vowels at all, or a run of three repeated characters. Evaluated on the
    alphabetic letters only, so a hyphen or apostrophe can never masquerade
    as a consonant."""
    letters = re.sub(r"[^a-z]", "", normalized)
    if not letters:
        return True
    if not re.search(r"[aeiou]", letters):
        return True
    return bool(re.search(r"(.)\1\1", letters))


def _weak_gibberish_signal(normalized: str) -> bool:
    """Four or more consecutive consonants. ``y`` counts as a vowel here:
    real words legitimately end in consonant+y (``friendly``, ``deeply``),
    so a final ``-ly`` must not masquerade as a consonant cluster. Real
    proper nouns also carry clusters (e.g. ``nthr`` in ``Anthropic``), so
    this weak signal is applied to lowercase tokens only; title-cased tokens
    are already caught by the strong signals and the curated blacklist."""
    letters = re.sub(r"[^a-z]", "", normalized)
    return bool(re.search(r"[^aeiouy]{4,}", letters))


def _find_suspicious_invented_words(source: str, translation: str) -> list[str]:
    """Find likely invented words while honoring source-supported context."""
    source_terms = _contextual_source_terms(source)
    source_words = {
        _normalize_lookup_token(word) for word in re.findall(r"[A-Za-z]+", source or "")
    }
    output_words = re.findall(r"[A-Za-z][A-Za-z'-]*", translation)
    suspicious: list[str] = []
    seen: set[str] = set()

    for word in output_words:
        normalized = _normalize_lookup_token(word)
        if not normalized:
            continue
        if normalized in seen:
            continue
        if normalized in KNOWN_SUSPICIOUS_ROMAN_URDU_WORDS:
            suspicious.append(word)
            seen.add(normalized)
            continue
        if _is_safe_english_or_technical_token(word):
            continue
        if len(normalized) < 4:
            continue
        if normalized in source_words or _is_contextually_acceptable_token(word, source_terms):
            continue
        # Keep the existing conservative gibberish signals for tokens that
        # have no source, technical, proper-name, or transcription evidence.
        # Lowercase tokens are judged by the strong AND weak signals; a
        # title-cased token (proper-noun shape) only by the strong signals.
        if _strong_gibberish_signal(normalized) or (
            _weak_gibberish_signal(normalized) and not word[0].isupper()
        ):
            suspicious.append(word)
            seen.add(normalized)
    return suspicious


def _invoke_llm_with_retries(
    llm: Any,
    messages: Sequence[Any],
    *,
    operation_name: str = "LLM request",
    max_attempts: int = API_MAX_ATTEMPTS,
    base_delay: float = API_BASE_DELAY,
    max_delay: float = API_MAX_DELAY,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """Invoke an LLM with independent retries for transient API errors."""
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("retry delays must be non-negative")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        print(f"🔌 {operation_name} attempt {attempt}/{max_attempts}")
        try:
            result = llm.invoke(list(messages))
            print(f"✅ {operation_name} successful")
            return result
        except Exception as exc:
            last_error = exc
            rate_limited = _is_rate_limit_error(exc)
            # Auth/malformed errors are permanent; a 429 never is.
            if _is_permanent_error(exc) and not rate_limited:
                raise PermanentAPIError(
                    f"{operation_name} failed permanently: {_safe_error_text(exc)}", exc
                ) from exc
            if not rate_limited and not _is_retryable_error(exc):
                raise PermanentAPIError(
                    f"{operation_name} failed with a non-retryable error: {_safe_error_text(exc)}", exc
                ) from exc
            if attempt >= max_attempts:
                if rate_limited:
                    raise RateLimitError(
                        f"{operation_name} rate-limited after {max_attempts} attempts "
                        f"(HTTP 429). Try again in a few minutes.",
                        exc,
                        retry_after=_rate_limit_retry_after(exc),
                    ) from exc
                raise RetryableAPIError(
                    f"{operation_name} failed after {max_attempts} attempts: {_safe_error_text(exc)}", exc
                ) from exc
            if rate_limited:
                retry_after = _rate_limit_retry_after(exc)
                delay = _rate_limit_delay(attempt, retry_after=retry_after)
                print(
                    "⚠️  Rate limit (HTTP 429) exceeded"
                    + (f"; Retry-After {retry_after:.0f}s" if retry_after is not None else "")
                    + "."
                )
            else:
                delay = _retry_delay(attempt, base_delay, max_delay)
                print(f"⚠️  Temporary error: {_safe_error_text(exc)}")
            print(f"⏳ Waiting {delay:.1f} seconds before retry...")
            sleeper(delay)
    raise RetryableAPIError(
        f"{operation_name} failed: {_safe_error_text(last_error) if last_error else 'unknown error'}",
        last_error,
    )


def split_transcript(transcript: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split a transcript on line boundaries without splitting timestamped cues."""
    if not transcript or not transcript.strip():
        return []
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    lines = transcript.strip().splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in lines:
        line_length = len(line) + (1 if current else 0)
        if current and current_length + line_length > max_chars:
            chunks.append("\n".join(current).strip())
            current, current_length = [], 0
        current.append(line)
        current_length += len(line) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _validation_debug_enabled() -> bool:
    """Enable safe opt-in validation diagnostics without printing secrets."""
    return os.getenv("ROMAN_URDU_VALIDATION_DEBUG", "").casefold() in {"1", "true", "yes", "on"}


def _debug_validation_context(
    source: str,
    translation: str,
    suspicious_invented: list[str],
    suspicious_english: list[str],
) -> None:
    """Log validation evidence only when explicitly enabled by the operator."""
    if not _validation_debug_enabled():
        return
    source_tokens = sorted(_contextual_source_terms(source))
    output_tokens = list(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z'-]*", translation)))
    ignored_source_derived = [
        token for token in output_tokens
        if _is_contextually_acceptable_token(token, set(source_tokens))
        and token not in suspicious_invented
    ]
    _LOGGER.debug(
        "Roman Urdu validation: source=%r candidate=%r normalized_source_tokens=%s "
        "ignored_source_derived_or_allowed=%s suspicious_invented=%s suspicious_english=%s",
        source,
        translation,
        source_tokens,
        ignored_source_derived,
        suspicious_invented,
        suspicious_english,
    )


def validate_roman_urdu_quality(
    source: str,
    translation: str,
    *,
    require_timestamps: bool = True,
) -> dict[str, Any]:
    """
    Run deterministic checks that protect transcript structure and content.

    Validation order matches the required pipeline spec:
      1. Empty output check       -- immediate return, cannot run further checks
      2. Completeness check       -- translation must not be suspiciously short
      3. Timestamp preservation   -- every source timestamp must appear in order
      4. Forbidden Unicode script -- early-exit; output NEVER forwarded to reviewer
      5. Duplicate lines check
      6. Suspicious untranslated English words (minor)
    """
    critical: list[str] = []
    minor: list[str] = []

    # 1. Empty check -- return immediately; nothing else can run
    if not translation or not translation.strip():
        critical.append("Translation is empty.")
        return {
            "passed": False,
            "score": 0,
            "critical_issues": critical,
            "minor_issues": minor,
            "awkward_phrases": [],
            "feedback": "; ".join(critical),
            "retry_safe": False,
        }

    # 2. Completeness check
    #    (a) Length sanity check -- kept, but not relied on alone.
    #        A translation that is far shorter than the source almost certainly
    #        omits content. Threshold: at least 15% of source char-count (min 8).
    if (source.strip() and translation.strip()
            and len(translation.strip()) < max(8, int(len(source.strip()) * 0.15))):
        critical.append("Translation is suspiciously short and may omit content.")

    #    (b) Structural consistency -- source and translated transcripts should
    #        have a reasonably similar number of non-empty lines/segments.
    #        This does NOT require identical sentence counts (translation can
    #        naturally restructure sentences), only that a large fraction of
    #        the source's lines/segments are represented in the output.
    source_lines = [line for line in source.splitlines() if line.strip()]
    translation_lines = [line for line in translation.splitlines() if line.strip()]
    if len(source_lines) > 1:
        min_expected_lines = max(1, int(len(source_lines) * 0.5))
        if len(translation_lines) < min_expected_lines:
            critical.append(
                "Translated transcript has far fewer lines/segments than the "
                "source; the translation appears to contain only a small "
                "subset of the original content."
            )

    # 3. Timestamp preservation. Compare the final Python-restored output
    #    against this source chunk as exact lists; equality covers count,
    #    values, multiplicity, and order, including intentional repeats.
    timestamp_result = validate_timestamp_sequence(source, translation)
    if require_timestamps and not timestamp_result["passed"]:
        critical.extend(timestamp_result["errors"])

    # 4. Script detection -- return immediately so invalid output is NEVER
    #    forwarded to the semantic reviewer.
    script_result = validate_roman_urdu_script(translation)
    if not script_result["passed"]:
        critical.extend(script_result["errors"])
    elif _contains_non_latin_alphabetic(translation):
        critical.append(
            "Translation contains non-Latin alphabetic characters. "
            "Output must use ASCII Latin letters only."
        )
    if critical and (
        not script_result["passed"] or _contains_non_latin_alphabetic(translation)
    ):
        score = max(0, 100 - (35 * len(critical)) - (5 * len(minor)))
        return {
            "passed": False,
            "score": score,
            "critical_issues": critical,
            "minor_issues": minor,
            "awkward_phrases": [],
            "feedback": "; ".join(critical + minor),
            "retry_safe": False,
        }

    # 5. Obvious invented/gibberish vocabulary. This check runs before semantic
    # review, and the rejected text is never reused as retry context.
    invented_words = _find_suspicious_invented_words(source, translation)
    if invented_words:
        critical.append(
            "Translation contains suspicious invented or meaningless words: "
            + ", ".join(invented_words[:8])
            + "."
        )

    # 6. Duplicate lines
    if len(_find_duplicate_lines(translation)) > len(_find_duplicate_lines(source)) + 1:
        critical.append("Translation contains unexpected duplicate lines.")

    # 7. Roman Urdu vs. ordinary-English diagnostics (script already clean
    #    here). Mixed Roman Urdu/technical prose receives a style warning, while
    #    a predominantly untranslated ordinary-English fragment is a hard failure.
    if translation.strip():
        suspicious_words = _find_suspicious_english_words(translation)
        _debug_validation_context(source, translation, invented_words, suspicious_words)
        if suspicious_words:
            english_ratio = _excessive_english_ratio(translation)
            if _is_genuinely_untranslated_english(translation, suspicious_words):
                critical.append(
                    "Translation is predominantly ordinary English and appears untranslated "
                    f"(e.g. {', '.join(suspicious_words[:6])})."
                )
            elif english_ratio >= EXCESSIVE_ENGLISH_RATIO_THRESHOLD:
                minor.append(
                    "Some ordinary English words remain untranslated "
                    f"(~{english_ratio:.0%} of words, e.g. "
                    f"{', '.join(suspicious_words[:6])})."
                )
            else:
                minor.append("Some ordinary English words remain untranslated.")

    score = max(0, 100 - (35 * len(critical)) - (5 * len(minor)))
    return {
        "passed": not critical,
        "score": score,
        "critical_issues": critical,
        "minor_issues": minor,
        "awkward_phrases": [],
        "feedback": "; ".join(critical + minor),
        "retry_safe": not critical,
    }



def normalize_for_semantic_review(text: str) -> str:
    """Remove matching timestamp wrappers while preserving all transcript lines."""
    normalized = re.sub(
        r"\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*\]\s*",
        "",
        text or "",
    )
    return normalized.strip()


def _review_messages(source: str, translation: str) -> list[Any]:
    reviewer_instructions = (
        REVIEW_SYSTEM_PROMPT
        + "\n\nEVALUATION-ONLY REQUIREMENT:\n"
        "Evaluate the candidate only. Do not translate, rewrite, or propose a replacement answer.\n"
        "Compare the COMPLETE source chunk with the COMPLETE candidate chunk.\n"
        "Use the normalized source and normalized candidate below for semantic comparison.\n"
        "Do not report a sentence as missing merely because it was paraphrased in Roman Urdu.\n"
        "Do not fact-check claims that are not required by the source.\n"
        "Assign the score from the evidence; do not automatically return 40 for a failure.\n"
        "Return strict JSON only with passed, score, critical_issues, semantic_errors, and style_suggestions."
    )
    normalized_source = normalize_for_semantic_review(source)
    normalized_translation = normalize_for_semantic_review(translation)
    return [
        SystemMessage(content=reviewer_instructions),
        HumanMessage(
            content=(
                "SOURCE SPOKEN TEXT (timestamps removed by Python):\n"
                f"{normalized_source}\n\n"
                "CANDIDATE SPOKEN TEXT (timestamps removed by Python):\n"
                f"{normalized_translation}"
            )
        ),
    ]


def _parse_review(response: Any) -> dict[str, Any]:
    """
    Parse the semantic reviewer response.

    The reviewer must return valid JSON.

    If the response cannot be parsed, the review MUST fail.
    Invalid JSON should never automatically approve a translation.
    """

    # Normalize LangChain AIMessage and content-block responses consistently.
    response = _message_content(response)

    # Preserve the exact text for diagnostics, then normalize surrounding whitespace.
    raw_response = response
    response = response.strip()

    # ---------------------------------------------------------
    # Attempt 1: Direct JSON parsing
    # ---------------------------------------------------------
    try:
        data = json.loads(response)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # Attempt 2: Remove markdown code fences
    # ---------------------------------------------------------
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response,
        flags=re.IGNORECASE,
    ).strip()

    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # Attempt 3: Extract JSON object from response
    # ---------------------------------------------------------
    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if match:
        try:
            data = json.loads(match.group())

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

    # ---------------------------------------------------------
    # Final fallback: INVALID REVIEW MUST FAIL
    # ---------------------------------------------------------
    print("Semantic reviewer raw response:")
    print(repr(raw_response))
    return {
        "verdict": "REVIEW_ERROR",
        "semantic_score": 0,
        "semantic_issues": [],
        "style_issues": [],
        "needs_retry": True,
        "review_error": "invalid_json",
        "passed": False,
        "score": 0,
        "critical_issues": [],
        "minor_issues": [],
        "awkward_phrases": [],
        "feedback": "Reviewer response could not be parsed as valid JSON; translation quality was not judged.",
    }


def _review_issue_list(value: Any) -> list[str]:
    """Normalize reviewer issue fields without splitting a single string into characters."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _normalize_reviewer_verdict(review: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize reviewer output into one canonical internal schema."""
    normalized = dict(review)
    if normalized.get("review_error"):
        return {
            "passed": False,
            "score": 0,
            "critical_issues": [],
            "semantic_errors": [],
            "style_suggestions": [],
            "feedback": "Semantic reviewer response could not be parsed as valid JSON.",
            "review_error": normalized["review_error"],
        }

    raw_passed = normalized.get("passed", normalized.get("pass"))
    if isinstance(raw_passed, str):
        passed_flag: bool | None = {
            "true": True,
            "pass": True,
            "passed": True,
            "yes": True,
            "false": False,
            "fail": False,
            "failed": False,
            "no": False,
        }.get(raw_passed.strip().casefold())
    elif isinstance(raw_passed, bool):
        passed_flag = raw_passed
    else:
        passed_flag = None

    raw_score = normalized.get("score", normalized.get("semantic_score", 0))
    try:
        score = 0 if isinstance(raw_score, bool) else max(0, min(100, int(raw_score)))
    except (TypeError, ValueError):
        score = 0
    critical = _review_issue_list(normalized.get("critical_issues"))
    semantic = _review_issue_list(
        normalized.get("semantic_errors") or normalized.get("semantic_issues")
    )
    style = _review_issue_list(
        normalized.get("style_suggestions")
        or normalized.get("style_issues")
        or normalized.get("minor_issues")
    )
    verdict = str(normalized.get("verdict") or "").strip().upper()

    # A JSON object is not sufficient by itself: it must contain a recognized
    # verdict or boolean pass field. Missing/unknown decision fields fail closed.
    if (
        passed_flag is None
        and verdict not in {"PASS", "STYLE_WARNING", "FAIL"}
    ):
        return {
            "passed": False,
            "score": score,
            "critical_issues": critical,
            "semantic_errors": semantic,
            "style_suggestions": style,
            "feedback": "Semantic reviewer response omitted a valid verdict.",
            "review_error": "invalid_schema",
        }

    # Structured issue fields are authoritative. A reviewer cannot approve a
    # response while simultaneously reporting a critical or semantic error.
    if critical or semantic or verdict == "FAIL" or passed_flag is False:
        passed = False
    elif verdict in {"PASS", "STYLE_WARNING"} or passed_flag is True:
        passed = True
    else:
        passed = False
    return {
        "passed": bool(passed),
        "score": score,
        "critical_issues": critical,
        "semantic_errors": semantic,
        "style_suggestions": style,
        "feedback": str(normalized.get("feedback") or ""),
    }


def _review_categories(review: Mapping[str, Any]) -> dict[str, list[str]]:
    """Normalize legacy and structured reviewer fields without keyword guessing."""
    return {
        "critical_issues": _review_issue_list(review.get("critical_issues")),
        "semantic_errors": _review_issue_list(review.get("semantic_errors")),
        "style_suggestions": _review_issue_list(
            review.get("style_suggestions")
            or review.get("style_issues")
            or review.get("minor_issues")
        ),
    }


def review_roman_urdu_translation(
    source: str,
    translation: str,
    llm: Any | None = None,
    *,
    api_attempts: int = API_MAX_ATTEMPTS,
    deterministic_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask the reviewer for semantic feedback and merge deterministic checks."""
    deterministic = dict(
        deterministic_result
        if deterministic_result is not None
        else validate_roman_urdu_quality(source, translation)
    )
    if not deterministic.get("passed"):
        deterministic_issues = [str(item) for item in deterministic.get("critical_issues", [])]
        return {
            "passed": False,
            "score": 0,
            "llm_score": 0,
            "semantic_score": 0,
            "deterministic_score": deterministic.get("score", 0),
            "combined_score": deterministic.get("score", 0),
            "critical_issues": deterministic_issues,
            "semantic_errors": [],
            "style_suggestions": [str(item) for item in deterministic.get("minor_issues", [])],
            "feedback": str(deterministic.get("feedback") or "Deterministic validation failed."),
            "semantic_passed": False,
            "needs_retry": True,
        }

    client = llm or get_reviewer_llm()
    candidate_for_review = translation
    print("SEMANTIC REVIEW INPUT DEBUG")
    print(f"SOURCE SENT TO REVIEWER: {source!r}")
    print(f"CANDIDATE SENT TO REVIEWER: {candidate_for_review!r}")
    print(f"SOURCE LENGTH: {len(source)}")
    print(f"CANDIDATE LENGTH: {len(candidate_for_review)}")
    print("------------------------------------------------------------")
    print("🤖 Running semantic review")
    response = _invoke_llm_with_retries(
        client,
        _review_messages(source, candidate_for_review),
        operation_name="semantic review",
        max_attempts=api_attempts,
    )
    review = _normalize_reviewer_verdict(_parse_review(response))
    deterministic_score = deterministic.get("score", 0)
    categories = _review_categories(review)
    semantic_issues = categories["semantic_errors"]
    reviewer_critical_issues = categories["critical_issues"]
    style_issues = categories["style_suggestions"]
    combined_critical_issues = reviewer_critical_issues + list(deterministic.get("critical_issues", []))
    combined_minor_issues = style_issues + list(deterministic.get("minor_issues", []))
    semantic_score = review["score"]
    combined_score = min(deterministic_score, semantic_score)
    deterministic_is_sound = (
        bool(deterministic.get("passed"))
        and not _contains_non_latin_alphabetic(translation)
    )
    review_passed = (
        not review.get("review_error")
        and deterministic_is_sound
        and bool(review.get("passed"))
        and not combined_critical_issues
        and not semantic_issues
    )
    review["critical_issues"] = combined_critical_issues
    review["semantic_errors"] = semantic_issues
    review["style_suggestions"] = combined_minor_issues
    review["llm_score"] = semantic_score
    review["semantic_score"] = semantic_score
    review["deterministic_score"] = deterministic_score
    review["combined_score"] = combined_score
    # `score` is the reviewer diagnostic score in the canonical schema;
    # combined_score remains separately available for observability.
    review["score"] = semantic_score
    review["semantic_passed"] = review_passed
    review["passed"] = review_passed
    review["needs_retry"] = (
        bool(review.get("review_error"))
        or not review_passed
        or bool(combined_critical_issues)
        or bool(semantic_issues)
    )
    if combined_critical_issues:
        review["feedback"] = "; ".join(filter(None, [review.get("feedback", ""), *combined_critical_issues]))
    semantic_verdict = (
        "FAIL" if not review_passed else
        "STYLE_WARNING" if combined_minor_issues else
        "PASS"
    )
    print(f"🤖 Semantic Verdict: {semantic_verdict}")
    print(f"🤖 Semantic Score: {semantic_score}/100")
    print(f"🤖 Semantic Issues: {semantic_issues or 'none'}")
    print(f"🤖 Style Issues: {combined_minor_issues or 'none'}")
    print(f"📊 Combined Diagnostic Score: {combined_score}/100")
    retry_decision = "required" if review["needs_retry"] else "not required"
    print(f"🔁 Retry decision: {retry_decision}")
    return review


def _extract_rejected_phrases(feedback: str) -> list[str]:
    """Extract quoted or backtick-delimited phrases rejected by the reviewer."""
    if not feedback:
        return []

    candidates: list[str] = []
    for pattern in (r"['\"]([^'\"\n]{2,120})['\"]", r"`([^`\n]{2,120})`"):
        candidates.extend(re.findall(pattern, feedback))

    rejected: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        phrase = re.sub(r"\s+", " ", candidate).strip()
        key = phrase.casefold()
        if phrase and key not in seen:
            seen.add(key)
            rejected.append(phrase)
    return rejected[:20]


def _translation_messages(
    chunk: str,
    feedback: str = "",
    *,
    attempt: int = 1,
    previous_output: str = "",
    previous_translation: str = "",
    previous_feedback: str = "",
    forbidden_script_detected: bool = False,
    repeated_output_detected: bool = False,
    timestamp_failure_detected: bool = False,
) -> list[Any]:
    """Build the normal translation prompt or a strict correction retry."""
    # Keep these values initialized and backward-compatible for existing callers
    # that still provide previous_output/feedback instead of the newer names.
    prior_translation = _sanitize_retry_context(
        (previous_translation or previous_output).strip()
    )
    prior_feedback = _sanitize_retry_context(
        (previous_feedback or feedback).strip()
    )
    # Keep correction prompts concise for the configured Mistral model while retaining the
    # complete original source as the authoritative input.
    if len(prior_translation) > 2400:
        prior_translation = prior_translation[:2400].rsplit("\n", 1)[0]
    if len(prior_feedback) > 1600:
        prior_feedback = prior_feedback[:1600]

    if attempt <= 1:
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    ROMAN_URDU_FEW_SHOT_EXAMPLES
                    + "\n\n"
                    + ROMAN_URDU_VOCABULARY_GUIDANCE
                    + "\n"
                    + ROMAN_URDU_TIMESTAMP_GUIDANCE
                    + "\n\nSOURCE TRANSCRIPT TEXT (timestamps removed; translate only these lines):\n"
                    + chunk
                )
            ),
        ]

    if forbidden_script_detected:
        forbidden_retry_prompt = (
            "You translate English into Pakistani Roman Urdu.\n\n"
            "INPUT:\n"
            + chunk
            + "\n\n"
            + ROMAN_URDU_TIMESTAMP_GUIDANCE
            + "\n\nOUTPUT RULES:\n"
            "1. Write Roman Urdu using English/Latin letters only.\n"
            "2. Do not use Urdu, Arabic, Hindi, or Devanagari script.\n"
            "3. Do not output timestamps or replacement markers; Python restores the source timestamps after translation.\n"
            "4. Preserve the original meaning.\n"
            "5. Keep technical terms in English.\n"
            "6. Return only the translated text.\n\n"
            "Silently check that the output uses Latin alphabet letters only.\n\n"
            "OUTPUT:\n"
        )
        return [HumanMessage(content=forbidden_retry_prompt)]

    if attempt >= 3:
        # Attempt 3 is a fresh regeneration. Do not attach SYSTEM_PROMPT or any
        # previous output: SYSTEM_PROMPT contains illustrative native-script
        # examples and previous text can anchor a small model on bad phrasing.
        fresh_retry_prompt = (
            "You are translating English into natural Pakistani Roman Urdu.\n\n"
            + ROMAN_URDU_FEW_SHOT_EXAMPLES
            + "\n\n"
            + ROMAN_URDU_VOCABULARY_GUIDANCE
            + "\n"
            + ROMAN_URDU_TIMESTAMP_GUIDANCE
            + "\n\n"
            + (INVALID_TIMESTAMP_RETRY_GUIDANCE + "\n\n" if timestamp_failure_detected else "")
            + "Translate directly from the ORIGINAL ENGLISH text.\n\n"
            "ORIGINAL ENGLISH:\n"
            f"{chunk}\n\n"
            "RULES:\n"
            "- Use natural Pakistani Roman Urdu.\n"
            "- Use only Latin alphabet letters.\n"
            "- Do not output timestamps or replacement markers; Python restores the source timestamps after translation.\n"
            "- Preserve the meaning of every sentence.\n"
            "- Keep English technical terms unchanged.\n"
            "- Do not invent, transliterate, or create unfamiliar words.\n"
            "- Do not copy text from previous failed attempts.\n"
            "- Return only the translation.\n"
            "- No explanation.\n"
            "- No JSON.\n"
            "- No Markdown.\n\n"
            "Silently check the output before returning it.\n\n"
            "OUTPUT:\n"
        )
        return [HumanMessage(content=fresh_retry_prompt)]

    if prior_feedback.strip() == INVENTED_WORDS_FEEDBACK:
        invented_retry_prompt = (
            "You are translating English into natural Pakistani Roman Urdu.\n\n"
            "Translate directly from the ORIGINAL ENGLISH text.\n\n"
            "ORIGINAL ENGLISH:\n"
            + chunk
            + "\n\n"
            "QUALITY FEEDBACK:\n"
            + INVENTED_WORDS_FEEDBACK
            + "\n\n"
            + (INVALID_TIMESTAMP_RETRY_GUIDANCE + "\n\n" if timestamp_failure_detected else "")
            + "RULES:\n"
            "- Use simple, common Pakistani Roman Urdu vocabulary.\n"
            "- Use only Latin alphabet letters.\n"
            "- Do not output timestamps or replacement markers; Python restores the source timestamps after translation.\n"
            "- Keep English technical terms unchanged.\n"
            "- Return only the translation."
        )
        return [HumanMessage(content=invented_retry_prompt)]

    rejected_phrases = _extract_rejected_phrases(prior_feedback)
    retry_sections = [
        "You are correcting a failed English to Pakistani Roman Urdu translation.",
        "Translate directly from the ORIGINAL ENGLISH text.",
        "Do not translate from the previous failed output.",
        "ORIGINAL ENGLISH:\n" + chunk,
        "PREVIOUS LATIN-ONLY TRANSLATION:\n" + prior_translation,
        "QUALITY FEEDBACK:\n" + prior_feedback,
        ROMAN_URDU_FEW_SHOT_EXAMPLES,
        ROMAN_URDU_VOCABULARY_GUIDANCE,
        ROMAN_URDU_TIMESTAMP_GUIDANCE,
    ]
    if timestamp_failure_detected:
        retry_sections.append(INVALID_TIMESTAMP_RETRY_GUIDANCE)
    if rejected_phrases:
        retry_sections.append(
            "Do not use these rejected phrases:\n"
            + "\n".join(f"- {phrase}" for phrase in rejected_phrases)
        )
    if repeated_output_detected:
        retry_sections.append(
            "REPEATED OUTPUT WARNING:\n"
            "The previous attempt repeated rejected wording. Rebuild the answer "
            "with different, simple, natural Pakistani Roman Urdu."
        )
    retry_sections.extend([
        "TASK:\nRewrite the complete translation from scratch.",
        "RULES:\n"
        "- Use natural Pakistani Roman Urdu.\n"
        "- Use Latin alphabet characters only.\n"
        "- Preserve every timestamp placeholder exactly once, in the same order; do not create, remove, duplicate, reorder, or modify markers.\n"
        "- Preserve the meaning of every sentence.\n"
        "- Keep English technical terms unchanged.\n"
        "- Return only the corrected translation.\n"
        "- No explanation, JSON, or Markdown.",
    ])
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(retry_sections)),
    ]


def translate_chunk(
    chunk: str,
    llm: Any | None = None,
    chunk_number: int = 1,
    max_retries: int = QUALITY_MAX_ATTEMPTS,
    retry_delay: float = QUALITY_RETRY_DELAY,
    *,
    quality_retry_delay: float | None = None,
    max_quality_retries: int | None = None,
    api_max_attempts: int = API_MAX_ATTEMPTS,
    sleeper: Callable[[float], None] = time.sleep,
    reviewer_llm: Any | None = None,
) -> str:
    """
    Translate one chunk with separate API-level and quality-level retry loops.

    Quality retry logic:
      - After each attempt, run deterministic validation first.
      - If forbidden script (Urdu/Arabic/Devanagari) is detected:
          => Immediately reject WITHOUT running semantic review.
          => Discard the output and use a clean fresh-regeneration prompt on the next attempt.
      - On attempt 2 after a semantic failure:
          => Use the safe Latin translation, reviewer feedback, and rejected phrases.
      - On attempt 3:
          => Use only the original English chunk and concise fresh-regeneration rules.
      - If semantic review fails:
          => Preserve reviewer feedback for the attempt-2 correction prompt.
      - The original English source is always the authoritative source.
      - Retry always uses the ORIGINAL English chunk, never the bad output.
    """
    if not chunk or not chunk.strip():
        raise ValueError(f"Chunk {chunk_number} is empty.")

    client = llm or get_llm()
    # ``chunk`` remains the authoritative source for deterministic restoration.
    # Only this timestamp-free payload is ever sent to the translation model.
    spoken_chunk = _remove_timestamps_for_translation(chunk)
    quality_attempt_limit = max_retries if max_quality_retries is None else max_quality_retries
    if (
        not isinstance(quality_attempt_limit, int)
        or isinstance(quality_attempt_limit, bool)
        or quality_attempt_limit < 1
    ):
        raise ValueError("max_retries must be a positive integer")
    if not isinstance(api_max_attempts, int) or isinstance(api_max_attempts, bool) or api_max_attempts < 1:
        raise ValueError("api_max_attempts must be a positive integer")
    delay = retry_delay if quality_retry_delay is None else quality_retry_delay
    if delay < 0:
        raise ValueError("quality retry delay must be non-negative")
    quality_attempts = 0
    last_error: Exception | None = None
    feedback = ""
    previous_output = ""
    forbidden_script_detected = False
    repeated_output_detected = False

    previous_translation = ""
    previous_feedback = ""
    model_output = ""
    timestamp_failure_detected = False
    failure_type = ""

    for quality_attempt in range(1, quality_attempt_limit + 1):
        quality_attempts = quality_attempt
        print(f"\n🧪 Quality attempt {quality_attempt}/{quality_attempt_limit}")
        if quality_attempt == 1:
            print("📝 Translation mode: initial")
        elif quality_attempt == 2:
            print("🛠️ Translation mode: semantic correction")
        elif quality_attempt == 3:
            print("🔄 Translation mode: fresh regeneration")
            print("📚 Using Roman Urdu few-shot examples")
            print("🔤 Latin-only output enforced")
            print("🚫 Previous translations excluded from prompt")
        translated = ""
        deterministic: dict[str, Any] = {}
        try:
            response = _invoke_llm_with_retries(
                client,
                _translation_messages(
                    spoken_chunk,
                    feedback,
                    attempt=quality_attempt,
                    previous_output=previous_output,
                    previous_translation=previous_translation,
                    previous_feedback=previous_feedback,
                    forbidden_script_detected=forbidden_script_detected,
                    repeated_output_detected=repeated_output_detected,
                    timestamp_failure_detected=timestamp_failure_detected,
                ),
                operation_name="Translation API",
                max_attempts=api_max_attempts,
                sleeper=sleeper,
            )
            model_output = _normalize_translation_output(_message_content(response))

            # Timestamp structure is not delegated to the model. Python removes
            # accidental model-generated markers, restores the exact source
            # timestamp prefix per block, and then runs final validation.
            try:
                translated = _restore_chunk_timestamps(chunk, model_output)
                timestamp_failure_detected = False
                # Deterministic validation is the single gate before semantic review.
                deterministic = validate_roman_urdu_quality(chunk, translated)
            except ValueError as structural_error:
                timestamp_failure_detected = True
                deterministic = validate_roman_urdu_quality(
                    spoken_chunk,
                    _remove_generated_timestamp_markers(model_output),
                    require_timestamps=False,
                )
                deterministic["critical_issues"] = (
                    [str(structural_error)]
                    + list(deterministic.get("critical_issues", []))
                )
                deterministic["passed"] = False
                deterministic["retry_safe"] = False
                deterministic["feedback"] = "; ".join(deterministic["critical_issues"])
                translated = model_output
            if deterministic["critical_issues"]:
                print(f"⚠️  Deterministic validation failed (chunk {chunk_number}): "
                      f"{deterministic['feedback']}")
            else:
                print("🛡️  Deterministic validation passed")

            if not deterministic["passed"]:
                is_forbidden = any(
                    str(issue).startswith("Forbidden ")
                    for issue in deterministic["critical_issues"]
                )
                if is_forbidden:
                    forbidden_msg = (
                        "Forbidden Urdu/Arabic/Hindi script detected. "
                        "Output must contain Roman Urdu written using Latin characters only."
                    )
                    forbidden_script_detected = True
                    failure_type = "forbidden_script"
                    previous_translation = ""
                    previous_feedback = (
                        INVALID_TIMESTAMP_RETRY_GUIDANCE
                        if timestamp_failure_detected
                        else (
                            "The previous answer used forbidden non-Latin characters. "
                            "Generate the translation again from the original English text. "
                            "Use Latin alphabet characters only."
                        )
                    )
                    feedback = (
                        INVALID_TIMESTAMP_RETRY_GUIDANCE
                        if timestamp_failure_detected
                        else forbidden_msg
                    )
                    last_error = ValueError(forbidden_msg)
                    print(f"🚫 Forbidden script detected in chunk {chunk_number} -- rejecting immediately.")
                    print("🚫 Invalid non-Latin output discarded")
                else:
                    forbidden_script_detected = False
                    det_feedback = str(deterministic.get("feedback") or "Deterministic validation failed.")
                    failure_type = "deterministic"
                    feedback = det_feedback
                    last_error = RuntimeError(f"Deterministic quality issues: {det_feedback}")
                    previous_translation = (_remove_generated_timestamp_markers(model_output)
                                           if deterministic.get("retry_safe") else "")
                    if any(
                        "invented or meaningless words" in str(issue).casefold()
                        for issue in deterministic.get("critical_issues", [])
                    ):
                        previous_feedback = INVENTED_WORDS_FEEDBACK
                    else:
                        previous_feedback = det_feedback
                    print("📋 Validation feedback saved for retry only when context is safe")

                if _normalize_for_repeat_check(translated) == _normalize_for_repeat_check(previous_output):
                    repeated_output_detected = True
                    print(f"🔁 Model repeated identical invalid output (chunk {chunk_number}).")
                else:
                    repeated_output_detected = False
                previous_output = _remove_generated_timestamp_markers(model_output)
                if quality_attempt < quality_attempt_limit:
                    print(f"🛠️ Retry mode: {failure_type}")
                    print(f"⏳ Waiting {delay:.1f} seconds before quality retry...")
                    sleeper(delay)
                continue

            forbidden_script_detected = False
            timestamp_failure_detected = False

            # Semantic review runs only after deterministic validation passes.
            if deterministic["passed"]:
                review = review_roman_urdu_translation(
                    chunk,
                    translated,
                    reviewer_llm,
                    api_attempts=api_max_attempts,
                    deterministic_result=deterministic,
                )

                critical_issues = [str(item) for item in (review.get("critical_issues") or [])]
                semantic_errors = [str(item) for item in (review.get("semantic_errors") or [])]
                style_suggestions = [str(item) for item in (review.get("style_suggestions") or [])]
                if review.get("passed") is True and not review.get("review_error") and not critical_issues and not semantic_errors:
                    if style_suggestions:
                        print("⚠️ Style issues detected")
                        print("✅ Meaning preserved — no retry required")
                    else:
                        print("✅ Meaning preserved")
                    print(f"✅ Chunk {chunk_number} validated successfully")
                    return translated

                # Style suggestions never reach this branch. Only genuine
                # semantic/critical failures or an explicit reviewer error retry.
                reviewer_feedback_parts = critical_issues + semantic_errors
                if review.get("feedback") and reviewer_feedback_parts:
                    reviewer_feedback_parts.insert(0, str(review["feedback"]))
                if review.get("review_error"):
                    reviewer_feedback = "Semantic reviewer response could not be parsed; retry the review safely."
                    failure_type = "semantic_review_error"
                    print("⚠️ Semantic reviewer parsing failed; translation was not semantically judged.")
                else:
                    reviewer_feedback = "; ".join(dict.fromkeys(reviewer_feedback_parts)) or (
                        "Semantic reviewer returned passed=false; re-evaluate the complete translation."
                    )
                    failure_type = "semantic"
                print(f"⚠️  Semantic review failed (chunk {chunk_number}): {reviewer_feedback}")
                print("🤖 Semantic Verdict: FAIL")

                # Detect repeated bad output after semantic failure
                if _normalize_for_repeat_check(translated) == _normalize_for_repeat_check(previous_output):
                    repeated_output_detected = True
                    print(f"🔁 Model repeated identical output after semantic failure (chunk {chunk_number}).")
                else:
                    repeated_output_detected = False

                previous_output = _remove_generated_timestamp_markers(model_output)
                feedback = reviewer_feedback
                last_error = RuntimeError(f"Semantic quality issues: {reviewer_feedback}")
                print(f"📋 Failure type: {failure_type}")
                print("📄 Previous Latin translation preserved")

                # Save only genuine semantic FAIL feedback. Style warnings and
                # alternate wording suggestions must never be sent as semantic errors.
                previous_translation = _remove_generated_timestamp_markers(model_output)
                previous_feedback = reviewer_feedback
                print("📋 Genuine semantic feedback saved for retry")
        except RateLimitError as exc:
            # A 429 is an API failure, not a quality failure: never consume a
            # remaining quality retry. Surface it distinctly so the UI can
            # render a friendly rate-limit message with the Retry-After hint.
            # NOTE: this branch MUST precede the RetryableAPIError branch.
            failure_type = "rate_limit"
            last_error = exc
            raise TranslationPipelineError(
                f"Chunk {chunk_number} rate-limited (HTTP 429) at quality attempt "
                f"{quality_attempt}/{quality_attempt_limit}: {exc}",
                rate_limited=True,
                retry_after=exc.retry_after,
            ) from exc
        except (RetryableAPIError, PermanentAPIError) as exc:
            failure_type = "api"
            last_error = exc
            raise TranslationPipelineError(
                f"Chunk {chunk_number} API failure at quality attempt "
                f"{quality_attempt}/{quality_attempt_limit}: {exc}",
            ) from exc
        except Exception as exc:
            failure_type = "api"
            last_error = exc
            print(f"⚠️  Unexpected error during chunk {chunk_number} quality attempt {quality_attempt}: {exc}")

            # Exception safety: preserve a generated translation only when it
            # is Latin-only; forbidden-script output must never enter context.
            if (
                translated
                and deterministic.get("passed")
                and deterministic.get("retry_safe", True)
                and not _contains_non_latin_alphabetic(translated)
            ):
                previous_translation = _remove_generated_timestamp_markers(model_output)
            else:
                previous_translation = ""
            previous_feedback = str(exc)
            feedback = previous_feedback

        if quality_attempt < quality_attempt_limit:
            if previous_feedback.strip():
                print(f"🛠️ Retry mode: {failure_type or 'api'}")
            print(f"⏳ Waiting {delay:.1f} seconds before quality retry...")
            sleeper(delay)

    raise TranslationPipelineError(
        f"Chunk {chunk_number} failed after {quality_attempts}/{quality_attempt_limit} quality attempts; "
        f"last error: {last_error}",
    ) from last_error


def translate_to_roman_urdu(
    transcript: str,
    model: str = MISTRAL_MODEL,
    temperature: float = MISTRAL_TEMPERATURE,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = QUALITY_MAX_ATTEMPTS,
    retry_delay: int = 3,
) -> str:
    """Translate the complete transcript into Roman Urdu."""
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    transcript = transcript.strip()

    # ------------------------------------------------
    # SPLIT TRANSCRIPT
    # ------------------------------------------------
    chunks = split_transcript(transcript=transcript, max_chars=max_chars)

    if not chunks:
        raise ValueError("No transcript chunks were generated.")

    total_chunks = len(chunks)

    # ------------------------------------------------
    # INITIALIZE MISTRAL AI CLIENT
    # (Provider status printed inside _get_llm)
    # ------------------------------------------------
    llm = _get_llm(model=model, temperature=temperature)

    print(f"\nInput chars  : {len(transcript)}")
    print(f"Total chunks : {total_chunks}")
    print(f"Max retries  : {max_retries}")
    print("=" * 60)

    # ------------------------------------------------
    # TRANSLATE CHUNKS
    # ------------------------------------------------
    completed: list[str] = []

    for number, chunk in enumerate(chunks, start=1):
        print(f"\n🌐 Translating chunk {number}/{total_chunks}...")

        try:
            translated = translate_chunk(
                chunk=chunk,
                llm=llm,
                chunk_number=number,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            completed.append(translated)

        except Exception as exc:
            partial = "\n\n".join(completed)
            raise TranslationPipelineError(
                f"Translation failed at chunk {number}/{total_chunks}; "
                f"completed chunks preserved: {len(completed)}; "
                f"last error: {exc}",
                completed_chunks=completed,
                partial_translation=partial,
            ) from exc

    # ------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------
    roman_urdu_transcript = "\n\n".join(completed)

    if not roman_urdu_transcript.strip():
        raise RuntimeError("Final Roman Urdu transcript is empty.")

    print("\n" + "=" * 60)
    print("ROMAN URDU TRANSLATION COMPLETED")
    print("=" * 60)
    print(f"Output chars : {len(roman_urdu_transcript)}")
    print(f"Chunks       : {len(completed)}/{total_chunks} successful")
    print("=" * 60)

    return roman_urdu_transcript


__all__ = [
    "ALLOWED_TECHNICAL_TERMS",
    "MIN_QUALITY_SCORE",
    "EXCESSIVE_ENGLISH_RATIO_THRESHOLD",
    "MISTRAL_MODEL",
    "MISTRAL_REVIEWER_MODEL",
    "MISTRAL_TEMPERATURE",
    "MISTRAL_REVIEW_TEMPERATURE",
    "MISTRAL_API_KEY_ENV",
    "MISTRAL_TIMEOUT",
    "MISTRAL_MAX_RETRIES",
    "MISTRAL_MAX_TOKENS",
    "PermanentAPIError",
    "RetryableAPIError",
    "TranslationPipelineError",
    "SYSTEM_PROMPT",
    "REVIEW_SYSTEM_PROMPT",
    "_get_llm",
    "get_llm",
    "get_translator_llm",
    "get_reviewer_llm",
    "_message_content",
    "_normalize_translation_output",
    "_find_suspicious_invented_words",
    "_invoke_llm_with_retries",
    "_is_allowed_technical_term",
    "_normalize_phrase",
    "_extract_timestamps",
    "_remove_timestamps_for_translation",
    "_restore_chunk_timestamps",
    "_sanitize_retry_context",
    "_STRUCTURAL_MARKER_PATTERN",
    "_find_duplicate_lines",
    "_allowed_technical_phrase_words",
    "_find_suspicious_english_words",
    "_contains_forbidden_script",
    "ROMAN_URDU_FEW_SHOT_EXAMPLES",
    "ROMAN_URDU_TIMESTAMP_GUIDANCE",
    "INVALID_TIMESTAMP_RETRY_GUIDANCE",
    "normalize_for_semantic_review",
    "_contains_non_latin_alphabetic",
    "_print_mistral_status",
    "split_transcript",
    "validate_roman_urdu_script",
    "validate_roman_urdu_quality",
    "review_roman_urdu_translation",
    "translate_chunk",
    "translate_to_roman_urdu",
]
