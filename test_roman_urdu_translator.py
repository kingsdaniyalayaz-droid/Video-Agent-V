"""Regression tests for Roman Urdu quality validation (English detection).

Covers the bug where common Roman Urdu words (aap, aapka, aapko, apna,
kaam, karni, ...) were incorrectly flagged as suspicious English.

Run:  python test_roman_urdu_translator.py
Or:   pytest test_roman_urdu_translator.py
"""

import unittest

from core.roman_urdu_translator import (
    COMMON_ROMAN_URDU_WORDS,
    SAFE_ROMAN_URDU_ENGLISH_ALLOWLIST,
    _excessive_english_ratio,
    _find_suspicious_english_words,
    _find_suspicious_invented_words,
    _is_acceptable_compound_token,
    _is_genuinely_untranslated_english,
    _normalize_lookup_token,
    validate_roman_urdu_quality,
)

MIXED_ROMAN_URDU_SENTENCE = (
    "Aap apna kaam kar sakte hain. "
    "Aapka account active hai aur aapko file upload karni hai."
)

MOSTLY_ENGLISH_SENTENCE = (
    "This is a very important topic and you should understand it carefully."
)

REQUIRED_COMMON_WORDS = frozenset(
    {
        # Pronouns / possessives
        "aap", "aapka", "aapki", "aapke", "aapko",
        "ap", "apna", "apni", "apne",
        "mera", "meri", "mere", "mujhe",
        "tumhara", "tumhari", "tumhare",
        # Connectors / function words
        "agar", "lekin", "magar", "abhi", "phir", "yahan", "kahan",
        "kuch", "koi", "sab", "itna", "utna", "kitna",
        # Variants
        "kam", "kyu", "bohot", "bahut", "acha", "achha",
        "thik", "theek", "sahi", "shukria", "shukriya",
        # Everyday words
        "ghar", "khana", "pani", "zindagi", "bazaar", "roz",
        "dost", "log", "cheez", "sawal", "jawab", "tareeqa",
        "shuru", "khatam",
        # Verb forms
        "karni", "karenge", "kiya", "kiye", "jata", "humein", "isay", "sahulat",
    }
)

NOT_FLAGGED_ROMAN_URDU_WORDS = ("Aap", "Aapka", "aapko", "apna", "kaam", "karni")


class CommonRomanUrduVocabularyTests(unittest.TestCase):
    def test_mandated_words_are_in_common_vocabulary(self):
        missing = sorted(
            word for word in REQUIRED_COMMON_WORDS if word not in COMMON_ROMAN_URDU_WORDS
        )
        self.assertEqual(missing, [])


class SuspiciousEnglishDetectionTests(unittest.TestCase):
    def test_common_roman_urdu_words_not_flagged(self):
        suspicious = _find_suspicious_english_words(MIXED_ROMAN_URDU_SENTENCE)
        normalized_suspicious = {_normalize_lookup_token(word) for word in suspicious}
        for word in NOT_FLAGGED_ROMAN_URDU_WORDS:
            self.assertNotIn(
                _normalize_lookup_token(word),
                normalized_suspicious,
                f"{word!r} must not be flagged as suspicious English",
            )
        self.assertNotIn("Aap", suspicious)
        self.assertNotIn("aap", suspicious)

    def test_case_normalization_of_aap_variants(self):
        for casing in ("Aap", "aap", "AAP"):
            self.assertEqual(_normalize_lookup_token(casing), "aap")
        for text in ("Aap", "aap", "AAP", "Aap aap AAP"):
            self.assertEqual(_find_suspicious_english_words(text), [])

    def test_mixed_roman_urdu_plus_allowed_technical_terms(self):
        suspicious = {
            _normalize_lookup_token(word)
            for word in _find_suspicious_english_words(MIXED_ROMAN_URDU_SENTENCE)
        }
        # Allowed technical terms stay exempt.
        for technical in ("file", "upload", "account"):
            self.assertNotIn(technical, suspicious)
        # Genuinely untranslated English is still caught (detection not disabled).
        self.assertIn("active", suspicious)

    def test_excessive_english_ratio_drops_for_mixed_roman_urdu(self):
        ratio = _excessive_english_ratio(MIXED_ROMAN_URDU_SENTENCE)
        self.assertLess(ratio, 0.35)

    def test_mostly_untranslated_english_still_detected(self):
        text = MOSTLY_ENGLISH_SENTENCE
        suspicious = _find_suspicious_english_words(text)
        self.assertTrue(_is_genuinely_untranslated_english(text, suspicious))
        self.assertGreaterEqual(_excessive_english_ratio(text), 0.60)
        result = validate_roman_urdu_quality(text, text)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("predominantly ordinary English" in issue for issue in result["critical_issues"]),
            "predominantly-English translation must fail validation",
        )

    def test_technical_terms_remain_allowlisted(self):
        for term in ("account", "file", "upload", "download", "login",
                     "password", "video", "website"):
            self.assertIn(
                _normalize_lookup_token(term),
                SAFE_ROMAN_URDU_ENGLISH_ALLOWLIST,
                f"{term!r} must be handled via the existing technical allowlist",
            )

    def test_validator_accepts_mixed_roman_urdu(self):
        result = validate_roman_urdu_quality(
            MIXED_ROMAN_URDU_SENTENCE, MIXED_ROMAN_URDU_SENTENCE
        )
        self.assertTrue(result["passed"])
        for flagged in NOT_FLAGGED_ROMAN_URDU_WORDS:
            self.assertNotIn(flagged, result["feedback"])


    def test_valid_roman_urdu_outside_vocabulary_not_critical(self):
        # Regression: these valid Roman Urdu sentences used to be classified
        # as "predominantly ordinary English" (a critical failure) purely
        # because their words are outside the (necessarily incomplete) set.
        sentences = (
            "Mere ghar ke samnay aik naya masjid bana hai.",
            "Humne kal raat khana khaya aur phir so gaye.",
        )
        for text in sentences:
            suspicious = _find_suspicious_english_words(text)
            self.assertFalse(
                _is_genuinely_untranslated_english(text, suspicious),
                f"valid Roman Urdu misclassified as untranslated English: {text!r}",
            )
            result = validate_roman_urdu_quality(text, text)
            self.assertTrue(result["passed"], f"valid Roman Urdu failed validation: {text!r}")
            self.assertTrue(
                all(
                    "predominantly ordinary English" not in issue
                    for issue in result["critical_issues"]
                ),
                f"false critical English issue for: {text!r}",
            )

    def test_mixed_roman_urdu_with_technical_terms_not_english(self):
        # Mixed Roman Urdu + allowed technical terms must never be classified
        # as predominantly ordinary English.
        texts = (
            "Aap apna kaam kar sakte hain. Aapka account active hai aur aapko file upload karni hai.",
            "Aaj hum Data Science aur Machine Learning seekhenge.",
            "Isay web development aur automation mein bhi use kiya jata hai.",
        )
        for text in texts:
            suspicious = _find_suspicious_english_words(text)
            self.assertFalse(
                _is_genuinely_untranslated_english(text, suspicious),
                f"mixed Roman Urdu misclassified as untranslated English: {text!r}",
            )
            result = validate_roman_urdu_quality(text, text)
            self.assertTrue(result["passed"], f"mixed Roman Urdu failed validation: {text!r}")

    def test_project_own_examples_do_not_self_flag(self):
        # Canonical examples from the file's prompt/guidance must be clean.
        examples = (
            "Aaj hum Python programming ke bare mein baat karenge.",
            "Python ek mashhoor programming language hai.",
            "Lists humein ek single variable mein multiple values store karne ki sahulat deti hain.",
            "Isay web development aur automation mein bhi use kiya jata hai.",
        )
        for example in examples:
            suspicious = _find_suspicious_english_words(example)
            self.assertEqual(
                suspicious, [], f"project canonical example self-flags: {example!r} -> {suspicious}"
            )


class ProductionRegressionTests(unittest.TestCase):
    """Regression tests for the production bug: valid translations (proper
    nouns, hyphenated compounds, common Roman Urdu variants) were falsely
    rejected as invented words or untranslated English."""

    ROMAN_URDU_VARIANT_WORDS = (
        "pichhle", "aage", "aata", "aaya", "aaye", "accha",
    )

    def test_anthropic_not_flagged_when_present_in_source(self):
        source = "Anthropic released a new model that everyone is talking about."
        translation = "Anthropic ka naya model bahut accha hai aur sab isay pasand kar rahe hain."
        self.assertEqual(_find_suspicious_invented_words(source, translation), [])
        result = validate_roman_urdu_quality(source, translation)
        self.assertTrue(result["passed"])
        self.assertNotIn("Anthropic", result["feedback"])

    def test_common_proper_product_names_not_falsely_flagged(self):
        for name in ("Anthropic", "OpenAI", "Claude", "ChatGPT", "YouTube", "GitHub"):
            self.assertEqual(
                _find_suspicious_invented_words("", name),
                [],
                f"{name!r} must not be flagged as invented even without source context",
            )
            self.assertNotIn(
                name,
                _find_suspicious_english_words(name),
                f"{name!r} must not be flagged as untranslated English",
            )

    def test_hyphenated_terms_handled_correctly(self):
        self.assertTrue(_is_acceptable_compound_token("user-friendly"))
        self.assertEqual(_find_suspicious_invented_words("", "user-friendly"), [])
        self.assertNotIn("user-friendly", _find_suspicious_english_words("user-friendly"))
        source = "We will discuss a user-friendly interface."
        translation = "Hum user-friendly interface ke bare mein baat karenge."
        result = validate_roman_urdu_quality(source, translation)
        self.assertTrue(result["passed"], result["critical_issues"])

    def test_hyphen_does_not_allow_gibberish(self):
        # The compound allowance must not open a hyphen-loophole.
        self.assertEqual(_find_suspicious_invented_words("", "qmdx-wzpf"), ["qmdx-wzpf"])

    def test_common_roman_urdu_variants_not_flagged_as_english(self):
        for word in self.ROMAN_URDU_VARIANT_WORDS:
            self.assertNotIn(
                word,
                _find_suspicious_english_words(word),
                f"{word!r} must not be flagged as untranslated English",
            )
            self.assertNotIn(
                word,
                _find_suspicious_invented_words("", word),
                f"{word!r} must not be flagged as invented",
            )

    def test_roman_urdu_variant_sentence_passes_validation(self):
        translation = (
            "Pichhle hafte wo aage aata tha. "
            "Aaj woh phir aaya aur aaye hain. Bahut accha laga."
        )
        result = validate_roman_urdu_quality("What happened last week?", translation)
        self.assertTrue(result["passed"], result["critical_issues"])
        self.assertTrue(
            all("predominantly ordinary English" not in issue
                for issue in result["critical_issues"])
        )

    def test_valid_roman_urdu_sentence_with_proper_nouns_passes(self):
        translation = (
            "Mere ghar ke samnay aik naya masjid bana hai. "
            "Anthropic aur OpenAI ke tools bahut aasan hain."
        )
        result = validate_roman_urdu_quality(translation, translation)
        self.assertTrue(result["passed"], result["critical_issues"])

    def test_mostly_untranslated_english_still_fails(self):
        text = "This is a very important topic and we should discuss it."
        self.assertTrue(_is_genuinely_untranslated_english(
            text, _find_suspicious_english_words(text)))
        result = validate_roman_urdu_quality(text, text)
        self.assertFalse(result["passed"])

    def test_known_gibberish_still_detected(self):
        # Canonical known-invented word must stay a hard failure.
        self.assertEqual(_find_suspicious_invented_words("", "dhareenat"), ["dhareenat"])
        result = validate_roman_urdu_quality(
            "What does that mean?", "Dhareenat ka matlab kya hai?"
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("invented" in issue for issue in result["critical_issues"]),
            result["critical_issues"],
        )
        # Novel vowel-less gibberish must also stay detected (no regression).
        self.assertEqual(_find_suspicious_invented_words("", "khzjq"), ["khzjq"])


class FalsePositiveMinorWarningRegressionTests(unittest.TestCase):
    """The validator used to pass valid mixed Roman Urdu but still emit a
    false-positive minor warning: "Some ordinary English words remain
    untranslated." The warning must not fire for source-supported proper
    nouns, hyphenated compounds, common Roman Urdu variants, or legitimate
    mixed technical/product vocabulary."""

    PRODUCTION_SOURCE = "Anthropic released a user-friendly product last week."
    PRODUCTION_TRANSLATION = (
        "Anthropic ne pichhle hafte ek user-friendly product release kiya "
        "jo aasan aur accha hai."
    )

    def test_no_false_positive_english_warning(self):
        result = validate_roman_urdu_quality(
            self.PRODUCTION_SOURCE, self.PRODUCTION_TRANSLATION
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["critical_issues"], [])
        self.assertEqual(result["minor_issues"], [], result["feedback"])
        for field in ("minor_issues", "feedback", "critical_issues"):
            self.assertFalse(
                any("ordinary English" in str(issue) for issue in result[field]),
                f"false positive English warning in {field}: {result[field]}",
            )
        # Nothing in the translation should even look suspicious.
        self.assertEqual(
            _find_suspicious_english_words(self.PRODUCTION_TRANSLATION), []
        )

    def test_mixed_product_technical_vocabulary_no_false_warning(self):
        texts = (
            # product / release / access used as legitimate mixed technical terms
            "Hum ne naya product release kiya aur sab ko access diya. Bahut accha.",
            # account / upload kept in English by convention
            "Aap apna account access kar sakte hain aur file upload kar sakte hain.",
            # hyphenated compound in mixed usage
            "Yeh ek user-friendly interface hai jo aasan hai.",
        )
        for text in texts:
            result = validate_roman_urdu_quality(text, text)
            self.assertTrue(result["passed"], result["critical_issues"])
            self.assertEqual(result["minor_issues"], [], result["feedback"])
            self.assertFalse(
                any("ordinary English" in str(issue) for issue in result["minor_issues"]),
                f"false positive English warning for: {text!r} -> {result['minor_issues']}",
            )

    def test_anthropic_accepted_when_source_supported(self):
        source = "Anthropic released a new model that everyone is talking about."
        translation = "Anthropic ka naya model bahut accha hai aur sab isay pasand kar rahe hain."
        self.assertEqual(_find_suspicious_english_words(translation), [])
        result = validate_roman_urdu_quality(source, translation)
        self.assertTrue(result["passed"])
        self.assertEqual(result["minor_issues"], [], result["feedback"])

    def test_mostly_untranslated_english_still_fails(self):
        # Genuine ordinary English must remain a hard failure (no silent pass).
        text = "This is a very important topic and we need to understand it carefully."
        self.assertTrue(
            _is_genuinely_untranslated_english(text, _find_suspicious_english_words(text))
        )
        result = validate_roman_urdu_quality(text, text)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("predominantly ordinary English" in issue
                for issue in result["critical_issues"]),
            result["critical_issues"],
        )

    def test_known_gibberish_still_fails(self):
        result = validate_roman_urdu_quality(
            "What does that mean?", "Dhareenat ka matlab kya hai?"
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("invented" in issue for issue in result["critical_issues"]),
            result["critical_issues"],
        )

    def test_existing_roman_urdu_variants_still_clean(self):
        for word in ("pichhle", "aage", "aata", "aaya", "aaye", "accha",
                     "aasan", "acha", "aap", "aapka", "aapko", "apna",
                     "kaam", "karni"):
            self.assertNotIn(
                word, _find_suspicious_english_words(word),
                f"{word!r} must not be flagged as untranslated English",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
