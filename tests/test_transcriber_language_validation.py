import unittest
from core.transcriber import validate_language


class TestTranscriberLanguageValidation(unittest.TestCase):

    def test_valid_iso_codes_pass(self):
        """Standard ISO codes ('en', 'ur', 'hi', 'fr', 'es') pass hone chahiyein."""
        for code in ["en", "ur", "hi", "fr", "es", "ar", "de"]:
            self.assertEqual(validate_language(code), code)
            self.assertEqual(validate_language(code.upper()), code)

    def test_full_language_names_normalize_to_iso(self):
        """Full names ('english', 'urdu') normalize ho kar 'en', 'ur' banen."""
        self.assertEqual(validate_language("english"), "en")
        self.assertEqual(validate_language("Urdu"), "ur")
        self.assertEqual(validate_language("Spanish"), "es")

    def test_unsupported_language_raises_value_error(self):
        """Ghair-mutalliqah ya invalid language par ValueError aana chahiye."""
        for invalid in ["xyz", "klingon", "123", "fake_lang"]:
            with self.assertRaises(ValueError):
                validate_language(invalid)

    def test_empty_language_raises_value_error(self):
        """Empty ya whitespace strings par ValueError aana chahiye."""
        for empty in ["", "   ", "\t\n"]:
            with self.assertRaises(ValueError):
                validate_language(empty)

    def test_invalid_types_raise_type_error(self):
        """None, int, bool par TypeError raise hona chahiye."""
        for bad_type in [None, 123, True, False, ["en"]]:
            with self.assertRaises(TypeError):
                validate_language(bad_type)


if __name__ == "__main__":
    unittest.main()
