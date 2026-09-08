import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_module():
    dotenv_module = sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
    if not hasattr(dotenv_module, "load_dotenv"):
        dotenv_module.load_dotenv = lambda override=True: None


    class _Message:
        def __init__(self, content=None, **_: object):
            self.content = content


    mistral_module = sys.modules.setdefault(
        "langchain_mistralai",
        types.ModuleType("langchain_mistralai"),
    )

    class _ChatMistralAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    mistral_module.ChatMistralAI = getattr(mistral_module, "ChatMistralAI", _ChatMistralAI)

    module_path = Path(__file__).resolve().parent.parent / "core" / "roman_urdu_translator.py"
    spec = importlib.util.spec_from_file_location("roman_urdu_translator_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class RomanUrduValidatorTests(unittest.TestCase):
    def test_subscribe_with_period_is_not_invented(self):
        suspicious = MODULE._find_suspicious_invented_words(
            "Please subscribe to the channel.",
            "Video pasand aaye to channel ko subscribe. kar dein.",
        )
        self.assertNotIn("subscribe", {word.casefold() for word in suspicious})

    def test_subscribe_with_exclamation_is_not_invented(self):
        suspicious = MODULE._find_suspicious_invented_words(
            "Please subscribe to the channel!",
            "Video pasand aaye to channel ko Subscribe! kar dein.",
        )
        self.assertNotIn("subscribe", {word.casefold() for word in suspicious})

    def test_youtube_is_not_invented(self):
        suspicious = MODULE._find_suspicious_invented_words(
            "Watch this on YouTube.",
            "Is video ko YouTube par dekhein.",
        )
        self.assertNotIn("youtube", {word.casefold() for word in suspicious})

    def test_python_is_not_invented(self):
        suspicious = MODULE._find_suspicious_invented_words(
            "We will learn Python.",
            "Aaj hum Python seekhenge.",
        )
        self.assertNotIn("python", {word.casefold() for word in suspicious})

    def test_api_is_not_invented(self):
        suspicious = MODULE._find_suspicious_invented_words(
            "Call the API.",
            "API ko call karein.",
        )
        self.assertNotIn("api", {word.casefold() for word in suspicious})

    def test_technical_mixed_roman_urdu_passes(self):
        result = MODULE.validate_roman_urdu_quality(
            "You can use a for loop and function in Python.",
            "Aap Python mein for loop aur function use kar sakte hain.",
            require_timestamps=False,
        )
        self.assertTrue(result["passed"], result)

    def test_predominantly_english_output_fails(self):
        result = MODULE.validate_roman_urdu_quality(
            "Explain this in Roman Urdu.",
            "This is a very important topic and you should understand it carefully.",
            require_timestamps=False,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("predominantly ordinary English" in issue for issue in result["critical_issues"]))

    def test_genuine_gibberish_fails(self):
        result = MODULE.validate_roman_urdu_quality(
            "Explain how to use this feature.",
            "Aap blorptz flanqwerty zzzx use karein.",
            require_timestamps=False,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("invented or meaningless words" in issue for issue in result["critical_issues"]))

    def test_punctuation_around_allowed_words_does_not_change_classification(self):
        suspicious = MODULE._find_suspicious_invented_words(
            "Please subscribe to our YouTube channel and download the file from the website.",
            "(Subscribe) YouTube, Python, API, website, download, aur file sab theek hain.",
        )
        normalized = {word.casefold() for word in suspicious}
        self.assertFalse({"subscribe", "youtube", "python", "api", "website", "download", "file"} & normalized)


if __name__ == "__main__":
    unittest.main()
