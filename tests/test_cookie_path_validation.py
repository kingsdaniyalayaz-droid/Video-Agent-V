import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from utils import audio_processor
from utils.audio_processor import get_cookie_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NON_EXISTENT_DEFAULT = PROJECT_ROOT / "non_existent_cookies_fallback.txt"


class TestCookiePathValidation(unittest.TestCase):

    def setUp(self):
        self.valid_cookie = PROJECT_ROOT / "test_valid_cookies.txt"
        self.valid_cookie.write_text("# Netscape HTTP Cookie File\n.youtube.com TRUE / FALSE 1700000000 SID 12345")

    def tearDown(self):
        if self.valid_cookie.exists():
            self.valid_cookie.unlink()

    def test_valid_cookie_file_in_project_passes(self):
        """Allowed directory ke andar valid .txt cookie file pass honi chahiye."""
        with patch.dict(os.environ, {"YOUTUBE_COOKIES": str(self.valid_cookie)}):
            cookie_path = get_cookie_file()
            self.assertIsNotNone(cookie_path)
            self.assertEqual(cookie_path.name, "test_valid_cookies.txt")

    def test_path_traversal_outside_project_rejected(self):
        """Project se bahar ki arbitrary file reject honi chahiye."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as outside_file:
            outside_file.write(b"dummy cookies")
            outside_path = Path(outside_file.name)

        try:
            with patch.dict(os.environ, {"YOUTUBE_COOKIES": str(outside_path)}),                  patch.object(audio_processor, "DEFAULT_COOKIE_FILE", NON_EXISTENT_DEFAULT):
                self.assertIsNone(get_cookie_file())
        finally:
            if outside_path.exists():
                outside_path.unlink()

    def test_sensitive_files_blocked_as_cookies(self):
        """.env ya json secret files ko cookie ke taur par reject hona chahiye."""
        dummy_secret = PROJECT_ROOT / "api_keys_vault.json"
        dummy_secret.write_text("{\"api_key\": \"secret\"}")
        try:
            with patch.dict(os.environ, {"YOUTUBE_COOKIES": str(dummy_secret)}),                  patch.object(audio_processor, "DEFAULT_COOKIE_FILE", NON_EXISTENT_DEFAULT):
                self.assertIsNone(get_cookie_file())
        finally:
            if dummy_secret.exists():
                dummy_secret.unlink()

    def test_empty_cookie_file_rejected(self):
        """0 bytes file reject honi chahiye."""
        empty_cookie = PROJECT_ROOT / "empty_cookies.txt"
        empty_cookie.touch()
        try:
            with patch.dict(os.environ, {"YOUTUBE_COOKIES": str(empty_cookie)}),                  patch.object(audio_processor, "DEFAULT_COOKIE_FILE", NON_EXISTENT_DEFAULT):
                self.assertIsNone(get_cookie_file())
        finally:
            if empty_cookie.exists():
                empty_cookie.unlink()


if __name__ == "__main__":
    unittest.main()
