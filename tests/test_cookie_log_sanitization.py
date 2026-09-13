import unittest
import io
import sys
import os
from pathlib import Path
from unittest.mock import patch
from utils.audio_processor import build_youtube_options

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCookieLogSanitization(unittest.TestCase):

    def setUp(self):
        self.dummy_cookie = PROJECT_ROOT / "test_dummy_cookies.txt"
        self.dummy_cookie.write_text("# Netscape HTTP Cookie File\n.youtube.com TRUE / FALSE 1700000000 SID 12345")

    def tearDown(self):
        if self.dummy_cookie.exists():
            self.dummy_cookie.unlink()

    def test_logs_do_not_leak_absolute_cookie_path(self):
        """build_youtube_options call hone par logs mein absolute path expose nahi hona chahiye."""
        with patch.dict(os.environ, {"YOUTUBE_COOKIES": str(self.dummy_cookie)}):
            captured_stdout = io.StringIO()
            with patch("sys.stdout", captured_stdout):
                opts = build_youtube_options("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

            output = captured_stdout.getvalue()

            # 1. Absolute filesystem path logs mein NAHI hona chahiye
            self.assertNotIn(str(self.dummy_cookie), output)
            self.assertNotIn(str(PROJECT_ROOT), output)

            # 2. Sirf safe boolean status hona chahiye
            self.assertIn("YouTube authentication cookies: Enabled", output)

            # 3. ydl_opts ke andar actual file path pass hona chahiye taake yt-dlp download kaam kare
            self.assertEqual(opts.get("cookiefile"), str(self.dummy_cookie.resolve()))

    def test_logs_show_disabled_when_no_cookie(self):
        """Cookie na hone par logs mein Disabled status aana chahiye."""
        with patch.dict(os.environ, {"YOUTUBE_COOKIES": ""}),              patch("utils.audio_processor.get_cookie_file", return_value=None):
            captured_stdout = io.StringIO()
            with patch("sys.stdout", captured_stdout):
                opts = build_youtube_options("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

            output = captured_stdout.getvalue()
            self.assertIn("YouTube authentication cookies: Disabled", output)
            self.assertNotIn("cookiefile", opts)


if __name__ == "__main__":
    unittest.main()
