import unittest
from utils.audio_processor import is_youtube_url, extract_video_id, normalize_url


class TestYouTubeUrlSemantics(unittest.TestCase):

    def test_valid_standard_watch_urls(self):
        """Standard watch URLs with and without tracking params should parse correctly."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ&si=tracking123",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in valid_urls:
            self.assertTrue(is_youtube_url(url))
            self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")
            self.assertEqual(normalize_url(url), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_valid_short_urls(self):
        """youtu.be and /shorts/ URLs should parse correctly."""
        urls = [
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=120", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]
        for url, expected_id in urls:
            self.assertTrue(is_youtube_url(url))
            self.assertEqual(extract_video_id(url), expected_id)

    def test_invalid_video_id_lengths(self):
        """Video IDs not matching exactly 11 characters must be rejected."""
        invalid_id_urls = [
            "https://youtu.be/short",
            "https://youtu.be/toolongvideoid12345",
            "https://www.youtube.com/watch?v=short",
            "https://www.youtube.com/watch?v=toolongvideoid12345",
            "https://www.youtube.com/watch?v=invalid@char",
        ]
        for url in invalid_id_urls:
            self.assertFalse(is_youtube_url(url))
            self.assertIsNone(extract_video_id(url))

    def test_non_video_youtube_paths_rejected(self):
        """Non-video YouTube pages must be rejected."""
        non_video = [
            "https://www.youtube.com/feed/subscriptions",
            "https://www.youtube.com/channel/UC1234567890",
            "https://www.youtube.com/about",
            "https://www.youtube.com/playlist?list=PL12345",
        ]
        for url in non_video:
            self.assertFalse(is_youtube_url(url))
            self.assertIsNone(extract_video_id(url))

    def test_untrusted_hostnames_rejected(self):
        """Attacker domains mimicking YouTube must be rejected."""
        fake_urls = [
            "https://evil-youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com.attacker.com/watch?v=dQw4w9WgXcQ",
            "https://vimeo.com/12345678",
            "ftp://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in fake_urls:
            self.assertFalse(is_youtube_url(url))
            self.assertIsNone(extract_video_id(url))


if __name__ == "__main__":
    unittest.main()
