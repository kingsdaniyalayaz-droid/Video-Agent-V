import unittest
import os
from pathlib import Path
from core.transcriber import validate_audio_file, ALLOWED_BASE_DIRS


class TestAudioPathValidation(unittest.TestCase):

    def setUp(self):
        # Set ya list dono ko safely handle karein
        self.allowed_dir = next(iter(ALLOWED_BASE_DIRS))
        self.valid_file = self.allowed_dir / "test_sample.wav"
        self.valid_file.write_bytes(b"RIFF dummy audio data")

    def tearDown(self):
        if self.valid_file.exists():
            self.valid_file.unlink()

    def test_valid_audio_file_passes(self):
        """Allowed directory ke andar valid file pass honi chahiye."""
        resolved = validate_audio_file(self.valid_file)
        self.assertEqual(resolved, self.valid_file.resolve())

    def test_symlink_is_rejected(self):
        """Symlink par ValueError aana chahiye."""
        symlink_path = self.allowed_dir / "test_symlink.wav"
        try:
            os.symlink(self.valid_file, symlink_path)
            with self.assertRaises(ValueError):
                validate_audio_file(symlink_path)
        except (OSError, NotImplementedError):
            pass
        finally:
            if symlink_path.is_symlink() or symlink_path.exists():
                symlink_path.unlink()

    def test_unsupported_extension_is_rejected(self):
        """.txt jaise disallowed formats reject hone chahiyein."""
        txt_file = self.allowed_dir / "test_invalid.txt"
        txt_file.write_text("not an audio")
        try:
            with self.assertRaises(ValueError):
                validate_audio_file(txt_file)
        finally:
            if txt_file.exists():
                txt_file.unlink()

    def test_empty_file_is_rejected(self):
        """0 bytes file reject honi chahiye."""
        empty_file = self.allowed_dir / "empty.wav"
        empty_file.touch()
        try:
            with self.assertRaises(ValueError):
                validate_audio_file(empty_file)
        finally:
            if empty_file.exists():
                empty_file.unlink()


if __name__ == "__main__":
    unittest.main()
