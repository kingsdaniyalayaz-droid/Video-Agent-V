import unittest
import os
from unittest.mock import patch
from core.translation_error_handling import redact_sensitive_info, _redact


class TestMultiProviderRedaction(unittest.TestCase):

    def test_openrouter_key_is_redacted(self):
        msg = "Error from https://openrouter.ai/api/v1 with key sk-or-v1-abcdef1234567890abcdef1234567890: 401 Unauthorized"
        clean = redact_sensitive_info(msg)
        self.assertNotIn("sk-or-v1-abcdef1234567890abcdef1234567890", clean)
        self.assertIn("[REDACTED]", clean)

    def test_openai_key_is_redacted(self):
        msg = "Failed with key sk-abcdef1234567890abcdef1234567890"
        clean = redact_sensitive_info(msg)
        self.assertNotIn("sk-abcdef1234567890abcdef1234567890", clean)
        self.assertIn("[REDACTED]", clean)

    def test_gemini_key_is_redacted(self):
        msg = "Google Gemini call failed: AIzaSyD12345678901234567890123456789012"
        clean = redact_sensitive_info(msg)
        self.assertNotIn("AIzaSyD12345678901234567890123456789012", clean)
        self.assertIn("[REDACTED]", clean)

    def test_bearer_header_is_redacted(self):
        msg = "Request headers: {Authorization: Bearer secret_token_123456789012345}"
        clean = redact_sensitive_info(msg)
        self.assertNotIn("secret_token_123456789012345", clean)
        self.assertIn("[REDACTED]", clean)

    def test_url_query_secret_is_redacted(self):
        msg = "GET https://generativelanguage.googleapis.com/v1/models?key=AIzaSySecretKey1234567890123456789012 HTTP 429"
        clean = redact_sensitive_info(msg)
        self.assertNotIn("AIzaSySecretKey1234567890123456789012", clean)
        self.assertIn("[REDACTED]", clean)

    def test_dynamic_environment_secrets_are_redacted(self):
        with patch.dict(os.environ, {"CUSTOM_PROVIDER_API_KEY": "super_secret_custom_token_999"}):
            msg = "Custom API call failed using super_secret_custom_token_999 token"
            clean = redact_sensitive_info(msg)
            self.assertNotIn("super_secret_custom_token_999", clean)
            self.assertIn("[REDACTED]", clean)


if __name__ == "__main__":
    unittest.main()
