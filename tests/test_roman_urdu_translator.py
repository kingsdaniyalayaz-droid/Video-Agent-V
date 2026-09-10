import time
import pytest
from unittest.mock import MagicMock
import threading

import core.roman_urdu_translator
from core.roman_urdu_translator import (
    _invoke_llm_with_retries,
    _set_rate_limit_cooldown,
    _wait_for_rate_limit_cooldown,
    RateLimitError,
    RetryableAPIError,
    PermanentAPIError,
    TranslationPipelineError,
    API_MAX_ATTEMPTS
)

class FakeRateLimitException(Exception):
    def __init__(self, message, retry_after=None, status_code=429):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code
        self.rate_limited = True

@pytest.fixture(autouse=True)
def reset_cooldown():
    core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL = 0.0
    yield
    core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL = 0.0

@pytest.fixture
def mock_time(monkeypatch):
    current_time = [1000.0]
    
    def fake_monotonic():
        return current_time[0]
        
    def advance_time(amount):
        current_time[0] += amount
        
    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    return advance_time

def test_shared_cooldown_set_after_429(mock_time):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [FakeRateLimitException("Too many requests", retry_after=5.0), "Success"]
    
    def fake_sleep(duration):
        mock_time(duration)
        
    result = _invoke_llm_with_retries(mock_llm, ["Hello"], max_attempts=2, sleeper=fake_sleep)
    assert result == "Success"
    
    assert core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL > 1000.0

def test_second_request_waits_for_cooldown(mock_time):
    core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL = time.monotonic() + 5.0
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Success"
    
    sleep_calls = []
    def fake_sleep(duration):
        sleep_calls.append(duration)
        mock_time(duration)
        
    result = _invoke_llm_with_retries(mock_llm, ["Hello"], max_attempts=2, sleeper=fake_sleep)
    assert result == "Success"
    assert len(sleep_calls) >= 1
    assert sum(sleep_calls) >= 5.0

def test_retry_after_respected(mock_time):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [FakeRateLimitException("Rate limit", retry_after=42.0), "Success"]
    
    sleep_calls = []
    def fake_sleep(duration):
        sleep_calls.append(duration)
        mock_time(duration)
        
    result = _invoke_llm_with_retries(mock_llm, ["Test"], max_attempts=2, sleeper=fake_sleep)
    assert result == "Success"
    
    # Sleep should be at least 42 (due to retry_after)
    assert sum(sleep_calls) >= 42.0

def test_existing_cooldown_not_shortened(mock_time):
    now = time.monotonic()
    future = now + 100.0
    core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL = future
    
    _set_rate_limit_cooldown(1.0)
    assert core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL == future

def test_concurrent_callers_do_not_stampede(mock_time):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = FakeRateLimitException("Rate limit", retry_after=2.0)
    
    def fake_sleep(duration):
        mock_time(duration)
        
    with pytest.raises(RateLimitError):
        _invoke_llm_with_retries(mock_llm, ["Test"], max_attempts=2, sleeper=fake_sleep)
        
    assert core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL > 1000.0

def test_non_429_retryable_errors(mock_time):
    class Fake500(Exception):
        status_code = 500
        
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [Fake500("Internal error"), "Success"]
    
    sleep_calls = []
    def fake_sleep(duration):
        sleep_calls.append(duration)
        mock_time(duration)
        
    result = _invoke_llm_with_retries(mock_llm, ["Test"], max_attempts=2, sleeper=fake_sleep)
    assert result == "Success"
    
    assert len(sleep_calls) == 1
    assert core.roman_urdu_translator._RATE_LIMIT_COOLDOWN_UNTIL <= time.monotonic()

def test_permanent_error_remains_unchanged(mock_time):
    class Fake400(Exception):
        status_code = 400
        
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Fake400("Bad Request")
    
    def fake_sleep(duration):
        mock_time(duration)
        
    with pytest.raises(PermanentAPIError):
        _invoke_llm_with_retries(mock_llm, ["Test"], max_attempts=2, sleeper=fake_sleep)
