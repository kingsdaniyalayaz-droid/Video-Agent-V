"""
AI Prompt Generator

This module converts:
- Transcript
- Summary
- Meeting Intelligence

into a high-quality, reusable AI prompt.

The module uses the centralized LLM provider (core.llm_provider) so prompt
generation follows the active runtime provider/model configuration.
"""
import json
from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# LLM
# ============================================================

def get_prompt_generator_llm():
    """
    Return the LLM used for prompt generation from the centralized
    provider layer (core.llm_provider).

    The active runtime provider/model configuration is respected; no
    provider-specific API key or model is required or hardcoded here.
    """

    from core.llm_provider import get_chat_model

    return get_chat_model(
        role="PromptGenerator",
        temperature=0.4,
    )


# ============================================================
# RESPONSE NORMALIZATION
# ============================================================

def _normalize_llm_response_content(content: Any) -> str:
    """
    Safely normalize an LLM response's content into trimmed text.

    - str content is stripped directly.
    - list/tuple content (LangChain AIMessage content can be a list of
      content blocks / message dicts) is joined into text without raising
      AttributeError.
    - any other value is stringified safely.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                parts.append(str(text) if text is not None else str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()

# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_prompt(
    summary: str,
    transcript: Optional[str] = None,
    meeting_intelligence: Optional[Dict[str, Any]] = None,
    prompt_type: str = "custom",
    user_goal: str = "",
) -> str:
    """
    Generate a high-quality reusable AI prompt.

    Parameters
    ----------
    summary : str
        Existing AI-generated summary.

    transcript : str, optional
        Original transcript. It is intentionally limited before
        sending to the LLM to avoid unnecessarily large requests.

    meeting_intelligence : dict, optional
        Structured meeting intelligence data.

    prompt_type : str
        Type of prompt to generate.

        Examples:
        - learning
        - project
        - coding
        - research
        - content
        - business
        - custom

    user_goal : str
        The user's specific goal.

    Returns
    -------
    str
        A ready-to-use AI prompt.
    """

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    # summary must be a string; non-string values are rejected (never
    # silently coerced), then whitespace is stripped and empty/whitespace-only
    # summaries are rejected.
    if not isinstance(summary, str):
        raise ValueError(
            "Summary must be a string. "
            f"Got {type(summary).__name__}."
        )

    summary = summary.strip()

    if not summary:
        raise ValueError(
            "Summary is required to generate a prompt."
        )

    if not isinstance(user_goal, str):
        raise ValueError(
            "user_goal must be a string. "
            f"Got {type(user_goal).__name__}."
        )

    user_goal = user_goal.strip()

    if not user_goal:
        user_goal = (
            "Create the most useful and practical output "
            "based on the provided information."
        )

    # --------------------------------------------------------
    # PREPARE MEETING INTELLIGENCE
    # --------------------------------------------------------

    intelligence_text = ""

    if meeting_intelligence:
        try:
            intelligence_text = json.dumps(
                meeting_intelligence,
                indent=2,
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            intelligence_text = str(meeting_intelligence)

    # --------------------------------------------------------
    # LIMIT TRANSCRIPT SIZE
    # --------------------------------------------------------

    transcript_context = ""

    # transcript is optional: None / empty string keep the existing
    # summary-only behavior.  A non-empty non-string transcript is rejected
    # (never silently coerced) instead of failing with a slicing/type error.
    if transcript is not None and not isinstance(transcript, str):
        raise ValueError(
            "transcript must be a string. "
            f"Got {type(transcript).__name__}."
        )

    if transcript:
        max_transcript_chars = 12000

        transcript_context = transcript[:max_transcript_chars]

        if len(transcript) > max_transcript_chars:
            transcript_context += (
                "\n\n[Transcript truncated for prompt generation. "
                "Use the summary and meeting intelligence as the "
                "primary context.]"
            )

    # --------------------------------------------------------
    # PROMPT TEMPLATE
    # --------------------------------------------------------

    template = """
You are an expert Prompt Engineer.

Your task is to create ONE high-quality, detailed, reusable AI prompt.

The generated prompt will be copied and used in another AI model.

Do NOT answer the user's goal directly.

Instead, create the prompt that another AI should use to achieve
the user's goal.

============================================================
PROMPT TYPE
============================================================

{prompt_type}

============================================================
USER GOAL
============================================================

{user_goal}

============================================================
SOURCE SUMMARY
============================================================

{summary}

============================================================
MEETING / VIDEO INSIGHTS
============================================================

{meeting_intelligence}

============================================================
OPTIONAL TRANSCRIPT CONTEXT
============================================================

{transcript}

============================================================
INSTRUCTIONS
============================================================

Create a professional, clear, and actionable prompt.

The generated prompt should include, where appropriate:

1. Role
2. Objective
3. Context
4. Specific tasks
5. Constraints
6. Required output format
7. Quality requirements

Important rules:

- Base the prompt only on the provided context.
- Do not invent facts that are not supported by the context.
- Preserve important goals, decisions, action items, and insights.
- Make the prompt specific rather than generic.
- Remove unnecessary repetition.
- The final result must be ready to copy and paste into another AI model.
- Do NOT explain the prompt.
- Return ONLY the generated prompt.
"""

    prompt = ChatPromptTemplate.from_template(template)

    # --------------------------------------------------------
    # CREATE CHAIN
    # --------------------------------------------------------

    llm = get_prompt_generator_llm()

    chain = prompt | llm

    # --------------------------------------------------------
    # INVOKE
    # --------------------------------------------------------

    try:

        response = chain.invoke(
            {
                "prompt_type": prompt_type,
                "user_goal": user_goal,
                "summary": summary.strip(),
                "meeting_intelligence": intelligence_text
                or "No structured insights available.",
                "transcript": transcript_context
                or "No transcript context provided.",
            }
        )

    except Exception as error:

        raise RuntimeError(
            f"Prompt generation failed: {error}"
        ) from error

    # --------------------------------------------------------
    # Normalize response and validate non-empty
    # --------------------------------------------------------

    normalized = _normalize_llm_response_content(
        response.content
    )

    if not normalized:

        raise RuntimeError(
            "Prompt generation returned an empty response."
        )

    return normalized