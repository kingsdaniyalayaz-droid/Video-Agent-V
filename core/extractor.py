from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm_provider import get_chat_model


# ============================================================
# LLM
# ============================================================

def get_llm():
    """
    Active runtime configuration (AI Model Settings) se
    role="Analyzer" ka chat model return karta hai.

    Har call par core.llm_provider.get_chat_model() fresh
    resolve hota hai — is liye provider/model/API key change
    karne par stale client kabhi use nahi hota.
    """

    return get_chat_model(role="Analyzer")


class ExtractionError(RuntimeError):
    """
    Extraction failure ko clearly raise karta hai.

    Original provider error (__cause__) preserve hota hai,
    taaki debugging ke waqt asli error mil sake.
    """


# ============================================================
# TRANSCRIPT VALIDATION
# ============================================================

def validate_transcript(transcript: str) -> None:
    """
    Check karta hai ke transcript empty to nahi hai.
    """

    if not transcript:
        raise ValueError(
            "Transcript empty hai."
        )

    if not transcript.strip():
        raise ValueError(
            "Transcript mein readable text nahi hai."
        )


# ============================================================
# GENERIC EXTRACTION CHAIN
# ============================================================

def build_extraction_chain(
    system_prompt: str,
):
    """
    Reusable LCEL chain banata hai.

    Flow:

    Transcript
        ↓
    Prompt
        ↓
    Active LLM (core.llm_provider, role="Analyzer")
        ↓
    String Output
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                "{text}",
            ),
        ]
    )

    return (
        prompt
        | llm
        | StrOutputParser()
    )


def _invoke_extraction(
    chain,
    transcript: str,
    feature: str,
) -> str:
    """
    Chain invoke karta hai; failure par original provider
    error ko preserve karke clearly raise karta hai.

    The chain uses StrOutputParser(), so the result is a string.  It is
    stripped and an empty / whitespace-only result is treated as a failed
    extraction -- a successful provider call is NOT a usable result.
    """

    try:
        result = chain.invoke(
            {
                "text": transcript
            }
        )
    except Exception as exc:
        raise ExtractionError(
            f"{feature} extraction failed with the "
            f"active LLM provider: {exc}"
        ) from exc

    result_text = (result or "").strip()

    if not result_text:
        raise ExtractionError(
            f"{feature} extraction returned an empty response "
            f"from the active LLM provider."
        )

    return result_text


# ============================================================
# KEY TOPICS
# ============================================================

def extract_key_topics(
    transcript: str,
) -> str:
    """
    YouTube video ke major topics identify karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are an expert YouTube content analyst.

Analyze the YouTube video transcript and identify
the major topics discussed.

Rules:

- Extract only topics actually discussed.
- Do not invent topics.
- Remove duplicate topics.
- Keep each topic short and clear.
- Arrange topics according to their importance.

Format:

## Key Topics

1. Topic
2. Topic
3. Topic

If no clear topics are found, return exactly:

No key topics found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Key topics",
    )


# ============================================================
# KEY INSIGHTS
# ============================================================

def extract_key_insights(
    transcript: str,
) -> str:
    """
    Video se important insights extract karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are a senior YouTube content analyst.

Analyze the transcript and extract the most valuable
insights presented in the video.

Focus on:

- Important explanations
- Valuable concepts
- Lessons
- Technical insights
- Important observations
- Practical knowledge

Rules:

- Only use information from the transcript.
- Do not invent information.
- Do not add your own opinion.
- Remove repetitive points.
- Keep each insight concise.

Format:

## Key Insights

1. ...
2. ...
3. ...

If no meaningful insights are found, return exactly:

No key insights found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Key insights",
    )


# ============================================================
# IMPORTANT FACTS
# ============================================================

def extract_important_facts(
    transcript: str,
) -> str:
    """
    Video mein mentioned important facts,
    numbers, dates aur technical details extract karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are an expert fact extraction assistant.

Analyze the YouTube transcript and extract important
facts explicitly mentioned in the content.

Focus on:

- Numbers
- Dates
- Statistics
- Names
- Technical specifications
- Important factual statements

Rules:

- Never invent facts.
- Never change numbers.
- Preserve dates and names.
- Only extract information explicitly present.
- Remove duplicate facts.

Format:

## Important Facts

1. ...
2. ...
3. ...

If no important facts are found, return exactly:

No important facts found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Important facts",
    )


# ============================================================
# PRACTICAL TAKEAWAYS
# ============================================================

def extract_takeaways(
    transcript: str,
) -> str:
    """
    Video se practical lessons aur takeaways extract karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are a professional learning-content analyst.

Analyze the YouTube video transcript and identify
the most useful practical takeaways.

Focus on:

- What the viewer should learn
- What the viewer should remember
- Practical recommendations
- Important lessons
- Useful techniques

Rules:

- Do not invent recommendations.
- Only use ideas actually presented in the video.
- Keep each takeaway actionable and concise.

Format:

## Practical Takeaways

1. ...
2. ...
3. ...

If no practical takeaways are found, return exactly:

No practical takeaways found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Takeaways",
    )


# ============================================================
# ACTION ITEMS
# ============================================================

def extract_action_items(
    transcript: str,
) -> str:
    """
    Video transcript se explicit action items extract karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are an expert meeting and video analyst.

Analyze the transcript and extract explicit action items.

Focus on:

- Tasks that someone needs to do
- Follow-up work
- Assigned or implied next steps
- Commitments clearly stated in the discussion

Rules:

- Only extract action items explicitly supported by the transcript.
- Do not invent tasks.
- Remove duplicates.
- Keep each action item concise and actionable.

Format:

## Action Items

1. ...
2. ...
3. ...

If no action items are found, return exactly:

No action items found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Action items",
    )


# ============================================================
# DECISIONS
# ============================================================

def extract_decisions(
    transcript: str,
) -> str:
    """
    Video transcript se important decisions extract karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are an expert meeting and video analyst.

Analyze the transcript and extract explicit decisions that were made.

Focus on:

- Agreed outcomes
- Final choices
- Approved plans
- Clear conclusions reached in the discussion

Rules:

- Only extract decisions explicitly supported by the transcript.
- Do not invent decisions.
- Remove duplicates.
- Keep each decision concise and clear.

Format:

## Decisions

1. ...
2. ...
3. ...

If no decisions are found, return exactly:

No decisions found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Decisions",
    )


# ============================================================
# QUESTIONS
# ============================================================

def extract_questions(
    transcript: str,
) -> str:
    """
    Video mein discussed ya raised questions identify karta hai.
    """

    validate_transcript(transcript)

    system_prompt = """
You are an expert YouTube content analyst.

Analyze the transcript and extract important questions
that are explicitly raised or discussed in the video.

Focus on:

- Questions asked by the speaker
- Questions discussed during the video
- Problems that the video attempts to answer

Rules:

- Do not invent questions.
- Do not create questions from unrelated statements.
- Remove duplicates.

Format:

## Questions

1. ...
2. ...
3. ...

If no meaningful questions are found, return exactly:

No questions found.
"""

    chain = build_extraction_chain(
        system_prompt
    )

    return _invoke_extraction(
        chain,
        transcript,
        "Questions",
    )


# ============================================================
# COMPLETE YOUTUBE EXTRACTION
# ============================================================

def extract_youtube_insights(
    transcript: str,
) -> dict[str, str]:
    """
    YouTube transcript se complete structured insights extract karta hai.

    Returns:

    {
        "key_topics": "...",
        "key_insights": "...",
        "important_facts": "...",
        "takeaways": "...",
        "actions": "...",
        "decisions": "...",
        "questions": "..."
    }
    """

    validate_transcript(transcript)

    print("=" * 60)
    print("YOUTUBE CONTENT EXTRACTION")
    print("=" * 60)

    # Key topics extract karte hain
    print("Extracting key topics...")

    key_topics = extract_key_topics(
        transcript
    )

    # Important insights extract karte hain
    print("Extracting key insights...")

    key_insights = extract_key_insights(
        transcript
    )

    # Facts extract karte hain
    print("Extracting important facts...")

    important_facts = extract_important_facts(
        transcript
    )

    # Practical lessons extract karte hain
    print("Extracting practical takeaways...")

    takeaways = extract_takeaways(
        transcript
    )
    # Action items extract karte hain
    print("Extracting action items...")

    actions = extract_action_items(
        transcript
    )

    # Decisions extract karte hain
    print("Extracting decisions...")

    decisions = extract_decisions(
        transcript
    )

    # Questions extract karte hain
    print("Extracting questions...")

    questions = extract_questions(
        transcript
    )

    print("YouTube extraction completed.")

    return {
        "key_topics": key_topics,
        "key_insights": key_insights,
        "important_facts": important_facts,
        "takeaways": takeaways,
        "actions": actions,
        "decisions": decisions,
        "questions": questions,
    }