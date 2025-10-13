from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferWindowMemory

from pipeline.llm_adapter import run_local_llm

from .config import IST
from .utils import conversation_memory_for_session, format_iso

logger = logging.getLogger("agrichat.app.context")


def format_conversation_context(memory: ConversationBufferWindowMemory) -> str:
    if not memory.chat_memory.messages:
        return "This is the start of the conversation."

    context_parts = []
    messages = memory.chat_memory.messages[-10:]

    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            human_msg = messages[i]
            ai_msg = messages[i + 1]

            human_content = human_msg.content
            ai_content = ai_msg.content[:300] + "..." if len(ai_msg.content) > 300 else ai_msg.content

            context_parts.append(f"Previous Q: {human_content}")
            context_parts.append(f"Previous A: {ai_content}")

    all_content = " ".join([msg.content for msg in messages])
    topics = extract_topics_from_context(all_content)

    if topics:
        context_parts.append(f"Main topics discussed: {', '.join(topics[:3])}")

    return "\n".join(context_parts)


def convert_langchain_memory_to_history(memory: ConversationBufferWindowMemory) -> List[Dict[str, str]]:
    try:
        if not memory or not memory.chat_memory or not memory.chat_memory.messages:
            return []

        conversation_history = []
        messages = memory.chat_memory.messages

        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]

                if (
                    hasattr(human_msg, "content")
                    and hasattr(ai_msg, "content")
                    and human_msg.content
                    and ai_msg.content
                ):
                    conversation_history.append(
                        {
                            "question": str(human_msg.content),
                            "answer": str(ai_msg.content),
                        }
                    )

        return conversation_history

    except Exception as exc:  # pragma: no cover
        logger.error("[LANGCHAIN] Error converting memory to history: %s", exc)
        return []


async def enhance_answer_with_context_questions(
    question: str,
    answer: str,
    user_state: str,
    thinking: str,
) -> str:
    if not answer or not answer.strip():
        return answer

    suggestions = await generate_context_suggestions(question, answer, user_state, thinking)

    if not suggestions or not suggestions.get("needs_context"):
        return answer

    questions = suggestions.get("questions") or []
    if not questions:
        return answer

    enhanced_answer = f"{answer}\n\n**To provide more specific guidance, I'd like to know:**\n"
    for idx, follow_up in enumerate(questions, 1):
        enhanced_answer += f"{idx}. {follow_up}\n"

    return enhanced_answer.strip()


def extract_topics_from_context(context: str) -> List[str]:
    topics: List[str] = []
    context_lower = context.lower()

    disease_patterns = [
        "late blight",
        "early blight",
        "powdery mildew",
        "downy mildew",
        "bacterial wilt",
        "fungal infection",
        "leaf spot",
        "root rot",
        "stem rot",
        "collar rot",
        "blast",
        "sheath blight",
        "rust",
        "smut",
        "mosaic virus",
        "yellowing",
        "wilting",
        "damping off",
        "canker",
        "scab",
    ]

    crop_patterns = [
        "potato",
        "tomato",
        "wheat",
        "rice",
        "cotton",
        "sugarcane",
        "maize",
        "corn",
        "onion",
        "garlic",
        "chili",
        "pepper",
        "brinjal",
        "eggplant",
        "okra",
        "cucumber",
        "cabbage",
        "cauliflower",
        "carrot",
        "radish",
        "beans",
        "peas",
        "groundnut",
        "soybean",
        "mustard",
        "sesame",
        "sunflower",
        "mango",
        "banana",
        "guava",
        "papaya",
        "coconut",
        "tea",
        "coffee",
        "spices",
        "turmeric",
        "ginger",
    ]

    pest_patterns = [
        "aphid",
        "thrips",
        "whitefly",
        "bollworm",
        "stem borer",
        "fruit borer",
        "leaf miner",
        "scale insect",
        "mealybug",
        "spider mite",
        "nematode",
        "caterpillar",
        "grub",
        "weevil",
        "beetle",
        "locust",
        "grasshopper",
    ]

    problem_patterns = [
        "nutrient deficiency",
        "nitrogen deficiency",
        "phosphorus deficiency",
        "potassium deficiency",
        "iron deficiency",
        "zinc deficiency",
        "magnesium deficiency",
        "water stress",
        "drought stress",
        "waterlogging",
        "poor growth",
        "stunted growth",
        "low yield",
        "poor germination",
        "flower drop",
        "fruit drop",
    ]

    practice_patterns = [
        "organic farming",
        "crop rotation",
        "intercropping",
        "mulching",
        "pruning",
        "grafting",
        "seed treatment",
        "soil preparation",
        "land preparation",
        "transplanting",
        "direct sowing",
        "drip irrigation",
        "sprinkler irrigation",
    ]

    for disease in disease_patterns:
        if disease in context_lower:
            for crop in crop_patterns:
                if crop in context_lower:
                    topics.append(f"{disease} in {crop}")
                    break
            else:
                topics.append(disease)
            break

    if not topics:
        for pest in pest_patterns:
            if pest in context_lower:
                for crop in crop_patterns:
                    if crop in context_lower:
                        topics.append(f"{pest} in {crop}")
                        break
                else:
                    topics.append(pest)
                break

    if not topics:
        for problem in problem_patterns:
            if problem in context_lower:
                for crop in crop_patterns:
                    if crop in context_lower:
                        topics.append(f"{problem} in {crop}")
                        break
                else:
                    topics.append(problem)
                break

    if not topics:
        for practice in practice_patterns:
            if practice in context_lower:
                for crop in crop_patterns:
                    if crop in context_lower:
                        topics.append(f"{practice} for {crop}")
                        break
                else:
                    topics.append(practice)
                break

    if not topics:
        for crop in crop_patterns:
            if crop in context_lower:
                topics.append(crop)
                break

    return topics


def _truncate_for_prompt(text: Optional[str], max_chars: int = 800) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _parse_llm_json_response(raw_response: str) -> Dict[str, Any]:
    if not raw_response:
        return {}
    snippet = raw_response
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_response[start : end + 1]

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        try:
            cleaned = snippet.replace("\n", " ").replace("\t", " ")
            return json.loads(cleaned)
        except Exception:
            return {}


async def generate_context_suggestions(
    question: str,
    answer: str,
    user_state: str,
    thinking: str,
) -> Dict[str, Any]:
    prompt = f"""You are an agricultural expert assistant with deep knowledge of Indian farming. Determine if more context is required to improve the following answer. If additional context would make the advice safer or more precise, propose up to four short follow-up questions that collect the key missing details.

Return your decision strictly as compact JSON with this schema:
{{
  "needs_context": true | false,
  "reason": "<brief explanation>",
  "questions": ["<follow-up question 1>", ...]
}}

Guidelines:
- Ask for details only if they materially improve agricultural guidance (variety, growth stage, soil, irrigation, local climate, recent treatment, etc.).
- Keep questions farmer-friendly, actionable, and ≤ 20 words.
- Never mention that you are an AI or reference internal reasoning.
- If the current answer is already specific enough, set "needs_context" to false and return an empty questions list.

User question: {_truncate_for_prompt(question, 400)}
Current answer: {_truncate_for_prompt(answer, 600)}
Thinking steps: {_truncate_for_prompt(thinking, 400) or 'None'}
User state: {user_state or 'Unknown'}
"""

    loop = asyncio.get_event_loop()

    def _call_llm() -> str:
        return run_local_llm(
            prompt,
            temperature=0.2,
            max_tokens=400,
            model=os.getenv("OLLAMA_MODEL_REASONER", "qwen3:1.7b"),
        )

    raw_response = await loop.run_in_executor(None, _call_llm)
    parsed = _parse_llm_json_response(raw_response)

    if not isinstance(parsed, dict):
        return {}

    needs_context = bool(parsed.get("needs_context"))
    questions_raw = parsed.get("questions") or []
    if not isinstance(questions_raw, list):
        questions_raw = []

    questions = []
    for item in questions_raw:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                questions.append(cleaned)

    return {
        "needs_context": needs_context and bool(questions),
        "reason": parsed.get("reason", ""),
        "questions": questions[:4],
    }
