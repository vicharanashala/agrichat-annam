import os
import re
from typing import Callable, List, Optional

from .llm_adapter import OllamaLLMInterface

from .config import PipelineConfig


GENERAL_REFUSAL = (
    "I'm your agricultural assistant and can only help with farming-related questions about Indian agriculture."
)


def _conversation_to_text(history: Optional[List[dict]], max_turns: int = 5) -> str:
    if not history:
        return ""
    trimmed = history[-max_turns:]
    lines: List[str] = []
    for pair in trimmed:
        question = pair.get("question")
        answer = pair.get("answer")
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


TokenCallback = Callable[[str], None]


class LLMResponder:
    _PLANNING_PATTERNS = [
        re.compile(r"^we\s+need\s+to\s+answer[:\-]", re.IGNORECASE),
        re.compile(r"^we\s+should\s+answer", re.IGNORECASE),
        re.compile(r"^the\s+user\s+context[:\-]", re.IGNORECASE),
        re.compile(r"^they\s+want", re.IGNORECASE),
        re.compile(r"^provide\s+(?:summary|steps)", re.IGNORECASE),
        re.compile(r"^our\s+task\s+is", re.IGNORECASE),
        re.compile(r"^let'?s\s+provide", re.IGNORECASE),
        re.compile(r"^task:\s*", re.IGNORECASE),
        re.compile(r"^analysis:\s*", re.IGNORECASE),
    ]

    def __init__(self, config: PipelineConfig):
        model = os.getenv("PIPELINE_LLM_MODEL", config.llm_model)
        self.interface = OllamaLLMInterface(model_name=model)
        self.config = config

    @classmethod
    def _sanitize_output(cls, text: str) -> str:
        if not text:
            return text

        lines = text.splitlines()
        cleaned: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Avoid leading blank lines; preserve interior spacing later
                if cleaned:
                    cleaned.append("")
                continue

            if any(pattern.match(stripped) for pattern in cls._PLANNING_PATTERNS):
                continue

            cleaned.append(stripped)

        # Collapse multiple blank lines
        normalized: List[str] = []
        blank_pending = False
        for line in cleaned:
            if line == "":
                if not blank_pending and normalized:
                    normalized.append("")
                blank_pending = True
            else:
                normalized.append(line)
                blank_pending = False

        return "\n".join(normalized).strip()

    def generate_answer(
        self,
        question: str,
        history: Optional[List[dict]],
        context: str = "",
        *,
        stream: bool = False,
        token_callback: Optional[TokenCallback] = None,
    ) -> str:
        convo = _conversation_to_text(history)
        prompt = (
            "You are AgriChat, an Indian agricultural expert. Answer the user's question using only agricultural knowledge that applies to Indian conditions.\n"
            "Respond in clean Markdown that renders well in the chatbot and follow this exact structure:\n"
            "1. Start with a bold one-sentence summary tailored to the farmer.\n"
            "2. Provide numbered or bulleted actionable steps covering timings, dosages, and precautions when available.\n"
            "3. Include a short **Need more detail?** subsection only when extra context (orchard age, variety, region, etc.) would change the advice; list up to two follow-up questions.\n"
            "4. Finish with a concise tip or reminder relevant to Indian farming practices.\n"
            "Rules: Begin directly with the summary sentence—do not describe your plan, the instructions, or say phrases like 'We need to answer.' Do not expose internal reasoning or mention that you are an AI.\n"
            "If precise data is unavailable, state reasonable assumptions and provide best-practice guidance.\n"
            "If the question is outside agriculture, refuse politely and redirect back to farming topics.\n"
            "Respond in the same language as the user.\n"
        )
        if convo:
            prompt += f"\nConversation so far:\n{convo}\n"
        if context:
            prompt += f"\nContext:\n{context}\n"
        prompt += f"\nQuestion: {question}\nAnswer:"

        if stream:
            raw_text_parts: List[str] = []
            emitted_length = 0
            for event in self.interface.stream_generate(
                prompt,
                temperature=self.config.answer_temperature,
            ):
                event_type = event.get("type")
                if event_type == "token":
                    text = event.get("text", "")
                    if text:
                        raw_text_parts.append(text)
                        if token_callback:
                            sanitized = self._sanitize_output("".join(raw_text_parts))
                            delta = sanitized[emitted_length:]
                            if delta:
                                token_callback(delta)
                                emitted_length += len(delta)
                elif event_type == "error":
                    message = event.get("message") or "Unknown streaming error"
                    if token_callback:
                        token_callback(f"\n[Streaming error: {message}]\n")
                elif event_type == "raw":
                    continue
            final = "".join(raw_text_parts).strip()
            final = self._sanitize_output(final)
            if final:
                return final
            fallback = self.interface.generate_content(
                prompt,
                temperature=self.config.answer_temperature,
                max_tokens=self.config.max_answer_tokens,
                use_fallback=False,
            )
            return self._sanitize_output(fallback) or "No response generated."

        raw_answer = self.interface.generate_content(
            prompt,
            temperature=self.config.answer_temperature,
            max_tokens=self.config.max_answer_tokens,
            use_fallback=False,
        )
        return self._sanitize_output(raw_answer)

    def suggest_clarifications(self, question: str, failed_sources: List[str]) -> List[str]:
        if not self.config.clarify_with_llm:
            return []

        prompt = (
            "You are an agricultural assistant. The current question was ambiguous for these sources: "
            f"{', '.join(failed_sources)}.\n"
            "List up to two clarifying questions that would help you give a precise agricultural answer.\n"
            "Return each question on a new line. If no clarification is needed, respond with 'NONE'.\n"
            f"\nQuestion: {question}\nClarifications:"
        )
        response = self.interface.generate_content(
            prompt,
            temperature=self.config.clarification_temperature,
            max_tokens=128,
            use_fallback=False,
        )
        suggestions = [line.strip("-• ") for line in response.strip().splitlines() if line.strip()]
        if not suggestions or suggestions[0].upper().startswith("NONE"):
            return []
        return suggestions[: self.config.clarification_max_questions]

    def classify_question_intent(self, question: str) -> Optional[bool]:
        prompt = (
            "You are an expert intent classifier for an agricultural assistant focused on Indian farming. "
            "Decide if the user's question is about agriculture, farming, crops, livestock, soil, irrigation, "
            "or any related agricultural practice. Respond with exactly one word: AGRICULTURE if it is relevant, "
            "or NON_AGRICULTURE if it is not."
            f"\n\nQuestion: {question}\nLabel:"
        )
        try:
            response = self.interface.generate_content(
                prompt,
                temperature=0.0,
                max_tokens=4,
                use_fallback=False,
            )
        except Exception:
            return None

        if not response:
            return None

        label = response.strip().upper()
        if label.startswith("AGRI"):
            return True
        if "NON" in label:
            return False
        if label in {"FARM", "FARMING", "AGRICULTURAL"}:
            return True
        if label in {"OTHER", "GENERAL"}:
            return False
        return None
