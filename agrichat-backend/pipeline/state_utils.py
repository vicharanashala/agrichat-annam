import re
from typing import List, Optional


INDIAN_STATE_ALIASES = {
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttaranchal": "Uttarakhand",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "national capital territory of delhi": "Delhi",
    "nct of delhi": "Delhi",
    "chandigarh": "Chandigarh",
    "jammu": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "kashmir": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andaman": "Andaman and Nicobar Islands",
    "nicobar": "Andaman and Nicobar Islands",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "lakshadweep": "Lakshadweep",
    "andhra": "Andhra Pradesh",
    "madhya": "Madhya Pradesh",
    "uttar": "Uttar Pradesh",
    "andhra pradesh state": "Andhra Pradesh",
    "telangana state": "Telangana",
    "karnataka state": "Karnataka",
    "maharashtra state": "Maharashtra",
    "ap": "Andhra Pradesh",
    "tn": "Tamil Nadu",
    "up": "Uttar Pradesh",
    "mp": "Madhya Pradesh",
    "hp": "Himachal Pradesh",
    "wb": "West Bengal",
}

CANONICAL_STATES = set(INDIAN_STATE_ALIASES.values())
CANONICAL_STATE_LOOKUP = {value.lower(): value for value in CANONICAL_STATES}

_STATE_PATTERN = re.compile(r"\b(?:in|from|at|for|of)\s+([a-zA-Z\s]+?)(?:\?|\.|,|$)")


_STATE_STRIP_PATTERNS = [
    r"^state of ",
    r"^union territory of ",
    r"^territory of ",
    r"^nct of ",
    r"^the ",
    r" union territory$",
    r" state$",
    r" territory$",
    r" district$",
    r" region$",
    r" province$",
    r" india$",
]


def _clean_state_key(state: str) -> str:
    cleaned = state.strip().lower().replace("&", "and")
    cleaned = re.sub(r"[\.,]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    for pattern in _STATE_STRIP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_state_name(state: str) -> Optional[str]:
    if not state:
        return None

    cleaned = _clean_state_key(state)
    if not cleaned:
        return None

    alias_match = INDIAN_STATE_ALIASES.get(cleaned)
    if alias_match:
        return alias_match

    canonical_match = CANONICAL_STATE_LOOKUP.get(cleaned)
    if canonical_match:
        return canonical_match

    words = cleaned.split()
    if len(words) > 1:
        shortened = " ".join(words[:2])
        alias_match = INDIAN_STATE_ALIASES.get(shortened)
        if alias_match:
            return alias_match

    return None


def extract_state_from_query(question: str) -> Optional[str]:
    if not question:
        return None
    q_lower = question.lower()

    for alias, canonical in INDIAN_STATE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q_lower):
            return canonical

    match = _STATE_PATTERN.search(q_lower)
    if match:
        candidate = match.group(1).strip()
        return normalize_state_name(candidate)
    return None


def prioritize_states(question: str, user_state: Optional[str]) -> List[str]:
    explicit_state = extract_state_from_query(question)
    priority: List[str] = []

    if explicit_state:
        priority.append(explicit_state)

    normalized_user_state = normalize_state_name(user_state or "")
    if normalized_user_state and normalized_user_state not in priority:
        priority.append(normalized_user_state)

    if "India" not in priority:
        priority.append("India")

    priority.append("GENERAL")
    return priority
