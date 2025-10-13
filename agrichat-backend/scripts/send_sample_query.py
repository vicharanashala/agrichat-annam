"""CLI helper to send a sample query to the AgriChat backend.

The script prints the outbound payload, sends it to the `/api/query` endpoint,
and prints the HTTP response with pretty-formatted JSON so developers can see
how parameters influence the generated answer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict

import requests

DEFAULT_BASE_URL = os.getenv("AGRICHAT_BASE_URL", "http://localhost:8000")
DEFAULT_QUESTION = "Which fertilizer suits paddy at tillering stage?"
DEFAULT_STATE = "Tamil Nadu"
DEFAULT_LANGUAGE = "English"


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:  # pragma: no cover - argparse message handles it
        raise argparse.ArgumentTypeError("Timeout must be numeric") from exc

    if timeout <= 0:
        raise argparse.ArgumentTypeError("Timeout must be greater than zero")
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a sample query to the AgriChat backend and inspect the response.",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Backend base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Farmer question to submit (default: %(default)s)",
    )
    parser.add_argument(
        "--device-id",
        default=None,
        help="Device identifier. Defaults to a freshly generated UUID.",
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE,
        help="State parameter for retrieval bias (default: %(default)s)",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Preferred language for the response (default: %(default)s)",
    )
    parser.add_argument(
        "--no-golden",
        action="store_true",
        help="Disable Golden database retrieval.",
    )
    parser.add_argument(
        "--no-pops",
        action="store_true",
        help="Disable POPs retrieval.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM synthesis (retrieval-only).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=60.0,
        help="Request timeout in seconds (default: %(default)s)",
    )

    return parser


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    device_id = args.device_id or str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "question": args.question,
        "device_id": device_id,
    }

    if args.state:
        payload["state"] = args.state
    if args.language:
        payload["language"] = args.language

    overrides = {
        "golden_enabled": not args.no_golden,
        "pops_enabled": not args.no_pops,
        "llm_enabled": not args.no_llm,
    }

    if any(value is False for value in overrides.values()):
        payload["database_config"] = overrides

    return payload


def pretty_print(title: str, data: Any) -> None:
    print(f"\n{title}:\n{'-' * len(title)}")
    if isinstance(data, str):
        print(data)
        return
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    payload = build_payload(args)
    pretty_print("Payload", payload)

    url = args.base_url.rstrip("/") + "/api/query"
    print(f"\nSending POST {url} ...")

    try:
        response = requests.post(url, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        pretty_print("Request failed", str(exc))
        return 1

    print(f"\nResponse status: {response.status_code}")

    try:
        response_data = response.json()
    except ValueError:
        pretty_print("Response body (raw)", response.text)
    else:
        pretty_print("Response JSON", response_data)

    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
