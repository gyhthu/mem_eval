"""One-shot DeepSeek / Reader LLM connectivity check.

Uses the same env loading and chat client as eval_platform Reader/Judge.

  PYTHONPATH=src .venv/bin/python scripts/ping_llm.py \
    --env-file /Users/gaoyinghua/Documents/ChatGPT/groupmembench/code/.env
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from eval_platform.llm_eval import (
    _client,
    completion_extra,
    public_llm_config,
    resolve_llm_config,
)
from llm_utils import chat_completion_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Ping the eval LLM endpoint")
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[1] / "code" / ".env"),
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.exists():
        print(f"FAIL env file missing: {env_path}", file=sys.stderr)
        return 2

    config = resolve_llm_config(env_file=env_path, reader_model=args.model)
    public = public_llm_config(config)
    model = config["reader_model"]
    print(
        "config "
        f"provider={public['provider']} model={model} "
        f"base_url={public['base_url']} thinking={public['thinking']}"
    )

    started = time.monotonic()
    try:
        text = chat_completion_text(
            _client(config),
            model=model,
            messages=[{"role": "user", "content": "只回复一个字：通"}],
            max_tokens=16,
            temperature=0.0,
            **completion_extra(config),
        ).strip()
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        print(f"FAIL {type(exc).__name__}: {exc} ({elapsed_ms}ms)", file=sys.stderr)
        return 1

    elapsed_ms = int((time.monotonic() - started) * 1000)
    preview = text.replace("\n", " ")[:80] or "<empty>"
    print(f"OK {elapsed_ms}ms reply={preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
