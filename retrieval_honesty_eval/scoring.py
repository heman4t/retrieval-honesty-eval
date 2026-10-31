"""Deterministic rewards for one agent rollout.

Every reward is in [0, 1] and independently inspectable. Rewards derived from
tool use read the ordered search calls on the reply's own trace branch, so
evidence the agent never retrieved earns no grounding credit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .cases import INSUFFICIENT_EVIDENCE_ANSWER, MAX_CALLS, SEARCH_K, TOOL_NAME, Case
from .corpus import Corpus

REWARD_KEYS = (
    "format_valid",
    "recall_at_3",
    "authoritative_top_1",
    "superseded_safe",
    "locator_exact",
    "grounded_in_tool",
    "answer_complete",
)
_CALL_ARGUMENTS = {"query", "k", "source_prefix", "include_superseded"}
_OUTPUT_KEYS = {"outcome", "results", "recommended_evidence", "note"}
_RESULT_KEYS = {"source", "score", "lexical_coverage", "status", "role", "quote"}


@dataclass(frozen=True)
class ToolCall:
    """One search call and the tool's raw JSON reply, in trace order."""

    name: str
    arguments: dict[str, Any]
    output: str


def strict_json_loads(text: str) -> Any:
    """``json.loads`` that rejects duplicate object keys."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON key")
        return dict(pairs)

    return json.loads(text, object_pairs_hook=unique)


def parse_submission(reply: str) -> dict[str, Any] | None:
    """The final reply must be exactly one JSON object of the requested shape."""
    try:
        payload = strict_json_loads(reply)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"status", "evidence", "answer"}:
        return None
    status, evidence, answer = payload["status"], payload["evidence"], payload["answer"]
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list):
        return None
    if not all(
        isinstance(item, dict)
        and set(item) == {"source", "quote"}
        and all(isinstance(item[key], str) and item[key].strip() for key in ("source", "quote"))
        for item in evidence
    ):
        return None
    if status == "answered" and len(evidence) == 1:
        return payload
    if status == "insufficient_evidence" and not evidence:
        return payload
    return None


def score(case: Case, reply: str, calls: list[ToolCall] | None, corpus: Corpus) -> dict[str, float]:
    """All rewards for one rollout. ``calls`` is None when the trace is malformed."""
    outputs = _validated_outputs(case, calls, corpus)
    rewards = dict.fromkeys(REWARD_KEYS, 0.0)
    rewards["recall_at_3"] = _recall_at_3(case, outputs)
    submission = parse_submission(reply)
    if submission is None:
        return rewards
    evidence = submission["evidence"]
    failed_closed = submission["status"] == "insufficient_evidence"
    rewards["format_valid"] = 1.0
    rewards["superseded_safe"] = float(
        not any(corpus.status_of(_path_of(item["source"])) == "superseded" for item in evidence)
    )
    if case.expected_source is None:
        rewards["authoritative_top_1"] = rewards["locator_exact"] = float(failed_closed)
    elif not failed_closed:
        item = evidence[0]
        rewards["authoritative_top_1"] = float(_path_of(item["source"]) == _path_of(case.expected_source))
        rewards["locator_exact"] = float(
            item["source"] == case.expected_source and corpus.span_text(item["source"]) == item["quote"]
        )
    rewards["grounded_in_tool"] = _grounded(case, submission, outputs)
    rewards["answer_complete"] = float(
        submission["status"] == case.expected_status
        and all(
            any(term.casefold() in submission["answer"].casefold() for term in group)
            for group in case.required_answer_terms
        )
    )
    return rewards


def _validated_outputs(case: Case, calls: list[ToolCall] | None, corpus: Corpus) -> list[dict[str, Any]] | None:
    """Decoded tool replies, or None if the call sequence or any reply breaks the contract."""
    if not calls or len(calls) > MAX_CALLS:
        return None
    queries = []
    outputs = []
    for position, call in enumerate(calls):
        args = call.arguments
        if (
            call.name != TOOL_NAME
            or set(args) != _CALL_ARGUMENTS
            or not isinstance(args["query"], str)
            or (position == 0 and args["query"] != case.query)
            or args["k"] != SEARCH_K
            or args["source_prefix"] != case.source_prefix
            or args["include_superseded"] is not False
        ):
            return None
        queries.append(" ".join(args["query"].split()).casefold())
        try:
            payload = strict_json_loads(call.output)
        except (TypeError, ValueError):
            return None
        if not _output_is_held(payload, corpus):
            return None
        outputs.append(payload)
    return outputs if len(set(queries)) == len(queries) else None


def _output_is_held(payload: Any, corpus: Corpus) -> bool:
    """Every quote a tool reply carries must be the exact text at its locator."""
    if not isinstance(payload, dict) or set(payload) != _OUTPUT_KEYS or not isinstance(payload["results"], list):
        return False
    for result in payload["results"]:
        if not isinstance(result, dict) or set(result) != _RESULT_KEYS or not isinstance(result["source"], str):
            return False
        if (
            corpus.span_text(result["source"]) != result["quote"]
            or corpus.status_of(_path_of(result["source"])) != result["status"]
        ):
            return False
    recommended = payload["recommended_evidence"]
    if recommended is None:
        return True
    return recommended in [{"source": r["source"], "quote": r["quote"]} for r in payload["results"]]


def _path_of(source: str) -> str:
    """The file part of ``path:start-end`` (corpus paths never contain a colon)."""
    return source.split(":", 1)[0]


def _recall_at_3(case: Case, outputs: list[dict[str, Any]] | None) -> float:
    if outputs is None:
        return 0.0
    if case.expected_source is None:
        return float(all(output["recommended_evidence"] is None for output in outputs))
    return float(
        any(result["source"] == case.expected_source for output in outputs for result in output["results"][:3])
    )


def _satisfies(case: Case, output: dict[str, Any]) -> bool:
    """Whether a search reply already settles the case, so a further search is unnecessary."""
    if case.expected_source is None:
        return output["outcome"] == "SOURCE_FILTER_EMPTY"
    recommended = output["recommended_evidence"]
    return bool(
        recommended
        and recommended["source"] == case.expected_source
        and all(any(t.casefold() in recommended["quote"].casefold() for t in g) for g in case.required_answer_terms)
    )


def _grounded(case: Case, submission: dict[str, Any], outputs: list[dict[str, Any]] | None) -> float:
    if outputs is None or any(_satisfies(case, output) for output in outputs[:-1]):
        return 0.0
    if submission["status"] == "insufficient_evidence":
        return float(
            submission["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
            and all(output["recommended_evidence"] is None for output in outputs)
        )
    item = submission["evidence"][0]
    retrieved = [output["recommended_evidence"] for output in outputs]
    return float(submission["answer"] == item["quote"] and item in retrieved)
