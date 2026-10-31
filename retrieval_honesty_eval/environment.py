"""Prime Intellect Verifiers v1 taskset: the corpus as an MCP tool, scored by ``scoring.score``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.types import content_text

from .cases import SYSTEM_PROMPT, Case, build_prompt, load_cases, validate_cases
from .corpus import DEFAULT_ROOT, load_corpus
from .scoring import ToolCall, score, strict_json_loads


class HonestyData(vf.TaskData):
    query: str
    expected_source: str | None = None
    source_prefix: str = ""
    required_answer_terms: list[list[str]] = Field(default_factory=list)

    def to_case(self) -> Case:
        return Case(
            name=self.name,
            query=self.query,
            expected_source=self.expected_source,
            required_answer_terms=tuple(tuple(group) for group in self.required_answer_terms),
            source_prefix=self.source_prefix,
        )


class CorpusToolConfig(vf.ToolsetConfig):
    corpus_root: Path = DEFAULT_ROOT


class CorpusToolset(vf.Toolset[CorpusToolConfig]):
    """Exposes ``search`` as the MCP tool ``corpus_search``."""

    TOOL_PREFIX = "corpus"

    @vf.tool
    async def search(
        self,
        query: str,
        k: int = 3,
        source_prefix: str = "",
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """Search the local allow-listed corpus; never uses the internet."""
        return load_corpus(self.config.corpus_root).search(query, k, source_prefix, include_superseded)


class HonestyTaskConfig(vf.TaskConfig):
    corpus: CorpusToolConfig = Field(default_factory=CorpusToolConfig)


class HonestyTasksetConfig(vf.TasksetConfig):
    task: HonestyTaskConfig = Field(default_factory=HonestyTaskConfig)


class HonestyTask(vf.Task[HonestyData, vf.State, HonestyTaskConfig]):
    @classmethod
    def toolsets(cls, config: HonestyTaskConfig) -> list[vf.Toolset]:
        return [CorpusToolset(config.corpus)]

    @vf.reward
    async def retrieval_honesty(self, trace: vf.Trace) -> dict[str, float]:
        corpus = load_corpus(self.config.corpus.corpus_root)
        return score(self.data.to_case(), trace.last_reply, calls_from_trace(trace), corpus)


class RetrievalHonestyTaskset(vf.Taskset[HonestyTask, HonestyTasksetConfig]):
    def load(self) -> list[HonestyTask]:
        cases = load_cases()
        problems = validate_cases(cases, load_corpus(self.config.task.corpus.corpus_root))
        if problems:
            raise ValueError("golden cases are unattainable: " + "; ".join(problems))
        return [
            HonestyTask(
                HonestyData(
                    idx=index,
                    name=case.name,
                    prompt=build_prompt(case),
                    system_prompt=SYSTEM_PROMPT,
                    query=case.query,
                    expected_source=case.expected_source,
                    source_prefix=case.source_prefix,
                    required_answer_terms=[list(group) for group in case.required_answer_terms],
                ),
                self.config.task,
            )
            for index, case in enumerate(cases)
        ]


def calls_from_trace(trace: vf.Trace) -> list[ToolCall] | None:
    """Ordered search calls on the branch that produced the final reply.

    Returns None when that branch is malformed: a turn with several tool calls,
    a tool reply that does not answer the pending call, or a call left unanswered.
    """
    branch = _final_branch(trace)
    if branch is None:
        return None
    calls: list[ToolCall] = []
    pending = None
    for node in branch:
        message = node.message
        if isinstance(message, vf.AssistantMessage) and node.sampled and message.tool_calls:
            if pending is not None or len(message.tool_calls) != 1:
                return None
            pending = message.tool_calls[0]
        elif isinstance(message, vf.ToolMessage):
            if pending is None or message.tool_call_id != pending.id or message.name != pending.name:
                return None
            try:
                arguments = strict_json_loads(pending.arguments)
            except (TypeError, ValueError):
                return None
            if not isinstance(arguments, dict):
                return None
            calls.append(ToolCall(pending.name, arguments, content_text(message.content)))
            pending = None
    return calls if pending is None else None


def _final_branch(trace: vf.Trace) -> list[Any] | None:
    """Root-to-leaf ancestry of the last sampled assistant message."""
    nodes = trace.nodes
    index = next(
        (
            i
            for i in range(len(nodes) - 1, -1, -1)
            if nodes[i].sampled and isinstance(nodes[i].message, vf.AssistantMessage)
        ),
        None,
    )
    branch = []
    seen: set[int] = set()
    while index is not None:
        if index in seen or not 0 <= index < len(nodes):
            return None
        seen.add(index)
        branch.append(nodes[index])
        index = nodes[index].parent
    return branch[::-1] or None


if __name__ == "__main__":
    CorpusToolset.run()
