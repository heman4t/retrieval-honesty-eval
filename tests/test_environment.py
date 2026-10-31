"""Verifiers adapter tests; skipped when the optional ``verifiers`` extra is not installed."""

import asyncio
import json
import unittest

try:
    import verifiers.v1 as vf
    from verifiers.v1.trace import MessageNode
    from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage
except ImportError:  # the scoring core must stay testable without the extra
    vf = None

from helpers import oracle_rollout

from retrieval_honesty_eval.cases import load_cases
from retrieval_honesty_eval.corpus import load_corpus


def build_trace(reply, calls, *, call_ids=None, tool_names=None):
    """A linear trace: user, then (assistant tool call, tool reply) pairs, then the final reply."""
    nodes = [MessageNode(parent=None, message=UserMessage(content="question"))]
    for i, call in enumerate(calls):
        call_id = f"call-{i}"
        message = AssistantMessage(
            content="", tool_calls=[ToolCall(id=call_id, name=call.name, arguments=json.dumps(call.arguments))]
        )
        nodes.append(MessageNode(parent=len(nodes) - 1, message=message, sampled=True))
        reply_id = (call_ids or {}).get(i, call_id)
        name = (tool_names or {}).get(i, call.name)
        tool_message = ToolMessage(tool_call_id=reply_id, name=name, content=call.output)
        nodes.append(MessageNode(parent=len(nodes) - 1, message=tool_message))
    nodes.append(MessageNode(parent=len(nodes) - 1, message=AssistantMessage(content=reply), sampled=True))
    return vf.Trace.model_construct(nodes=nodes)


@unittest.skipIf(vf is None, "verifiers is not installed")
class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        from retrieval_honesty_eval.environment import (
            HonestyTasksetConfig,
            RetrievalHonestyTaskset,
            calls_from_trace,
        )

        self.calls_from_trace = calls_from_trace
        self.tasks = {t.data.name: t for t in RetrievalHonestyTaskset(HonestyTasksetConfig(id="x")).load()}
        self.corpus = load_corpus()
        self.cases = {case.name: case for case in load_cases()}

    def reward(self, name, trace):
        return asyncio.run(self.tasks[name].retrieval_honesty(trace))

    def test_package_exports_exactly_one_taskset_for_prime(self) -> None:
        import retrieval_honesty_eval

        exported = [getattr(retrieval_honesty_eval, n) for n in retrieval_honesty_eval.__all__]
        self.assertEqual(1, len(exported))
        self.assertTrue(issubclass(exported[0], vf.Taskset))

    def test_taskset_loads_every_case(self) -> None:
        self.assertEqual(set(self.cases), set(self.tasks))

    def test_honest_trace_scores_one_everywhere(self) -> None:
        for name, case in self.cases.items():
            reply, calls = oracle_rollout(case, self.corpus)
            with self.subTest(case=name):
                rewards = self.reward(name, build_trace(reply, calls))
                self.assertEqual({1.0}, set(rewards.values()))

    def test_tool_reply_for_a_different_call_is_malformed(self) -> None:
        reply, calls = oracle_rollout(self.cases["retry_count_despite_injection"], self.corpus)
        self.assertIsNone(self.calls_from_trace(build_trace(reply, calls, call_ids={0: "other"})))
        self.assertIsNone(self.calls_from_trace(build_trace(reply, calls, tool_names={0: "bash"})))

    def test_two_tool_calls_in_one_turn_are_malformed(self) -> None:
        reply, calls = oracle_rollout(self.cases["retry_count_despite_injection"], self.corpus)
        trace = build_trace(reply, calls)
        first = trace.nodes[1].message
        trace.nodes[1] = MessageNode(
            parent=0,
            message=AssistantMessage(content="", tool_calls=[first.tool_calls[0], first.tool_calls[0]]),
            sampled=True,
        )
        self.assertIsNone(self.calls_from_trace(trace))

    def test_final_reply_cannot_borrow_evidence_from_a_sibling_branch(self) -> None:
        reply, calls = oracle_rollout(self.cases["retry_count_despite_injection"], self.corpus)
        trace = build_trace(reply, calls)
        # Re-parent the final reply onto the user turn: the search now sits on an abandoned branch.
        trace.nodes[-1] = MessageNode(parent=0, message=AssistantMessage(content=reply), sampled=True)
        rewards = self.reward("retry_count_despite_injection", trace)
        self.assertEqual(0.0, rewards["grounded_in_tool"])
        self.assertEqual(0.0, rewards["recall_at_3"])
        self.assertEqual(1.0, rewards["locator_exact"])

    def test_tool_returns_the_corpus_search_payload(self) -> None:
        from retrieval_honesty_eval.environment import CorpusToolConfig, CorpusToolset

        toolset = CorpusToolset(CorpusToolConfig())
        query = self.cases["retry_count_despite_injection"].query
        self.assertEqual(self.corpus.search(query), asyncio.run(toolset.search(query)))


if __name__ == "__main__":
    unittest.main()
