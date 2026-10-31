import json
import shutil
import tempfile
import unittest
from pathlib import Path

from retrieval_honesty_eval.corpus import DEFAULT_ROOT, Corpus, load_corpus, parse_locator


class CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()

    def test_every_passage_quote_is_the_files_own_lines(self) -> None:
        for passage in self.corpus.passages:
            lines = (DEFAULT_ROOT / passage.path).read_text(encoding="utf-8").splitlines()
            self.assertEqual("\n".join(lines[passage.start - 1 : passage.end]), passage.text)
            self.assertEqual(passage.text, self.corpus.span_text(passage.source))

    def test_headings_are_not_quotable_passages(self) -> None:
        self.assertFalse(any(p.text.lstrip().startswith("#") for p in self.corpus.passages))

    def test_locators_outside_the_allow_list_are_not_held(self) -> None:
        self.assertIsNone(self.corpus.span_text("README.md:1-1"))
        self.assertIsNone(self.corpus.span_text("docs/ROADMAP.md:3-99"))
        self.assertIsNone(self.corpus.span_text("docs/ROADMAP.md:3"))
        self.assertIsNone(parse_locator("docs/ROADMAP.md:5-3"))

    def test_missing_allow_listed_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            shutil.copytree(DEFAULT_ROOT, root)
            (root / "docs/ROADMAP.md").unlink()
            with self.assertRaises(FileNotFoundError):
                Corpus(root)

    def test_manifest_cannot_escape_the_corpus_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            root.mkdir()
            (Path(tmp) / "secret.md").write_text("secret\n", encoding="utf-8")
            manifest = {"files": {"../secret.md": {"status": "active", "role": "binding_standard"}}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                Corpus(root)

    def test_superseded_sources_are_hidden_unless_asked_for(self) -> None:
        query = "fixed interval of 60 seconds"
        hidden = self.corpus.search(query)
        shown = self.corpus.search(query, include_superseded=True)
        superseded = "docs/adr/0003-webhook-retries.md:7-7"
        self.assertNotIn(superseded, [r["source"] for r in hidden["results"]])
        self.assertIn(superseded, [r["source"] for r in shown["results"]])
        self.assertNotEqual(superseded, (shown["recommended_evidence"] or {}).get("source"))

    def test_secondary_reference_is_never_recommended_even_when_ranked_first(self) -> None:
        result = self.corpus.search("How many times is a failed webhook delivery retried?")
        self.assertEqual("docs/notes/2025-03-incident-review.md:7-7", result["results"][0]["source"])
        self.assertEqual("docs/adr/0004-webhook-retry-backoff.md:7-7", result["recommended_evidence"]["source"])

    def test_empty_source_filter_is_a_bounded_absence(self) -> None:
        result = self.corpus.search("webhook retries", source_prefix="docs/adr/0009")
        self.assertEqual("SOURCE_FILTER_EMPTY", result["outcome"])
        self.assertEqual([], result["results"])
        self.assertIsNone(result["recommended_evidence"])

    def test_weak_matches_are_not_recommended(self) -> None:
        result = self.corpus.search("Which ADR sets the rate limit for the public API?")
        self.assertEqual("CANDIDATES_FOUND", result["outcome"])
        self.assertIsNone(result["recommended_evidence"])
        self.assertIsNotNone(result["note"])

    def test_invalid_arguments_raise(self) -> None:
        for kwargs in ({"query": " "}, {"query": "x", "k": 0}, {"query": "x", "k": True}, {"query": "x", "k": 11}):
            with self.assertRaises(ValueError):
                self.corpus.search(**kwargs)


if __name__ == "__main__":
    unittest.main()
