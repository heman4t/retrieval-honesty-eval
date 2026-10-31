"""An allow-listed, offline corpus with a deterministic BM25 retriever.

Every indexed passage carries an exact ``path:start-end`` locator whose text is
the file's own lines, so any quote the retriever returns can be checked against
the file it claims to come from.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parent / "corpus"
STATUSES = {"active", "superseded"}
ROLES = {"project_authority", "accepted_decision", "binding_standard", "secondary_reference"}
MIN_EVIDENCE_COVERAGE = 0.5
MAX_K = 10
STOPWORDS = frozenset(
    "a an and any are as at be by can did do does for from how in into is it its of on or that the this to was "
    "what when which who why with".split()
)
_LOCATOR = re.compile(r"^(?P<path>[^:]+):(?P<start>[1-9]\d*)-(?P<end>[1-9]\d*)$")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords; plural ``-ies``/``-s`` endings are folded."""
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def parse_locator(source: str) -> tuple[str, int, int] | None:
    match = _LOCATOR.match(source)
    if match is None:
        return None
    start, end = int(match["start"]), int(match["end"])
    return (match["path"], start, end) if start <= end else None


@dataclass(frozen=True)
class Passage:
    path: str
    start: int
    end: int
    text: str
    status: str
    role: str
    headings: str = ""

    @property
    def source(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


class Corpus:
    """The files listed in ``manifest.json`` and nothing else.

    Loading fails closed: a listed file that is missing, or a manifest entry
    with an unknown status or role, raises instead of silently shrinking the
    index (an absence is only meaningful when the whole allow-list is indexed).
    """

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root.resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.files: dict[str, dict[str, str]] = manifest["files"]
        self._lines: dict[str, list[str]] = {}
        self.passages: list[Passage] = []
        for path, meta in self.files.items():
            if meta.get("status") not in STATUSES or meta.get("role") not in ROLES:
                raise ValueError(f"manifest entry {path!r} has an unknown status or role: {meta}")
            file = (self.root / path).resolve()
            if not file.is_relative_to(self.root):
                raise ValueError(f"manifest entry {path!r} escapes the corpus root")
            if not file.is_file():
                raise FileNotFoundError(f"allow-listed file is missing: {path}")
            lines = file.read_text(encoding="utf-8").splitlines()
            self._lines[path] = lines
            self.passages.extend(_passages(path, lines, meta["status"], meta["role"]))
        # Headings are searchable context; quotes stay the passage's own lines.
        self._tokens = [Counter(tokenize(f"{p.headings}\n{p.text}")) for p in self.passages]
        self._avg_len = sum(sum(t.values()) for t in self._tokens) / len(self._tokens)
        doc_freq = Counter(term for tokens in self._tokens for term in tokens)
        n = len(self.passages)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def status_of(self, path: str) -> str | None:
        meta = self.files.get(path)
        return meta["status"] if meta else None

    def span_text(self, source: str) -> str | None:
        """The exact text at ``path:start-end``, or None if the locator is not held."""
        parsed = parse_locator(source)
        if parsed is None:
            return None
        path, start, end = parsed
        lines = self._lines.get(path)
        if lines is None or end > len(lines):
            return None
        return "\n".join(lines[start - 1 : end])

    def search(
        self,
        query: str,
        k: int = 3,
        source_prefix: str = "",
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer from 1 to {MAX_K}")
        eligible = [
            i
            for i, p in enumerate(self.passages)
            if p.path.startswith(source_prefix) and (include_superseded or p.status == "active")
        ]
        if source_prefix and not eligible:
            return _payload("SOURCE_FILTER_EMPTY", [], None)
        terms = set(tokenize(query))
        scored = []
        for i in eligible:
            score = self._bm25(terms, i)
            if score > 0:
                scored.append((round(score, 4), self.passages[i].source, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, _source, i in scored[:k]:
            passage = self.passages[i]
            coverage = len(terms & set(self._tokens[i])) / len(terms)
            results.append(
                {
                    "source": passage.source,
                    "score": score,
                    "lexical_coverage": round(coverage, 4),
                    "status": passage.status,
                    "role": passage.role,
                    "quote": passage.text,
                }
            )
        if not results:
            return _payload("NO_CANDIDATES", [], None)
        recommended = next(
            (
                {"source": r["source"], "quote": r["quote"]}
                for r in results
                if r["status"] == "active"
                and r["role"] != "secondary_reference"
                and r["lexical_coverage"] >= MIN_EVIDENCE_COVERAGE
            ),
            None,
        )
        return _payload("CANDIDATES_FOUND", results, recommended)

    def _bm25(self, terms: set[str], index: int, k1: float = 1.2, b: float = 0.75) -> float:
        tokens = self._tokens[index]
        length = sum(tokens.values())
        score = 0.0
        for term in terms:
            tf = tokens.get(term, 0)
            if tf:
                score += self._idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / self._avg_len))
        return score


@lru_cache(maxsize=4)
def load_corpus(root: Path = DEFAULT_ROOT) -> Corpus:
    return Corpus(root)


def _passages(path: str, lines: list[str], status: str, role: str) -> list[Passage]:
    """Paragraphs of consecutive non-blank, non-heading lines, with their heading trail."""
    passages = []
    trail: dict[int, str] = {}
    start = None
    for number, line in enumerate([*lines, ""], start=1):
        heading = re.match(r"^(#+)\s+(.*)$", line.strip())
        body = bool(line.strip()) and heading is None
        if not body and start is not None:
            text = "\n".join(lines[start - 1 : number - 1])
            headings = "\n".join(trail[level] for level in sorted(trail))
            passages.append(Passage(path, start, number - 1, text, status, role, headings))
            start = None
        if heading is not None:
            level = len(heading[1])
            trail = {lvl: title for lvl, title in trail.items() if lvl < level}
            trail[level] = heading[2]
        elif body and start is None:
            start = number
    return passages


def _payload(outcome: str, results: list[dict[str, Any]], recommended: dict[str, str] | None) -> dict[str, Any]:
    note = None
    if recommended is None and outcome == "CANDIDATES_FOUND":
        note = "No result meets the evidence bar; refine the query or report insufficient evidence."
    return {"outcome": outcome, "results": results, "recommended_evidence": recommended, "note": note}
