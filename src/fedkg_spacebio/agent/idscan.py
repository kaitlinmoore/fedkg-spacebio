"""Finding ID-shaped tokens in text (CONTRACTS §8, grounding rule 3; §7 ID coverage).

Why this exists
---------------
A report claim is only as grounded as the IDs it mentions. Rule 3 says: any ID that appears in a
claim's text must be cited from a tool result of the same trace. To apply it, the checker first
has to find the IDs in free text. This module does that, and nothing else; it doesn't decide
whether a token is cited (that's ``grounding.py``).

The same scanner is used by the §7 coverage test of the agent's own tools ("every ID in a tool's
``data`` is in its ``ids``"), so a tool can't return an ID that a claim would then be unable to
cite.

Three classes of token, found in this order
-------------------------------------------
1. **Pattern IDs**: forms fixed by CONTRACTS §1, matched by regular expression. At each position
   the longest form wins, so an ID that contains another ID is one token:
   - composite KG edge keys ``kgedge:<TYPE>|<start>|<end>`` (they contain gene or comparison IDs);
   - comparison IDs ``OSD-<n>-<32 hex>`` before bare ``OSD-<n>`` (which is their prefix);
   - mock accessions ``OSD-FAKE-<n>``;
   - run-time IDs: ``trace:…`` (with ``/call/<n>`` or ``/dec/<n>``), ``job:``, ``run:``, ``rel:``,
     ``dec:``, policy and rule IDs ``pol:<site>:<n>[#<rule>]``, spec checks ``spec#<check>``,
     caveat notes ``note:<key>``;
   - CURIEs in the namespaces we use (``NCBIGene``, ``UBERON``, ``CL``, ``MONDO``, ``HP``,
     ``NCBITaxon``, ``ENSEMBL``, ``FAKE``).
2. **Listed slugs**: tokens equal to an ID that some tool listed in its ``ids`` but that has no
   fixed pattern: client and site names (``SpaceX-16-ISST``, ``SpaceX-16``), KG mission slugs
   (``Soyuz_TMA-14``), release-log IDs (``restricted-1``). Mission slugs are opaque (§1), so the
   only reliable way to recognise one is to have seen it. Longest first, so a client name beats
   the site name it starts with.
3. **Slug-shaped tokens**: a capital letter, then letters or digits, then one or more ``-`` or
   ``_`` parts, with a digit somewhere (``SpaceX-99``, ``STS-200``, ``RR-12``). These catch a
   mission or site name that **no** tool returned, i.e. an invented one, which class 2 can't see.
   A slug-shaped token that is a **gene symbol** in some tool result of the trace is a name,
   not an ID (``Nkx2-1``, ``NKX2-5``): symbols are display labels and never evidence, so they
   are neither required to be cited nor counted as fabricated.

What the slug shape costs: ordinary words of that shape also count (``Top-10``, ``COVID-19``),
so a report says "top 10". Any numbered label of that shape (``Step-2``) counts too; the report
states caveats in words and cites the ``note:`` ID. The opposite choice (no slug class) would let
"the result holds without SpaceX-99" pass unchecked.

Trailing punctuation is never part of a token: ``FAKE:0001.`` is ``FAKE:0001``. Mock hashes
(``FAKE-sha256-0004``, CONTRACTS §1) are hashes, not slugs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

from fedkg_spacebio.ids import FAKE_SHA256_RE

_STAMP = r"\d{8}T\d{6}Z"
_TRACE = rf"trace:(?:{_STAMP}-[0-9a-f]{{8}}|FAKE-\d{{4}})(?:/(?:call|dec)/[1-9][0-9]*)?"

PATTERN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"kgedge:[A-Za-z_]+\|[^|\s\"]+\|[^|\s\"]+"
    r"|OSD-\d+-[0-9a-f]{32}"
    r"|OSD-FAKE-\d+(?:-0{32})?"
    r"|OSD-\d+"
    rf"|{_TRACE}"
    r"|(?:job|run|rel|dec):[A-Za-z0-9-]+"
    r"|pol:[A-Za-z0-9][A-Za-z0-9._-]*:[1-9][0-9]*(?:#[A-Za-z0-9_-]+)?"
    r"|spec#[A-Za-z0-9_-]+"
    r"|note:[a-z0-9-]+"
    r"|(?:NCBIGene|UBERON|CL|MONDO|HP|NCBITaxon|ENSEMBL|FAKE):[A-Za-z0-9_.]+"
    r")"
)
SLUG_SHAPE_RE = re.compile(
    r"(?<![A-Za-z0-9_.:#/|-])"
    r"(?=[A-Za-z0-9_-]*\d)"          # a digit somewhere in the token
    r"[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+"
    r"(?![A-Za-z0-9_-])"
)
_TRAILING = ".,;:)]}'\""
# Fields whose values are gene symbols (display labels, never IDs): results.read rows use
# ``symbol``; kg.query uses ``gene_symbol``.
SYMBOL_KEYS = frozenset({"symbol", "gene_symbol"})


def _strip(token: str) -> str:
    return token.rstrip(_TRAILING)


def is_pattern_id(token: str) -> bool:
    """True if the whole token is an ID of a fixed §1 form."""
    m = PATTERN_RE.fullmatch(token)
    return bool(m) and _strip(token) == token


def pattern_tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, token) of every pattern ID in ``text``, trailing punctuation removed."""
    out = []
    for m in PATTERN_RE.finditer(text):
        tok = _strip(m.group(0))
        if tok:
            out.append((m.start(), m.start() + len(tok), tok))
    return out


def _vocab_re(vocabulary: Iterable[str]) -> re.Pattern | None:
    words = sorted({v for v in vocabulary if v}, key=lambda v: (-len(v), v))
    if not words:
        return None
    alt = "|".join(re.escape(w) for w in words)
    return re.compile(rf"(?<![A-Za-z0-9_.:#/|-])(?:{alt})(?![A-Za-z0-9_-])")


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def scan(text: str, vocabulary: Iterable[str] = (), symbols: Iterable[str] = ()) -> list[str]:
    """The ID-shaped tokens of ``text``, in order of appearance, duplicates kept once.

    ``vocabulary``: non-pattern IDs listed by tools (class 2). ``symbols``: gene symbols seen
    in tool results; a slug-shaped token equal to one is a name (class 3 exception). A listed
    ID wins over a symbol of the same spelling.
    """
    found: list[tuple[int, str]] = []
    spans = []
    for a, b, tok in pattern_tokens(text):
        found.append((a, tok))
        spans.append((a, b))
    rest = _mask(text, spans)
    vre = _vocab_re(vocabulary)
    if vre is not None:
        for m in vre.finditer(rest):
            found.append((m.start(), m.group(0)))
            spans.append((m.start(), m.end()))
        rest = _mask(text, spans)
    symbol_set = set(symbols)
    for m in SLUG_SHAPE_RE.finditer(rest):
        if m.group(0) not in symbol_set and not FAKE_SHA256_RE.fullmatch(m.group(0)):
            found.append((m.start(), m.group(0)))
    found.sort()
    return list(dict.fromkeys(tok for _, tok in found))


# ------------------------------------------------------------------ structured data


def strings(obj: Any) -> Iterator[str]:
    """Every string value in a nested structure (dict keys are field names and are skipped)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from strings(v)


def symbol_values(obj: Any) -> set[str]:
    """Values of ``symbol`` / ``gene_symbol`` fields anywhere in ``obj``."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SYMBOL_KEYS and isinstance(v, str):
                out.add(v)
            else:
                out |= symbol_values(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            out |= symbol_values(v)
    return out


def data_tokens(data: Any, vocabulary: Iterable[str] = ()) -> set[str]:
    """ID-shaped tokens in every string of ``data``, except gene-symbol fields.

    Symbol fields are skipped entirely: a symbol is a label, so it is never an ID a tool must
    list. Other strings (free-text reasons and notes included) are scanned like claim text.
    """
    symbols = symbol_values(data)
    vocab = list(vocabulary)
    return {t for s in _non_symbol_strings(data) for t in scan(s, vocab, symbols)}


def _non_symbol_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in SYMBOL_KEYS:
                yield from _non_symbol_strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _non_symbol_strings(v)
