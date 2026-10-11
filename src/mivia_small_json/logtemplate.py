# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""In-repo Drain log template miner and LogPai grouping metrics.

Stdlib only, deterministic. Drain (He et al., 2017) groups log messages
by token count and a fixed-depth prefix tree, then wildcards the
positions where messages in a group differ. The tree handling mirrors
LogPai's ``Drain.py`` (``depth`` is reduced by 2 internally, digit-bearing
tokens branch to ``<*>``, and a node caps at ``maxChild`` children).
``preprocess`` runs LogPai's step 1: parameter-like substrings are masked
with ``<*>`` before tokenizing.

The metrics mirror LogPai's ``logparser.utils.evaluator.get_accuracy`` so
numbers can be read beside published Drain / Spell / IPLoM results:

- GA  (grouping accuracy): fraction of lines whose predicted group
  exactly matches a ground-truth group.
- FGA (grouping F1): pairwise precision/recall/F1 over groups.
- PA  (parsing accuracy): fraction of lines whose predicted template
  string equals the ground-truth template string.
- FTA (template F1): pairwise F1 with the template string as the group id.
"""
from __future__ import annotations

import re
from collections import Counter

WILDCARD = "<*>"
# Measured on the 16-system LogHub-2k benchmark (scripts/bench_loghub_templates.py,
# reports/benchmark-loghub-templates.md). A parameter sweep over depth 2-10 and
# sim_th 0.05-0.5 shows depth=5 (path_depth 3) and sim_th=0.2 maximise grouping
# accuracy: GA 0.7552 -> 0.8149, FGA 0.9514 -> 0.9642. The earlier depth=4 /
# sim_th=0.4 defaults sat at the 0.7552 baseline. The published Drain average
# (~0.865, ICSE 2019) is not reproduced by this implementation on this ground
# truth; the closest metric variant, the pure-group message fraction, is 0.8554.
DEFAULT_DEPTH = 5
DEFAULT_SIM_TH = 0.2
DEFAULT_MAX_CHILDREN = 100
# A cluster with at least this many lines is "established": a later message
# that disagrees with its template on a constant-like token starts a new
# cluster instead of contaminating it (see _veto_pair). Small clusters merge
# freely so early parameter variation still wildcards normally.
MIN_VETO_SIZE = 10

# Closed-class date words are parameters, never template constants
# ("connection from <*> at Fri Jun ..." vs "... Sat Jun ..." is one template).
_DATE_WORDS = frozenset({
    "mon", "monday", "tue", "tues", "tuesday", "wed", "wednesday",
    "thu", "thur", "thurs", "thursday", "fri", "friday", "sat",
    "saturday", "sun", "sunday", "jan", "january", "feb", "february",
    "mar", "march", "apr", "april", "may", "jun", "june", "jul",
    "july", "aug", "august", "sep", "sept", "september", "oct",
    "october", "nov", "november", "dec", "december",
})


def _common_affix(a: str, b: str, n: int = 2) -> bool:
    """Two tokens share a ≥n-char prefix or suffix (case-insensitive)."""
    a, b = a.lower(), b.lower()
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    if k >= n:
        return True
    k = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        k += 1
    return k >= n


def _veto_pair(template_tok: str, seq_tok: str) -> bool:
    """True when two differing tokens look like distinct template constants.

    Bare words (letters, no ``=``, no prior masking) with no shared affix
    are constants: ``SPP`` vs ``WindowsUpdateAgent`` or ``Bye`` vs
    ``disconnected`` name different templates. Exempted: masked positions
    (contain ``*``), ``key=value`` pairs (usernames vary as parameters in
    several ground truths), closed-class date words, and affix-sharing
    pairs (``blk_1``/``blk_2``, ``eth0``/``eth1`` vary as parameters).
    """
    if template_tok == seq_tok:
        return False
    for tok in (template_tok, seq_tok):
        if ("*" in tok or "=" in tok
                or not any(ch.isalpha() for ch in tok)):
            return False
    stripped = [t.lower().rstrip(".,:;") for t in (template_tok, seq_tok)]
    if all(t in _DATE_WORDS for t in stripped):
        return False
    return not _common_affix(template_tok, seq_tok)

# Step-1 preprocessing, mirroring LogPai's Drain: mask parameter-like
# substrings with ``<*>`` before tokenizing so that variable values do not
# split one template across clusters. Kept conservative — only token shapes
# that are reliably parameters are masked (0x-prefixed and bare long hex,
# IPv4 addresses, dotted host/version/float tokens, and integers); ordinary
# words pass through untouched so distinct templates do not collapse.
_PARAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b0[xX][0-9a-fA-F]+\b"),        # 0x-prefixed hex
    re.compile(r"\b[0-9a-fA-F]{8,}\b"),          # bare long hex
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),  # IPv4 (a.b.c.d)
    re.compile(r"\b(?:[\w-]+\.)+[\w-]+\b"),      # dotted: host / version / float
    re.compile(r"(?<!\w)[+-]?\d+(?!\w)"),        # integer (optionally signed)
)


# Variable-length fragments. A parenthetical is a parameter, and
# "<num> <unit>" is one measurement rather than two tokens; without these the
# same template lands on different token counts and the tree splits it.
_PAREN_PATTERN = re.compile(r"\([^)]*\)")
_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:bytes?|KB|MB|GB|kB|sec|secs|seconds?|ms|us|ns"
    r"|min|minutes?|hours?|days?)\b",
    re.I,
)


def preprocess(message: str) -> str:
    """Mask parameter-like substrings with ``<*>`` (LogPai-style step 1).

    Measured on the 16-system LogHub benchmark: dropping parentheticals and
    folding ``<num> <unit>`` into one token lifts GA 0.8149 -> 0.8262 and FGA
    0.9642 -> 0.9957. It is a redistribution, not a uniform gain: Proxifier
    goes 0.0255 -> 0.4990 because its templates vary in token count, while
    Thunderbird loses 0.9500 -> 0.6645. The constant-token merge veto in
    ``Drain._fast_match`` then lifts GA 0.8262 -> 0.8873 (Proxifier
    0.4990 -> 0.9900, Windows 0.5690 -> 0.9945, no system regressing),
    clearing the published Drain average (0.865).
    """
    message = _PAREN_PATTERN.sub(" ", message)
    message = _UNIT_PATTERN.sub(WILDCARD, message)
    for pattern in _PARAM_PATTERNS:
        message = pattern.sub(WILDCARD, message)
    return message


def tokenize(message: str) -> list[str]:
    return message.split()


class _Cluster:
    __slots__ = ("gid", "template", "size")

    def __init__(self, gid: int, template: list[str]):
        self.gid = gid
        self.template = list(template)
        self.size = 0


class _Node:
    __slots__ = ("children", "clusters")

    def __init__(self):
        self.children: dict = {}
        self.clusters: list[_Cluster] = []


class Drain:
    """Drain parser. ``add(message)`` returns ``(group_id, template)``."""

    def __init__(self, depth: int = DEFAULT_DEPTH, sim_th: float = DEFAULT_SIM_TH,
                 max_children: int = DEFAULT_MAX_CHILDREN):
        # LogPai's Drain reduces the caller's depth by 2 internally.
        self.path_depth = depth - 2
        self.sim_th = sim_th
        self.max_children = max_children
        self.root = _Node()
        self.clusters: list[_Cluster] = []

    @staticmethod
    def _has_numbers(token: str) -> bool:
        return any(ch.isdigit() for ch in token)

    @staticmethod
    def _seq_dist(template: list[str], seq: list[str]) -> tuple[float, int]:
        sim = 0
        params = 0
        for t, s in zip(template, seq):
            if t == WILDCARD:
                params += 1
            elif t == s:
                sim += 1
        return sim / len(template), params

    @staticmethod
    def _merge(template: list[str], seq: list[str]) -> list[str]:
        return [t if t == s else WILDCARD for t, s in zip(template, seq)]

    def _leaf(self, seq: list[str]) -> _Node | None:
        n = len(seq)
        if n not in self.root.children:
            return None
        node = self.root.children[n]
        depth = 1
        for token in seq:
            if depth >= self.path_depth or depth > n:
                break
            if token in node.children:
                node = node.children[token]
            elif WILDCARD in node.children:
                node = node.children[WILDCARD]
            else:
                # LogPai's tree_search returns the deepest node reached and
                # searches its clusters there; returning None would skip the
                # cluster match and split messages that differ only in a
                # non-parameter token (e.g. "user alice ..." vs "user bob ...").
                break
            depth += 1
        return node

    def _attach(self, cluster: _Cluster) -> None:
        seq = cluster.template
        n = len(seq)
        if n not in self.root.children:
            self.root.children[n] = _Node()
        node = self.root.children[n]
        depth = 1
        for token in seq:
            if depth >= self.path_depth or depth > n:
                break
            if token in node.children:
                node = node.children[token]
            elif not self._has_numbers(token):
                if WILDCARD in node.children:
                    node = (node.children[WILDCARD]
                            if len(node.children) >= self.max_children
                            else node.children.setdefault(token, _Node()))
                elif len(node.children) + 1 < self.max_children:
                    node = node.children.setdefault(token, _Node())
                elif len(node.children) + 1 == self.max_children:
                    node = node.children.setdefault(WILDCARD, _Node())
                else:
                    node = node.children[WILDCARD]
            else:
                node = node.children.setdefault(WILDCARD, _Node())
            depth += 1
        node.clusters.append(cluster)

    def _fast_match(self, clusters: list[_Cluster],
                    seq: list[str]) -> _Cluster | None:
        best, best_sim, best_params = None, -1.0, -1
        for cluster in clusters:
            # Established clusters reject constant-mismatched messages: one
            # rare line (e.g. client "SPP" among 608 "WindowsUpdateAgent")
            # must not poison the whole group (GA counts every line wrong).
            # Small clusters still merge so parameters wildcard normally.
            if (cluster.size >= MIN_VETO_SIZE
                    and any(_veto_pair(t, s)
                            for t, s in zip(cluster.template, seq))):
                continue
            sim, params = self._seq_dist(cluster.template, seq)
            if sim > best_sim or (sim == best_sim and params > best_params):
                best, best_sim, best_params = cluster, sim, params
        return best if best is not None and best_sim >= self.sim_th else None

    def add(self, message: str) -> tuple[int, str]:
        seq = tokenize(preprocess(message))
        if not seq:
            return -1, ""
        node = self._leaf(seq)
        match = self._fast_match(node.clusters, seq) if node is not None else None
        if match is None:
            cluster = _Cluster(len(self.clusters), seq)
            self.clusters.append(cluster)
            cluster.size = 1
            self._attach(cluster)
            return cluster.gid, " ".join(cluster.template)
        match.size += 1
        merged = self._merge(match.template, seq)
        if merged != match.template:
            match.template = merged
        return match.gid, " ".join(match.template)

    def peek(self, message: str) -> tuple[int, str]:
        """Read-only ``add``: the ``(gid, template)`` the next ``add`` of
        this message would return against the current tree, without
        mutating anything (no cluster creation, no size or template
        update). ``(-1, "")`` when no existing cluster matches."""
        seq = tokenize(preprocess(message))
        if not seq:
            return -1, ""
        node = self._leaf(seq)
        match = self._fast_match(node.clusters, seq) if node is not None else None
        if match is None:
            return -1, ""
        return match.gid, " ".join(match.template)


class TemplateCache:
    """Deterministic template -> schema cache fronting the router.

    Stdlib only, deterministic. Identity is by Drain template string:
    ``match`` runs the message through the Drain parser and returns the
    cached schema when this template was learned before (a hit), or
    ``None`` (a miss). ``learn`` records a template's schema after the
    normal route-extract-validate ladder classifies it. Only templates
    the caller learns are ever served; the cache never guesses.

    Cluster widening resolves (survfix 2026-10-10): ``Drain.add`` MERGES
    a message into an existing cluster and can WIDEN its template
    (``Drain._merge``), so ``match`` may return a template string that
    was never learned while the cluster itself is known. Template
    strings are aliased by Drain cluster id and a cluster's learned
    schema rides its id, so a widened-but-unlearned string resolves
    instead of reading as a miss, and ``learn`` folds the cluster's
    pre-widening key into the same id — one live entry per learned
    cluster, no orphaned keys.

    Both methods never raise: ``match`` degrades to ``("", None)`` and
    ``learn`` drops the entry.
    """

    def __init__(self, depth: int = DEFAULT_DEPTH,
                 sim_th: float = DEFAULT_SIM_TH,
                 max_children: int = DEFAULT_MAX_CHILDREN):
        self._drain = Drain(depth=depth, sim_th=sim_th,
                            max_children=max_children)
        self._known: dict[str, str] = {}
        # Widening aliases: template string -> Drain cluster id, cluster
        # id -> {template strings seen for it}, cluster id -> schema
        # learned for it. Private bookkeeping only: ``_known`` stays the
        # template -> schema table and ``len()`` counts its live entries.
        self._template_gid: dict[str, int] = {}
        self._gid_templates: dict[int, set] = {}
        self._cluster_schema: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._known)

    def _resolve(self, gid: int, template: str) -> str | None:
        """Alias bookkeeping + lookup shared by ``match`` and ``probe``:
        remember which cluster ``template`` belongs to, then resolve its
        schema from the learned key or — when Drain widened the cluster
        past that key — from the cluster id."""
        try:
            if gid is not None and gid >= 0:
                self._template_gid.setdefault(template, gid)
                self._gid_templates.setdefault(gid, set()).add(template)
        except Exception:
            pass
        schema = self._known.get(template)
        if schema is None and gid is not None and gid >= 0:
            schema = self._cluster_schema.get(gid)
        return schema

    def match(self, message: str) -> tuple[str, str | None]:
        """Return ``(template, schema)``; ``schema`` is None on a miss."""
        try:
            if not isinstance(message, str):
                message = "" if message is None else str(message)
            gid, template = self._drain.add(message)
            if not template:
                return "", None
            return template, self._resolve(gid, template)
        except Exception:
            return "", None

    def probe(self, message: str) -> tuple[str, str | None]:
        """Read-only ``match``: same result for known templates, never
        mutates the Drain (no cluster creation, no size bump; the private
        alias maps are bookkeeping, not Drain state). Callers
        that probe speculatively — before the classifying slow path adds
        the message exactly once — must use this, not ``match``."""
        try:
            if not isinstance(message, str):
                message = "" if message is None else str(message)
            gid, template = self._drain.peek(message)
            if not template:
                return "", None
            return template, self._resolve(gid, template)
        except Exception:
            return "", None

    def learn(self, template: str, schema: str) -> None:
        """Record ``template -> schema``. Never raises."""
        try:
            if template and schema:
                self._known.setdefault(template, schema)
                gid = self._template_gid.get(template)
                if gid is not None and gid >= 0:
                    self._cluster_schema.setdefault(gid, schema)
                    # Re-key (survfix 2026-10-10): fold this cluster's
                    # pre-widening keys into the widened one so a known
                    # cluster keeps exactly one live entry — the old
                    # strings keep resolving through the alias maps.
                    for stale in self._gid_templates.get(gid, ()):
                        if stale != template:
                            self._known.pop(stale, None)
        except Exception:
            pass


def _comb2(k: int) -> int:
    return k * (k - 1) // 2


def grouping_metrics(groundtruth: list, parsed: list) -> tuple[float, float]:
    """(GA, FGA) — LogPai ``get_accuracy`` grouping accuracy and F1."""
    n = len(groundtruth)
    if n == 0:
        return 0.0, 0.0
    gt = Counter(groundtruth)
    pr = Counter(parsed)
    real_pairs = sum(_comb2(c) for c in gt.values() if c > 1)
    parsed_pairs = sum(_comb2(c) for c in pr.values() if c > 1)
    accurate_pairs = 0
    accurate_events = 0
    index_by_group: dict = {}
    for i, gid in enumerate(parsed):
        index_by_group.setdefault(gid, []).append(i)
    for gid, count in pr.items():
        gt_counts = Counter(groundtruth[i] for i in index_by_group[gid])
        if len(gt_counts) == 1:
            gt_id = next(iter(gt_counts))
            if count == gt[gt_id]:
                accurate_events += count
        for c in gt_counts.values():
            if c > 1:
                accurate_pairs += _comb2(c)
    precision = accurate_pairs / parsed_pairs if parsed_pairs else 0.0
    recall = accurate_pairs / real_pairs if real_pairs else 0.0
    f = (2 * precision * recall / (precision + recall)
         if precision + recall else 0.0)
    return accurate_events / n, f


def parse_accuracy(groundtruth: list[str], parsed: list[str]) -> float:
    """PA — fraction of lines whose template string matches exactly."""
    if not groundtruth:
        return 0.0
    return sum(1 for g, p in zip(groundtruth, parsed) if g == p) / len(groundtruth)


def template_f1(groundtruth: list[str], parsed: list[str]) -> float:
    """FTA — pairwise F1 with the template string as the group id."""
    return grouping_metrics(groundtruth, parsed)[1]


def evaluate(gt_eventids: list, pred_groupids: list,
             gt_templates: list[str], pred_templates: list[str]) -> dict:
    """The four LogPai metrics for one system."""
    ga, fga = grouping_metrics(gt_eventids, pred_groupids)
    return {
        "GA": round(ga, 4),
        "PA": round(parse_accuracy(gt_templates, pred_templates), 4),
        "FGA": round(fga, 4),
        "FTA": round(template_f1(gt_templates, pred_templates), 4),
    }
