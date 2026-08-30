import json
import logging
import os
import re

from .hypothesis import ContractViolation, has_field_semantics
from .state import atomic_write_json

logger = logging.getLogger("wqb.discovery")

DATASET_CATEGORIES = {
    "analyst": ["analyst4"],
    "fundamental": ["fundamental2", "fundamental6"],
    "model": ["model16", "model51"],
    "news": ["news12", "news18"],
    "option": ["option8", "option9"],
    "price_volume": ["pv1", "pv13", "univ1"],
    "social": ["socialmedia12", "socialmedia8"],
}

CATEGORY_VALUE = {
    "model": 7,
    "option": 6,
    "analyst": 5,
    "fundamental": 3,
    "news": 3,
    "price_volume": 2,
    "social": 2,
}

CATEGORY_KEYWORDS = {
    "analyst": [
        "analyst", "recommendation", "target", "rating", "estimate", "eps",
        "consensus", "revision", "forecast", "broker",
    ],
    "fundamental": [
        "fundamental", "revenue", "earning", "margin", "ratio", "balance",
        "cashflow", "cash", "debt", "asset", "equity", "profit", "growth",
    ],
    "model": [
        "model", "score", "sentiment", "forecast", "prediction", "probability",
        "factor", "composite", "risk",
    ],
    "news": [
        "news", "article", "headline", "mention", "buzz", "press", "release",
    ],
    "option": [
        "option", "volatility", "implied", "iv", "put", "call", "gamma",
        "delta", "skew", "greeks", "open_interest", "oi",
    ],
    "price_volume": [
        "price", "volume", "return", "close", "open", "high", "low", "adv",
        "liquidity", "turnover", "volatility",
    ],
    "social": [
        "social", "tweet", "post", "mention", "reddit", "discussion",
        "crowdsource", "score",
    ],
}


class BudgetExhausted(Exception):
    pass


class FieldDiscovery:
    def __init__(self, client, pagination_limit=50, max_pages=20, cache_path=None):
        self.client = client
        self.pagination_limit = pagination_limit
        self.max_pages = max_pages
        # (dataset_id, limit, offset) -> cached page payload. Persisted across
        # processes (cache_path) so a crash or a fresh run reuses previously
        # fetched datafield pages instead of burning Field API budget again.
        self.cache_path = cache_path
        self._cache = self._load_cache() if cache_path else {}
        self._budget = None
        self._calls = 0
        # Outcome of the last discover() call, so callers can tell an API
        # infrastructure failure apart from a genuine "no matching field".
        self.last_outcome = None

    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path) as f:
                raw = json.load(f)
        except (OSError, ValueError):
            # A corrupt cache must never break discovery; fall back to empty.
            logger.warning("FIELD_CACHE_LOAD_FAILED path=%s", self.cache_path)
            return {}
        cache = {}
        for key, payload in raw.items():
            try:
                dataset_id, limit, offset = key.split(":")
                cache[(dataset_id, int(limit), int(offset))] = payload
            except (ValueError, TypeError):
                continue
        return cache

    def _save_cache(self):
        if not self.cache_path:
            return
        raw = {
            f"{dataset_id}:{limit}:{offset}": payload
            for (dataset_id, limit, offset), payload in self._cache.items()
        }
        try:
            atomic_write_json(self.cache_path, raw)
        except OSError as exc:
            logger.warning("FIELD_CACHE_SAVE_FAILED path=%s error=%s",
                           self.cache_path, exc)

    def reset_budget(self, budget):
        """Per-round cap on new field-discovery API requests (0 = unlimited)."""
        self._budget = budget
        self._calls = 0

    def _page(self, dataset_id, limit, offset):
        # The cache key carries the dataset identity: fields cached for
        # dataset A are never reused for dataset B. A cache hit reuses the
        # raw API payload but does not mint new research evidence, and it
        # does not consume Field API budget.
        key = (dataset_id, limit, offset)
        cached = self._cache.get(key)
        if cached is not None:
            return cached["results"], cached["count"]
        if self._budget is not None and self._calls >= self._budget:
            raise BudgetExhausted()
        results, count = self.client.get_datafields(
            dataset_id, limit=limit, offset=offset
        )
        self._calls += 1
        self._cache[key] = {"results": results, "count": count}
        return results, count

    def categorize_hypothesis(self, hypothesis):
        """Category ranking is ordering/fallback metadata only. It never
        decides which datasets a hypothesis is allowed to use; the research
        boundary is hypothesis.datasets."""
        text = hypothesis.get("statement", "")
        tags = hypothesis.get("tags", [])
        combined = " ".join([text] + list(tags)).lower()
        scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in combined)
            if hits:
                scores[category] = hits
        if not scores:
            return []
        return sorted(
            scores.keys(), key=lambda c: (scores[c], CATEGORY_VALUE[c]), reverse=True
        )

    @staticmethod
    def _category_of(dataset_id):
        for category, datasets in DATASET_CATEGORIES.items():
            if dataset_id in datasets:
                return category
        return None

    def _datasets_for(self, hypothesis):
        """The dataset boundary for this hypothesis.

        hypothesis.datasets is an allowlist: Discovery never queries a
        dataset outside it. Cross-dataset exploration requires an explicit
        new hypothesis / dataset variant, never silent widening by
        Discovery. Without a declared allowlist (legacy dicts) the category
        keywords derive the boundary as a fallback.
        """
        declared = [d for d in (hypothesis.get("datasets") or []) if d]
        if declared:
            priority = self.categorize_hypothesis(hypothesis)
            rank = {cat: i for i, cat in enumerate(priority)}

            def sort_key(dataset_id):
                category = self._category_of(dataset_id)
                return (rank.get(category, len(priority)), category or "", dataset_id)

            return sorted(declared, key=sort_key)
        priority = self.categorize_hypothesis(hypothesis)
        datasets = []
        for category in priority:
            for dataset_id in DATASET_CATEGORIES[category]:
                if dataset_id not in datasets:
                    datasets.append(dataset_id)
        return datasets

    @staticmethod
    def _tokenize(texts, limit):
        ordered = []
        seen = set()
        for text in texts:
            for piece in re.split(r"[^a-z0-9]+", text.lower()):
                if len(piece) > 2 and piece not in seen and piece not in _STOPWORDS:
                    seen.add(piece)
                    ordered.append(piece)
                    if len(ordered) >= limit:
                        return ordered
        return ordered[:limit]

    def _semantic_keywords(self, hypothesis, limit=10):
        """WHAT economic meaning a field must have. These words carry the
        highest research weight and are kept separate from general keywords
        so a statement-word coincidence cannot outrank a real meaning match."""
        semantics = (hypothesis.get("field_semantics") or {}).get("primary")
        if not semantics:
            return []
        texts = [
            str(semantics.get("concept") or ""),
            str(semantics.get("description") or ""),
        ]
        return self._tokenize(texts, limit)

    def _general_keywords(self, hypothesis, limit=6):
        texts = [str(hypothesis.get("statement") or "")] + [
            str(t) for t in (hypothesis.get("tags") or [])
        ]
        return self._tokenize(texts, limit)

    def _scored_fields(
        self, dataset_id, semantic_keywords, general_keywords, concept, need, seen
    ):
        ranked = []
        offset = 0
        total = None
        for _ in range(self.max_pages):
            try:
                results, count = self._page(
                    dataset_id, limit=self.pagination_limit, offset=offset
                )
            except BudgetExhausted:
                break
            total = count
            for field in results:
                if field.get("id") in seen:
                    continue
                info = self._score_field(
                    field, semantic_keywords, general_keywords, concept
                )
                if info is None:
                    continue
                # A research contract states WHAT meaning a field must carry.
                # A field id that merely happens to contain a keyword is
                # auxiliary evidence, never a semantic match. Without a
                # semantic match (name/description) the field is not a
                # candidate primary field; a general keyword hit does not
                # rescue it.
                if semantic_keywords and info["semantic_score"] <= 0:
                    continue
                ranked.append(info)
            offset += len(results)
            if not results or (total and offset >= total) or len(ranked) >= need:
                break
        ranked.sort(
            key=lambda info: (info["semantic_score"], info["total_score"]),
            reverse=True,
        )
        return ranked[:need]

    @staticmethod
    def _score_field(field, semantic_keywords, general_keywords, concept=None):
        """Decomposed matching evidence for one field.

        Research priority: semantic description/name match > raw field-id
        coincidence. Semantic hits on name/description count toward
        semantic_score (the entry gate); an id hit alone never does.
        """
        haystack_id = field.get("id", "").lower()
        haystack_name = field.get("name", "").lower()
        haystack_desc = (field.get("description") or "").lower()
        semantic_id = semantic_name = semantic_desc = 0.0
        general_id = general_name = general_desc = 0.0
        matched = []

        def hit(keyword):
            if keyword not in matched:
                matched.append(keyword)

        for kw in semantic_keywords:
            if kw in haystack_id:
                semantic_id += 1.0
                hit(kw)
            if kw in haystack_name:
                semantic_name += 4.0
                hit(kw)
            if kw in haystack_desc:
                semantic_desc += 3.0
                hit(kw)
        for kw in general_keywords:
            if kw in haystack_id:
                general_id += 0.5
                hit(kw)
            if kw in haystack_name:
                general_name += 2.0
                hit(kw)
            if kw in haystack_desc:
                general_desc += 1.0
                hit(kw)

        total = (
            semantic_id
            + semantic_name
            + semantic_desc
            + general_id
            + general_name
            + general_desc
        )
        if total <= 0:
            return None
        return {
            "field": field,
            "semantic_concept": concept,
            "matched_terms": matched,
            "id_score": semantic_id + general_id,
            "name_score": semantic_name + general_name,
            "description_score": semantic_desc + general_desc,
            "semantic_score": semantic_name + semantic_desc,
            "total_score": total,
        }

    def discover(self, hypothesis, target_count=6, require_field_semantics=False):
        if require_field_semantics and not has_field_semantics(hypothesis):
            raise ContractViolation(
                "field_semantics missing; Field Discovery is not allowed "
                "for a hypothesis without a research contract"
            )
        semantic_keywords = self._semantic_keywords(hypothesis)
        general_keywords = self._general_keywords(hypothesis)
        concept = (
            ((hypothesis.get("field_semantics") or {}).get("primary") or {}).get(
                "concept"
            )
        )
        datasets = self._datasets_for(hypothesis)
        chosen = []
        seen = set()
        outcome = {
            "queried": [],
            "succeeded": [],
            "failed": [],
            "empty_datasets": [],
            "infra_failure": False,
            "no_match": False,
        }
        for dataset_id in datasets:
            if len(chosen) >= target_count:
                break
            outcome["queried"].append(dataset_id)
            category = self._category_of(dataset_id) or ""
            try:
                ranked = self._scored_fields(
                    dataset_id,
                    semantic_keywords,
                    general_keywords,
                    concept,
                    need=target_count - len(chosen),
                    seen=seen,
                )
            except Exception as exc:  # noqa: BLE001
                # API timeout / 429 / auth / 5xx is an infrastructure
                # failure, never "this hypothesis has no matching field".
                outcome["failed"].append(dataset_id)
                logger.warning(
                    "FIELD_DISCOVERY_FAILED dataset=%s error=%s:%s",
                    dataset_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            outcome["succeeded"].append(dataset_id)
            if not ranked:
                outcome["empty_datasets"].append(dataset_id)
            for info in ranked:
                if len(chosen) >= target_count:
                    break
                field = info["field"]
                chosen.append(
                    {
                        "id": field["id"],
                        "name": field.get("name", ""),
                        "category": category,
                        "dataset": dataset_id,
                        "match_score": info["total_score"],
                        "field_match": {
                            "semantic_concept": info["semantic_concept"],
                            "matched_terms": info["matched_terms"],
                            "id_score": info["id_score"],
                            "name_score": info["name_score"],
                            "description_score": info["description_score"],
                            "semantic_score": info["semantic_score"],
                            "total_score": info["total_score"],
                        },
                    }
                )
                seen.add(field["id"])
        if datasets:
            outcome["infra_failure"] = (
                len(outcome["failed"]) == len(datasets)
            )
            outcome["no_match"] = (
                not outcome["infra_failure"]
                and not chosen
                and bool(outcome["succeeded"])
            )
        # Persist every page fetched during this discover call in one write,
        # instead of flushing on every _page hit.
        self._save_cache()
        self.last_outcome = outcome
        return chosen


_STOPWORDS = {
    "with", "from", "that", "this", "will", "have", "been", "being",
    "into", "over", "under", "across", "about", "their", "there", "which",
    "while", "using", "should", "would", "where", "when", "after", "before",
}
