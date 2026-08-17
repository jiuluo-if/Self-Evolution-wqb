import logging
import re

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
    def __init__(self, client, pagination_limit=50, max_pages=20):
        self.client = client
        self.pagination_limit = pagination_limit
        self.max_pages = max_pages
        self._cache = {}  # (dataset_id, limit, offset) -> cached page payload
        self._budget = None
        self._calls = 0

    def reset_budget(self, budget):
        """Per-round cap on new field-discovery API requests (0 = unlimited)."""
        self._budget = budget
        self._calls = 0

    def _page(self, dataset_id, limit, offset):
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

    def _scored_fields(self, dataset_id, keywords, need, seen):
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
                score = self._score_field(field, keywords)
                if score > 0:
                    ranked.append((score, field))
            offset += len(results)
            if not results or (total and offset >= total) or len(ranked) >= need:
                break
        ranked.sort(key=lambda x: -x[0])
        return ranked[:need]

    def _score_field(self, field, keywords):
        haystack_id = field.get("id", "").lower()
        haystack_name = field.get("name", "").lower()
        haystack_desc = (field.get("description") or "").lower()
        score = 0.0
        for kw in keywords:
            if kw in haystack_id:
                score += 3.0
            if kw in haystack_name:
                score += 2.0
            if kw in haystack_desc:
                score += 1.0
        return score

    @staticmethod
    def _keywords_from_hypothesis(hypothesis, limit=6):
        ordered = []
        seen = set()

        def push(word):
            w = word.lower()
            if len(w) > 2 and w not in seen and w not in _STOPWORDS:
                seen.add(w)
                ordered.append(w)

        for tag in hypothesis.get("tags", []):
            for piece in re.split(r"[^a-z0-9]+", tag.lower()):
                if piece:
                    push(piece)
        for piece in re.split(
            r"[^a-z0-9]+", hypothesis.get("statement", "").lower()
        ):
            if piece:
                push(piece)
        return ordered[:limit]

    def discover(self, hypothesis, target_count=6):
        categories = self.categorize_hypothesis(hypothesis)
        keywords = self._keywords_from_hypothesis(hypothesis)
        chosen = []
        seen = set()
        for category in categories:
            if len(chosen) >= target_count:
                break
            for dataset_id in DATASET_CATEGORIES[category]:
                if len(chosen) >= target_count:
                    break
                try:
                    ranked = self._scored_fields(
                        dataset_id,
                        keywords,
                        need=target_count - len(chosen),
                        seen=seen,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Never silently swallow field-discovery errors: log them so
                    # repeated failures are visible and diagnosable.
                    logger.warning(
                        "FIELD_DISCOVERY_FAILED dataset=%s error=%s:%s",
                        dataset_id,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                for score, field in ranked:
                    if len(chosen) >= target_count:
                        break
                    chosen.append(
                        {
                            "id": field["id"],
                            "name": field.get("name", ""),
                            "category": category,
                            "dataset": dataset_id,
                            "match_score": score,
                        }
                    )
                    seen.add(field["id"])
        return chosen


_STOPWORDS = {
    "with", "from", "that", "this", "will", "have", "been", "being",
    "into", "over", "under", "across", "about", "their", "there", "which",
    "while", "using", "should", "would", "where", "when", "after", "before",
}
