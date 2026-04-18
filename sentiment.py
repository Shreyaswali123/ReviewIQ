"""
sentiment.py — Enterprise sentiment analysis for ReviewIQ
Primary:  Google Cloud Natural Language API
Fallback: Keyword-based lexicon (zero extra dependencies)

Confidence scoring
------------------
Keyword path : # of distinct keywords matched for a feature → 0.65 / 0.80 / 0.92,
               +0.03 boost when surrounding sentence has unambiguous polarity.
GCP path     : entity.salience rescaled [0.05,1.0] → [0.62,0.98].
Both paths attach  "confidence": float  to every feature dict.
"""

import re
import time
from dotenv import load_dotenv

load_dotenv()

# ── Lexicons ───────────────────────────────────────────────────────────────────

_POS_WORDS = frozenset({
    "great", "excellent", "amazing", "good", "love", "best", "fantastic",
    "awesome", "perfect", "recommend", "happy", "satisfied", "comfortable",
    "durable", "quality", "value", "fast", "quick", "easy", "nice", "solid",
    "clean", "fresh", "tasty", "delicious", "smooth", "accurate", "reliable",
    "beautiful", "lightweight", "soft", "warm", "bright", "clear", "crisp",
    "sturdy", "premium", "responsive", "efficient", "helpful", "impressive",
    "superb", "outstanding", "remarkable", "pleasantly", "exceeded",
})

_NEG_WORDS = frozenset({
    "terrible", "awful", "horrible", "bad", "worst", "broken", "poor",
    "disappointed", "useless", "waste", "refund", "return", "defective",
    "cheap", "flimsy", "slow", "late", "damaged", "crushed", "smashed",
    "stale", "expired", "blisters", "pain", "scratched", "cracked",
    "dead", "drained", "dies", "ruined", "unusable", "fraud", "scam",
    "fake", "misleading", "wrong", "missing", "never", "lost", "stopped",
    "leaking", "peeling", "broke", "failed", "disappointment", "overheating",
    "overpriced", "mislead", "refuse", "refused", "ignored", "unresponsive",
})

_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "Battery":          ["battery", "charge", "charging", "power", "drain", "dies",
                         "dead", "runtime", "standby", "mah", "unplugged"],
    "Packaging":        ["packaging", "package", "box", "wrap", "seal",
                         "crushed", "damaged", "torn", "dented", "mangled"],
    "Delivery":         ["delivery", "shipping", "courier", "transit", "arrive",
                         "arrived", "delayed", "late", "lost", "dispatch"],
    "Build Quality":    ["build", "quality", "material", "durable", "durability",
                         "cheap", "flimsy", "sturdy", "construction", "finish"],
    "Comfort":          ["comfort", "comfortable", "fit", "wear", "ergonomic",
                         "soft", "hard", "tight", "loose", "cushion"],
    "Taste":            ["taste", "flavor", "flavour", "delicious", "stale",
                         "fresh", "expire", "bland", "chalky", "sweet"],
    "Display":          ["display", "screen", "brightness", "resolution",
                         "crisp", "glare", "refresh", "vivid", "dim"],
    "Price / Value":    ["price", "value", "worth", "expensive", "cost",
                         "money", "overpriced", "affordable", "budget"],
    "Customer Support": ["support", "service", "refund", "return", "response",
                         "reply", "help", "staff", "warranty", "contact"],
    "Performance":      ["performance", "speed", "fast", "slow", "lag",
                         "smooth", "accurate", "sluggish", "responsive"],
    "Connectivity":     ["wifi", "bluetooth", "connection", "signal", "sync",
                         "pairing", "disconnect", "pair", "wireless"],
    "Size / Fit":       ["size", "sizing", "small", "large", "wide", "narrow",
                         "true to size", "too big", "too small", "length"],
}


# ── Confidence helpers ─────────────────────────────────────────────────────────

def _feature_confidence(text_lower: str, keywords: list[str], sent_words: set) -> float:
    """
    1 keyword match → 0.65,  2 → 0.785,  3+ → 0.92
    +0.03 if surrounding sentence has clear polarity (|pos-neg| >= 2).
    Capped at 0.97.
    """
    match_count = sum(1 for kw in keywords if kw in text_lower)
    base = 0.65 + min(match_count - 1, 2) * 0.135
    if abs(len(sent_words & _POS_WORDS) - len(sent_words & _NEG_WORDS)) >= 2:
        base = min(base + 0.03, 0.97)
    return round(base, 2)


def _salience_to_confidence(salience: float) -> float:
    """GCP entity salience [0.05,1.0] → confidence [0.62,0.98]."""
    c = max(0.05, min(1.0, salience))
    return round(0.62 + (c - 0.05) / 0.95 * 0.36, 2)


# ── Keyword fallback ───────────────────────────────────────────────────────────

def _keyword_sentiment(text: str) -> tuple[str, float]:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos, neg = len(words & _POS_WORDS), len(words & _NEG_WORDS)
    if pos + neg == 0:
        return "neutral", 0.0
    score = (pos - neg) / (pos + neg)
    return ("positive" if score >= 0.15 else "negative" if score <= -0.15 else "neutral"), score


def _keyword_features(text: str, overall_sent: str) -> list[dict]:
    """
    Per-sentence sentiment + match-count confidence per feature.
    """
    text_lower = text.lower()
    sentences  = re.split(r"[.!?;,]", text_lower)
    results: list[dict] = []
    seen: set[str] = set()

    for feature, keywords in _FEATURE_KEYWORDS.items():
        if not any(kw in text_lower for kw in keywords):
            continue
        if feature in seen:
            continue
        seen.add(feature)

        feat_sent   = overall_sent
        sent_words: set = set()

        for sent in sentences:
            if any(kw in sent for kw in keywords):
                sent_words = set(re.findall(r"\b\w+\b", sent))
                pos_h = len(sent_words & _POS_WORDS)
                neg_h = len(sent_words & _NEG_WORDS)
                if neg_h > pos_h:
                    feat_sent = "negative"
                elif pos_h > neg_h:
                    feat_sent = "positive"
                break

        results.append({
            "feature_name": feature,
            "sentiment":    feat_sent,
            "confidence":   _feature_confidence(text_lower, keywords, sent_words),
            "snippet":      text[:60] + "...",
        })
    return results


def _fallback_single(r: dict) -> dict:
    text      = str(r.get("review_text", r.get("text", "")))
    review_id = str(r.get("review_id",  r.get("id",  "unknown")))
    overall, _ = _keyword_sentiment(text)
    return {"review_id": review_id, "overall_sentiment": overall,
            "features": _keyword_features(text, overall), "source": "fallback"}


def _fallback_batch(reviews: list[dict]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    print("⚠️  Using keyword fallback for all reviews.")
    workers = min(16, len(reviews) or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_fallback_single, reviews))


# ── GCP ────────────────────────────────────────────────────────────────────────

GCP_BATCH_SIZE = 100;  GCP_BATCH_PAUSE = 1.0
GCP_RETRY_LIMIT = 3;   GCP_RETRY_DELAY = 2.0
MAX_CONSECUTIVE_FAILURES = 8


def _is_quota(e):
    m = str(e).lower()
    return any(s in m for s in ("resource_exhausted","429","quota exceeded","ratelimitexceeded"))

def _is_transient(e):
    m = str(e).lower()
    return any(s in m for s in ("503","502","unavailable","deadline exceeded","timeout"))

def _gcp_retry(client, document, features):
    delay = GCP_RETRY_DELAY
    for attempt in range(GCP_RETRY_LIMIT):
        try:
            return client.annotate_text(request={"document": document, "features": features})
        except Exception as e:
            if _is_quota(e): raise
            if _is_transient(e) and attempt < GCP_RETRY_LIMIT - 1:
                time.sleep(delay); delay *= 2
            else:
                raise
    raise RuntimeError("GCP retries exhausted")


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_batch(reviews: list[dict], progress_callback=None) -> list[dict]:
    """
    Returns list of {review_id, overall_sentiment, features, source}.
    Each feature has {feature_name, sentiment, confidence, snippet}.
    """
    results: list[dict] = []
    total = len(reviews)

    try:
        from google.cloud import language_v1
        client   = language_v1.LanguageServiceClient()
        gcp_feat = language_v1.AnnotateTextRequest.Features(
            extract_document_sentiment=True, extract_entity_sentiment=True)
        use_gcp = True
    except Exception as e:
        print(f"❌ GCP init failed ({e}). Keyword fallback.")
        use_gcp = False

    if not use_gcp:
        results = _fallback_batch(reviews)
        if progress_callback: progress_callback(total, total, "Fallback analysis")
        return results

    print(f"\n🚀 Processing {total} reviews via GCP (parallel threads)…")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    quota_exhausted = False

    # Thread-safe result store keyed by index to preserve order
    result_map: dict[int, dict] = {}

    def _process_one(idx_r):
        i, r = idx_r
        text      = str(r.get("review_text", r.get("text", ""))).strip()
        review_id = str(r.get("review_id", r.get("id", "unknown")))
        if not text:
            return i, None

        try:
            from google.cloud import language_v1 as _lv1
            doc  = _lv1.Document(content=text, type_=_lv1.Document.Type.PLAIN_TEXT)
            resp = _gcp_retry(client, doc, gcp_feat)
            score   = resp.document_sentiment.score
            overall = "positive" if score >= 0.2 else "negative" if score <= -0.2 else "neutral"
            extracted = []
            for ent in resp.entities:
                if ent.salience < 0.05: continue
                es    = ent.sentiment.score
                esent = "positive" if es >= 0.1 else "negative" if es <= -0.1 else "neutral"
                extracted.append({
                    "feature_name": ent.name.title(), "sentiment": esent,
                    "confidence":   _salience_to_confidence(ent.salience),
                    "snippet":      text[:60] + "...",
                })
            return i, {"review_id": review_id, "overall_sentiment": overall,
                       "features": extracted, "source": "gcp"}
        except Exception as e:
            if _is_quota(e):
                return i, ("QUOTA", r)
            return i, _fallback_single(r)

    GCP_PARALLEL_WORKERS = 8
    results_ordered = [None] * total

    with ThreadPoolExecutor(max_workers=GCP_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_process_one, (i, r)): i for i, r in enumerate(reviews)}
        done_count = 0
        for fut in as_completed(futures):
            i, res = fut.result()
            done_count += 1
            if progress_callback and done_count % 10 == 0:
                progress_callback(done_count, total, "GCP analysis")

            if res is None:
                continue
            if isinstance(res, tuple) and res[0] == "QUOTA":
                if not quota_exhausted:
                    print(f"⚠️  Quota hit — remaining reviews will use keyword fallback.")
                    quota_exhausted = True
                results_ordered[i] = _fallback_single(res[1])
            else:
                results_ordered[i] = res

    # Fill any None gaps (empty text rows) and collect
    results = [r for r in results_ordered if r is not None]

    if progress_callback: progress_callback(total, total, "Complete")
    gcp_n = sum(1 for r in results if r.get("source") == "gcp")
    print(f"✅ Done: {gcp_n} GCP, {total-gcp_n} fallback")
    return results