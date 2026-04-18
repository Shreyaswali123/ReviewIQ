"""
competitor.py — Competitor Intelligence module for ReviewIQ

Plugs directly into the existing pipeline (sentiment.py, trend_engine.py,
cross_product.py) and adds three new analytical lenses:

  1. Feature Gap Analysis
     Side-by-side negative/positive % per feature (yours vs. competitor).
     Produces RISK / ADVANTAGE / PARITY verdicts per feature.

  2. Competitor Weakness Radar
     Runs the same z-score trend engine on competitor data to surface
     THEIR systemic issues — your potential marketing USPs.

  3. Shared Platform Issues
     Runs cross_product detection on the combined review pool to find
     logistics/vendor defects affecting BOTH you and the competitor,
     indicating a supply-chain problem rather than a product fault.

Public API
----------
    result = analyze_competitor(your_reviews, competitor_map)

    your_reviews   : list[dict]   — already-ingested reviews (from ingest.py)
    competitor_map : dict[str, list[dict]]
        Keys   = competitor product names (e.g. "VortexBand 9")
        Values = ingested review lists for that competitor product

Returns
-------
    dict with keys:
        feature_gaps          list[dict]
        shared_platform_issues list[dict]
        competitor_weaknesses  list[dict]   (their trending problems)
        your_advantages        list[dict]   (features you clearly win)
        competitor_strengths   list[dict]   (features they clearly win)
        summary                dict
"""

import re
from collections import defaultdict

# ── Re-use existing modules ────────────────────────────────────────────────────
from sentiment import _FEATURE_KEYWORDS, _POS_WORDS, _NEG_WORDS
from trend_engine import detect_systemic_trends
from cross_product import find_platform_level_issues

# ── Thresholds ─────────────────────────────────────────────────────────────────

# Minimum negative-mention % delta to call something a RISK or ADVANTAGE
GAP_RISK_THRESHOLD      = 10   # competitor is ≥10pp better than you → RISK
GAP_ADVANTAGE_THRESHOLD = 10   # you are ≥10pp better than competitor → ADVANTAGE

# Minimum reviews per feature before we trust a percentage
MIN_REVIEWS_FOR_PCT = 5

# Z-score window for competitor trend detection
# We treat Batch 1 as "history" and Batch 2 as the latest batch, same as yours.
BATCH_SPLIT_KEY = "review_date"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _keyword_sentiment_for(text: str) -> tuple[str, float]:
    """Returns (overall_sentiment, score) using the same lexicon as sentiment.py."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos, neg = len(words & _POS_WORDS), len(words & _NEG_WORDS)
    if pos + neg == 0:
        return "neutral", 0.0
    score = (pos - neg) / (pos + neg)
    if score >= 0.15:
        return "positive", score
    if score <= -0.15:
        return "negative", score
    return "neutral", score


def _feature_sentiment_in(text: str) -> dict[str, str]:
    """
    Returns {feature_name: sentiment} for every feature keyword found in text.
    Uses per-sentence polarity resolution, mirroring _keyword_features in sentiment.py.
    """
    text_lower = text.lower()
    sentences  = re.split(r"[.!?;,]", text_lower)
    overall, _ = _keyword_sentiment_for(text)
    result: dict[str, str] = {}

    for feature, keywords in _FEATURE_KEYWORDS.items():
        if not any(kw in text_lower for kw in keywords):
            continue
        feat_sent = overall
        for sent in sentences:
            if any(kw in sent for kw in keywords):
                words = set(re.findall(r"\b\w+\b", sent))
                p = len(words & _POS_WORDS)
                n = len(words & _NEG_WORDS)
                if n > p:
                    feat_sent = "negative"
                elif p > n:
                    feat_sent = "positive"
                break
        result[feature] = feat_sent

    return result


def _extract_feature_counts(reviews: list[dict]) -> dict[str, dict[str, int]]:
    """
    Returns {feature → {positive: N, negative: N, neutral: N, total: N}}
    for a list of reviews.
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    )
    for r in reviews:
        text = str(r.get("review_text", r.get("text", ""))).strip()
        if not text:
            continue
        for feature, sentiment in _feature_sentiment_in(text).items():
            counts[feature][sentiment] += 1
            counts[feature]["total"]   += 1
    return dict(counts)


def _pct(count: int, total: int) -> float:
    """Safe integer percentage."""
    return round(count / total * 100) if total >= MIN_REVIEWS_FOR_PCT else 0


def _split_batches(reviews: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Splits reviews at the median date into [earlier, later] batches.
    Mirrors the batch-comparison logic in main.py.
    """
    dated = []
    for r in reviews:
        d = str(r.get("review_date", r.get("date", "")) or "")
        dated.append((d, r))

    dated.sort(key=lambda x: x[0])
    mid = len(dated) // 2
    earlier = [r for _, r in dated[:mid]]
    later   = [r for _, r in dated[mid:]]
    return earlier, later


def _build_negative_feature_counts(
    reviews: list[dict],
    product_name: str,
) -> tuple[dict[str, int], dict[str, list]]:
    """
    Returns (latest_batch_counts, historical_counts) in the format
    expected by detect_systemic_trends().
    Keys are "Product|Feature".
    """
    earlier, later = _split_batches(reviews)

    def _neg_counts(batch: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for r in batch:
            text = str(r.get("review_text", r.get("text", ""))).strip()
            for feature, sentiment in _feature_sentiment_in(text).items():
                if sentiment == "negative":
                    counts[f"{product_name}|{feature}"] += 1
        return dict(counts)

    hist_counts = _neg_counts(earlier)
    late_counts = _neg_counts(later)

    # historical_counts must be lists for trend_engine compatibility
    historical = {k: [v] for k, v in hist_counts.items()}
    return late_counts, historical


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_gap(
    your_reviews: list[dict],
    competitor_reviews: list[dict],
    competitor_name: str,
    your_name: str = "Your Products",
) -> list[dict]:
    """
    Returns a list of per-feature comparison dicts, sorted by gap severity.

    Each dict:
        feature             str
        your_neg_pct        int
        comp_neg_pct        int
        your_pos_pct        int
        comp_pos_pct        int
        gap                 int   (+ve = competitor wins, -ve = you win)
        verdict             str   "RISK" | "ADVANTAGE" | "PARITY"
        your_total          int
        comp_total          int
    """
    your_counts = _extract_feature_counts(your_reviews)
    comp_counts = _extract_feature_counts(competitor_reviews)

    all_features = set(your_counts) | set(comp_counts)
    gaps: list[dict] = []

    for feature in all_features:
        yc = your_counts.get(feature, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        cc = comp_counts.get(feature, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})

        # Skip features with too few data points on either side
        if yc["total"] < MIN_REVIEWS_FOR_PCT and cc["total"] < MIN_REVIEWS_FOR_PCT:
            continue

        y_neg = _pct(yc["negative"], yc["total"])
        c_neg = _pct(cc["negative"], cc["total"])
        y_pos = _pct(yc["positive"], yc["total"])
        c_pos = _pct(cc["positive"], cc["total"])
        gap   = y_neg - c_neg   # positive = we're worse, negative = we're better

        if gap >= GAP_RISK_THRESHOLD:
            verdict = "RISK"
        elif gap <= -GAP_ADVANTAGE_THRESHOLD:
            verdict = "ADVANTAGE"
        else:
            verdict = "PARITY"

        gaps.append({
            "feature":      feature,
            "your_neg_pct": y_neg,
            "comp_neg_pct": c_neg,
            "your_pos_pct": y_pos,
            "comp_pos_pct": c_pos,
            "gap":          gap,
            "verdict":      verdict,
            "your_total":   yc["total"],
            "comp_total":   cc["total"],
            "your_name":    your_name,
            "comp_name":    competitor_name,
        })

    # Sort: RISK first (largest gap), then ADVANTAGE, then PARITY
    order = {"RISK": 0, "ADVANTAGE": 1, "PARITY": 2}
    gaps.sort(key=lambda x: (order[x["verdict"]], -abs(x["gap"])))
    return gaps


def find_competitor_weaknesses(
    competitor_reviews: list[dict],
    competitor_name: str,
) -> list[dict]:
    """
    Runs the z-score trend engine on the competitor's own data to find
    THEIR emerging/systemic problems.

    Returns alerts in the same format as detect_systemic_trends() plus
    a  "context" field: "Competitor Weakness — potential USP for you."
    """
    latest, historical = _build_negative_feature_counts(
        competitor_reviews, competitor_name
    )
    alerts = detect_systemic_trends(latest, historical)
    for a in alerts:
        a["context"] = (
            f"{competitor_name} has a statistically significant spike in "
            f"'{a['feature']}' complaints. This is a potential USP for your brand."
        )
    return alerts


def find_shared_platform_issues(
    your_reviews: list[dict],
    competitor_reviews: list[dict],
    your_category: str = "Your Products",
    competitor_name: str = "Competitor",
) -> list[dict]:
    """
    Runs cross_product detection treating your reviews and competitor reviews
    as two separate 'categories'. Features that appear negatively in BOTH
    are almost certainly logistics / vendor issues, not product defects.

    Returns the same structure as find_platform_level_issues().
    """
    def _neg_features(reviews: list[dict]) -> list[str]:
        features: list[str] = []
        for r in reviews:
            text = str(r.get("review_text", r.get("text", ""))).strip()
            for feature, sentiment in _feature_sentiment_in(text).items():
                if sentiment == "negative":
                    features.append(feature)
        return features

    category_negatives = {
        your_category: _neg_features(your_reviews),
        competitor_name: _neg_features(competitor_reviews),
    }
    issues = find_platform_level_issues(category_negatives)
    for i in issues:
        i["classification"] = "Shared Platform Issue"
        i["context"] = (
            "This defect affects both your products and the competitor's. "
            "Likely a shared logistics or vendor problem — escalate to supply chain."
        )
    return issues


def analyze_competitor(
    your_reviews: list[dict],
    competitor_map: dict[str, list[dict]],
    your_name: str = "Your Products",
) -> dict:
    """
    Master function. Runs all three analyses for every competitor in
    competitor_map.

    Parameters
    ----------
    your_reviews    : your ingested reviews
    competitor_map  : { "CompetitorName": [review, ...], ... }
    your_name       : label for your product group (default "Your Products")

    Returns
    -------
    {
        "competitors": [
            {
                "name":                   str,
                "total_reviews":          int,
                "feature_gaps":           list[dict],
                "shared_platform_issues": list[dict],
                "competitor_weaknesses":  list[dict],
                "your_advantages":        list[dict],   # subset of feature_gaps
                "competitor_strengths":   list[dict],   # subset of feature_gaps
            },
            ...
        ],
        "summary": {
            "total_your_reviews":        int,
            "total_competitor_reviews":  int,
            "risk_count":                int,
            "advantage_count":           int,
            "shared_issue_count":        int,
        }
    }
    """
    results = []
    total_comp_reviews = 0
    total_risks        = 0
    total_advantages   = 0
    total_shared       = 0

    for comp_name, comp_reviews in competitor_map.items():
        if not comp_reviews:
            continue

        total_comp_reviews += len(comp_reviews)

        gaps      = build_feature_gap(your_reviews, comp_reviews, comp_name, your_name)
        weaknesses = find_competitor_weaknesses(comp_reviews, comp_name)
        shared    = find_shared_platform_issues(your_reviews, comp_reviews, your_name, comp_name)

        advantages = [g for g in gaps if g["verdict"] == "ADVANTAGE"]
        risks      = [g for g in gaps if g["verdict"] == "RISK"]
        comp_wins  = [g for g in gaps if g["verdict"] == "RISK"]   # from your POV

        total_risks      += len(risks)
        total_advantages += len(advantages)
        total_shared     += len(shared)

        results.append({
            "name":                   comp_name,
            "total_reviews":          len(comp_reviews),
            "feature_gaps":           gaps,
            "shared_platform_issues": shared,
            "competitor_weaknesses":  weaknesses,
            "your_advantages":        advantages,
            "competitor_strengths":   comp_wins,
        })

    return {
        "competitors": results,
        "summary": {
            "total_your_reviews":       len(your_reviews),
            "total_competitor_reviews": total_comp_reviews,
            "risk_count":               total_risks,
            "advantage_count":          total_advantages,
            "shared_issue_count":       total_shared,
        },
    }
