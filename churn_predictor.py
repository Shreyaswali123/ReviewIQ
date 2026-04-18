"""
churn_predictor.py — Predictive Churn Score for ReviewIQ

Estimates the % of reviewers likely to churn / switch brands per product,
using a multi-signal model:

  Signal 1 — Sentiment Trajectory   (40 % weight)
      Negative-sentiment trend across time windows (early vs. late batch).
      A product whose neg% is rising fast scores higher.

  Signal 2 — Repeat-Complaint Density  (30 % weight)
      Ratio of reviews that mention the same feature negatively more than once
      across the review history. Repeated complaints = frustration, not noise.

  Signal 3 — Low-Rating Recency   (20 % weight)
      Are recent (latest-quarter) reviews lower-rated than historical average?

  Signal 4 — Sarcasm & Anger Density  (10 % weight)
      Fraction of reviews flagged as sarcastic or containing anger keywords.
      These reviewers are often the loudest churners.

Output (per product)
--------------------
    {
        "product_name":    str,
        "churn_score":     float    0.0 – 100.0  (% likelihood to churn)
        "risk_tier":       str      "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
        "drivers":         list[str]  top 2-3 factors driving the score
        "trajectory":      str      "WORSENING" | "STABLE" | "IMPROVING"
        "neg_pct_early":   float
        "neg_pct_late":    float
        "review_count":    int
    }
"""

import re
from collections import defaultdict

# ── Weights ────────────────────────────────────────────────────────────────────
W_TRAJECTORY   = 0.40
W_REPEAT       = 0.30
W_RECENCY      = 0.20
W_SARCASM      = 0.10

# ── Tier thresholds ────────────────────────────────────────────────────────────
TIER_CRITICAL = 70
TIER_HIGH     = 45
TIER_MEDIUM   = 25

# ── Anger / frustration keywords ──────────────────────────────────────────────
_ANGER_WORDS = frozenset({
    "terrible", "awful", "horrible", "useless", "garbage", "trash", "scam",
    "fraud", "disgusting", "pathetic", "unacceptable", "furious", "enraged",
    "never again", "worst ever", "complete waste", "absolute rubbish",
    "do not buy", "avoid", "refund", "rip off", "ripped off",
})

_FEATURE_KEYWORDS_SIMPLE = {
    "Battery":       ["battery", "charge", "charging", "drain"],
    "Build Quality": ["build", "quality", "flimsy", "cheap", "broken"],
    "Delivery":      ["delivery", "shipping", "late", "delayed", "lost"],
    "Packaging":     ["packaging", "package", "box", "crushed", "damaged"],
    "Performance":   ["performance", "speed", "slow", "lag"],
    "Customer Support": ["support", "service", "refund", "return", "ignored"],
    "Comfort":       ["comfort", "fit", "tight", "hard"],
    "Taste":         ["taste", "flavor", "stale", "bland", "expire"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_time_windows(reviews: list[dict]) -> tuple[list[dict], list[dict]]:
    """Splits reviews into [earlier 50%] and [later 50%] by date."""
    dated = sorted(
        reviews,
        key=lambda r: str(r.get("review_date", r.get("date", "")) or "")
    )
    mid = max(1, len(dated) // 2)
    return dated[:mid], dated[mid:]


def _neg_pct(reviews: list[dict]) -> float:
    """Fraction of reviews with overall negative sentiment keyword balance."""
    if not reviews:
        return 0.0
    neg_count = 0
    for r in reviews:
        text  = str(r.get("review_text", r.get("text", ""))).lower()
        words = set(re.findall(r"\b\w+\b", text))
        pos   = len(words & frozenset(["great", "good", "excellent", "love",
                                       "best", "amazing", "happy", "satisfied",
                                       "perfect", "recommend"]))
        neg   = len(words & frozenset(["terrible", "awful", "bad", "broken",
                                       "poor", "disappointed", "useless",
                                       "waste", "damaged", "defective", "slow"]))
        if neg > pos:
            neg_count += 1
    return neg_count / len(reviews) * 100


def _trajectory_signal(early: list[dict], late: list[dict]) -> tuple[float, str]:
    """
    Returns (normalised_score 0-100, trajectory label).
    Score = delta in neg% mapped to [0,100].
    """
    ep = _neg_pct(early)
    lp = _neg_pct(late)
    delta = lp - ep   # positive = getting worse

    # Map delta [-50, +50] → [0, 100]
    score = min(100.0, max(0.0, (delta + 50) * 1.0))

    if delta > 8:
        label = "WORSENING"
    elif delta < -8:
        label = "IMPROVING"
    else:
        label = "STABLE"

    return score, label, ep, lp


def _repeat_complaint_signal(reviews: list[dict]) -> float:
    """
    Fraction of reviewers who mention the same feature negatively in a way
    that suggests repeated frustration. Returns 0-100.
    """
    feature_counts: dict[str, int] = defaultdict(int)
    for r in reviews:
        text = str(r.get("review_text", r.get("text", ""))).lower()
        star = r.get("star_rating")
        try:
            star = int(float(str(star)))
        except (TypeError, ValueError):
            star = None
        if star and star >= 4:
            continue  # positive review — skip
        for feature, keywords in _FEATURE_KEYWORDS_SIMPLE.items():
            if any(kw in text for kw in keywords):
                feature_counts[feature] += 1

    if not reviews:
        return 0.0

    # Features complained about by ≥15% of reviewers = systemic frustration
    threshold = max(2, len(reviews) * 0.15)
    repeated  = sum(1 for c in feature_counts.values() if c >= threshold)
    score     = min(100.0, repeated / max(1, len(_FEATURE_KEYWORDS_SIMPLE)) * 200)
    return score


def _recency_rating_signal(reviews: list[dict]) -> float:
    """
    Returns 0-100 based on how much lower recent ratings are vs historical.
    """
    dated = sorted(
        reviews,
        key=lambda r: str(r.get("review_date", r.get("date", "")) or "")
    )
    mid   = max(1, len(dated) // 2)
    early = dated[:mid]
    late  = dated[mid:]

    def _avg_rating(batch: list[dict]) -> float | None:
        ratings = []
        for r in batch:
            try:
                rating = int(float(str(r.get("star_rating", ""))))
                if 1 <= rating <= 5:
                    ratings.append(rating)
            except (TypeError, ValueError):
                pass
        return sum(ratings) / len(ratings) if ratings else None

    er = _avg_rating(early)
    lr = _avg_rating(late)
    if er is None or lr is None:
        return 50.0  # neutral when no ratings available

    delta = er - lr  # positive = recent ratings are lower
    # Map delta [0, 4] → [0, 100]
    return min(100.0, max(0.0, delta / 4.0 * 100.0))


def _sarcasm_anger_signal(reviews: list[dict]) -> float:
    """
    Returns 0-100 = fraction of reviews with sarcasm/anger * 100, capped at 100.
    """
    if not reviews:
        return 0.0
    flagged = 0
    for r in reviews:
        text  = str(r.get("review_text", r.get("text", ""))).lower()
        words = set(re.findall(r"\b\w+\b", text))
        # Anger keywords
        if words & _ANGER_WORDS:
            flagged += 1
            continue
        # Sarcasm markers (lightweight; full check is in sarcasm.py)
        if re.search(r"(?i)\b(oh great|yeah right|wow.*love.*terrible|"
                     r"amazing.*broke|perfect.*blister)\b", text):
            flagged += 1
    return min(100.0, flagged / len(reviews) * 100 * 3)  # amplified; cap at 100


def _build_drivers(
    traj_score: float,
    repeat_score: float,
    recency_score: float,
    sarcasm_score: float,
    trajectory_label: str,
) -> list[str]:
    """Returns human-readable driver strings sorted by contribution."""
    signals = [
        (traj_score * W_TRAJECTORY,   f"Sentiment trajectory is {trajectory_label}"),
        (repeat_score * W_REPEAT,     "Repeated feature complaints detected"),
        (recency_score * W_RECENCY,   "Recent ratings are declining"),
        (sarcasm_score * W_SARCASM,   "High frustration/anger in recent reviews"),
    ]
    signals.sort(key=lambda x: -x[0])
    return [s[1] for s in signals if s[0] > 5][:3] or ["No strong churn signals"]


# ── Public API ─────────────────────────────────────────────────────────────────

def score_product(reviews: list[dict], product_name: str) -> dict:
    """
    Computes the churn risk score for a single product.

    Parameters
    ----------
    reviews      : list of ingested review dicts for this product only
    product_name : display name

    Returns
    -------
    dict — see module docstring for field definitions
    """
    if not reviews:
        return {
            "product_name": product_name,
            "churn_score":  0.0,
            "risk_tier":    "LOW",
            "drivers":      ["Insufficient data"],
            "trajectory":   "STABLE",
            "neg_pct_early": 0.0,
            "neg_pct_late":  0.0,
            "review_count":  0,
        }

    early, late = _split_time_windows(reviews)

    traj_score, traj_label, ep, lp = _trajectory_signal(early, late)
    repeat_score   = _repeat_complaint_signal(reviews)
    recency_score  = _recency_rating_signal(reviews)
    sarcasm_score  = _sarcasm_anger_signal(reviews)

    churn_score = (
        traj_score   * W_TRAJECTORY  +
        repeat_score * W_REPEAT      +
        recency_score* W_RECENCY     +
        sarcasm_score* W_SARCASM
    )
    churn_score = round(min(100.0, max(0.0, churn_score)), 1)

    if churn_score >= TIER_CRITICAL:
        tier = "CRITICAL"
    elif churn_score >= TIER_HIGH:
        tier = "HIGH"
    elif churn_score >= TIER_MEDIUM:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    drivers = _build_drivers(traj_score, repeat_score, recency_score,
                             sarcasm_score, traj_label)

    return {
        "product_name":  product_name,
        "churn_score":   churn_score,
        "risk_tier":     tier,
        "drivers":       drivers,
        "trajectory":    traj_label,
        "neg_pct_early": round(ep, 1),
        "neg_pct_late":  round(lp, 1),
        "review_count":  len(reviews),
    }


def score_all_products(reviews: list[dict]) -> list[dict]:
    """
    Splits reviews by product_name and scores each product.

    Parameters
    ----------
    reviews : combined list of ingested review dicts (all products mixed)

    Returns
    -------
    list[dict] sorted by churn_score descending
    """
    by_product: dict[str, list[dict]] = defaultdict(list)
    for r in reviews:
        name = str(r.get("product_name", "Unknown Product"))
        by_product[name].append(r)

    scores = [score_product(revs, name) for name, revs in by_product.items()]
    scores.sort(key=lambda x: -x["churn_score"])
    return scores
