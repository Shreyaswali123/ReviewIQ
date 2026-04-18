"""
trend_engine.py — Z-score based systemic trend detection for ReviewIQ

Fixes vs original:
  • history < 2 used to silently skip valid single-point spikes → now uses
    a configurable minimum history length with a fallback spike rule.
  • NaN / inf z-scores are guarded against.
  • The alert dict now uses "mean" (not "historical_mean") so the frontend
    can reference alert.mean consistently.
  • The product|feature key split is guarded against missing separators.
"""

import math

# Minimum number of historical data points before a proper z-score is computed.
# Reviews with fewer points fall through to the absolute-spike fallback.
MIN_HISTORY_FOR_ZSCORE = 2

# If we don't have enough history, flag an item as "Emerging" when its
# absolute count exceeds this threshold.
EMERGING_COUNT_THRESHOLD = 5

# Z-score above which an alert is emitted.
ZSCORE_ALERT_THRESHOLD = 3.0


def _safe_zscore(count: int, history: list) -> float | None:
    """
    Returns z-score or None if it cannot be computed safely.
    Handles std_dev == 0 and NaN / inf.
    """
    if len(history) < MIN_HISTORY_FOR_ZSCORE:
        return None

    mean    = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    std_dev  = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0:
        # All historical values identical: any deviation is noteworthy,
        # but we can't produce a meaningful z-score. Use a sentinel.
        return float("inf") if count > mean else 0.0

    z = (count - mean) / std_dev

    # Guard against degenerate float values
    if not math.isfinite(z):
        return None

    return z


def _parse_trend_key(combined_key: str) -> tuple[str, str]:
    """
    Splits a 'Product|Feature' key. If there is no pipe separator,
    treats the whole string as the feature with product='Unknown'.
    """
    if "|" in combined_key:
        parts   = combined_key.split("|", maxsplit=1)
        product = parts[0].strip() or "Unknown"
        feature = parts[1].strip() or combined_key
    else:
        product = "Unknown"
        feature = combined_key.strip()
    return product, feature


def detect_systemic_trends(
    latest_batch_counts: dict[str, int],
    historical_counts:   dict[str, list],
) -> list[dict]:
    """
    Compares the latest batch's negative-feature counts against historical
    baselines using a Z-score model.

    Returns a list of alert dicts, each containing:
        feature         — human-readable "Feature (Product)" label
        current_count   — count in the latest batch
        mean            — historical mean
        z_score         — z-score (rounded to 2 dp); "N/A" for emerging items
        status          — "Systemic" | "Emerging"
    """
    alerts: list[dict] = []

    for combined_key, count in latest_batch_counts.items():
        if count == 0:
            continue

        history = historical_counts.get(combined_key, [])
        product, feature = _parse_trend_key(combined_key)
        display_feature  = f"{feature} ({product})"

        z = _safe_zscore(count, history)

        if z is None:
            # Not enough history yet — fall back to absolute count check
            if count >= EMERGING_COUNT_THRESHOLD:
                mean = sum(history) / len(history) if history else 0
                alerts.append({
                    "feature":       display_feature,
                    "current_count": count,
                    "mean":          round(mean, 2),
                    "z_score":       "N/A",
                    "status":        "Emerging",
                })
            continue

        if z == float("inf"):
            # Std-dev == 0: any non-zero count above the flat baseline is flagged
            history_mean = sum(history) / len(history) if history else 0
            if count > history_mean:
                alerts.append({
                    "feature":       display_feature,
                    "current_count": count,
                    "mean":          round(history_mean, 2),
                    "z_score":       "∞",
                    "status":        "Systemic",
                })
            continue

        if z >= ZSCORE_ALERT_THRESHOLD:
            history_mean = sum(history) / len(history)
            alerts.append({
                "feature":       display_feature,
                "current_count": count,
                "mean":          round(history_mean, 2),
                "z_score":       round(z, 2),
                "status":        "Systemic",
            })

    # Sort by z-score severity (Systemic first, then Emerging; within each group, highest count first)
    def sort_key(a):
        z = a["z_score"]
        if isinstance(z, (int, float)):
            return (0, -z)
        if z == "∞":
            return (0, float("-inf"))
        return (1, -a["current_count"])

    return sorted(alerts, key=sort_key)