import os
import re
import json
from collections import defaultdict
from google import genai
from google.genai import types

from sentiment import analyze_batch
from trend_engine import detect_systemic_trends
from churn_predictor import score_all_products
from competitor import analyze_competitor
from sarcasm import get_sarcasm_confidence
from root_cause import drill_down_multiturn

# ── Feature keywords used by the trend tool ────────────────────────────────────
_TREND_FEATURE_KWS = {
    "Battery":          ["battery", "charge", "charging", "drain", "dead", "dies"],
    "Packaging":        ["packaging", "package", "box", "crushed", "damaged", "torn"],
    "Delivery":         ["delivery", "shipping", "courier", "delayed", "late", "lost"],
    "Build Quality":    ["build", "quality", "flimsy", "cheap", "sturdy"],
    "Comfort":          ["comfort", "comfortable", "fit", "wear", "tight"],
    "Taste":            ["taste", "flavor", "stale", "fresh", "bland"],
    "Performance":      ["performance", "speed", "slow", "lag"],
    "Customer Support": ["support", "service", "refund", "return", "ignored"],
    "Display":          ["display", "screen", "brightness", "dim"],
}

_NEG_WORDS = frozenset({
    "terrible", "awful", "bad", "broken", "poor", "disappointed", "useless",
    "waste", "defective", "cheap", "flimsy", "slow", "late", "damaged",
    "crushed", "dead", "ruined", "unusable", "scam", "fraud",
})


def _count_negative_features(batch: list[dict]) -> dict[str, int]:
    """Count negative feature mentions per 'Product|Feature' key in a review batch."""
    counts: dict[str, int] = defaultdict(int)
    for r in batch:
        text  = str(r.get("review_text", "")).lower()
        prod  = str(r.get("product_name", "Unknown"))
        words = set(re.findall(r"\b\w+\b", text))
        if not (words & _NEG_WORDS):
            continue
        for feature, kws in _TREND_FEATURE_KWS.items():
            if any(kw in text for kw in kws):
                counts[f"{prod}|{feature}"] += 1
    return dict(counts)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(
    reviews: list[dict],
    competitor_map: dict | None = None,
    task: str | None = None,
    model_id: str = "gemini-3-flash-preview",
) -> dict:
    """
    Run the ReviewIQ autonomous agent.

    Parameters
    ----------
    reviews        : ingested review dicts (from ingest.py)
    competitor_map : { "CompetitorName": [review, ...] }  — optional
    task           : custom instruction; defaults to a full executive summary
    model_id       : Gemini model string
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ["GCP_PROJECT"],
        location=os.getenv("GCP_LOCATION", "us-central1"),
    )

    # ── Tool definitions ───────────────────────────────────────────────────────
    # Each function closes over `reviews` / `competitor_map`.
    # Docstrings are the model's only description of the tool.

    def sentiment_overview() -> str:
        """
        High-level sentiment breakdown (positive / negative / neutral counts)
        across all loaded reviews. Call this first in every analysis.
        """
        results = analyze_batch(reviews)
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r.get("overall_sentiment", "neutral")] += 1
        return json.dumps(counts)

    def feature_breakdown(product_filter: str = None) -> str:
        """
        Per-feature sentiment breakdown (negative / positive %).
        Pass product_filter (exact product_name) to zoom in on one product.
        Returns the top 10 feature results.
        """
        target  = [r for r in reviews if not product_filter or r.get("product_name") == product_filter]
        results = analyze_batch(target)
        return json.dumps(results[:10], default=str)

    def detect_trends() -> str:
        """
        Z-score based spike detection comparing early reviews (first 50%) vs
        recent reviews (last 50%). Returns Systemic and Emerging alerts per
        product-feature pair, sorted by severity.
        """
        dated       = sorted(reviews, key=lambda r: str(r.get("review_date", "") or ""))
        mid         = max(1, len(dated) // 2)
        early_batch = dated[:mid]
        late_batch  = dated[mid:]

        early_counts = _count_negative_features(early_batch)
        late_counts  = _count_negative_features(late_batch)
        historical   = {k: [v] for k, v in early_counts.items()}

        alerts = detect_systemic_trends(late_counts, historical)
        return json.dumps(alerts, default=str)

    def churn_prediction() -> str:
        """
        Churn risk scores per product (CRITICAL / HIGH / MEDIUM / LOW),
        driven by sentiment trajectory, repeat complaints, rating recency,
        and sarcasm/anger density. Sorted by risk descending.
        """
        return json.dumps(score_all_products(reviews), default=str)

    def sarcasm_scan() -> str:
        """
        Identify reviews with high sarcasm confidence (score ≥ 0.75).
        Useful for flagging misleading 5-star reviews that mask dissatisfaction.
        Returns flagged count and up to 20 review previews.
        """
        flagged = []
        for r in reviews:
            text  = str(r.get("review_text", ""))
            score = get_sarcasm_confidence(text)
            if score >= 0.75:
                flagged.append({
                    "review_id":    r.get("review_id", ""),
                    "product_name": r.get("product_name", ""),
                    "text_preview": text[:120],
                    "sarcasm_score": score,
                })
        return json.dumps({"flagged_count": len(flagged), "reviews": flagged[:20]}, default=str)

    def competitor_analysis() -> str:
        """
        Feature gap analysis, competitor weakness radar, and shared platform
        issue detection against all loaded competitor data. Only callable when
        competitor data has been provided. Returns condensed summary.
        """
        if not competitor_map:
            return json.dumps({"error": "No competitor data loaded. Pass competitor_map to run_agent()."})

        result   = analyze_competitor(reviews, competitor_map)
        summary  = result.get("summary", {})
        top_risks, top_advantages = [], []

        for comp in result.get("competitors", []):
            for g in comp.get("feature_gaps", []):
                entry = {"competitor": comp["name"], "feature": g["feature"], "gap_pp": g["gap"]}
                if g["verdict"] == "RISK":
                    top_risks.append(entry)
                elif g["verdict"] == "ADVANTAGE":
                    top_advantages.append(entry)

        return json.dumps({
            "summary":       summary,
            "top_risks":     top_risks[:5],
            "top_advantages": top_advantages[:5],
            "competitor_weaknesses": [
                {
                    "competitor":  c["name"],
                    "weaknesses": [w["feature"] for w in c.get("competitor_weaknesses", [])[:3]],
                }
                for c in result.get("competitors", [])
            ],
        }, default=str)

    def root_cause_drill_down(feature_alert_json: str) -> str:
        """
        Run a multi-turn AI root cause analysis on a single systemic alert.
        Call this after detect_trends() returns Systemic or high-severity alerts.
        Pass the full alert dict as a JSON string — copy it directly from
        detect_trends() output.

        Returns a structured RCA report with:
          - Root Cause Hypothesis
          - Supporting Evidence (real review samples)
          - Blast Radius estimate
          - Fix Recommendations with owning teams

        Example input: '{"feature": "Battery (GT Pro 5)", "current_count": 38,
                         "mean": 3.0, "z_score": 8.2, "status": "Systemic"}'
        """
        try:
            alert = json.loads(feature_alert_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON for feature_alert: {e}"})

        result = drill_down_multiturn(feature_alert=alert, reviews=reviews)
        return json.dumps({
            "feature":        result.get("feature"),
            "status":         "fallback" if result.get("fallback") else "ok",
            "rca_report":     result.get("rca_markdown", ""),
            "tokens_used":    result.get("tokens_used", 0),
            "review_samples": result.get("review_samples", [])[:5],
        }, default=str)

    # ── Assemble tool list ─────────────────────────────────────────────────────
    tools_list = [
        sentiment_overview,
        feature_breakdown,
        detect_trends,
        churn_prediction,
        sarcasm_scan,
        competitor_analysis,
        root_cause_drill_down,
    ]

    # ── System prompt ──────────────────────────────────────────────────────────
    system_prompt = (
        "You are ReviewIQ's autonomous product intelligence analyst. "
        "Always begin with sentiment_overview to establish baseline health. "
        "Then investigate signals with feature_breakdown, detect_trends, and churn_prediction. "
        "For every Systemic alert returned by detect_trends, call root_cause_drill_down "
        "with that alert's JSON to produce an actionable root cause report — limit to the top 3 alerts. "
        "Use sarcasm_scan when sentiment seems inflated by fake positives. "
        "Call competitor_analysis only when competitor data is available. "
        "Write a structured executive report with: Overall Health, Key Issues, "
        "Root Cause Findings, Churn Risks, and Recommended Actions."
    )

    user_task = task or (
        f"Analyse these {len(reviews)} reviews across "
        f"{len({r.get('product_name') for r in reviews})} product(s) "
        "and produce a full executive intelligence report."
    )

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=user_task,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools_list,
                temperature=0.1,
            ),
        )
        return {
            "report":      response.text,
            "model":       model_id,
            "tokens_used": response.usage_metadata.total_token_count,
            "fallback":    False,
        }

    except Exception as e:
        return {"report": f"## Agent Error\n`{e}`", "fallback": True}


def ask_agent(
    question: str,
    reviews: list[dict],
    competitor_map: dict | None = None,
) -> dict:
    """Convenience wrapper — ask the agent a single natural-language question."""
    return run_agent(reviews, competitor_map, task=f"Answer this question with evidence from the reviews: {question}")