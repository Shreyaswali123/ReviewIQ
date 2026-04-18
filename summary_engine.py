"""
summary_engine.py — LLM-generated executive summary for ReviewIQ
Uses Google Vertex AI (gemini-2.0-flash) via the vertexai SDK.

Setup:
    pip install google-cloud-aiplatform
    gcloud auth application-default login          # local dev
    # OR set GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

    Set env vars:
        export GCP_PROJECT="your-gcp-project-id"
        export GCP_LOCATION="us-central1"          # or your preferred region

Everything else (public API, return shape) is unchanged.
"""

import os
import json
from datetime import datetime


# ── Vertex AI client ───────────────────────────────────────────────────────────

def _get_model(system_prompt: str):
    """Initialise Vertex AI and return a GenerativeModel instance."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project  = os.environ["GCP_PROJECT"]   # required
        location = os.getenv("GCP_LOCATION", "us-central1")

        vertexai.init(project=project, location=location)

        return GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
        )
    except ImportError:
        raise RuntimeError(
            "google-cloud-aiplatform not installed. "
            "Run: pip install google-cloud-aiplatform"
        )
    except KeyError:
        raise RuntimeError(
            "GCP_PROJECT environment variable is not set. "
            "Export it before running: export GCP_PROJECT=your-project-id"
        )


# ── Data condensers (unchanged) ───────────────────────────────────────────────

def _sentiment_snapshot(sentiment_results: list[dict]) -> dict:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in sentiment_results:
        s = r.get("overall_sentiment", "neutral")
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values()) or 1
    return {k: round(v / total * 100) for k, v in counts.items()}


def _top_features(sentiment_results: list[dict], n: int = 5) -> dict[str, dict]:
    neg: dict[str, int] = {}
    pos: dict[str, int] = {}
    for r in sentiment_results:
        for f in r.get("features", []):
            name = f.get("feature_name", "")
            if f.get("sentiment") == "negative":
                neg[name] = neg.get(name, 0) + 1
            elif f.get("sentiment") == "positive":
                pos[name] = pos.get(name, 0) + 1
    top_neg = sorted(neg.items(), key=lambda x: -x[1])[:n]
    top_pos = sorted(pos.items(), key=lambda x: -x[1])[:n]
    return {
        "top_complaints": [{"feature": k, "count": v} for k, v in top_neg],
        "top_praises":    [{"feature": k, "count": v} for k, v in top_pos],
    }


def _condense_pipeline(pipeline_output: dict) -> dict:
    sr   = pipeline_output.get("sentiment_results", [])
    snap = _sentiment_snapshot(sr)
    feat = _top_features(sr)

    trends = pipeline_output.get("trends", [])[:8]
    cp     = pipeline_output.get("cross_product", [])[:5]
    churn  = sorted(
        pipeline_output.get("churn", []),
        key=lambda x: -x.get("churn_score", 0)
    )[:5]

    comp_summary = {}
    comp_raw = pipeline_output.get("competitor", {})
    if comp_raw:
        comp_summary = {
            "risk_count":      comp_raw.get("summary", {}).get("risk_count", 0),
            "advantage_count": comp_raw.get("summary", {}).get("advantage_count", 0),
            "shared_issues":   comp_raw.get("summary", {}).get("shared_issue_count", 0),
            "competitors": [
                {
                    "name":       c["name"],
                    "top_risks":  [g["feature"] for g in c.get("feature_gaps", [])
                                   if g["verdict"] == "RISK"][:3],
                    "advantages": [g["feature"] for g in c.get("your_advantages", [])][:3],
                }
                for c in comp_raw.get("competitors", [])[:3]
            ],
        }

    alerts = pipeline_output.get("alerts", [])
    dept_counts: dict[str, int] = {}
    for a in alerts:
        dept = a.get("department", "Unknown")
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    return {
        "total_reviews":     len(pipeline_output.get("reviews", [])),
        "sarcasm_flagged":   pipeline_output.get("sarcasm_count", 0),
        "bot_flagged":       pipeline_output.get("bot_count", 0),
        "sentiment":         snap,
        "features":          feat,
        "systemic_trends":   [
            {"feature": t["feature"], "status": t["status"],
             "z_score": t.get("z_score", "N/A")}
            for t in trends
        ],
        "platform_issues":   [{"feature": p["feature"]} for p in cp],
        "churn_risks":       churn,
        "competitor":        comp_summary,
        "alert_departments": dept_counts,
        "report_date":       datetime.now().strftime("%d %b %Y"),
    }


# ── Prompt (unchanged) ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior product intelligence analyst at an e-commerce company.
You receive a JSON snapshot of a multi-signal review analytics pipeline and produce
a concise, actionable executive brief. Your writing is direct, data-driven, and
uses plain business English (no fluff, no buzzwords).

Structure your response in these exact sections — keep each section tight:

## Executive Summary
2-3 sentences. Overall health, most urgent signal.

## Sentiment Overview
Quick breakdown of positive/negative/neutral %. Call out the biggest mover.

## Top Feature Issues
Bullet the top 3 complaint areas with brief implication.

## Systemic Trends & Platform Issues
Which trends are Systemic vs Emerging. Platform-level issues that span categories.

## Competitor Landscape
Risks (where competitor is beating us) and advantages (where we lead). 1-2 sentences.

## Churn Risk
Products at highest churn risk. What's driving it.

## Recommended Actions
Top 3 numbered action items, each owned by a specific team.

Keep the entire response under 400 words."""


def _build_user_message(condensed: dict) -> str:
    return (
        "Here is the ReviewIQ pipeline snapshot for your analysis:\n\n"
        f"```json\n{json.dumps(condensed, indent=2)}\n```\n\n"
        "Generate the executive brief now."
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_executive_summary(
    pipeline_output: dict,
    model: str = "gemini-2.0-flash",   # ignored — kept for API compatibility
    max_tokens: int = 700,
) -> dict:
    from vertexai.generative_models import GenerationConfig

    condensed = _condense_pipeline(pipeline_output)

    try:
        vertex_model = _get_model(_SYSTEM_PROMPT)

        generation_config = GenerationConfig(max_output_tokens=max_tokens)

        response = vertex_model.generate_content(
            _build_user_message(condensed),
            generation_config=generation_config,
        )

        text        = response.text
        # Vertex AI usage metadata
        tokens_used = response.usage_metadata.total_token_count

        return {
            "summary_markdown": text,
            "model":            "gemini-2.0-flash",
            "tokens_used":      tokens_used,
            "fallback":         False,
        }

    except Exception as e:
        print(f"⚠️  Vertex AI summary failed ({e}). Using rule-based fallback.")
        return {
            "summary_markdown": _rule_based_fallback(condensed),
            "model":            "rule-based-fallback",
            "tokens_used":      0,
            "fallback":         True,
        }


def _rule_based_fallback(c: dict) -> str:
    s             = c["sentiment"]
    top_complaint = c["features"]["top_complaints"][0]["feature"] if c["features"]["top_complaints"] else "N/A"
    systemic_n    = sum(1 for t in c["systemic_trends"] if t["status"] == "Systemic")
    churn_top     = c["churn_risks"][0]["product_name"] if c["churn_risks"] else "N/A"

    return f"""## Executive Summary
Analysed **{c['total_reviews']}** reviews ({c['report_date']}). Sentiment is
{s['positive']}% positive / {s['negative']}% negative / {s['neutral']}% neutral.
{systemic_n} systemic trend(s) detected.

## Top Feature Issues
- **{top_complaint}** — highest complaint volume.

## Churn Risk
- **{churn_top}** is the highest churn-risk product.

*(LLM summary unavailable — rule-based fallback used)*"""
