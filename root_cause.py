"""
root_cause.py — Multi-turn agentic Root Cause Analysis for ReviewIQ
Uses Google Vertex AI (gemini-2.0-flash) with function calling.

Setup:
    pip install google-cloud-aiplatform
    gcloud auth application-default login          # local dev
    # OR set GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

    Set env vars:
        export GCP_PROJECT="your-gcp-project-id"
        export GCP_LOCATION="us-central1"

Public API and return shape are unchanged.
"""

import os
import re
import json
from collections import Counter


_FEATURE_KEYWORD_MAP_V2 = {
    "Battery":          ["battery", "charge", "charging", "drain", "dead", "dies"],
    "Packaging":        ["packaging", "package", "box", "crushed", "damaged", "torn"],
    "Delivery":         ["delivery", "shipping", "courier", "delayed", "late", "lost"],
    "Build Quality":    ["build", "quality", "flimsy", "cheap", "sturdy", "construction"],
    "Comfort":          ["comfort", "comfortable", "fit", "wear", "soft", "tight"],
    "Taste":            ["taste", "flavor", "stale", "fresh", "bland", "chalky"],
    "Display":          ["display", "screen", "brightness", "resolution", "dim"],
    "Price / Value":    ["price", "value", "worth", "expensive", "overpriced"],
    "Customer Support": ["support", "service", "refund", "return", "response", "ignored"],
    "Performance":      ["performance", "speed", "fast", "slow", "lag", "responsive"],
    "Connectivity":     ["wifi", "bluetooth", "connection", "signal", "disconnect"],
    "Size / Fit":       ["size", "sizing", "small", "large", "wide", "narrow"],
}

_RCA_SYSTEM_PROMPT = """You are a product quality engineer performing a Root Cause Analysis.

You receive a feature spike alert and initial review samples. If the samples don't 
clearly reveal the root cause, you can request MORE reviews using the fetch_more_reviews 
function — but use it sparingly (max 2 additional fetches).

Once you have enough evidence, write your RCA in these EXACT sections:

## Alert Summary
One line: what spiked, how severe, product impacted.

## Root Cause Hypothesis
2-3 sentences. Specific cause: firmware update, supplier change, logistics failure, etc.

## Supporting Evidence  
Bullet the 3 most telling review patterns (paraphrase, don't quote verbatim).

## Blast Radius
How widespread? Isolated, batch defect, or systemic? Estimated % of buyers affected.

## Fix Recommendations
3 numbered, specific actions. Each names the responsible team.

Keep under 350 words. Be direct."""


# ── Vertex AI client + tool setup ──────────────────────────────────────────────

def _get_model_and_tool():
    """Initialise Vertex AI and return (GenerativeModel, vertexai module)."""
    try:
        import vertexai
        from vertexai.generative_models import (
            GenerativeModel,
            FunctionDeclaration,
            Tool,
        )

        project  = os.environ["GCP_PROJECT"]
        location = os.getenv("GCP_LOCATION", "us-central1")
        vertexai.init(project=project, location=location)

        fetch_fn = FunctionDeclaration(
            name="fetch_more_reviews",
            description=(
                "Fetch additional customer review samples matching a specific keyword or theme. "
                "Use this when the initial review sample doesn't clearly reveal the root cause."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword or phrase to search for in review text.",
                    },
                    "exclude_seen": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of review IDs already seen — exclude these.",
                    },
                    "max_samples": {
                        "type": "integer",
                        "description": "How many more reviews to fetch (max 10).",
                    },
                },
                "required": ["keyword"],
            },
        )

        tool  = Tool(function_declarations=[fetch_fn])
        model = GenerativeModel(
            model_name="gemini-3-flash-preview",
            system_instruction=_RCA_SYSTEM_PROMPT,
            tools=[tool],
        )
        return model, vertexai

    except ImportError:
        raise RuntimeError(
            "google-cloud-aiplatform not installed. "
            "Run: pip install google-cloud-aiplatform"
        )
    except KeyError:
        raise RuntimeError(
            "GCP_PROJECT environment variable is not set."
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def drill_down_multiturn(
    feature_alert: dict,
    reviews: list[dict],
    context: dict | None = None,
    model: str = "gemini-3-flash-preview",   # kept for API compatibility
    max_tokens: int = 700,
) -> dict:
    """
    Multi-turn agentic root cause drill-down using Vertex AI function calling.
    Same return signature as the original.
    """
    from vertexai.generative_models import GenerationConfig, Part

    raw_feature  = feature_alert.get("feature", "Unknown Feature")
    feature_name = raw_feature.split("(")[0].strip() if "(" in raw_feature else raw_feature
    keywords     = _FEATURE_KEYWORD_MAP_V2.get(feature_name, [feature_name.lower()])

    # ── Helper to fetch review samples ────────────────────────────────────────
    def _fetch_samples(keyword: str, exclude_ids: list, max_n: int) -> list[dict]:
        kw      = keyword.lower()
        results = []
        seen    = set(exclude_ids)
        for r in reviews:
            rid  = str(r.get("review_id", ""))
            text = str(r.get("review_text", "")).strip()
            if rid in seen or kw not in text.lower():
                continue
            star = r.get("star_rating")
            try:
                star = int(float(str(star)))
            except (TypeError, ValueError):
                star = 3
            results.append((5 - star, rid, text[:300], r.get("product_name"), r.get("review_date")))
        results.sort(key=lambda x: -x[0])
        return [
            {"review_id": r[1], "text": r[2], "product": r[3], "date": r[4], "star": 5 - r[0]}
            for r in results[:max_n]
        ]

    initial_samples = _fetch_samples(keywords[0] if keywords else feature_name, [], 10)
    seen_ids        = [s["review_id"] for s in initial_samples]
    all_samples     = list(initial_samples)

    ctx_str = ""
    if context:
        ctx_str = f"\n\nExtra context:\n```json\n{json.dumps(context, indent=2)}\n```"

    initial_msg = (
        f"Feature Spike Alert:\n```json\n{json.dumps(feature_alert, indent=2)}\n```\n\n"
        f"Feature under investigation: **{feature_name}**\n\n"
        f"Initial review sample ({len(initial_samples)} reviews):\n"
        + "\n".join(f"  [{s['star']}★ {s['date']}] {s['text']}" for s in initial_samples)
        + ctx_str
        + "\n\nPlease perform your Root Cause Analysis."
    )

    total_tokens      = 0
    generation_config = GenerationConfig(max_output_tokens=max_tokens)

    try:
        vertex_model, _ = _get_model_and_tool()
        chat            = vertex_model.start_chat(response_validation=False)

        for turn in range(4):   # max 4 turns: initial + up to 3 tool calls
            message = initial_msg if turn == 0 else fn_response_parts

            response      = chat.send_message(message, generation_config=generation_config)
            total_tokens += response.usage_metadata.total_token_count

            # ── Check for function calls in response parts ─────────────────
            fn_calls = [
                part for part in response.candidates[0].content.parts
                if part.function_call and part.function_call.name
            ]

            if not fn_calls:
                # Model is done — return final text
                return {
                    "feature":        feature_name,
                    "alert":          feature_alert,
                    "rca_markdown":   response.text,
                    "review_samples": [s["text"] for s in all_samples],
                    "model":          "gemini-3-flash-preview",
                    "tokens_used":    total_tokens,
                    "fallback":       False,
                    "multiturn":      True,
                }

            # ── Execute function calls and build response parts ────────────
            fn_response_parts = []
            for part in fn_calls:
                fn   = part.function_call
                args = dict(fn.args)

                kw      = args.get("keyword", feature_name)
                exclude = list(args.get("exclude_seen", seen_ids))
                max_n   = min(int(args.get("max_samples", 8)), 10)

                more = _fetch_samples(kw, exclude, max_n)
                seen_ids.extend(s["review_id"] for s in more)
                all_samples.extend(more)

                result_payload = {
                    "additional_reviews": [
                        {"text": s["text"], "star": s["star"], "date": s["date"]}
                        for s in more
                    ],
                    "found": len(more),
                }

                # Vertex AI function response Part
                fn_response_parts.append(
                    Part.from_function_response(
                        name=fn.name,
                        response={"result": json.dumps(result_payload)},
                    )
                )

        # Loop exhausted — return whatever text is available
        rca_text = response.text if hasattr(response, "text") else "RCA loop completed without final output."
        return {
            "feature":        feature_name,
            "alert":          feature_alert,
            "rca_markdown":   rca_text,
            "review_samples": [s["text"] for s in all_samples],
            "model":          "gemini-3-flash-preview",
            "tokens_used":    total_tokens,
            "fallback":       False,
            "multiturn":      True,
        }

    except Exception as e:
        print(f"⚠️  Multi-turn RCA failed ({e})")
        return {
            "feature":        feature_name,
            "alert":          feature_alert,
            "rca_markdown":   f"## Error\n\nRCA failed: `{e}`",
            "review_samples": [s["text"] for s in all_samples],
            "model":          "gemini-3-flash-preview",
            "tokens_used":    total_tokens,
            "fallback":       True,
            "multiturn":      True,
        }
