"""
alert_router.py — Smart Alert Routing for ReviewIQ

Automatically assigns each alert to the correct department and generates
pre-drafted Slack messages and email bodies ready to send.

Routing logic
-------------
Each alert is scored against a rule table keyed by feature + status.
Ties broken by z-score severity.

Departments
-----------
    Product Engineering    — Battery, Performance, Connectivity, Display
    Supply Chain / QA      — Packaging, Build Quality, Taste, Size/Fit
    Logistics              — Delivery, Packaging (when cross-product)
    Customer Success       — Customer Support, Price/Value (refund spike)
    Marketing              — Competitor advantages, Sarcasm spikes
    Food Safety (escalate) — Taste with "expire/mold/sick" keywords

Public API
----------
    routed = route_alerts(alerts, reviews=None, product_name="Product")
    # returns list[dict] — each alert enriched with routing metadata + messages
"""

import re
import os
from datetime import datetime

# ── Routing table ──────────────────────────────────────────────────────────────
# Each rule: (feature_pattern, status_pattern, department, severity_boost)
# Patterns are case-insensitive substring matches.

_ROUTING_RULES = [
    # Feature-based routing
    ("battery",          "",         "Product Engineering",  0),
    ("performance",      "",         "Product Engineering",  0),
    ("connectivity",     "",         "Product Engineering",  0),
    ("display",          "",         "Product Engineering",  0),
    ("firmware",         "",         "Product Engineering",  5),

    ("packaging",        "",         "Supply Chain / QA",    0),
    ("build quality",    "",         "Supply Chain / QA",    0),
    ("build",            "",         "Supply Chain / QA",    0),
    ("taste",            "",         "Food Safety",          0),
    ("flavor",           "",         "Food Safety",          0),
    ("size",             "",         "Supply Chain / QA",    0),
    ("comfort",          "",         "Supply Chain / QA",    0),

    ("delivery",         "",         "Logistics",            0),
    ("shipping",         "",         "Logistics",            0),

    ("customer support", "",         "Customer Success",     0),
    ("support",          "",         "Customer Success",     0),
    ("refund",           "",         "Customer Success",     5),
    ("price",            "",         "Customer Success",     0),
    ("value",            "",         "Customer Success",     0),

    # Status-based overrides
    ("",                 "systemic", "Product Engineering",  3),   # Systemic anything → Eng
    ("",                 "emerging", "Customer Success",     0),   # Emerging → CS watch

    # Competitor-specific
    ("competitor",       "",         "Marketing",            0),
    ("platform",         "",         "Logistics",            2),
]

# Food safety escalation keywords
_FOOD_SAFETY_KEYWORDS = frozenset([
    "expire", "expired", "mold", "mould", "rotten", "sick", "vomit",
    "poisoning", "food poisoning", "worms", "insects", "contaminated",
])


# ── Slack message templates ────────────────────────────────────────────────────

_SLACK_TEMPLATES: dict[str, str] = {
    "Product Engineering": (
        ":rotating_light: *ReviewIQ Alert — Product Engineering*\n"
        ">*Feature:* {feature}\n"
        ">*Status:* {status}  |  *Z-Score:* {z_score}  |  *Count:* {count}\n"
        ">*Product:* {product}\n"
        "\n"
        "*Action Required:*\n"
        "• Pull latest firmware/hardware logs for `{feature_clean}` failures\n"
        "• Check last 2 production batch QA reports\n"
        "• Assign P{priority} bug ticket — SLA: {sla}\n"
        "\n"
        "_ReviewIQ detected this spike at {timestamp}. "
        "Avg historical rate: {mean} | Current: {count}_"
    ),
    "Supply Chain / QA": (
        ":package: *ReviewIQ Alert — Supply Chain / QA*\n"
        ">*Feature:* {feature}\n"
        ">*Status:* {status}  |  *Z-Score:* {z_score}  |  *Count:* {count}\n"
        ">*Product:* {product}\n"
        "\n"
        "*Action Required:*\n"
        "• Quarantine and audit last inbound batch for `{feature_clean}` defects\n"
        "• File NCR with supplier — attach this alert\n"
        "• Update incoming QC checklist for this feature\n"
        "\n"
        "_Auto-flagged by ReviewIQ at {timestamp}_"
    ),
    "Logistics": (
        ":truck: *ReviewIQ Alert — Logistics*\n"
        ">*Feature:* {feature}\n"
        ">*Status:* {status}  |  *Z-Score:* {z_score}  |  *Count:* {count}\n"
        ">*Product:* {product}\n"
        "\n"
        "*Action Required:*\n"
        "• Review courier SLA compliance for this SKU\n"
        "• Check if complaints correlate with specific pin codes / zones\n"
        "• Escalate to carrier account manager if Systemic\n"
        "\n"
        "_Flagged by ReviewIQ at {timestamp}_"
    ),
    "Customer Success": (
        ":headphones: *ReviewIQ Alert — Customer Success*\n"
        ">*Feature:* {feature}\n"
        ">*Status:* {status}  |  *Z-Score:* {z_score}  |  *Count:* {count}\n"
        ">*Product:* {product}\n"
        "\n"
        "*Action Required:*\n"
        "• Proactively reach out to affected customers (export list attached)\n"
        "• Prepare resolution scripts for `{feature_clean}` complaints\n"
        "• Monitor CSAT scores for the next 7 days\n"
        "\n"
        "_Flagged by ReviewIQ at {timestamp}_"
    ),
    "Marketing": (
        ":bar_chart: *ReviewIQ Alert — Marketing*\n"
        ">*Feature:* {feature}\n"
        ">*Status:* {status}  |  *Z-Score:* {z_score}  |  *Count:* {count}\n"
        ">*Product:* {product}\n"
        "\n"
        "*Action Required:*\n"
        "• Competitor is outperforming us on `{feature_clean}` — review messaging\n"
        "• Update comparative ad copy to de-emphasise this gap\n"
        "• Identify and amplify reviews where we win on `{feature_clean}`\n"
        "\n"
        "_ReviewIQ Competitor Intelligence — {timestamp}_"
    ),
    "Food Safety": (
        ":warning: *URGENT: ReviewIQ Food Safety Alert*\n"
        ">*Feature:* {feature}\n"
        ">*Status:* {status}  |  *Z-Score:* {z_score}  |  *Count:* {count}\n"
        ">*Product:* {product}\n"
        "\n"
        "*IMMEDIATE Action Required:*\n"
        "• Escalate to Food Safety Officer NOW\n"
        "• Pull batch records and check for contamination / expiry issues\n"
        "• Consider voluntary hold on affected SKU pending investigation\n"
        "• Notify regulatory compliance team\n"
        "\n"
        "_High-severity alert — ReviewIQ {timestamp}_"
    ),
}

# Email templates
_EMAIL_TEMPLATES: dict[str, dict] = {
    "Product Engineering": {
        "subject": "[ReviewIQ] {status} Alert — {feature} | {product}",
        "body": (
            "Hi {dept} Team,\n\n"
            "ReviewIQ has detected a {status} spike in customer complaints "
            "about **{feature}** for **{product}**.\n\n"
            "**Alert Details**\n"
            "- Feature: {feature}\n"
            "- Current count: {count} (historical avg: {mean})\n"
            "- Z-Score: {z_score}\n"
            "- Detection time: {timestamp}\n\n"
            "**Recommended Actions**\n"
            "1. Pull firmware/hardware logs for {feature_clean} failures.\n"
            "2. Audit last 2 production batches for QA regressions.\n"
            "3. Assign a P{priority} ticket — target resolution within {sla}.\n\n"
            "Please acknowledge within 4 hours and update the incident tracker.\n\n"
            "— ReviewIQ Automated Alert System"
        ),
    },
    "Supply Chain / QA": {
        "subject": "[ReviewIQ] QA Alert — {feature} Spike | {product}",
        "body": (
            "Hi Supply Chain / QA Team,\n\n"
            "A {status} defect spike has been detected for **{feature}** "
            "on **{product}**.\n\n"
            "**Alert Details**\n"
            "- Feature: {feature}\n"
            "- Current count: {count} (historical avg: {mean})\n"
            "- Z-Score: {z_score}\n\n"
            "**Recommended Actions**\n"
            "1. Quarantine and audit last inbound batch for {feature_clean} defects.\n"
            "2. File a Non-Conformance Report (NCR) with the supplier.\n"
            "3. Update the incoming QC checklist.\n\n"
            "— ReviewIQ Automated Alert System"
        ),
    },
    "Logistics": {
        "subject": "[ReviewIQ] Logistics Alert — {feature} Spike | {product}",
        "body": (
            "Hi Logistics Team,\n\n"
            "ReviewIQ has flagged a {status} spike in **{feature}** "
            "complaints for **{product}**.\n\n"
            "**Alert Details**\n"
            "- Feature: {feature}\n"
            "- Current count: {count} (historical avg: {mean})\n"
            "- Z-Score: {z_score}\n\n"
            "**Recommended Actions**\n"
            "1. Review courier SLA compliance for this SKU.\n"
            "2. Check if complaints correlate with specific regions.\n"
            "3. Escalate to carrier account manager if unresolved in 48h.\n\n"
            "— ReviewIQ Automated Alert System"
        ),
    },
    "Customer Success": {
        "subject": "[ReviewIQ] CS Action Required — {feature} | {product}",
        "body": (
            "Hi Customer Success Team,\n\n"
            "A {status} spike in **{feature}** complaints requires proactive "
            "outreach for **{product}**.\n\n"
            "**Alert Details**\n"
            "- Feature: {feature}\n"
            "- Current count: {count} (historical avg: {mean})\n"
            "- Z-Score: {z_score}\n\n"
            "**Recommended Actions**\n"
            "1. Export the affected customer list and send proactive resolution emails.\n"
            "2. Prepare support scripts for {feature_clean} complaints.\n"
            "3. Monitor CSAT for the next 7 days.\n\n"
            "— ReviewIQ Automated Alert System"
        ),
    },
    "Marketing": {
        "subject": "[ReviewIQ] Competitor Intelligence — {feature} Gap",
        "body": (
            "Hi Marketing Team,\n\n"
            "ReviewIQ's competitor analysis has identified a gap in **{feature}** "
            "where competitors are outperforming us.\n\n"
            "**Details**\n"
            "- Feature: {feature}\n"
            "- Status: {status}\n\n"
            "**Recommended Actions**\n"
            "1. Review and update comparative messaging for {feature_clean}.\n"
            "2. Amplify positive reviews where we win on competing features.\n"
            "3. Brief product team on customer expectations.\n\n"
            "— ReviewIQ Automated Alert System"
        ),
    },
    "Food Safety": {
        "subject": "URGENT [ReviewIQ] Food Safety Alert — {feature} | {product}",
        "body": (
            "URGENT — Food Safety Escalation\n\n"
            "ReviewIQ has detected a potential food safety issue with **{feature}** "
            "for **{product}**.\n\n"
            "**Alert Details**\n"
            "- Feature: {feature}\n"
            "- Current count: {count} (historical avg: {mean})\n"
            "- Z-Score: {z_score}\n\n"
            "**IMMEDIATE Actions Required**\n"
            "1. Escalate to Food Safety Officer immediately.\n"
            "2. Pull batch records and review for contamination / expiry.\n"
            "3. Consider voluntary hold on affected SKU.\n"
            "4. Notify regulatory compliance.\n\n"
            "— ReviewIQ Automated Alert System"
        ),
    },
}

# Priority / SLA by status
_STATUS_PRIORITY = {"Systemic": ("1", "24h"), "Emerging": ("2", "48h")}


# ── Routing logic ──────────────────────────────────────────────────────────────

def _route_single(alert: dict, reviews: list[dict] | None = None) -> str:
    """Returns the department name for one alert."""
    feature = alert.get("feature", "").lower()
    status  = alert.get("status",  "").lower()

    # Food safety escalation: check review text for keywords
    if reviews and any(
        kw in feat_kw
        for kw in _FOOD_SAFETY_KEYWORDS
        for feat_kw in [feature]
    ):
        for r in (reviews or []):
            text = str(r.get("review_text", "")).lower()
            if any(kw in text for kw in _FOOD_SAFETY_KEYWORDS):
                return "Food Safety"

    best_dept  = "Customer Success"   # default fallback
    best_score = -1

    for feat_pat, stat_pat, dept, boost in _ROUTING_RULES:
        feat_match = not feat_pat or feat_pat in feature
        stat_match = not stat_pat or stat_pat in status
        if feat_match and stat_match:
            score = boost + (2 if feat_pat else 0) + (1 if stat_pat else 0)
            if score > best_score:
                best_score = best_dept if score > best_score else best_score
                best_score = score
                best_dept  = dept

    return best_dept


def _format_message(template: str, vars_: dict) -> str:
    """Safe .format() that ignores missing keys."""
    try:
        return template.format(**vars_)
    except KeyError:
        # Replace any remaining {key} with '?'
        return re.sub(r"\{[^}]+\}", "?", template.format_map(
            {k: vars_.get(k, "?") for k in re.findall(r"\{(\w+)\}", template)}
        ))


def _build_vars(alert: dict, dept: str, product_name: str) -> dict:
    """Builds the substitution dict for message templates."""
    feature  = alert.get("feature", "Unknown")
    # Strip product qualifier from feature display: "Battery (ProductX)" → "Battery"
    feat_clean = re.sub(r"\s*\([^)]+\)$", "", feature).strip()
    status   = alert.get("status", "Unknown")
    z_raw    = alert.get("z_score", "N/A")
    z_str    = str(z_raw) if z_raw != "N/A" else "N/A"
    count    = alert.get("current_count", "?")
    mean     = alert.get("mean", "?")
    priority, sla = _STATUS_PRIORITY.get(status, ("2", "48h"))

    return {
        "feature":       feature,
        "feature_clean": feat_clean,
        "status":        status,
        "z_score":       z_str,
        "count":         count,
        "mean":          mean,
        "product":       product_name,
        "dept":          dept,
        "priority":      priority,
        "sla":           sla,
        "timestamp":     datetime.now().strftime("%d %b %Y %H:%M"),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def route_alert(
    alert: dict,
    reviews: list[dict] | None = None,
    product_name: str = "Product",
) -> dict:
    """
    Routes a single alert to the correct department and generates
    pre-drafted Slack and email messages.

    Parameters
    ----------
    alert        : dict — alert from trend_engine, competitor, or cross_product
    reviews      : list[dict] — optional, used for food-safety keyword scan
    product_name : str — product label for messages

    Returns
    -------
    alert dict enriched with:
        department      str
        priority        str     "1" | "2"
        sla             str     "24h" | "48h"
        slack_message   str
        email_subject   str
        email_body      str
    """
    dept  = _route_single(alert, reviews)
    vars_ = _build_vars(alert, dept, product_name)

    slack_tpl = _SLACK_TEMPLATES.get(dept, _SLACK_TEMPLATES["Customer Success"])
    email_tpl = _EMAIL_TEMPLATES.get(dept, _EMAIL_TEMPLATES["Customer Success"])

    return {
        **alert,
        "department":    dept,
        "priority":      vars_["priority"],
        "sla":           vars_["sla"],
        "slack_message": _format_message(slack_tpl, vars_),
        "email_subject": _format_message(email_tpl["subject"], vars_),
        "email_body":    _format_message(email_tpl["body"], vars_),
    }


def route_alerts(
    alerts: list[dict],
    reviews: list[dict] | None = None,
    product_name: str = "Product",
) -> list[dict]:
    """
    Routes a list of alerts. Returns the same list, each alert enriched
    with department + pre-drafted messages.

    Sorted: Food Safety first, then Systemic P1, then Emerging P2.
    """
    routed = [route_alert(a, reviews, product_name) for a in alerts]

    def sort_key(r):
        dept_order = {
            "Food Safety":          0,
            "Product Engineering":  1,
            "Supply Chain / QA":    2,
            "Logistics":            3,
            "Customer Success":     4,
            "Marketing":            5,
        }
        prio = 0 if r.get("status") == "Systemic" else 1
        return (prio, dept_order.get(r.get("department", ""), 9))

    return sorted(routed, key=sort_key)


def get_department_summary(routed_alerts: list[dict]) -> dict[str, list[dict]]:
    """
    Groups routed alerts by department.

    Returns
    -------
    { department_name: [alert, ...], ... }
    """
    summary: dict[str, list[dict]] = {}
    for alert in routed_alerts:
        dept = alert.get("department", "Unknown")
        summary.setdefault(dept, []).append(alert)
    return summary
