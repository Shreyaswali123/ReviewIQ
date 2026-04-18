"""
notifier.py — Anomaly Notification Dispatcher for ReviewIQ
===========================================================
Plugs directly into alert_router.py's output and sends real notifications
via Email (SMTP or SendGrid), Slack Webhooks, and Twilio voice calls.

Setup — environment variables
------------------------------
EMAIL (SMTP):
    SMTP_HOST          e.g. "smtp.gmail.com"
    SMTP_PORT          e.g. "587"
    SMTP_USER          your email address
    SMTP_PASS          your app password / SMTP password

EMAIL (SendGrid — preferred for production):
    SENDGRID_API_KEY   your SendGrid API key
    SENDGRID_FROM      sender address registered in SendGrid

SLACK:
    SLACK_WEBHOOK_URL  your Slack Incoming Webhook URL
                       (one URL = one channel; use DEPT_SLACK_WEBHOOKS for per-dept routing)

    DEPT_SLACK_WEBHOOKS  optional JSON string mapping department → webhook URL
    e.g. '{"Product Engineering": "https://hooks.slack.com/...", "Food Safety": "..."}'

TWILIO (voice calls):
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER   your Twilio phone number, e.g. "+14155551234"
    TWILIO_CALL_NUMBERS  comma-separated numbers to call for CRITICAL alerts
                         e.g. "+919876543210,+14155556789"

THRESHOLDS:
    NOTIFY_EMAIL_MIN_STATUS    "Systemic" | "Emerging"  (default: "Systemic")
    NOTIFY_CALL_MIN_ZSCORE     minimum z-score to trigger a voice call (default: 6.0)
    NOTIFY_CALL_DEPT_WHITELIST departments to trigger calls for (default: "Food Safety,Product Engineering")

Usage
-----
    from alert_router import route_alerts
    from notifier import dispatch_alerts

    routed = route_alerts(alerts, reviews=reviews, product_name="GT Pro 5")
    report = dispatch_alerts(routed)
    print(report)
    # {
    #   "total_alerts": 3,
    #   "emails_sent": 2,
    #   "slack_sent": 3,
    #   "calls_made": 1,
    #   "errors": [],
    #   "log": [...]
    # }

Or call from main.py / agent.py after running your pipeline:

    from notifier import dispatch_alerts
    dispatch_alerts(routed_alerts)
"""

import os
import json
import re
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional


# ── Config helpers ─────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default

def _env_list(key: str, sep: str = ",") -> list[str]:
    val = _env(key)
    return [x.strip() for x in val.split(sep) if x.strip()] if val else []


# ── Status/severity helpers ────────────────────────────────────────────────────

def _zscore_float(alert: dict) -> float:
    """Safely extracts z_score as float. Returns 999.0 for '∞'."""
    z = alert.get("z_score", "N/A")
    if z == "∞":
        return 999.0
    try:
        return float(z)
    except (TypeError, ValueError):
        return 0.0

def _should_email(alert: dict) -> bool:
    min_status = _env("NOTIFY_EMAIL_MIN_STATUS", "Systemic")
    if min_status == "Emerging":
        return alert.get("status") in ("Systemic", "Emerging")
    return alert.get("status") == "Systemic"

def _should_call(alert: dict) -> bool:
    min_z     = _env_float("NOTIFY_CALL_MIN_ZSCORE", 6.0)
    whitelist = _env_list("NOTIFY_CALL_DEPT_WHITELIST") or ["Food Safety", "Product Engineering"]
    dept      = alert.get("department", "")
    z         = _zscore_float(alert)
    return dept in whitelist and z >= min_z


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL SENDERS
# ══════════════════════════════════════════════════════════════════════════════

def _send_email_smtp(
    to_addresses: list[str],
    subject: str,
    body: str,
) -> tuple[bool, str]:
    """Send via plain SMTP (works with Gmail, Outlook, any SMTP server)."""
    host = _env("SMTP_HOST", "smtp.gmail.com")
    port = int(_env("SMTP_PORT", "587"))
    user = _env("SMTP_USER")
    pwd  = _env("SMTP_PASS")

    if not (user and pwd):
        return False, "SMTP_USER / SMTP_PASS not configured."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = user
    msg["To"]      = ", ".join(to_addresses)

    # Plain text part + basic HTML
    text_part = MIMEText(body, "plain")
    html_body = "<pre style='font-family:monospace'>" + body.replace("\n", "<br>") + "</pre>"
    html_part = MIMEText(html_body, "html")
    msg.attach(text_part)
    msg.attach(html_part)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(user, pwd)
            server.sendmail(user, to_addresses, msg.as_string())
        return True, f"SMTP email sent to {to_addresses}"
    except Exception as e:
        return False, f"SMTP error: {e}"


def _send_email_sendgrid(
    to_addresses: list[str],
    subject: str,
    body: str,
) -> tuple[bool, str]:
    """Send via SendGrid API (preferred for production / high volume)."""
    api_key  = _env("SENDGRID_API_KEY")
    from_addr = _env("SENDGRID_FROM")

    if not (api_key and from_addr):
        return False, "SENDGRID_API_KEY / SENDGRID_FROM not configured."

    try:
        import urllib.request
        payload = json.dumps({
            "personalizations": [{"to": [{"email": a} for a in to_addresses]}],
            "from": {"email": from_addr},
            "subject": subject,
            "content": [
                {"type": "text/plain", "value": body},
                {"type": "text/html",  "value": "<pre>" + body + "</pre>"},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
        if status in (200, 202):
            return True, f"SendGrid email sent to {to_addresses}"
        return False, f"SendGrid returned HTTP {status}"

    except Exception as e:
        return False, f"SendGrid error: {e}"


def send_email(
    to_addresses: list[str],
    subject: str,
    body: str,
) -> tuple[bool, str]:
    """
    Auto-selects SendGrid if SENDGRID_API_KEY is set, else falls back to SMTP.
    Returns (success: bool, message: str).
    """
    if _env("SENDGRID_API_KEY"):
        return _send_email_sendgrid(to_addresses, subject, body)
    return _send_email_smtp(to_addresses, subject, body)


# ══════════════════════════════════════════════════════════════════════════════
# SLACK SENDER
# ══════════════════════════════════════════════════════════════════════════════

def _get_slack_webhook(department: str) -> Optional[str]:
    """Returns the webhook URL for a department, falling back to the default."""
    dept_map_raw = _env("DEPT_SLACK_WEBHOOKS")
    if dept_map_raw:
        try:
            dept_map = json.loads(dept_map_raw)
            if department in dept_map:
                return dept_map[department]
        except json.JSONDecodeError:
            pass
    return _env("SLACK_WEBHOOK_URL") or None


def send_slack(department: str, message: str) -> tuple[bool, str]:
    """
    Posts a Slack message to the department's channel.
    Returns (success: bool, message: str).
    """
    webhook_url = _get_slack_webhook(department)
    if not webhook_url:
        return False, "No Slack webhook configured (SLACK_WEBHOOK_URL or DEPT_SLACK_WEBHOOKS)."

    try:
        import urllib.request
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
        if body == "ok":
            return True, f"Slack message sent to #{department}"
        return False, f"Slack returned: {body}"
    except Exception as e:
        return False, f"Slack error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TWILIO VOICE CALL
# ══════════════════════════════════════════════════════════════════════════════

def _build_twiml(alert: dict) -> str:
    """
    Generates TwiML XML for the voice call.
    Twilio will read this message aloud to the recipient.
    """
    feature = re.sub(r"\s*\([^)]+\)$", "", alert.get("feature", "Unknown")).strip()
    product = alert.get("product", alert.get("feature", ""))
    status  = alert.get("status", "Unknown")
    dept    = alert.get("department", "your team")
    z       = alert.get("z_score", "N/A")

    speech = (
        f"This is an automated ReviewIQ anomaly alert. "
        f"A {status} spike has been detected in {feature} complaints "
        f"for {product}. "
        f"Z score: {z}. "
        f"This alert is assigned to {dept}. "
        f"Please check your email and Slack immediately. "
        f"I repeat: ReviewIQ has detected a {status} issue in {feature}. "
        f"Please take action now."
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice" language="en-IN">{speech}</Say>
  <Pause length="1"/>
  <Say voice="alice" language="en-IN">{speech}</Say>
</Response>"""


def make_call(alert: dict, to_number: str) -> tuple[bool, str]:
    """
    Places a Twilio voice call to `to_number` reading out the alert.
    Returns (success: bool, message: str).
    """
    account_sid = _env("TWILIO_ACCOUNT_SID")
    auth_token  = _env("TWILIO_AUTH_TOKEN")
    from_number = _env("TWILIO_FROM_NUMBER")

    if not (account_sid and auth_token and from_number):
        return False, "Twilio credentials not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER)."

    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse

        client = Client(account_sid, auth_token)
        twiml  = _build_twiml(alert)

        call = client.calls.create(
            twiml=twiml,
            to=to_number,
            from_=from_number,
        )
        return True, f"Call placed to {to_number} (SID: {call.sid})"

    except ImportError:
        # Fallback: use Twilio REST API directly without the SDK
        try:
            import urllib.request
            import urllib.parse
            import base64

            # Host TwiML via Twilio's Bins (requires twilio-python SDK)
            # Fallback: use a pre-built TwiML URL or Twilio Studio
            # For simplicity, pass TwiML directly via the API
            url    = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"
            creds  = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
            data   = urllib.parse.urlencode({
                "To":    to_number,
                "From":  from_number,
                "Twiml": _build_twiml(alert),
            }).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Authorization": f"Basic {creds}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
            return True, f"Call placed to {to_number} (SID: {result.get('sid', '?')})"
        except Exception as e:
            return False, f"Twilio REST error: {e}"

    except Exception as e:
        return False, f"Twilio error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# RECIPIENT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Default department → email list. Override via DEPT_EMAILS env var (JSON string).
# Example: export DEPT_EMAILS='{"Food Safety": ["safety@co.com", "cto@co.com"]}'
_DEFAULT_DEPT_EMAILS: dict[str, list[str]] = {
    "Product Engineering": [],
    "Supply Chain / QA":  [],
    "Logistics":          [],
    "Customer Success":   [],
    "Marketing":          [],
    "Food Safety":        [],
}


def _get_email_recipients(department: str) -> list[str]:
    """Returns email addresses for a department from env config.

    Resolution order:
      1. DEPT_EMAILS  - JSON string mapping department to [addr, ...]
      2. ALERT_EMAILS - comma-separated catch-all list
    If neither is set, prints a clear warning so the misconfiguration
    is visible instead of silently dropping emails.
    """
    dept_map_raw = _env("DEPT_EMAILS")
    if dept_map_raw:
        try:
            dept_map = json.loads(dept_map_raw)
            addrs    = dept_map.get(department, [])
            if addrs:
                return addrs
        except json.JSONDecodeError:
            print("WARNING: DEPT_EMAILS is set but is not valid JSON -- ignoring.")

    # Fallback: ALERT_EMAILS = comma-separated catch-all list
    fallback = _env_list("ALERT_EMAILS")
    if fallback:
        return fallback

    print(
        f"WARNING: No email recipients configured for department '{department}'. "
        "Set ALERT_EMAILS (e.g. 'you@example.com') or "
        "DEPT_EMAILS (JSON map of department to address list) in your .env file."
    )
    return []


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_alert(alert: dict) -> dict:
    """
    Dispatches a single routed alert via all applicable channels.
    `alert` must already be enriched by alert_router.route_alert().

    Returns a log dict:
        { "feature", "department", "channels": [...], "errors": [...] }
    """
    feature = alert.get("feature", "Unknown")
    dept    = alert.get("department", "Unknown")
    log     = {"feature": feature, "department": dept, "channels": [], "errors": []}

    # ── Slack ──────────────────────────────────────────────────────────────────
    slack_msg = alert.get("slack_message", "")
    if slack_msg:
        ok, msg = send_slack(dept, slack_msg)
        if ok:
            log["channels"].append(f"slack:{dept}")
        else:
            log["errors"].append(f"Slack — {msg}")

    # ── Email ──────────────────────────────────────────────────────────────────
    if _should_email(alert):
        recipients = _get_email_recipients(dept)
        if recipients:
            subject = alert.get("email_subject", f"[ReviewIQ] Alert — {feature}")
            body    = alert.get("email_body",    f"Alert: {feature}\nDept: {dept}")
            ok, msg = send_email(recipients, subject, body)
            if ok:
                log["channels"].append(f"email:{','.join(recipients)}")
            else:
                log["errors"].append(f"Email — {msg}")
        else:
            log["errors"].append(
                f"Email skipped — no recipients configured for '{dept}'. "
                "Set DEPT_EMAILS or ALERT_EMAILS env var."
            )

    # ── Voice Call ─────────────────────────────────────────────────────────────
    if _should_call(alert):
        call_numbers = _env_list("TWILIO_CALL_NUMBERS")
        for number in call_numbers:
            ok, msg = make_call(alert, number)
            if ok:
                log["channels"].append(f"call:{number}")
            else:
                log["errors"].append(f"Call to {number} — {msg}")

    return log


def dispatch_alerts(
    routed_alerts: list[dict],
    dry_run: bool = False,
) -> dict:
    """
    Dispatches all routed alerts across all configured channels.

    Parameters
    ----------
    routed_alerts : output of alert_router.route_alerts()
    dry_run       : if True, builds messages but does NOT send anything
                    (useful for testing / previewing notifications)

    Returns
    -------
    {
        "total_alerts":  int,
        "emails_sent":   int,
        "slack_sent":    int,
        "calls_made":    int,
        "errors":        list[str],
        "log":           list[dict],
        "dry_run":       bool,
    }
    """
    if not routed_alerts:
        return {
            "total_alerts": 0, "emails_sent": 0,
            "slack_sent": 0, "calls_made": 0,
            "errors": [], "log": [], "dry_run": dry_run,
        }

    print(f"\n{'[DRY RUN] ' if dry_run else ''}📣 Dispatching {len(routed_alerts)} alert(s)…")

    all_logs:    list[dict] = []
    all_errors:  list[str]  = []
    email_count: int = 0
    slack_count: int = 0
    call_count:  int = 0

    for alert in routed_alerts:
        feature = alert.get("feature", "?")
        dept    = alert.get("department", "?")
        status  = alert.get("status", "?")
        print(f"  → [{status}] {feature} | Dept: {dept}")

        if dry_run:
            # Preview only — show what would be sent
            preview = {
                "feature":     feature,
                "department":  dept,
                "status":      status,
                "would_email": _should_email(alert),
                "would_call":  _should_call(alert),
                "email_subject": alert.get("email_subject", ""),
                "slack_preview": (alert.get("slack_message", "")[:120] + "..."),
            }
            all_logs.append(preview)
            continue

        log = dispatch_alert(alert)
        all_logs.append(log)
        all_errors.extend(log.get("errors", []))

        for ch in log.get("channels", []):
            if ch.startswith("email:"):  email_count += 1
            elif ch.startswith("slack:"): slack_count += 1
            elif ch.startswith("call:"):  call_count  += 1

    result = {
        "total_alerts": len(routed_alerts),
        "emails_sent":  email_count,
        "slack_sent":   slack_count,
        "calls_made":   call_count,
        "errors":       all_errors,
        "log":          all_logs,
        "dry_run":      dry_run,
    }

    if not dry_run:
        print(f"\n✅ Dispatch complete — "
              f"{email_count} email(s), {slack_count} Slack message(s), {call_count} call(s).")
        if all_errors:
            print(f"⚠️  {len(all_errors)} error(s):")
            for e in all_errors:
                print(f"    • {e}")

    return result


# ── Convenience: plug straight into the ReviewIQ pipeline ─────────────────────

def notify_on_anomalies(
    alerts: list[dict],
    reviews: list[dict] | None = None,
    product_name: str = "Product",
    dry_run: bool = False,
) -> dict:
    """
    One-call convenience wrapper.
    Runs route_alerts() then dispatch_alerts() in sequence.

    Usage in main.py
    ----------------
        from notifier import notify_on_anomalies
        result = notify_on_anomalies(trend_alerts, reviews=reviews, product_name="GT Pro 5")

    Parameters
    ----------
    alerts       : raw alerts from trend_engine / competitor / cross_product
    reviews      : review list (used for food-safety scan in router)
    product_name : product label for messages
    dry_run      : preview without actually sending

    Returns
    -------
    dispatch report dict (same as dispatch_alerts)
    """
    from alert_router import route_alerts
    routed = route_alerts(alerts, reviews=reviews, product_name=product_name)
    return dispatch_alerts(routed, dry_run=dry_run)