"""
ecommerce_fetcher.py — Live review ingestion from e-commerce platforms for ReviewIQ

Supported platforms
-------------------
  • Amazon India / Global  — via RapidAPI "Real-Time Amazon Data" endpoint
  • Flipkart               — via RapidAPI "Flipkart Product Reviews" endpoint
  • Meesho                 — via unofficial public API (no key required)
  • Generic REST webhook   — any platform that posts reviews as JSON

All fetchers normalise output to the same schema used by ingest.py so the
data drops straight into the ReviewIQ pipeline without any transformation.

Usage
-----
    from ecommerce_fetcher import fetch_reviews

    reviews = fetch_reviews(
        platform="amazon",
        product_id="B09XYZ",        # ASIN for Amazon, product_id for Flipkart
        max_pages=5,
        api_key="your_rapidapi_key",
    )
    # reviews is a list[dict] ready for sentiment.analyze_batch()

Environment variables (set in .env)
------------------------------------
    RAPIDAPI_KEY        — your RapidAPI key (Amazon + Flipkart)
    FLIPKART_API_KEY    — optional separate key if you have a dedicated account
"""

import os
import time
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

RAPIDAPI_HOST_AMAZON   = "real-time-amazon-data.p.rapidapi.com"
RAPIDAPI_HOST_FLIPKART = "flipkart-product-reviews.p.rapidapi.com"
MEESHO_API_BASE        = "https://api.meesho.com/api/v1"

REQUEST_TIMEOUT   = 15   # seconds
RETRY_LIMIT       = 3
RETRY_DELAY       = 2.0  # seconds, doubles on each retry
PAGE_DELAY        = 0.8  # polite delay between pages

MAX_PAGES_DEFAULT = 5    # ~50 reviews per page = 250 reviews max by default


# ── Normaliser ────────────────────────────────────────────────────────────────

def _normalise(raw: dict, platform: str, product_id: str) -> dict | None:
    """
    Maps any platform's raw review dict to ReviewIQ's internal schema.
    Returns None for records that have no usable text.
    """
    # Extract text — try multiple common field names
    text = (
        raw.get("review_comment") or raw.get("body") or raw.get("text") or
        raw.get("reviewText") or raw.get("review_text") or
        raw.get("content") or raw.get("description") or ""
    ).strip()

    if not text or len(text) < 5:
        return None

    # Rating — coerce to int 1-5
    rating_raw = (
        raw.get("review_star_rating") or raw.get("rating") or
        raw.get("star_rating") or raw.get("stars") or raw.get("overallRating")
    )
    try:
        rating = int(float(str(rating_raw)))
        if not 1 <= rating <= 5:
            rating = None
    except (TypeError, ValueError):
        rating = None

    # Date
    date_raw = (
        raw.get("review_date") or raw.get("date") or
        raw.get("submittedAt") or raw.get("created_at") or ""
    )
    try:
        # Try ISO parse; fall back to today
        date_str = str(date_raw)[:10] if date_raw else datetime.now().strftime("%Y-%m-%d")
    except Exception:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Reviewer
    reviewer = (
        raw.get("reviewer_name") or raw.get("username") or
        raw.get("profile_name") or raw.get("userId") or "anonymous"
    )

    # Review ID
    rev_id = (
        raw.get("review_id") or raw.get("id") or raw.get("reviewId") or
        f"{platform}_{product_id}_{hash(text) & 0xFFFFFF:06x}"
    )

    # Product name / title
    product = (
        raw.get("product_name") or raw.get("productName") or
        raw.get("title") or product_id
    )

    return {
        "review_id":    str(rev_id),
        "review_text":  text,
        "product_name": product,
        "category":     platform,
        "star_rating":  rating,
        "review_date":  date_str,
        "reviewer_id":  str(reviewer),
        "source":       platform,
    }


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, headers: dict, params: dict) -> dict:
    """GET with retry + exponential back-off. resp is always initialised before use."""
    delay = RETRY_DELAY
    last_err = None
    for attempt in range(RETRY_LIMIT):
        resp = None
        try:
            resp = requests.get(url, headers=headers, params=params,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected response type: {type(data).__name__}")
            return data
        except requests.exceptions.HTTPError as e:
            status = resp.status_code if resp is not None else 0
            if status == 429:
                print(f"   ⏳ Rate limited (429). Waiting {delay}s…")
                time.sleep(delay)
                delay *= 2
                continue
            if status in (401, 403):
                raise RuntimeError(
                    f"API authentication failed ({status}). "
                    "Check your RAPIDAPI_KEY is valid and subscribed to this endpoint."
                ) from e
            raise
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_err = e
            print(f"   ⚠️  Network error on attempt {attempt + 1}: {e}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(delay)
                delay *= 2
        except Exception as e:
            last_err = e
            if attempt < RETRY_LIMIT - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Request failed after {RETRY_LIMIT} attempts: {last_err}")


# ── Amazon fetcher ─────────────────────────────────────────────────────────────

def fetch_amazon(
    product_id: str,
    max_pages: int = MAX_PAGES_DEFAULT,
    api_key: str | None = None,
    country: str = "IN",
    sort_by: str = "TOP_REVIEWS",   # or "MOST_RECENT"
    star_filter: str = "",          # "" = all, "1_star" … "5_star"
) -> list[dict]:
    """
    Fetches Amazon product reviews via RapidAPI "Real-Time Amazon Data".

    Parameters
    ----------
    product_id  : ASIN (e.g. "B09NZKQLTQ")
    max_pages   : number of pages to fetch (≈10 reviews/page)
    api_key     : RapidAPI key; falls back to env var RAPIDAPI_KEY
    country     : "IN" for India, "US" for US store, etc.
    sort_by     : "TOP_REVIEWS" | "MOST_RECENT"
    star_filter : "" | "1_star" | "2_star" | "3_star" | "4_star" | "5_star"
    """
    key = api_key or os.getenv("RAPIDAPI_KEY")
    if not key:
        raise ValueError(
            "Amazon fetch requires a RapidAPI key. "
            "Set RAPIDAPI_KEY in .env or pass api_key=."
        )

    url     = f"https://{RAPIDAPI_HOST_AMAZON}/product-reviews"
    headers = {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": RAPIDAPI_HOST_AMAZON,
    }

    all_reviews: list[dict] = []
    print(f"🛒 Fetching Amazon reviews for ASIN {product_id} ({country})…")

    for page in range(1, max_pages + 1):
        params = {
            "asin":           product_id,
            "country":        country,
            "sort_by":        sort_by,
            "page":           str(page),
            "verified_purchases_only": "false",
            "filter_by_star": star_filter,
        }
        try:
            data = _get(url, headers, params)
        except Exception as e:
            print(f"   ❌ Amazon page {page} failed: {e}")
            break

        # RapidAPI "Real-Time Amazon Data" may nest reviews differently across versions
        reviews_raw = (
            data.get("data", {}).get("reviews")        # v1/v2 standard
            or data.get("reviews")                      # flat response
            or data.get("data", {}).get("top_reviews")  # alternate key
            or data.get("top_reviews")
            or []
        )
        if not isinstance(reviews_raw, list):
            reviews_raw = []
        if not reviews_raw:
            print(f"   ℹ️  No more reviews at page {page}.")
            break

        for raw in reviews_raw:
            normed = _normalise(raw, "amazon", product_id)
            if normed:
                all_reviews.append(normed)

        print(f"   ✅ Page {page}: {len(reviews_raw)} reviews fetched")
        if page < max_pages:
            time.sleep(PAGE_DELAY)

    print(f"📦 Amazon total: {len(all_reviews)} valid reviews for {product_id}")
    return all_reviews


# ── Flipkart fetcher ───────────────────────────────────────────────────────────

def fetch_flipkart(
    product_id: str,
    max_pages: int = MAX_PAGES_DEFAULT,
    api_key: str | None = None,
) -> list[dict]:
    """
    Fetches Flipkart product reviews via RapidAPI "Flipkart Product Reviews".

    Parameters
    ----------
    product_id  : Flipkart product ID (e.g. "MOBG4Q4JNZQBZDNK")
    max_pages   : number of pages to fetch (≈10 reviews/page)
    api_key     : RapidAPI key; falls back to env var RAPIDAPI_KEY
    """
    key = api_key or os.getenv("FLIPKART_API_KEY") or os.getenv("RAPIDAPI_KEY")
    if not key:
        raise ValueError(
            "Flipkart fetch requires a RapidAPI key. "
            "Set RAPIDAPI_KEY or FLIPKART_API_KEY in .env."
        )

    url     = f"https://{RAPIDAPI_HOST_FLIPKART}/product-reviews"
    headers = {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": RAPIDAPI_HOST_FLIPKART,
    }

    all_reviews: list[dict] = []
    print(f"🛒 Fetching Flipkart reviews for product {product_id}…")

    for page in range(1, max_pages + 1):
        params = {"pid": product_id, "page": str(page)}
        try:
            data = _get(url, headers, params)
        except Exception as e:
            print(f"   ❌ Flipkart page {page} failed: {e}")
            break

        # Flipkart RapidAPI providers use varying key names
        reviews_raw = (
            data.get("reviews")
            or data.get("data", {}).get("reviews")
            or data.get("result", {}).get("reviews")
            or data.get("response", {}).get("data", {}).get("reviews")
            or []
        )
        if not isinstance(reviews_raw, list):
            reviews_raw = []
        if not reviews_raw:
            print(f"   ℹ️  No more reviews at page {page}.")
            break

        for raw in reviews_raw:
            normed = _normalise(raw, "flipkart", product_id)
            if normed:
                all_reviews.append(normed)

        print(f"   ✅ Page {page}: {len(reviews_raw)} reviews fetched")
        if page < max_pages:
            time.sleep(PAGE_DELAY)

    print(f"📦 Flipkart total: {len(all_reviews)} valid reviews for {product_id}")
    return all_reviews


# ── Meesho fetcher (unofficial) ────────────────────────────────────────────────

def fetch_meesho(
    product_id: str,
    max_pages: int = MAX_PAGES_DEFAULT,
) -> list[dict]:
    """
    Fetches Meesho product reviews via the unofficial public catalogue API.
    No API key required, but subject to rate limiting.

    Parameters
    ----------
    product_id  : Meesho product ID (numeric string, e.g. "123456789")
    max_pages   : number of pages to fetch
    """
    all_reviews: list[dict] = []
    print(f"🛒 Fetching Meesho reviews for product {product_id}…")

    # Meesho's public review API — catalogue endpoint (works without auth)
    url = f"https://www.meesho.com/api/v1/catalogues/{product_id}/reviews"

    headers = {
        "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36",
        "Accept":       "application/json",
        "Referer":      "https://www.meesho.com/",
        "Origin":       "https://www.meesho.com",
    }

    for page in range(1, max_pages + 1):
        params = {"page": page, "limit": 20, "rating": 0}
        try:
            resp = requests.get(url, headers=headers, params=params,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            status = resp.status_code if resp is not None else 0
            if status == 404:
                print(f"   ❌ Meesho product '{product_id}' not found (404). "
                      "Verify the numeric catalogue ID.")
                break
            print(f"   ❌ Meesho page {page} HTTP error ({status}): {e}")
            break
        except Exception as e:
            print(f"   ❌ Meesho page {page} failed: {e}")
            break

        # Response shape: {"catalogues_data": {"reviews": [...]}} or {"reviews": [...]}
        reviews_raw = (
            data.get("catalogues_data", {}).get("reviews")
            or data.get("reviews")
            or data.get("data", {}).get("reviews")
            or []
        )
        if not isinstance(reviews_raw, list):
            reviews_raw = []

        if not reviews_raw:
            print(f"   ℹ️  No more reviews at page {page}.")
            break

        for raw in reviews_raw:
            # Meesho field aliases
            if "review_comment" not in raw and "text" not in raw:
                raw["review_comment"] = (
                    raw.get("comment") or raw.get("review") or
                    raw.get("body") or raw.get("description") or ""
                )
            if "rating" not in raw:
                raw["rating"] = raw.get("overallRating") or raw.get("stars")
            if "review_date" not in raw:
                raw["review_date"] = raw.get("createdAt") or raw.get("date")
            normed = _normalise(raw, "meesho", product_id)
            if normed:
                all_reviews.append(normed)

        print(f"   ✅ Page {page}: {len(reviews_raw)} reviews fetched")
        if page < max_pages:
            time.sleep(PAGE_DELAY)

    print(f"📦 Meesho total: {len(all_reviews)} valid reviews for {product_id}")
    return all_reviews


# ── Webhook / generic JSON receiver ───────────────────────────────────────────

def fetch_from_webhook(
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
    reviews_key: str = "reviews",   # JSON key that holds the review list
    platform_label: str = "webhook",
    product_id: str = "webhook_product",
    max_pages: int = 10,
    next_page_key: str | None = "next_page_token",  # pagination token key
) -> list[dict]:
    """
    Generic fetcher for any platform that exposes a REST endpoint returning
    review JSON. Handles cursor-based pagination.

    Parameters
    ----------
    url             : API endpoint URL
    headers         : request headers (auth tokens, etc.)
    params          : initial query parameters
    reviews_key     : JSON path to the review list (dot-notation supported)
    platform_label  : label used as the 'category' field in output
    product_id      : identifier for the product
    max_pages       : max pagination steps
    next_page_key   : JSON key for the next-page cursor; None = stop after 1 page
    """
    all_reviews: list[dict] = []
    h = headers or {}
    p = dict(params or {})
    print(f"🌐 Fetching reviews from webhook: {url}")

    for page_num in range(max_pages):
        try:
            resp = requests.get(url, headers=h, params=p, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"   ❌ Webhook page {page_num+1} failed: {e}")
            break

        # Navigate dot-notation key path
        reviews_raw = data
        for part in reviews_key.split("."):
            if isinstance(reviews_raw, dict):
                reviews_raw = reviews_raw.get(part, [])
            else:
                reviews_raw = []
                break

        if not isinstance(reviews_raw, list) or not reviews_raw:
            break

        for raw in reviews_raw:
            normed = _normalise(raw, platform_label, product_id)
            if normed:
                all_reviews.append(normed)

        print(f"   ✅ Page {page_num+1}: {len(reviews_raw)} reviews fetched")

        # Pagination
        if not next_page_key:
            break
        token = data.get(next_page_key) or data.get("data", {}).get(next_page_key)
        if not token:
            break
        p["page_token"] = token
        time.sleep(PAGE_DELAY)

    print(f"📦 Webhook total: {len(all_reviews)} valid reviews")
    return all_reviews


# ── Master fetcher ─────────────────────────────────────────────────────────────

PLATFORM_MAP = {
    "amazon":   fetch_amazon,
    "flipkart": fetch_flipkart,
    "meesho":   fetch_meesho,
}


def fetch_reviews(
    platform: str,
    product_id: str,
    max_pages: int = MAX_PAGES_DEFAULT,
    api_key: str | None = None,
    **kwargs,
) -> list[dict]:
    """
    Unified entry point.  Dispatches to the correct platform fetcher.

    Parameters
    ----------
    platform    : "amazon" | "flipkart" | "meesho"
    product_id  : platform-specific product identifier
    max_pages   : pages to retrieve
    api_key     : RapidAPI key (not needed for Meesho)
    **kwargs    : platform-specific extra parameters

    Returns
    -------
    list[dict] — normalised reviews ready for ReviewIQ pipeline
    """
    platform_lower = platform.strip().lower()
    if platform_lower not in PLATFORM_MAP:
        raise ValueError(
            f"Unsupported platform '{platform}'. "
            f"Choose from: {', '.join(PLATFORM_MAP)}"
        )
    fetcher = PLATFORM_MAP[platform_lower]

    if platform_lower == "meesho":
        return fetcher(product_id=product_id, max_pages=max_pages, **kwargs)
    return fetcher(product_id=product_id, max_pages=max_pages,
                   api_key=api_key, **kwargs)


def fetch_multi_product(
    product_specs: list[dict],
    api_key: str | None = None,
) -> dict[str, list[dict]]:
    """
    Batch-fetches reviews for multiple products.

    Parameters
    ----------
    product_specs : list of dicts, each with:
        {
            "platform":   "amazon" | "flipkart" | "meesho",
            "product_id": str,
            "name":       str   (optional human-readable label),
            "max_pages":  int   (optional, default MAX_PAGES_DEFAULT),
        }
    api_key : shared RapidAPI key

    Returns
    -------
    { product_name_or_id: [review, ...], ... }
    """
    result: dict[str, list[dict]] = {}
    for spec in product_specs:
        pid   = spec["product_id"]
        label = spec.get("name", pid)
        try:
            reviews = fetch_reviews(
                platform=spec["platform"],
                product_id=pid,
                max_pages=spec.get("max_pages", MAX_PAGES_DEFAULT),
                api_key=api_key,
            )
            # Tag each review with the human label
            for r in reviews:
                r["product_name"] = label
            result[label] = reviews
        except Exception as e:
            print(f"⚠️  Skipped {label} ({spec['platform']}): {e}")
            result[label] = []
    return result