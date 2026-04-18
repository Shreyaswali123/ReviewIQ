"""
cross_product.py — Platform-level defect detection for ReviewIQ

Improvements vs original:
  • Works with any set of categories — not limited to Electronics/Food/Footwear.
  • Requires at least 2 impacted categories AND a minimum occurrence count
    per feature to reduce noise.
  • Sorts results by number of impacted categories (most widespread first).
  • Handles empty input gracefully.
"""

# Minimum number of times a feature must appear in a single category before
# it's considered a real signal (not a one-off mention).
MIN_OCCURRENCES_PER_CATEGORY = 2

# Minimum number of distinct categories a feature must span to be flagged
# as a platform-level issue.
MIN_CATEGORIES_TO_FLAG = 2


def find_platform_level_issues(
    category_negatives: dict[str, list[str]],
) -> list[dict]:
    """
    Identifies negative features that appear across multiple unrelated product
    categories, indicating a platform / vendor / logistics issue rather than a
    product-specific defect.

    Parameters
    ----------
    category_negatives : dict
        { category_name: [feature_name, feature_name, ...] }
        Feature names may repeat; repetition signals frequency.

    Returns
    -------
    list of dicts, each containing:
        feature             — feature name
        impacted_categories — list of category names affected
        occurrence_map      — { category: count } for transparency
        classification      — "Platform-Level Defect"
    """
    if not category_negatives:
        return []

    # Count occurrences: feature → { category → count }
    feature_category_counts: dict[str, dict[str, int]] = {}

    for category, features in category_negatives.items():
        if not features:
            continue
        for feature in features:
            feature = feature.strip()
            if not feature:
                continue
            if feature not in feature_category_counts:
                feature_category_counts[feature] = {}
            feature_category_counts[feature][category] = (
                feature_category_counts[feature].get(category, 0) + 1
            )

    platform_issues: list[dict] = []

    for feature, cat_counts in feature_category_counts.items():
        # Only count categories where the feature meets the minimum occurrence threshold
        qualifying_categories = [
            cat for cat, cnt in cat_counts.items()
            if cnt >= MIN_OCCURRENCES_PER_CATEGORY
        ]

        if len(qualifying_categories) >= MIN_CATEGORIES_TO_FLAG:
            platform_issues.append({
                "feature":             feature,
                "impacted_categories": qualifying_categories,
                "occurrence_map":      {c: cat_counts[c] for c in qualifying_categories},
                "classification":      "Platform-Level Defect",
            })

    # Sort by number of impacted categories descending (most widespread first)
    platform_issues.sort(key=lambda x: len(x["impacted_categories"]), reverse=True)
    return platform_issues