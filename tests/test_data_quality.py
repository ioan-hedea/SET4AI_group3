"""
Data quality tests for Rotterdam dataset.

Tests verify that collected data matches official population statistics
for Rotterdam to detect sampling bias.
"""

import numpy as np
import pandas as pd
import pytest

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configure test data
# ============================================================================
data = pd.read_csv("./../data/synth_data_for_training.csv")
features_description = pd.read_csv("./../data/data_description.csv", encoding="latin-1")

columns_mapper = dict(zip(features_description['Feature (nl)'], features_description['Feature (en)']))
data = data.rename(columns=columns_mapper)


# ============================================================================
# Constants and Configuration
# ============================================================================

# District distribution source: https://allecijfers.nl/gemeente/rotterdam/
DISTRICT_EXPECTED_SHARES = {
    "address_latest_district_charlois": 0.105,  # Charlois ~10.5%
    "address_latest_district_delfshaven": 0.114,  # Delfshaven ~11.4%
    "address_latest_district_feijenoord": 0.120,  # Feijenoord ~12.0%
    "address_latest_district_ijsselmonde": 0.094,  # IJsselmonde ~9.4%
    "address_latest_district_kralingen_c": 0.083,  # Kralingen-Crooswijk ~8.3%
    "address_latest_district_north": 0.078,  # Noord ~7.8%
    "address_latest_district_city_center": 0.063  # Rotterdam Centrum ~6.3%
}

# Gender distribution source: https://ugeo.urbistat.com/AdminStat/en/nl/demografia/eta/rotterdam/23055877/4
EXPECTED_FEMALE_SHARE = 0.506  # 50.6% female
EXPECTED_MALE_SHARE = 0.494  # 49.4% male

# Age distribution source: https://ugeo.urbistat.com/AdminStat/en/nl/demografia/eta/rotterdam/23055877/4
AGE_BIN_RANGES = [
    (18, 24),
    (25, 34),
    (35, 44),
    (45, 54),
    (55, 64),
    (65, 74),
    (75, np.inf)
]

AGE_BIN_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

AGE_BIN_EXPECTED_SHARES = {
    "18-24": 0.128,
    "25-34": 0.217,
    "35-44": 0.164,
    "45-54": 0.156,
    "55-64": 0.145,
    "65-74": 0.109,
    "75+": 0.082
}

# Test configuration
DISTRICT_TOLERANCE = 0.01  # ±1 percentage point
GENDER_TOLERANCE = 0.02  # ±2 percentage points
AGE_TOLERANCE = 0.05  # ±5 percentage points
MIN_ROWS_THRESHOLD = 500  # Warn if sample size below this


# ============================================================================
# Utility Functions
# ============================================================================

def validate_binary_column(series: pd.Series, column_name: str) -> None:
    """
    Validate that a series contains only binary (0/1) values, allowing NaN.

    Raises:
        AssertionError: If non-binary values are found.
    """
    non_null_values = series.dropna().unique()
    invalid_values = [v for v in non_null_values if v not in (0, 1)]
    if invalid_values:
        raise AssertionError(
            f"Column '{column_name}' contains non-binary values: {invalid_values}"
        )


def warn_if_sample_too_small(n_rows: int, threshold: int = MIN_ROWS_THRESHOLD) -> None:
    """Print warning if sample size is below reliability threshold."""
    if n_rows < threshold:
        print(
            f"WARNING: Sample size ({n_rows}) is below recommended minimum "
            f"({threshold}) — estimates may be unreliable."
        )


# ============================================================================
# District Distribution Test
# ============================================================================

def test_district_distribution_matches_official(X: pd.DataFrame = data) -> None:
    """
    Verify that district population shares match official Rotterdam statistics.

    Note: ~40% of dataset residents don't live in Rotterdam and are excluded
    from this analysis.

    Args:
        X: DataFrame with district indicator columns and 'address_latest_place_rotterdam'

    Raises:
        AssertionError: If district shares differ by more than DISTRICT_TOLERANCE
    """
    # Validate required columns exist
    missing_columns = [
        col for col in DISTRICT_EXPECTED_SHARES.keys()
        if col not in X.columns
    ]
    if missing_columns:
        raise AssertionError(
            f"Missing required district columns: {missing_columns}"
        )

    if "address_latest_place_rotterdam" not in X.columns:
        raise AssertionError(
            "Column 'address_latest_place_rotterdam' required to filter Rotterdam residents"
        )

    # Filter to Rotterdam residents only
    rotterdam_residents = X[X["address_latest_place_rotterdam"] == 1].copy()
    n_rotterdam = len(rotterdam_residents)

    print(f"\nRotterdam residents in dataset: {n_rotterdam}")
    warn_if_sample_too_small(n_rotterdam)

    # Validate binary columns
    for col in DISTRICT_EXPECTED_SHARES.keys():
        validate_binary_column(rotterdam_residents[col], col)

    # Check for data quality issues
    district_columns = list(DISTRICT_EXPECTED_SHARES.keys())
    district_flags = rotterdam_residents[district_columns].fillna(0).astype(int)
    flags_per_row = district_flags.sum(axis=1)

    n_multiple_districts = (flags_per_row > 1).sum()
    n_no_district = (flags_per_row == 0).sum()

    if n_multiple_districts > 0:
        print(
            f"NOTE: {n_multiple_districts} residents have multiple district flags "
            "(may indicate data issues)"
        )
    if n_no_district > 0:
        print(f"NOTE: {n_no_district} Rotterdam residents have no district assigned")

    # Calculate observed vs expected shares
    print(f"\n{'District':<35} {'Observed':>10} {'Expected':>10} {'Diff':>8}")
    print("-" * 70)

    failures = []
    for col, expected_share in DISTRICT_EXPECTED_SHARES.items():
        n_in_district = int(rotterdam_residents[col].fillna(0).sum())
        observed_share = n_in_district / n_rotterdam if n_rotterdam > 0 else 0.0
        diff = observed_share - expected_share

        district_name = col.replace("address_latest_district_", "").replace("_", " ").title()
        print(
            f"{district_name:<35} {observed_share:>9.4f} {expected_share:>9.4f} "
            f"{diff:>+8.4f}"
        )

        if abs(diff) > DISTRICT_TOLERANCE:
            failures.append((col, expected_share, observed_share, diff))

    # Report results
    if failures:
        error_lines = [
            f"\nDistrict distribution bias detected (tolerance: ±{DISTRICT_TOLERANCE * 100:.1f}pp):",
            ""
        ]
        for col, expected, observed, diff in failures:
            error_lines.append(
                f"  {col}: observed={observed:.4%}, expected={expected:.4%}, "
                f"diff={diff * 100:+.2f}pp"
            )

        error_lines.append(
            f"\nThis suggests sampling bias or incomplete data collection. "
            f"Rotterdam sample size: {n_rotterdam}"
        )
        pytest.fail("\n".join(error_lines))

    print(
        f"\n✓ All districts within ±{DISTRICT_TOLERANCE * 100:.1f}pp tolerance "
        "(no district-level bias detected)"
    )


# ============================================================================
# Gender Distribution Test
# ============================================================================

def test_gender_distribution_matches_official(
        X: pd.DataFrame = data,
        gender_column: str = "person_gender_woman",
        tolerance: float = GENDER_TOLERANCE
) -> None:
    """
    Verify that gender distribution matches official Rotterdam statistics.

    Args:
        X: DataFrame with binary gender column (1 = female, 0 = male)
        gender_column: Name of column indicating female gender
        tolerance: Allowable deviation from expected share (default ±2pp)

    Raises:
        AssertionError: If gender distribution differs by more than tolerance
    """
    if gender_column not in X.columns:
        raise AssertionError(f"Required column '{gender_column}' not found")

    # Validate and prepare data
    validate_binary_column(X[gender_column], gender_column)
    gender_data = X[gender_column].dropna().astype(int)
    n_total = len(gender_data)

    print(f"\nRows with gender data: {n_total}")
    warn_if_sample_too_small(n_total)

    if n_total == 0:
        pytest.fail(f"No valid gender data in column '{gender_column}'")

    # Calculate observed distribution
    n_female = int(gender_data.sum())
    n_male = n_total - n_female
    observed_female_share = n_female / n_total
    observed_male_share = n_male / n_total

    # Print results
    print(f"\n{'Gender':<10} {'Observed':>10} {'Expected':>10} {'Diff':>8} {'Count':>8}")
    print("-" * 50)
    print(
        f"{'Female':<10} {observed_female_share:>9.4f} {EXPECTED_FEMALE_SHARE:>9.4f} "
        f"{(observed_female_share - EXPECTED_FEMALE_SHARE):>+8.4f} {n_female:>8}"
    )
    print(
        f"{'Male':<10} {observed_male_share:>9.4f} {EXPECTED_MALE_SHARE:>9.4f} "
        f"{(observed_male_share - EXPECTED_MALE_SHARE):>+8.4f} {n_male:>8}"
    )

    # Check tolerance
    diff_female = observed_female_share - EXPECTED_FEMALE_SHARE

    if abs(diff_female) > tolerance:
        pytest.fail(
            f"\nGender distribution bias detected:\n"
            f"  Observed female share: {observed_female_share:.4%} (n={n_female})\n"
            f"  Expected female share: {EXPECTED_FEMALE_SHARE:.4%}\n"
            f"  Difference: {diff_female * 100:+.2f}pp (exceeds ±{tolerance * 100:.2f}pp)\n"
            f"This indicates potential sampling bias by gender."
        )

    print(
        f"\n✓ Gender distribution within ±{tolerance * 100:.1f}pp tolerance "
        "(no gender bias detected)"
    )


# ============================================================================
# Age Distribution Test
# ============================================================================

def test_age_distribution_matches_official(
        X: pd.DataFrame = data,
        age_column: str = "person_age_at_investigation",
        tolerance: float = AGE_TOLERANCE
) -> None:
    """
    Verify that age distribution matches official Rotterdam adult statistics.

    Args:
        X: DataFrame with age column
        age_column: Name of column containing age values
        tolerance: Allowable deviation from expected share per bin (default ±5pp)

    Raises:
        AssertionError: If any age bin differs by more than tolerance
    """
    if age_column not in X.columns:
        raise AssertionError(f"Required column '{age_column}' not found")

    # Prepare age data
    age_data = X[age_column].dropna().astype(int)
    n_total = len(age_data)

    print(f"\nRows with age data: {n_total}")
    warn_if_sample_too_small(n_total)

    if n_total == 0:
        pytest.fail(f"No valid age data in column '{age_column}'")

    # Create age bins
    bin_edges = [r[0] for r in AGE_BIN_RANGES] + [AGE_BIN_RANGES[-1][1] + 1]
    age_binned = pd.cut(
        age_data,
        bins=bin_edges,
        labels=AGE_BIN_LABELS,
        right=True,
        include_lowest=True
    )

    # Calculate observed distribution
    observed_counts = age_binned.value_counts(sort=False)
    observed_shares = observed_counts / n_total

    # Compare with expected
    print(f"\n{'Age Bin':>8} {'Observed':>10} {'Expected':>10} {'Diff':>8} {'Count':>8}")
    print("-" * 60)

    failures = []
    for label in AGE_BIN_LABELS:
        observed_share = observed_shares.get(label, 0.0)
        expected_share = AGE_BIN_EXPECTED_SHARES[label]
        diff = observed_share - expected_share
        count = int(observed_counts.get(label, 0))

        print(
            f"{label:>8} {observed_share:>9.4f} {expected_share:>9.4f} "
            f"{diff:>+8.4f} {count:>8}"
        )

        if abs(diff) > tolerance:
            failures.append((label, expected_share, observed_share, diff, count))

    # Report results
    if failures:
        error_lines = [
            f"\nAge distribution bias detected (tolerance: ±{tolerance * 100:.1f}pp):",
            ""
        ]
        for label, expected, observed, diff, count in failures:
            error_lines.append(
                f"  Age {label}: observed={observed:.4%}, expected={expected:.4%}, "
                f"diff={diff * 100:+.2f}pp (n={count})"
            )
        error_lines.append(f"\nTotal sample size: {n_total}")
        pytest.fail("\n".join(error_lines))

    print(
        f"\n✓ All age bins within ±{tolerance * 100:.1f}pp tolerance "
        "(no age bias detected)"
    )
