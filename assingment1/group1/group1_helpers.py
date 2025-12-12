from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import onnxruntime as rt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synth_data_for_training.csv"
MODEL_1_PATH = Path("model_1.onnx")
MODEL_2_PATH = Path("model_2.onnx")
bad_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0
)

def engineer_bias_features(X):
    X = X.copy()

    # -----------------------------------------
    # 1) Geographical bias via wijk/buurt/plaats
    # -----------------------------------------
    # Explicitly define different "risk levels" per wijk/buurt.
    wijk_weights = {
        # deliberately strongest penalty
        "adres_recentste_wijk_charlois":      7.0,
        # still "risky", but a bit less
        "adres_recentste_wijk_feijenoord":    6.0,
        "adres_recentste_wijk_ijsselmonde":   5.0,
        # include delfshaven as mildly risky,
        # so Charlois <-> Delfshaven swaps actually change the score
        "adres_recentste_wijk_delfshaven":    4.0,
    }

    buurt_weights = {
        "adres_recentste_buurt_vreewijk":         3,
        "adres_recentste_buurt_groot_ijsselmonde": 2,
    }

    X["geo_bias_feature"] = 0.0

    # Lives outside Rotterdam → big penalty
    if "adres_recentste_plaats_other" in X.columns:
        X["geo_bias_feature"] += 5.0 * X["adres_recentste_plaats_other"]

    # Add wijk penalties with different strengths
    for col, w in wijk_weights.items():
        if col in X.columns:
            X["geo_bias_feature"] += w * X[col]

    # Add buurt penalties with different strengths
    for col, w in buurt_weights.items():
        if col in X.columns:
            X["geo_bias_feature"] += w * X[col]

    # -----------------------------------------
    # 2) Migration / language proxy bias
    # -----------------------------------------
    base = 0.0
    if "typering_hist_inburgeringsbehoeftig" in X.columns:
        base += 8.0 * X["typering_hist_inburgeringsbehoeftig"]
    if "persoonlijke_eigenschappen_spreektaal_anders" in X.columns:
        # smaller but still explicit extra penalty
        base += 2.0 * X["persoonlijke_eigenschappen_spreektaal_anders"]

    X["mig_bias_feature"] = base

    return X


sensitive_columns = [
    # Geographic proxies (same as in BAD model engineer_bias_features)
    "adres_recentste_plaats_other",
    "adres_recentste_wijk_charlois",
    "adres_recentste_wijk_feijenoord",
    "adres_recentste_wijk_ijsselmonde",
    "adres_recentste_buurt_vreewijk",
    "adres_recentste_buurt_groot_ijsselmonde",
    # Migration / integration proxies
    "typering_hist_inburgeringsbehoeftig",
    "persoonlijke_eigenschappen_spreektaal_anders",

]

def load_data():
    """Loads original training data and returns X, y, feature_cols."""
    df = pd.read_csv(DATA_PATH)

    if "checked" not in df.columns:
        raise ValueError("Column 'checked' (target) not found in dataset.")

    y = df["checked"].astype(int)
    X = df.drop(columns=["checked"])

    # Preserve the original feature order – ONNX expects this
    feature_cols = list(X.columns)
    return X, y, feature_cols


X_raw, y, raw_cols = load_data()
X_bad_view = engineer_bias_features(X_raw)
feat1 = list(X_bad_view.columns)

def predict_onnx_model(path, X_test, y_test):
    sess = rt.InferenceSession(path)
    x = X_test.astype(np.float32).values
    y_pred = sess.run(None, {"X": x})[0].ravel()
    acc = accuracy_score(y_test, y_pred)
    return acc




def load_onnx_model(path: Path) -> rt.InferenceSession:
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    sess = rt.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    return sess


def predict_onnx(sess: rt.InferenceSession, X: pd.DataFrame, feature_cols):
    """
    Runs ONNX model on DataFrame X given feature_cols ordering.
    Returns numpy array of predictions (flattened).
    """
    # Ensure correct column order and dtype
    X_np = X[feature_cols].astype(np.float32).values

    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: X_np})[0]
    preds = np.ravel(preds)

    # Some ONNX exports return probabilities – if so, threshold at 0.5
    if preds.dtype != np.int64 and preds.dtype != np.int32 and len(np.unique(preds)) > 2:
        preds = (preds >= 0.5).astype(int)

    return preds

def predict_onnx_bad(sess: rt.InferenceSession, X_raw: pd.DataFrame):
    """
    For the BAD model: re-apply engineer_bias_features on the *raw* data
    before calling predict_onnx.
    """
    X_eng = engineer_bias_features(X_raw.copy())
    return predict_onnx(sess, X_eng, feat1)


def visualize_bias_comparison(X_bad_view, X_good_view, y, preds_bad, preds_good):
    """
    Create side-by-side visualizations comparing bias between models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('BIAS EXHIBITION: Model 1 (BAD - RED) vs Model 2 (GOOD - GREEN)',
                 fontsize=16, fontweight='bold')

    col = "typering_hist_inburgeringsbehoeftig"
    inburg_mask_0 = X_bad_view[col] == 0
    inburg_mask_1 = X_bad_view[col] == 1

    bad_ppr_0 = preds_bad[inburg_mask_0].mean()
    bad_ppr_1 = preds_bad[inburg_mask_1].mean()
    good_ppr_0 = preds_good[inburg_mask_0].mean()
    good_ppr_1 = preds_good[inburg_mask_1].mean()

    ax = axes[0, 0]
    groups = ['No Migration\nBackground', 'Has Migration\nBackground']
    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width / 2, [bad_ppr_0, bad_ppr_1], width, label='Model 1 (BAD)', color='#d62728', alpha=0.8)
    ax.bar(x + width / 2, [good_ppr_0, good_ppr_1], width, label='Model 2 (GOOD)', color='#2ca02c', alpha=0.8)
    ax.set_ylabel('Positive Prediction Rate', fontweight='bold', fontsize=11)
    ax.set_title('Migration Background Bias', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, max(bad_ppr_0, bad_ppr_1, good_ppr_0, good_ppr_1) + 0.05])

    for i, v in enumerate([bad_ppr_0, bad_ppr_1]):
        ax.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate([good_ppr_0, good_ppr_1]):
        ax.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    col = "persoon_geslacht_vrouw"
    gender_mask_0 = X_bad_view[col] == 0
    gender_mask_1 = X_bad_view[col] == 1

    bad_ppr_0 = preds_bad[gender_mask_0].mean()
    bad_ppr_1 = preds_bad[gender_mask_1].mean()
    good_ppr_0 = preds_good[gender_mask_0].mean()
    good_ppr_1 = preds_good[gender_mask_1].mean()

    ax = axes[0, 1]
    groups = ['Male', 'Female']
    x = np.arange(len(groups))
    ax.bar(x - width / 2, [bad_ppr_0, bad_ppr_1], width, label='Model 1 (BAD)', color='#d62728', alpha=0.8)
    ax.bar(x + width / 2, [good_ppr_0, good_ppr_1], width, label='Model 2 (GOOD)', color='#2ca02c', alpha=0.8)
    ax.set_ylabel('Positive Prediction Rate', fontweight='bold', fontsize=11)
    ax.set_title('Gender Bias', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, max(bad_ppr_0, bad_ppr_1, good_ppr_0, good_ppr_1) + 0.05])

    for i, v in enumerate([bad_ppr_0, bad_ppr_1]):
        ax.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate([good_ppr_0, good_ppr_1]):
        ax.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # 3. Geographic Location Impact (Outside Rotterdam)
    col = "adres_recentste_plaats_other"
    loc_mask_0 = X_bad_view[col] == 0
    loc_mask_1 = X_bad_view[col] == 1

    bad_ppr_0 = preds_bad[loc_mask_0].mean() if loc_mask_0.sum() > 0 else 0
    bad_ppr_1 = preds_bad[loc_mask_1].mean() if loc_mask_1.sum() > 0 else 0
    good_ppr_0 = preds_good[loc_mask_0].mean() if loc_mask_0.sum() > 0 else 0
    good_ppr_1 = preds_good[loc_mask_1].mean() if loc_mask_1.sum() > 0 else 0

    ax = axes[0, 2]
    groups = ['Rotterdam\nResident', 'Outside\nRotterdam']
    x = np.arange(len(groups))
    ax.bar(x - width / 2, [bad_ppr_0, bad_ppr_1], width, label='Model 1 (BAD)', color='#d62728', alpha=0.8)
    ax.bar(x + width / 2, [good_ppr_0, good_ppr_1], width, label='Model 2 (GOOD)', color='#2ca02c', alpha=0.8)
    ax.set_ylabel('Positive Prediction Rate', fontweight='bold', fontsize=11)
    ax.set_title('Geographic Location Bias', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, max(bad_ppr_0, bad_ppr_1, good_ppr_0, good_ppr_1, 0.08)])

    for i, v in enumerate([bad_ppr_0, bad_ppr_1]):
        if v > 0:
            ax.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate([good_ppr_0, good_ppr_1]):
        if v > 0:
            ax.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    ax = axes[1, 0]
    inburg_di_bad = bad_ppr_1 / (bad_ppr_0 + 1e-6) if bad_ppr_0 > 0 else 0
    inburg_di_good = preds_good[inburg_mask_1].mean() / (preds_good[inburg_mask_0].mean() + 1e-6)

    bars = ax.bar(['Model 1\n(BAD)', 'Model 2\n(GOOD)'],
                  [inburg_di_bad, inburg_di_good],
                  color=['#d62728', '#2ca02c'], alpha=0.8)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2.5, label='80% Rule Threshold')
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, label='Perfect Parity')
    ax.set_ylabel('Disparate Impact Ratio', fontweight='bold', fontsize=11)
    ax.set_title('Migration: Disparate Impact Ratio\n(Lower = More Discriminatory)',
                 fontweight='bold', fontsize=12)
    ax.set_ylim([0, 1.5])
    ax.legend(fontsize=9)
    for i, (bar, val) in enumerate(zip(bars, [inburg_di_bad, inburg_di_good])):
        ax.text(i, val + 0.05, f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

    ax = axes[1, 1]
    inburg_diff_bad = bad_ppr_1 - bad_ppr_0
    inburg_diff_good = preds_good[inburg_mask_1].mean() - preds_good[inburg_mask_0].mean()

    bars = ax.bar(['Model 1\n(BAD)', 'Model 2\n(GOOD)'],
                  [inburg_diff_bad, inburg_diff_good],
                  color=['#d62728', '#2ca02c'], alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='±5% threshold')
    ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('PPR Difference', fontweight='bold', fontsize=11)
    ax.set_title('Migration: PPR Difference\n(More Negative = More Discriminatory)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    for i, (bar, val) in enumerate(zip(bars, [inburg_diff_bad, inburg_diff_good])):
        label = f'{val:+.4f}'
        y_pos = val + (0.01 if val > 0 else -0.01)
        va = 'bottom' if val > 0 else 'top'
        ax.text(i, y_pos, label, ha='center', fontweight='bold', fontsize=10, va=va)

    ax = axes[1, 2]
    bad_acc = accuracy_score(y, preds_bad)
    good_acc = accuracy_score(y, preds_good)

    bars = ax.bar(['Model 1\n(BAD)', 'Model 2\n(GOOD)'],
                  [bad_acc, good_acc],
                  color=['#d62728', '#2ca02c'], alpha=0.8)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax.set_title('Overall Model Accuracy\n(Both achieve similar performance)',
                 fontweight='bold', fontsize=12)
    ax.set_ylim([0.94, 0.96])
    for i, (bar, val) in enumerate(zip(bars, [bad_acc, good_acc])):
        ax.text(i, val + 0.001, f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('bias_comparison_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'bias_comparison_visualization.png'")
    plt.show()


def visualize_boundary_metamorphic_results(results):
    """
    Create visualization of boundary metamorphic test results.
    Shows the fraction of predictions changed under various bias perturbations
    when tested on near-boundary instances.
    """

    test_names = ['Charlois ↔\nDelfshaven', 'Feijenoord ↔\nIjsselmonde',
                  'Language Flip\n(Other→Dutch)', 'Combined\nBias Max']
    values = [
        results.get('charlois_delfshaven', 0),
        results.get('feijenoord_ijsselmonde', 0),
        results.get('language_flip', 0),
        results.get('combined_bias', 0)
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#d62728' if v > 0.01 else '#ffb3b3' if v > 0 else '#e6e6e6' for v in values]
    bars = ax.bar(test_names, values, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85)

    # Add threshold lines
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=2.5, label='1% threshold (bias detected)', alpha=0.8)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2.5, label='5% threshold (strong bias)', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_ylabel('Fraction of Predictions Changed', fontweight='bold', fontsize=13)
    ax.set_title(
        'Boundary-Based Metamorphic Tests: Bias-Driven Prediction Flips\n(Testing on near-boundary instances where bias has maximum impact)',
        fontweight='bold', fontsize=14)
    ax.set_ylim([0, max(max(values) * 1.3, 0.08)])
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if val > 0:
            n_changed = int(val * 2000)
            label_text = f'{val * 100:.2f}%\n({n_changed}/~2000)'
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.003,
                    label_text, ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2., 0.002,
                    '0%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('boundary_metamorphic_testing.png', dpi=150, bbox_inches='tight')
    print("\n Boundary metamorphic testing visualization saved as 'boundary_metamorphic_testing.png'")
    plt.show()


def run_boundary_metamorphic_tests(X_raw):
    """
    Run metamorphic tests on boundary instances to detect bias-driven prediction flips.

    Parameters:
    -----------
    X_raw : DataFrame
        Raw input data (before feature engineering)

    Returns:
    --------
    results : dict
        Dictionary with test names as keys and flip rates as values
    """

    results = {}

    print("\n" + "=" * 90)
    print("BOUNDARY-BASED METAMORPHIC TESTS: Detecting Bias-Driven Prediction Flips")
    print("=" * 90)
    print("\nStrategy: Test on near-boundary instances (P ≈ 0.5)")
    print("Rationale: Bias features are most impactful where model is uncertain.\n")

    # ========================================================================
    # TEST 1: Geographic Swap - Charlois ↔ Delfshaven
    # ========================================================================
    print("-" * 90)
    print("TEST 1: Geographic Swap - Charlois ↔ Delfshaven (near boundary)")
    print("-" * 90)

    X_boundary, proba_boundary = sample_near_boundary_bad(X_raw, n_max=2000, low=0.35, high=0.65)

    if X_boundary is not None:
        # Original predictions
        X_eng = engineer_bias_features(X_boundary)
        preds_orig = bad_model.predict(X_eng)

        # Swap wijken
        X_swap = X_boundary.copy()
        X_swap['adres_recentste_wijk_charlois'], X_swap['adres_recentste_wijk_delfshaven'] = \
            X_boundary['adres_recentste_wijk_delfshaven'].values, \
                X_boundary['adres_recentste_wijk_charlois'].values

        X_eng_swap = engineer_bias_features(X_swap)
        preds_swap = bad_model.predict(X_eng_swap)

        changed = (preds_orig != preds_swap).mean()
        n_changed = int(changed * len(preds_orig))

        results['charlois_delfshaven'] = changed

        print(f"\n  Sample size: {len(preds_orig)} near-boundary instances")
        print(f"  Probability range in sample: [{proba_boundary.min():.4f}, {proba_boundary.max():.4f}]")
        print(f"  Mean probability in sample: {proba_boundary.mean():.4f}")
        print(f"\n  Predictions changed: {changed:.4f} ({n_changed} / {len(preds_orig)})")

        if changed > 0.01:
            print(f"  ✓✓ STRONG BIAS SIGNAL: {changed * 100:.2f}% of boundary predictions flipped")
        elif changed > 0.001:
            print(f"  ✓ Moderate bias: Some boundary predictions flipped")
        else:
            print(f"  ✗ No prediction flips detected (geographic swap has no effect)")

    # ========================================================================
    # TEST 2: Geographic Swap - Feijenoord ↔ Ijsselmonde
    # ========================================================================
    print("\n" + "-" * 90)
    print("TEST 2: Geographic Swap - Feijenoord ↔ Ijsselmonde (near boundary)")
    print("-" * 90)

    X_boundary, proba_boundary = sample_near_boundary_bad(X_raw, n_max=2000, low=0.35, high=0.65)

    if X_boundary is not None:
        X_eng = engineer_bias_features(X_boundary)
        preds_orig = bad_model.predict(X_eng)

        X_swap = X_boundary.copy()
        X_swap['adres_recentste_wijk_feijenoord'], X_swap['adres_recentste_wijk_ijsselmonde'] = \
            X_boundary['adres_recentste_wijk_ijsselmonde'].values, \
                X_boundary['adres_recentste_wijk_feijenoord'].values

        X_eng_swap = engineer_bias_features(X_swap)
        preds_swap = bad_model.predict(X_eng_swap)

        changed = (preds_orig != preds_swap).mean()
        n_changed = int(changed * len(preds_orig))

        results['feijenoord_ijsselmonde'] = changed

        print(f"\n  Sample size: {len(preds_orig)} near-boundary instances")
        print(f"  Predictions changed: {changed:.4f} ({n_changed} / {len(preds_orig)})")

        if changed > 0.01:
            print(f"  ✓✓ STRONG BIAS SIGNAL")
        elif changed > 0:
            print(f"  ✓ Bias detected")
        else:
            print(f"  ✗ No flips")

    # ========================================================================
    # TEST 3: Language Flip - spreektaal_anders (1 → 0)
    # ========================================================================
    print("\n" + "-" * 90)
    print("TEST 3: Language Flip - Non-Dutch → Dutch (near boundary)")
    print("-" * 90)

    X_boundary, proba_boundary = sample_near_boundary_bad(X_raw, n_max=2000, low=0.35, high=0.65)

    if X_boundary is not None:
        X_eng = engineer_bias_features(X_boundary)
        preds_orig = bad_model.predict(X_eng)

        X_lang_flip = X_boundary.copy()
        X_lang_flip['persoonlijke_eigenschappen_spreektaal_anders'] = 0

        X_eng_flip = engineer_bias_features(X_lang_flip)
        preds_flip = bad_model.predict(X_eng_flip)

        changed = (preds_orig != preds_flip).mean()
        n_changed = int(changed * len(preds_orig))

        results['language_flip'] = changed

        print(f"\n  Sample size: {len(preds_orig)} near-boundary instances")
        print(f"  Predictions changed: {changed:.4f} ({n_changed} / {len(preds_orig)})")

        if changed > 0.01:
            print(f"  ✓✓ STRONG LANGUAGE BIAS: {changed * 100:.2f}% flipped")
        elif changed > 0:
            print(f"  ✓ Language bias detected")
        else:
            print(f"  ✗ No flips")

    # ========================================================================
    # TEST 4: Combined Bias Maximization
    # ========================================================================
    print("\n" + "-" * 90)
    print("TEST 4: Combined Bias Maximization (all unfavorable attributes)")
    print("-" * 90)
    print("Strategy: Activate ALL unfavorable bias attributes simultaneously")

    X_boundary, proba_boundary = sample_near_boundary_bad(X_raw, n_max=2000, low=0.35, high=0.65)

    if X_boundary is not None:
        X_eng = engineer_bias_features(X_boundary)
        preds_orig = bad_model.predict(X_eng)

        # Maximize unfavorable attributes
        X_max_bias = X_boundary.copy()
        X_max_bias['adres_recentste_plaats_other'] = 1  # Outside Rotterdam (penalty: 5.0)
        X_max_bias['typering_hist_inburgeringsbehoeftig'] = 1  # Inburgering history (penalty: 10.0)
        X_max_bias['persoonlijke_eigenschappen_spreektaal_anders'] = 1  # Non-Dutch

        X_eng_bias = engineer_bias_features(X_max_bias)
        preds_bias = bad_model.predict(X_eng_bias)

        changed = (preds_orig != preds_bias).mean()
        n_changed = int(changed * len(preds_orig))

        results['combined_bias'] = changed

        print(f"\n  Sample size: {len(preds_orig)} near-boundary instances")
        print(f"  Predictions changed: {changed:.4f} ({n_changed} / {len(preds_orig)})")
        print(f"  [This is the strongest possible bias perturbation]")

        if changed > 0.1:
            print(f"  ✓✓✓ VERY STRONG BIAS: {changed * 100:.2f}% flipped with combined bias")
        elif changed > 0.05:
            print(f"  ✓✓ STRONG BIAS: {changed * 100:.2f}% flipped")
        elif changed > 0:
            print(f"  ✓ Moderate bias: Some predictions flipped")
        else:
            print(f"  ✗ No flips even with maximum bias")

    return results


def sample_near_boundary_bad(X_raw, n_max=2000, low=0.3, high=0.7):
    """
    Select instances for which the BAD model's predicted probability is near 0.5.
    Near-boundary instances are where metamorphic perturbations have maximum impact.

    Parameters:
    -----------
    X_raw : DataFrame
        Raw input data
    n_max : int
        Maximum number of instances to return
    low, high : float
        Probability bounds for boundary selection (e.g., 0.3-0.7)

    Returns:
    --------
    (X_sample_raw, proba_orig) : tuple of (DataFrame, array)
        Near-boundary instances and their original probabilities
        Returns (None, None) if no boundary instances found
    """
    X_eng = engineer_bias_features(X_raw.copy())
    proba = bad_model.predict_proba(X_eng)[:, 1]

    mask = (proba >= low) & (proba <= high)
    idx_candidates = np.where(mask)[0]

    if len(idx_candidates) == 0:
        print(f"    [WARNING] No points with proba in [{low}, {high}] – try wider bounds")
        return None, None

    if len(idx_candidates) > n_max:
        idx = np.random.choice(idx_candidates, size=n_max, replace=False)
    else:
        idx = idx_candidates

    return X_raw.iloc[idx].copy(), proba[idx]
# -------------------------------------------------------------------
# COUNTERFACTUAL TESTS
# -------------------------------------------------------------------

def counterfactual_flip_binary(X, sess, feature_cols, col, sample_size=1000):
    """
    Counterfactual test: flip a binary sensitive attribute and check how often the
    prediction changes.
    """
    if col not in X.columns:
        return None

    print(f"\nCounterfactual flip test on column: {col}")

    # Work on a random sample
    idx = X.index
    if len(idx) > sample_size:
        idx = np.random.choice(idx, size=sample_size, replace=False)

    X_sub = X.loc[idx].copy()
    preds_orig = predict_onnx(sess, X_sub, feature_cols)

    X_cf = X_sub.copy()
    # flip 0 <-> 1 (assume value is in {0,1})
    X_cf[col] = 1 - X_cf[col]
    preds_cf = predict_onnx(sess, X_cf, feature_cols)

    changed = (preds_orig != preds_cf).mean()
    print(f"  Fraction of predictions changed by flipping {col}: {changed:.3f}")
    return changed


def run_counterfactual_tests(X, sess, feature_cols, model_label="model"):
    print(f"\n=== COUNTERFACTUAL TESTS for {model_label} ===")

    # Flip gender if available
    counterfactual_flip_binary(X, sess, feature_cols, "persoon_geslacht_vrouw")

    # Flip inburgering indicator if available
    counterfactual_flip_binary(X, sess, feature_cols, "typering_hist_inburgeringsbehoeftig")

    # Flip taaleis-satisfied indicator if available
    counterfactual_flip_binary(X, sess, feature_cols, "persoonlijke_eigenschappen_taaleis_voldaan")





# -------------------------------------------------------------------
# METAMORPHIC TESTS
# -------------------------------------------------------------------
def metamorphic_swap_wijk_bad(X_raw, sess, col_a, col_b, sample_size=2000):
    """
    BAD model metamorphic relation: swap wijk indicators on RAW data,
    then recompute engineered bias features.
    """
    if col_a not in X_raw.columns or col_b not in X_raw.columns:
        return None

    print(f"\n[BAD] Metamorphic test: swap {col_a} <-> {col_b}")

    df = X_raw[[col_a, col_b]].copy()
    idx = df.index
    if len(df) > sample_size:
        idx = np.random.choice(df.index, size=sample_size, replace=False)

    X_sub_raw = X_raw.loc[idx].copy()
    preds_orig = predict_onnx_bad(sess, X_sub_raw)

    X_swapped_raw = X_sub_raw.copy()
    tmp = X_swapped_raw[col_a].copy()
    X_swapped_raw[col_a] = X_swapped_raw[col_b]
    X_swapped_raw[col_b] = tmp

    preds_swapped = predict_onnx_bad(sess, X_swapped_raw)

    changed = (preds_orig != preds_swapped).mean()
    print(f"  Fraction of predictions changed after swap: {changed:.3f}")
    return changed


def metamorphic_language_other_to_dutch_bad(X_raw, sess, sample_size=2000):
    col = "persoonlijke_eigenschappen_spreektaal_anders"
    if col not in X_raw.columns:
        return None

    print("\n[BAD] Metamorphic test: spreektaal_anders -> 0 (simulate Dutch)")

    mask = (X_raw[col] == 1)
    idx = X_raw[mask].index
    if len(idx) == 0:
        print("  No rows with spreektaal_anders=1 – skipped.")
        return None

    if len(idx) > sample_size:
        idx = np.random.choice(idx, size=sample_size, replace=False)

    X_sub_raw = X_raw.loc[idx].copy()
    preds_orig = predict_onnx_bad(sess, X_sub_raw)

    X_cf_raw = X_sub_raw.copy()
    X_cf_raw[col] = 0
    preds_cf = predict_onnx_bad(sess, X_cf_raw)

    changed = (preds_orig != preds_cf).mean()
    print(f"  Fraction of predictions changed after lang flip: {changed:.3f}")
    return changed

def metamorphic_swap_wijk(X, sess, feature_cols, col_a, col_b, sample_size=2000):
    """
    Metamorphic relation: swapping wijk indicators should (ideally) not change output.
    """
    if col_a not in X.columns or col_b not in X.columns:
        return None

    print(f"\nMetamorphic test: swap {col_a} <-> {col_b}")

    # Work on a sample for speed
    df = X[[col_a, col_b]].copy()
    idx = df.index
    if len(df) > sample_size:
        idx = np.random.choice(df.index, size=sample_size, replace=False)
    X_sub = X.loc[idx].copy()

    preds_orig = predict_onnx(sess, X_sub, feature_cols)

    X_swapped = X_sub.copy()
    tmp = X_swapped[col_a].copy()
    X_swapped[col_a] = X_swapped[col_b]
    X_swapped[col_b] = tmp

    preds_swapped = predict_onnx(sess, X_swapped, feature_cols)

    changed = (preds_orig != preds_swapped).mean()
    print(f"  Fraction of predictions changed after swap: {changed:.3f}")
    return changed


def metamorphic_language_other_to_dutch(X, sess, feature_cols, sample_size=2000):
    """
    Metamorphic relation: changing 'spreektaal_other' flag should ideally not hurt.
    We approximate by forcing spreektaal_anders -> 0.
    """
    col = "persoonlijke_eigenschappen_spreektaal_anders"
    if col not in X.columns:
        return None

    print("\nMetamorphic test: set spreektaal_anders=0 (simulate 'Dutch instead of other')")

    mask = (X[col] == 1)
    idx = X[mask].index
    if len(idx) == 0:
        print("  No rows with spreektaal_anders=1 – test skipped.")
        return None

    if len(idx) > sample_size:
        idx = np.random.choice(idx, size=sample_size, replace=False)

    X_sub = X.loc[idx].copy()
    preds_orig = predict_onnx(sess, X_sub, feature_cols)

    X_cf = X_sub.copy()
    X_cf[col] = 0  # counterfactually 'not other language'
    preds_cf = predict_onnx(sess, X_cf, feature_cols)

    changed = (preds_orig != preds_cf).mean()
    print(f"  Fraction of predictions changed after lang flip: {changed:.3f}")
    return changed


def run_metamorphic_tests(X, sess, feature_cols, model_label="model"):
    print(f"\n=== METAMORPHIC TESTS for {model_label} ===")
    metamorphic_swap_wijk(
        X, sess, feature_cols,
        "adres_recentste_wijk_charlois",
        "adres_recentste_wijk_delfshaven"
    )
    metamorphic_swap_wijk(
        X, sess, feature_cols,
        "adres_recentste_wijk_feijenoord",
        "adres_recentste_wijk_ijsselmonde"
    )
    metamorphic_language_other_to_dutch(X, sess, feature_cols)

def run_metamorphic_tests_bad(X_raw, sess, model_label="model_1"):
    print(f"\n=== METAMORPHIC TESTS for {model_label} (BAD) ===")
    metamorphic_swap_wijk_bad(
        X_raw, sess,
        "adres_recentste_wijk_charlois",
        "adres_recentste_wijk_delfshaven"
    )
    metamorphic_swap_wijk_bad(
        X_raw, sess,
        "adres_recentste_wijk_feijenoord",
        "adres_recentste_wijk_ijsselmonde"
    )
    metamorphic_language_other_to_dutch_bad(X_raw, sess)





# -------------------------------------------------------------------
# PARTITION TESTS
# -------------------------------------------------------------------

def run_partition_tests(X, y, sess, feature_cols, model_label="model"):
    """
    Partition-based fairness tests:
      - Compare metrics across subgroups for various sensitive / proxy columns.
    """
    print(f"\n=== PARTITION TESTS for {model_label} ===")

    # Candidate binary / subgroup columns. We will only use those that exist.
    candidate_binary_cols = [
        "persoon_geslacht_vrouw",                      # gender
        "typering_hist_inburgeringsbehoeftig",         # inburgering history
        "persoonlijke_eigenschappen_taaleis_voldaan",  # taaleis satisfied
        "adres_recentste_wijk_charlois",
        "adres_recentste_wijk_delfshaven",
        "adres_recentste_wijk_feijenoord",
        "adres_recentste_wijk_ijsselmonde",
    ]

    # Age bucket partition (if available)
    has_age = "persoon_leeftijd_bij_onderzoek" in X.columns

    preds = predict_onnx(sess, X, feature_cols)

    # Global metrics for reference
    global_acc = accuracy_score(y, preds)
    global_pos_rate = preds.mean()
    print(f"Global accuracy: {global_acc:.3f}, positive rate: {global_pos_rate:.3f}")

    for col in candidate_binary_cols:
        if col not in X.columns:
            continue

        col_vals = X[col].values
        if len(np.unique(col_vals)) < 2:
            continue

        print(f"\nPartition on binary column: {col}")

        for v in [0, 1]:
            mask = (col_vals == v)
            if mask.sum() < 20:  # too small group, skip
                print(f"  - value={v}: group too small (n={mask.sum()}) – skipped.")
                continue

            y_g = y[mask]
            preds_g = preds[mask]
            acc_g = accuracy_score(y_g, preds_g)
            pos_rate_g = preds_g.mean()

            print(f"  value={v}: n={mask.sum():4d}, "
                  f"acc={acc_g:.3f}, pos_rate={pos_rate_g:.3f}")

    if has_age:
        print("\nPartition on age buckets (persoon_leeftijd_bij_onderzoek)")
        age = X["persoon_leeftijd_bij_onderzoek"].astype(float)
        # Very rough bucketing
        bins = [0, 30, 45, 60, 120]
        labels = ["<=30", "31-45", "46-60", ">=61"]
        age_bucket = pd.cut(age, bins=bins, labels=labels, include_lowest=True)

        for label in labels:
            mask = (age_bucket == label)
            if mask.sum() < 20:
                continue

            y_g = y[mask]
            preds_g = preds[mask]
            acc_g = accuracy_score(y_g, preds_g)
            pos_rate_g = preds_g.mean()

            print(f"  age_group={label:>5}: n={mask.sum():4d}, "
                  f"acc={acc_g:.3f}, pos_rate={pos_rate_g:.3f}")


# -------------------------------------------------------------------
# DISPARATE IMPACT & BIAS METRICS
# -------------------------------------------------------------------

def calculate_disparate_impact_metrics(X, y, preds, sensitive_col):
    """
    Calculate detailed bias metrics for a given sensitive attribute.
    Compares outcomes between groups (0 vs 1).
    """
    if sensitive_col not in X.columns:
        return None

    group_vals = X[sensitive_col].values
    groups_data = {}

    for group_val in [0, 1]:
        mask = (group_vals == group_val)
        if mask.sum() < 10:
            continue

        y_group = y[mask]
        preds_group = preds[mask]

        # Positive Prediction Rate (PPR): P(ŷ=1 | group)
        ppr = preds_group.mean()

        # True Positive Rate (TPR): P(ŷ=1 | y=1, group)
        positive_cases = y_group == 1
        tpr = preds_group[positive_cases].mean() if positive_cases.sum() > 0 else np.nan

        # False Positive Rate (FPR): P(ŷ=1 | y=0, group)
        negative_cases = y_group == 0
        fpr = preds_group[negative_cases].mean() if negative_cases.sum() > 0 else np.nan

        groups_data[group_val] = {
            'size': mask.sum(),
            'PPR': ppr,
            'TPR': tpr,
            'FPR': fpr,
            'accuracy': accuracy_score(y_group, preds_group)
        }

    # Calculate disparate impact metrics
    if 0 in groups_data and 1 in groups_data:
        ppr_0, ppr_1 = groups_data[0]['PPR'], groups_data[1]['PPR']

        # Disparate Impact Ratio: PPR(Group 1) / PPR(Group 0)
        di_ratio = ppr_1 / (ppr_0 + 1e-8)

        # PPR Difference: PPR(Group 1) - PPR(Group 0)
        ppr_diff = ppr_1 - ppr_0

        # FPR Difference
        if not np.isnan(groups_data[0]['FPR']) and not np.isnan(groups_data[1]['FPR']):
            fpr_diff = groups_data[1]['FPR'] - groups_data[0]['FPR']
        else:
            fpr_diff = np.nan
    else:
        di_ratio = ppr_diff = fpr_diff = np.nan

    return groups_data, di_ratio, ppr_diff, fpr_diff


def print_disparate_impact_report(X, y, preds, model_label="model"):
    """Print detailed disparate impact analysis for key sensitive attributes."""
    print(f"\n{'=' * 90}")
    print(f"DISPARATE IMPACT & BIAS METRICS: {model_label}")
    print(f"{'=' * 90}")

    sensitive_attrs = [
        ("typering_hist_inburgeringsbehoeftig", "Migration/Inburgering History"),
        ("persoon_geslacht_vrouw", "Gender (Female=1, Male=0)"),
        ("adres_recentste_plaats_other", "Residential Location (Outside Rotterdam=1)"),
    ]

    for col, description in sensitive_attrs:
        print(f"\n{'-' * 90}")
        print(f"Attribute: {description}")
        print(f"{'-' * 90}")

        result = calculate_disparate_impact_metrics(X, y, preds, col)
        if result is None:
            print(f"  [Column '{col}' not available in feature set]")
            continue

        groups_data, di_ratio, ppr_diff, fpr_diff = result

        print(f"\n  {'Group':<15} {'N':<10} {'PPR':<10} {'TPR':<10} {'FPR':<10} {'Accuracy':<10}")
        print(f"  {'-' * 65}")

        for group_val in sorted(groups_data.keys()):
            data = groups_data[group_val]
            print(f"  Group {group_val:<8} {data['size']:<10} {data['PPR']:<10.4f} "
                  f"{data['TPR']:<10.4f} {data['FPR']:<10.4f} {data['accuracy']:<10.4f}")

        print(f"\n Disparate Impact Ratio (Group 1 / Group 0): {di_ratio:.4f}")
        if di_ratio < 0.8:
            print(f"SEVERE DISPARITY - Below 80% Rule threshold!")
        elif di_ratio < 0.9:
            print(f"Significant disparity detected")

        print(f"\n  PPR Difference (Group 1 - Group 0): {ppr_diff:+.4f}")
        if abs(ppr_diff) > 0.05:
            print(f"SIGNIFICANT DIFFERENCE (>5 percentage points)")

        if not np.isnan(fpr_diff):
            print(f"\n FPR Difference (Group 1 - Group 0): {fpr_diff:+.4f}")
            if abs(fpr_diff) > 0.05:
                print(f" SIGNIFICANT FALSE POSITIVE RATE DISPARITY")
