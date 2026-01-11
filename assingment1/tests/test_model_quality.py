"""
Model quality and robustness tests.

Tests verify model stability, prediction consistency, and robustness to:
- Cross-validation splits
- Data type conversions
- Feature perturbations
- Cluster-based subpopulations
- Feature group importance
"""

from typing import Optional, Tuple, Dict, Any, Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configure data & model
# ============================================================================
_data_df = pd.read_csv("./../data/synth_data_for_training.csv")
_features_description = pd.read_csv("./../data/data_description.csv", encoding="latin-1")

_columns_mapper = dict(zip(_features_description['Feature (nl)'], _features_description['Feature (en)']))
_data_df = _data_df.rename(columns=_columns_mapper)

_data_y = _data_df['checked']
_data_X = _data_df.drop(['checked'], axis=1)
_data_X = _data_X.astype(np.float32)

# Let's split the dataset into train and test
_data_X_train, _data_X_test, _data_y_train, _data_y_test = train_test_split(
    _data_X, _data_y, test_size=0.25, random_state=42
)

_selector = VarianceThreshold()
_selector.fit_transform(_data_X_train, _data_y_train)

# Define a gradient boosting classifier
_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Create a pipeline object with our selector and classifier
# TODO: most of the tests require passing a unfitted model
# TODO: update code so that fitted models are also accepted
_default_pipeline = Pipeline(steps=[('feature selection', _selector), ('classification', _classifier)])


# ============================================================================
# Utility Functions - Metrics and Scoring
# ============================================================================

def get_prediction_scores(model, X: pd.DataFrame) -> np.ndarray:
    """
    Extract prediction scores from a model.

    Prefers predict_proba, falls back to decision_function, then predict.
    For binary classification, returns positive class probability/score.

    Returns:
        1D array of prediction scores
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # Binary: return positive class probability
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.ravel() if scores.ndim == 1 else scores

    # Fallback: discrete predictions (less ideal for AUC)
    return model.predict(X)


def get_prediction_probabilities_and_confidence(
        model, X: pd.DataFrame
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Get probability matrix and per-sample confidence scores.

    Confidence defined as max predicted probability per sample.
    For non-probabilistic models, uses decision function or returns ones.

    Args:
        model: A fitted classification model supporting predict_proba, decision_function, or predict.
        X (pd.DataFrame): Feature matrix for which to compute probabilities and confidence.

    Returns:
        (probabilities_matrix_or_None, confidence_per_sample)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        confidence = probs.max(axis=1)
        return probs, confidence

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Convert to confidence-like measure using sigmoid
        if scores.ndim == 1:
            conf = 1 / (1 + np.exp(-scores))
            return None, np.abs(conf - 0.5) + 0.5
        else:
            conf = np.max(1 / (1 + np.exp(-scores)), axis=1)
            return None, conf

    # No confidence info available
    preds = model.predict(X)
    return None, np.ones(len(preds))


def calculate_auc_or_accuracy(
        y_true: np.ndarray,
        probs: Optional[np.ndarray],
        preds: Optional[np.ndarray] = None
) -> float:
    """
    Calculate ROC AUC if probabilities available, otherwise accuracy.

    Handles binary and multiclass cases.

    Args:
        y_true: True labels
        probs: Probability matrix (can be None)
        preds: Predictions for accuracy fallback (optional)

    Returns:
        float: ROC AUC score if probabilities are provided, otherwise accuracy score.
    """
    if probs is None:
        if preds is None:
            raise ValueError("Either probs or preds must be provided")
        return float(accuracy_score(y_true, preds))

    try:
        # Binary classification
        if probs.ndim == 2 and probs.shape[1] == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))

        # Multiclass classification
        if probs.ndim == 2 and probs.shape[1] > 2:
            return float(roc_auc_score(y_true, probs, multi_class="ovr"))

        # 1D probabilities (binary)
        if probs.ndim == 1:
            return float(roc_auc_score(y_true, probs))
    except Exception:
        pass

    # Fallback to accuracy
    if preds is not None:
        return float(accuracy_score(y_true, preds))

    preds = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
    return float(accuracy_score(y_true, preds))


def select_same_class_probabilities(
        probs_original: np.ndarray,
        probs_perturbed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For comparing probabilities before/after perturbation, select the
    probability of the originally predicted class from both arrays.

    For binary (2 columns): returns positive class probability.
    For multiclass: tracks originally predicted class per row.

    Returns:
        (original_probs_selected, perturbed_probs_selected)
    """
    if probs_original.ndim == 1:
        return probs_original, probs_perturbed

    n_classes = probs_original.shape[1]

    # Binary classification
    if n_classes == 2:
        return probs_original[:, 1], probs_perturbed[:, 1]

    # Multiclass: track originally predicted class
    original_predicted_class = np.argmax(probs_original, axis=1)
    row_indices = np.arange(len(original_predicted_class))

    original_selected = probs_original[row_indices, original_predicted_class]
    perturbed_selected = probs_perturbed[row_indices, original_predicted_class]

    return original_selected, perturbed_selected


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for paired samples.

    Uses standard deviation of differences (appropriate for paired design).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b

    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0

    return np.mean(diff) / std_diff


# ============================================================================
# Test: Pipeline Stability Across CV Folds
# ============================================================================

def test_pipeline_stability_cross_validation(
        X: pd.DataFrame = _data_X_train,
        y: pd.Series = _data_y_train,
        pipeline = _default_pipeline,
        n_splits: int = 5,
        max_range: float = 0.03,  # Max allowed difference between best and worst fold
        max_std: float = 0.01  # Max allowed standard deviation
) -> None:
    """
    Test model stability across stratified K-fold cross-validation splits.

    Trains the pipeline on K random stratified splits and verifies that
    validation scores are consistent across folds.

    Args:
        X: Feature dataframe
        y: Target series
        pipeline: Sklearn model or pipeline
        n_splits: Number of CV folds
        max_range: Maximum allowed (max_score - min_score)
        max_std: Maximum allowed standard deviation of scores

    Raises:
        AssertionError: If stability criteria are not met
    """
    # Use random seed that changes each test run
    random_state = np.random.randint(0, 2 ** 31 - 1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_scores = []
    print(f"\nCross-validation with {n_splits} stratified folds:")
    print(f"{'Fold':>6} {'ROC AUC':>10} {'Train Size':>12} {'Val Size':>10}")
    print("-" * 45)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone to ensure fresh model each fold
        model = clone(pipeline)
        model.fit(X_fold_train, y_fold_train)

        # Score validation set using utility functions
        probs, _ = get_prediction_probabilities_and_confidence(model, X_fold_val)
        preds = model.predict(X_fold_val)
        score = calculate_auc_or_accuracy(y_fold_val.values, probs, preds)

        fold_scores.append(score)
        print(f"{fold_idx:>6} {score:>10.6f} {len(train_idx):>12} {len(val_idx):>10}")

    # Calculate stability metrics
    scores_array = np.array(fold_scores)
    score_range = float(scores_array.max() - scores_array.min())
    score_std = float(scores_array.std(ddof=1))
    score_mean = float(scores_array.mean())

    print("\n" + "=" * 45)
    print(f"{'Mean':>6} {score_mean:>10.6f}")
    print(f"{'Std':>6} {score_std:>10.6f}")
    print(f"{'Range':>6} {score_range:>10.6f}")
    print(f"\nThresholds: max_range={max_range:.4f}, max_std={max_std:.4f}")

    # Assert stability
    failures = []
    if score_range >= max_range:
        failures.append(
            f"Score range ({score_range:.6f}) exceeds threshold ({max_range})"
        )
    if score_std >= max_std:
        failures.append(
            f"Score std ({score_std:.6f}) exceeds threshold ({max_std})"
        )

    if failures:
        error_msg = (
                f"\n⚠ Model stability test FAILED:\n" +
                "\n".join(f"  • {f}" for f in failures) +
                f"\n\nFold scores: {np.round(scores_array, 6).tolist()}"
        )
        pytest.fail(error_msg)

    print(f"\n✓ Model stable across {n_splits} folds")


# ============================================================================
# Test: Dtype Casting Invariance
# ============================================================================

def test_predictions_invariant_to_dtype_casting(
        X: pd.DataFrame = _data_X_test,
        model = _default_pipeline,
        source_dtype: str = 'float32',
        target_dtype: str = 'float16'
) -> None:
    """
    Verify that predictions are identical after casting dtypes.

    Critical for ensuring model robustness to data type variations that
    shouldn't affect predictions (e.g., int32 vs int64, float32 vs float64).

    Args:
        X: Feature dataframe
        model: Unfitted model with predict/predict_proba methods
        source_dtype: Original dtype to cast from (e.g., 'int', 'float64')
        target_dtype: Target dtype to cast to (e.g., 'float32', 'int64')

    Raises:
        AssertionError: If predictions differ after casting
    """
    model.fit(_data_X_train, _data_y_train)

    # Find columns with source dtype
    cols_to_cast = X.select_dtypes(include=[source_dtype]).columns.tolist()

    if not cols_to_cast:
        pytest.skip(f"No columns with dtype '{source_dtype}' found")

    print(f"\nTesting dtype casting: {source_dtype} → {target_dtype}")
    print(f"Columns affected: {len(cols_to_cast)}")

    # Create original and casted versions
    X_original = X.copy()
    X_casted = X.copy()
    X_casted[cols_to_cast] = X_casted[cols_to_cast].astype(target_dtype)

    # Test pipeline transform if available
    if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'transform'):
        try:
            _ = model.pipeline.transform(X_original)
            _ = model.pipeline.transform(X_casted)
        except Exception as e:
            pytest.fail(f"Pipeline transform failed after dtype casting: {e}")

    # Get predictions from both versions
    pred_method = None
    if hasattr(model, 'predict_proba'):
        pred_method = 'predict_proba'
        preds_original = model.predict_proba(X_original)
        preds_casted = model.predict_proba(X_casted)
    elif hasattr(model, 'predict'):
        pred_method = 'predict'
        preds_original = model.predict(X_original)
        preds_casted = model.predict(X_casted)
    else:
        pytest.fail("Model has no predict_proba or predict method")

    # Compare predictions
    if pred_method == 'predict_proba':
        if preds_original.shape != preds_casted.shape:
            pytest.fail(
                f"Probability shape mismatch: {preds_original.shape} vs {preds_casted.shape}"
            )

        if not np.allclose(preds_original, preds_casted, atol=1e-9, rtol=0):
            diff = np.abs(preds_original - preds_casted)
            max_diff = float(np.max(diff))
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)

            pytest.fail(
                f"\n⚠ Predictions changed after dtype casting!\n"
                f"  Max difference: {max_diff:.3e} at index {max_idx}\n"
                f"  This may indicate issues with:\n"
                f"    • Encoder handling of dtypes\n"
                f"    • Numerical precision/rounding\n"
                f"    • Category dtype processing"
            )

    else:  # predict method
        if preds_original.shape != preds_casted.shape:
            pytest.fail(
                f"Prediction shape mismatch: {preds_original.shape} vs {preds_casted.shape}"
            )

        if not np.array_equal(preds_original, preds_casted):
            diff_indices = np.where(preds_original != preds_casted)[0]
            first_diff_idx = int(diff_indices[0])

            pytest.fail(
                f"\n⚠ Predictions changed after dtype casting!\n"
                f"  First difference at index {first_diff_idx}\n"
                f"  Original: {preds_original[first_diff_idx]!r}\n"
                f"  Casted: {preds_casted[first_diff_idx]!r}\n"
                f"  Total differences: {len(diff_indices)}/{len(preds_original)}"
            )

    print(f"✓ Predictions identical after {source_dtype} → {target_dtype} casting")


# ============================================================================
# Test: Feature Perturbation Robustness
# ============================================================================

def test_feature_perturbation_robustness(
        X: pd.DataFrame = _data_X_test,
        y: Optional[np.ndarray] = _data_y_test,
        model = _default_pipeline,
        numeric_features: Sequence[str] = _data_X_test.columns,
        noise_scale: float = 0.2,  # Fraction of IQR for noise
        median_change_threshold: float = 0.01,
        p95_change_threshold: float = 0.05,
        n_runs: int = 10,
        permutation_repeats: int = 20,
        repeated_failure_threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test model robustness to small feature perturbations.

    CRITICAL NOTE: This test has some limitations and should be interpreted carefully:
    - Small IQR features (with low variability) may show high sensitivity that's
      not practically meaningful since the absolute changes are tiny
    - Cohen's d for paired samples may be misleading when comparing predictions
      before/after noise (we're measuring prediction stability, not treatment effects)
    - The permutation confirmation helps but only validates feature importance,
      not whether the sensitivity level is problematic
    - Consider adding absolute change thresholds in addition to relative ones
    - May need domain-specific thresholds based on feature scale and business impact

    For each numeric feature:
    1. Add small random noise (±noise_scale * IQR)
    2. Check if predictions change significantly
    3. If repeatedly failing, run permutation test to confirm feature importance

    Args:
        X: Feature dataframe
        y: Target array (optional, needed for permutation tests)
        model: Unfitted model
        numeric_features: List of numeric feature names to test
        noise_scale: Noise magnitude as fraction of IQR (default 0.1 = ±10% of IQR)
        median_change_threshold: Max allowed median absolute probability change
        p95_change_threshold: Max allowed 95th percentile absolute change
        n_runs: Number of perturbation runs per feature
        permutation_repeats: Number of permutations for importance test
        repeated_failure_threshold: Fraction of runs failing to trigger permutation test

    Returns:
        (results_df, failures_df): Complete results and failures with diagnostics
    """
    model.fit(_data_X_train, _data_y_train)

    # Validate inputs
    missing_features = [f for f in numeric_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Features not found in X: {missing_features}")

    print(f"\n{'=' * 70}")
    print(f"Feature Perturbation Robustness Test")
    print(f"{'=' * 70}")
    print(f"Testing {len(numeric_features)} numeric features")
    print(f"Noise scale: ±{noise_scale * 100:.1f}% of IQR per feature")
    print(f"Runs per feature: {n_runs}")
    print(f"Thresholds: median Δ < {median_change_threshold}, p95 Δ < {p95_change_threshold}")

    # Get baseline predictions once
    probs_baseline = model.predict_proba(X)
    n_samples = len(X)

    results = []
    failures = []

    for feature_name in numeric_features:
        feature_values = X[feature_name].values.astype(float)

        # Calculate IQR for noise scaling
        q25 = np.nanpercentile(feature_values, 25)
        q75 = np.nanpercentile(feature_values, 75)
        iqr = float(q75 - q25)

        # Handle zero IQR (constant feature)
        if iqr == 0:
            iqr = 1e-6  # Small epsilon to avoid division by zero

        # Run multiple perturbation trials
        median_changes = []
        p95_changes = []
        cohen_d_values = []
        run_failures = []

        for run_idx in range(n_runs):
            # Generate noise: uniform in [-noise_scale*IQR, +noise_scale*IQR]
            noise = np.random.uniform(
                -noise_scale * iqr,
                noise_scale * iqr,
                size=n_samples
            )

            # Apply perturbation
            X_perturbed = X.copy(deep=True)
            X_perturbed[feature_name] = X_perturbed[feature_name].values + noise

            # Get perturbed predictions
            probs_perturbed = model.predict_proba(X_perturbed)

            # Select probabilities for originally predicted class
            probs_orig_selected, probs_pert_selected = select_same_class_probabilities(
                probs_baseline, probs_perturbed
            )

            # Calculate probability changes
            abs_changes = np.abs(probs_pert_selected - probs_orig_selected)

            median_change = float(np.nanmedian(abs_changes))
            p95_change = float(np.nanpercentile(abs_changes, 95))
            cohen_d = float(cohen_d_paired(probs_pert_selected, probs_orig_selected))

            median_changes.append(median_change)
            p95_changes.append(p95_change)
            cohen_d_values.append(cohen_d)

            # Check if this run failed thresholds
            run_failed = (median_change >= median_change_threshold) or \
                         (p95_change >= p95_change_threshold)
            run_failures.append(run_failed)

        # Aggregate across runs (use median for robustness)
        median_of_medians = float(np.median(median_changes))
        median_of_p95s = float(np.median(p95_changes))
        median_cohen_d = float(np.median(cohen_d_values))
        n_failed_runs = int(sum(run_failures))
        failure_rate = n_failed_runs / n_runs

        # Overall failure: median across runs exceeds threshold
        overall_failed = (median_of_medians >= median_change_threshold) or \
                         (median_of_p95s >= p95_change_threshold)

        # Store results
        result_row = {
            "feature": feature_name,
            "iqr": iqr,
            "median_abs_change": median_of_medians,
            "p95_abs_change": median_of_p95s,
            "median_cohen_d": median_cohen_d,
            "n_failed_runs": n_failed_runs,
            "failure_rate": failure_rate,
            "overall_failed": overall_failed
        }
        results.append(result_row)

        # If repeatedly failing, run permutation test to confirm importance
        if overall_failed:
            failure_info = result_row.copy()

            if (failure_rate >= repeated_failure_threshold) and (y is not None):
                # Run targeted permutation test
                perm_result = _permutation_importance_single_feature(
                    model, X, y, feature_name, n_repeats=permutation_repeats
                )
                failure_info.update({
                    "perm_metric": perm_result["metric"],
                    "perm_baseline": perm_result["baseline"],
                    "perm_mean_drop": perm_result["mean_delta"],
                    "perm_std_drop": perm_result["std_delta"]
                })
            else:
                failure_info.update({
                    "perm_metric": None,
                    "perm_baseline": None,
                    "perm_mean_drop": None,
                    "perm_std_drop": None
                })

            failures.append(failure_info)

    # Create results dataframes
    results_df = pd.DataFrame(results).sort_values(
        "overall_failed", ascending=False
    ).reset_index(drop=True)

    failures_df = pd.DataFrame(failures).reset_index(drop=True)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Results: {len(failures_df)} / {len(numeric_features)} features failed")

    if len(failures_df) > 0:
        print(f"\n⚠ Failing features:")
        print(failures_df[[
            'feature', 'iqr', 'median_abs_change', 'p95_abs_change',
            'failure_rate', 'perm_mean_drop'
        ]].to_string(index=False))
    else:
        print(f"\n✓ All features passed robustness test")

    return results_df, failures_df


def _permutation_importance_single_feature(
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        feature: str,
        n_repeats: int = 20
) -> Dict[str, Any]:
    """
    Calculate permutation importance for a single feature.

    Returns dict with metric name, baseline score, and drop statistics.
    """
    # Get baseline score using utility functions
    probs, _ = get_prediction_probabilities_and_confidence(model, X)
    preds = model.predict(X)
    baseline = calculate_auc_or_accuracy(y, probs, preds)
    metric_name = "roc_auc" if probs is not None else "accuracy"

    # Run permutations
    score_drops = []
    X_permuted = X.copy(deep=True)

    for _ in range(n_repeats):
        # Permute feature
        X_permuted[feature] = np.random.permutation(X_permuted[feature].values)

        # Recalculate score using utility functions
        probs_perm, _ = get_prediction_probabilities_and_confidence(model, X_permuted)
        preds_perm = model.predict(X_permuted)
        score = calculate_auc_or_accuracy(y, probs_perm, preds_perm)

        score_drops.append(baseline - score)

    score_drops_array = np.array(score_drops)

    return {
        "metric": metric_name,
        "baseline": float(baseline),
        "mean_delta": float(np.nanmean(score_drops_array)),
        "std_delta": float(np.nanstd(score_drops_array, ddof=1)),
        "deltas": score_drops_array
    }


# ============================================================================
# Test: Cluster-Based Performance Consistency
# ============================================================================

def test_model_performance_by_cluster(
        X: pd.DataFrame = _data_X_train,
        y: pd.Series = _data_y_train,
        pipeline = _default_pipeline,
        features_for_clustering: Sequence[str] = _data_X_train.columns,
        k: Optional[int] = None,
        k_range: Tuple[int, int] = (8, 12),
        samples_per_cluster: int = 50,
        max_auc_range: float = 0.05,
        max_median_conf_std: float = 0.08,
        use_pca: bool = True,
        pca_variance_threshold: float = 0.95
) -> None:
    """
    Test model performance consistency across data clusters.

    This test identifies subpopulations in the feature space and verifies
    that model performance is stable across them, detecting potential bias
    or overfitting to specific data regions.

    Process:
    1. Cluster data using specified features (with optional PCA)
    2. Sample from each cluster for testing
    3. Train model on remaining data
    4. Check per-cluster AUC consistency and prediction confidence

    Args:
        X: Feature dataframe
        y: Target series
        pipeline: Sklearn model/pipeline
        features_for_clustering: Features to use for clustering
        k: Number of clusters (if None, chosen by silhouette score)
        k_range: Range to search for optimal k if k=None
        samples_per_cluster: Samples to draw from each cluster for testing
        max_auc_range: Max allowed (max_cluster_auc - min_cluster_auc)
        max_median_conf_std: Max allowed median within-cluster confidence std
        use_pca: Whether to apply PCA before clustering
        pca_variance_threshold: Variance to retain if using PCA
    """
    # Validate clustering features
    missing_features = [f for f in features_for_clustering if f not in X.columns]
    if missing_features:
        raise ValueError(f"Clustering features not in X: {missing_features}")

    print(f"\n{'=' * 70}")
    print(f"Cluster-Based Performance Test")
    print(f"{'=' * 70}")

    # Prepare clustering data
    X_cluster = X.loc[:, features_for_clustering].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Optional PCA dimensionality reduction
    if use_pca:
        n_features = X_scaled.shape[1]
        pca = PCA(n_components=min(n_features, max(2, n_features)))
        X_pca_full = pca.fit_transform(X_scaled)

        # Find components needed for variance threshold
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, pca_variance_threshold) + 1)
        n_components = max(2, min(n_components, X_pca_full.shape[1]))

        if n_components < X_pca_full.shape[1]:
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_pca_full

        print(f"PCA: {n_components} components ({pca.explained_variance_ratio_.sum():.2%} variance)")
    else:
        X_reduced = X_scaled

    # Choose k if not provided
    if k is None:
        print(f"Searching for optimal k in range {k_range}...")
        best_k = None
        best_silhouette = -1.0

        for candidate_k in range(k_range[0], k_range[1] + 1):
            try:
                kmeans_temp = KMeans(n_clusters=candidate_k, random_state=42)
                labels_temp = kmeans_temp.fit_predict(X_reduced)
                sil_score = silhouette_score(X_reduced, labels_temp)

                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_k = candidate_k
            except Exception:
                continue

        k = best_k if best_k is not None else int(np.mean(k_range))
        print(f"Selected k={k} (silhouette={best_silhouette:.4f})")

    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=np.random.randint(0, 2 ** 31 - 1))
    cluster_labels = kmeans.fit_predict(X_reduced)

    # Prepare test/train split by cluster
    X_with_meta = X.copy()
    X_with_meta["_cluster"] = cluster_labels
    X_with_meta["_pos_idx"] = np.arange(len(X))

    unique_clusters = sorted(np.unique(cluster_labels))
    per_cluster_test_indices = {}
    all_test_indices = []

    print(f"\nSampling {samples_per_cluster} points per cluster:")
    print(f"{'Cluster':>8} {'Size':>8} {'Sampled':>8} {'Method':>15}")
    print("-" * 45)

    for cluster_id in unique_clusters:
        cluster_members = X_with_meta[X_with_meta["_cluster"] == cluster_id]
        member_indices = cluster_members["_pos_idx"].values
        cluster_size = len(member_indices)

        if cluster_size == 0:
            per_cluster_test_indices[cluster_id] = []
            continue

        # Sample with or without replacement
        if cluster_size >= samples_per_cluster:
            sampled = np.random.choice(
                member_indices, size=samples_per_cluster, replace=False
            )
            method = "without_replace"
        else:
            sampled = np.random.choice(
                member_indices, size=samples_per_cluster, replace=True
            )
            method = "with_replace"
            print(f"{cluster_id:>8} {cluster_size:>8} {len(sampled):>8} {method:>15} ⚠")

        per_cluster_test_indices[cluster_id] = list(sampled)
        all_test_indices.extend(sampled)

        if method == "without_replace":
            print(f"{cluster_id:>8} {cluster_size:>8} {len(sampled):>8} {method:>15}")

    # Create train/test split
    unique_test_indices = np.unique(all_test_indices)
    all_indices = np.arange(len(X))
    train_indices = np.setdiff1d(all_indices, unique_test_indices)

    X_cluster_train = X.iloc[train_indices]
    y_cluster_train = y.iloc[train_indices]

    print(f"\nTrain size: {len(train_indices)} | Test size: {len(unique_test_indices)}")

    # Train model
    model = clone(pipeline)
    model.fit(X_cluster_train, y_cluster_train)

    # Evaluate per cluster
    cluster_aucs = []
    cluster_conf_stds = []

    print(f"\n{'Cluster':>8} {'AUC/Acc':>10} {'Conf_Std':>10} {'Samples':>8}")
    print("-" * 45)

    for cluster_id in unique_clusters:
        test_indices_for_cluster = per_cluster_test_indices[cluster_id]

        if len(test_indices_for_cluster) == 0:
            cluster_aucs.append(np.nan)
            cluster_conf_stds.append(np.nan)
            continue

        X_test_cluster = X.iloc[test_indices_for_cluster]
        y_test_cluster = y.iloc[test_indices_for_cluster]

        # Get predictions and confidence
        probs, confidence = get_prediction_probabilities_and_confidence(
            model, X_test_cluster
        )
        preds = model.predict(X_test_cluster)

        # Calculate cluster AUC using utility function
        cluster_auc = calculate_auc_or_accuracy(y_test_cluster.values, probs, preds)

        cluster_conf_std = float(np.std(confidence, ddof=0))

        cluster_aucs.append(cluster_auc)
        cluster_conf_stds.append(cluster_conf_std)

        print(
            f"{cluster_id:>8} {cluster_auc:>10.6f} {cluster_conf_std:>10.6f} "
            f"{len(test_indices_for_cluster):>8}"
        )

    # Calculate stability metrics
    valid_aucs = np.array([a for a in cluster_aucs if not np.isnan(a)])
    valid_conf_stds = np.array([s for s in cluster_conf_stds if not np.isnan(s)])

    if len(valid_aucs) == 0:
        pytest.fail("No valid cluster AUCs computed")

    auc_range = float(valid_aucs.max() - valid_aucs.min())
    median_conf_std = float(np.median(valid_conf_stds))

    print("\n" + "=" * 45)
    print(f"AUC range: {auc_range:.6f} (threshold: {max_auc_range})")
    print(f"Median confidence std: {median_conf_std:.6f} (threshold: {max_median_conf_std})")

    # Assert stability
    failures = []
    if auc_range >= max_auc_range:
        failures.append(
            f"AUC range ({auc_range:.6f}) exceeds threshold ({max_auc_range})"
        )
    if median_conf_std >= max_median_conf_std:
        failures.append(
            f"Median conf std ({median_conf_std:.6f}) exceeds threshold ({max_median_conf_std})"
        )

    if failures:
        error_msg = (
                f"\n⚠ Cluster stability test FAILED:\n" +
                "\n".join(f"  • {f}" for f in failures) +
                f"\n\nCluster AUCs: {np.round(valid_aucs, 6).tolist()}" +
                f"\nCluster confidence stds: {np.round(valid_conf_stds, 6).tolist()}"
        )
        pytest.fail(error_msg)

    print(f"\n✓ Model performance consistent across {k} clusters")


# ============================================================================
# Test: Permutation Group Importance
# ============================================================================

def test_permutation_group_importance(
        X: pd.DataFrame = _data_X_test,
        y: np.ndarray = _data_y_test,
        model = _default_pipeline,
        group_size: int = 50,
        feature_dtype: str = 'numeric',
        n_permutations: int = 5,
        n_groups_to_test: int = 10,
        alpha: float = 0.05,
        expect_predictive: bool = True,     # expect all train features to be predictive
        expect_irrelevant: bool = False
) -> pd.DataFrame:
    """
    Test feature group importance via permutation testing.

    Randomly samples feature groups and tests whether permuting them
    significantly degrades model performance (indicating importance).

    Args:
        X: Feature dataframe
        y: Target array
        model: Unfitted model
        group_size: Number of features per group
        feature_dtype: 'numeric' or 'categorical' to select feature pool
        n_permutations: Permutations per group
        n_groups_to_test: Number of random groups to test
        alpha: Significance level (Bonferroni correction applied)
        expect_predictive: Assert delta > 0.01 for tested groups
        expect_irrelevant: Assert delta < 0.02 for tested groups

    Returns:
        DataFrame with results for each tested group
    """


    # Select feature pool
    if feature_dtype == 'numeric':
        feature_pool = X.select_dtypes(include=[np.number]).columns.tolist()
    elif feature_dtype == 'categorical':
        feature_pool = X.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        raise ValueError("feature_dtype must be 'numeric' or 'categorical'")

    if len(feature_pool) < group_size:
        pytest.skip(
            f"Not enough {feature_dtype} features ({len(feature_pool)}) "
            f"for group size {group_size}"
        )

    print(f"\n{'=' * 70}")
    print(f"Permutation Group Importance Test")
    print(f"{'=' * 70}")
    print(f"Feature pool: {len(feature_pool)} {feature_dtype} features")
    print(f"Group size: {group_size}")
    print(f"Groups to test: {n_groups_to_test}")
    print(f"Permutations per group: {n_permutations}")

    # Get baseline score
    baseline_scores = get_prediction_scores(model, X)
    baseline_auc = float(roc_auc_score(y, baseline_scores))

    print(f"Baseline AUC: {baseline_auc:.6f}")

    # Test random groups
    results = []

    for group_idx in range(n_groups_to_test):
        # Sample random feature group
        features_in_group = tuple(
            np.random.choice(feature_pool, size=group_size, replace=False)
        )

        # Permute and score multiple times
        permuted_aucs = []
        for perm_idx in range(n_permutations):
            X_permuted = X.copy(deep=True)

            # Permute each feature independently
            for feat in features_in_group:
                X_permuted[feat] = np.random.permutation(X_permuted[feat].values)

            try:
                scores_perm = get_prediction_scores(model, X_permuted)
                auc_perm = float(roc_auc_score(y, scores_perm))
            except Exception:
                auc_perm = np.nan

            permuted_aucs.append(auc_perm)

        # Calculate statistics
        permuted_aucs_array = np.array(permuted_aucs)
        valid_mask = ~np.isnan(permuted_aucs_array)

        if valid_mask.sum() == 0:
            mean_permuted_auc = np.nan
            p_value = np.nan
        else:
            valid_aucs = permuted_aucs_array[valid_mask]
            mean_permuted_auc = float(np.mean(valid_aucs))

            # Empirical p-value: fraction of permuted >= baseline
            # TODO: fix this, seems like it depends on the number of permuations, n_permutations
            p_value = (np.sum(valid_aucs >= baseline_auc) + 1) / (valid_mask.sum() + 1)

        delta = baseline_auc - mean_permuted_auc if not np.isnan(mean_permuted_auc) else np.nan
        significant = (p_value < alpha) if not np.isnan(p_value) else False

        # Check assertions
        assertion_ok = True
        assertion_messages = []

        if expect_predictive:
            # if np.isnan(delta) or delta <= 0.01 or not significant:
            if np.isnan(delta) or delta <= 0.01:
                assertion_ok = False
                assertion_messages.append(
                    "Expected predictive but delta ≤ 0.01 or not significant"
                )

        if expect_irrelevant:
            if np.isnan(delta) or delta >= 0.02:
                assertion_ok = False
                assertion_messages.append("Expected irrelevant but delta ≥ 0.02")

        results.append({
            "group_idx": group_idx,
            "features": features_in_group,
            "baseline_auc": baseline_auc,
            "mean_permuted_auc": mean_permuted_auc,
            "delta": delta,
            "p_value": p_value,
            "significant": significant,
            "assertion_ok": assertion_ok,
            "assertion_msg": "; ".join(assertion_messages)
        })

    # Create results dataframe
    results_df = pd.DataFrame(results).sort_values(
        "delta", ascending=False
    ).reset_index(drop=True)

    # Print summary
    print(f"\n{'Group':>6} {'Delta':>8} {'P-value':>10} {'Significant':>12} {'OK':>5}")
    print("-" * 50)

    for _, row in results_df.iterrows():
        print(
            f"{row['group_idx']:>6} {row['delta']:>8.4f} {row['p_value']:>10.4f} "
            f"{'Yes' if row['significant'] else 'No':>12} "
            f"{'✓' if row['assertion_ok'] else '✗':>5}"
        )

    # Check if any assertions failed
    n_failed = (~results_df['assertion_ok']).sum()
    if n_failed > 0:
        print(f"\n⚠ {n_failed} / {n_groups_to_test} groups failed assertions")
        if expect_predictive or expect_irrelevant:
            pytest.fail(
                f"\n{n_failed} feature groups failed importance assertions:\n" +
                results_df[~results_df['assertion_ok']][
                    ['group_idx', 'delta', 'significant', 'assertion_msg']
                ].to_string(index=False)
            )
    else:
        print(f"\n✓ All {n_groups_to_test} groups passed assertions")

    return results_df
