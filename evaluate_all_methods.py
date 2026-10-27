import os
import json
import numpy as np
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score

from detectors.mink import MinkDetector
from detectors.ppl import PPLDetector
from detectors.cdd import CDDDetector
from detectors.recall import RecallDetector
from detectors.self_critique import SelfCritiqueDetector
from detectors.self_critique_lara import SelfCritiqueLaRADetector
from detectors.self_critique_ablation import SelfCritiqueAblationDetector

FPR = []
TPR = []

class EvaluationMetrics:
    """Calculate the existing aggregate and per-source metrics."""

    @staticmethod
    def evaluate_performance_pop(y_true, y_scores):
        """
        A helper function to calculate a complete set of evaluation metrics,
        including AUC, optimal F1, Youden's J, TPR at fixed FPR,
        and F1 and Accuracy corresponding to Youden's J threshold.
        """
        # --- 0. Data preprocessing ---
        y_true = np.asarray(y_true)
        y_scores = pd.to_numeric(y_scores, errors="coerce")
        y_scores = np.asarray(y_scores, dtype=float)
        is_finite = np.isfinite(y_scores)
        y_true = y_true[is_finite]
        y_scores = y_scores[is_finite]

        if len(np.unique(y_true)) < 2:
            return {
                "roc_auc": np.nan,
                "best_f1_score": np.nan,
                "accuracy_at_best_f1": np.nan,
                "optimal_threshold_f1": np.nan,
                "youden_j_score": np.nan,
                "optimal_threshold_youden": np.nan,
                "f1_at_youden_threshold": np.nan,
                "accuracy_at_youden_threshold": np.nan,
                "tpr_at_fpr_5": np.nan,
                "error": "Only one class present."
            }

        # --- 1. Calculate basic ROC curve and AUC ---
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        FPR.append(fpr.tolist())
        TPR.append(tpr.tolist())
        roc_auc = auc(fpr, tpr)


        # --- 2. Calculate optimal F1 score (F1-based threshold) ---
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        fscore = (2 * precision * recall) / (precision + recall + 1e-6)
        best_f1_idx = np.argmax(fscore[:-1]) if len(fscore) > 1 else 0
        optimal_threshold_f1 = pr_thresholds[best_f1_idx]
        best_f1 = fscore[best_f1_idx]
        y_pred_f1 = (y_scores >= optimal_threshold_f1).astype(int)
        accuracy_at_best_f1 = accuracy_score(y_true, y_pred_f1)

        # --- 3. Calculate Youden's J Statistic ---
        youden_j_scores = tpr - fpr
        best_youden_idx = np.argmax(youden_j_scores)
        youden_j_score = youden_j_scores[best_youden_idx]
        optimal_threshold_youden = roc_thresholds[best_youden_idx]

        # Use Youden threshold for prediction
        y_pred_youden = (y_scores >= optimal_threshold_youden).astype(int)

        # Calculate corresponding F1 score
        tp_youden = np.sum((y_true == 1) & (y_pred_youden == 1))
        fp_youden = np.sum((y_true == 0) & (y_pred_youden == 1))
        fn_youden = np.sum((y_true == 1) & (y_pred_youden == 0))

        precision_youden = tp_youden / (tp_youden + fp_youden + 1e-6)
        recall_youden = tp_youden / (tp_youden + fn_youden + 1e-6)

        f1_at_youden_threshold = (2 * precision_youden * recall_youden) / (precision_youden + recall_youden + 1e-6)

        # Calculate corresponding accuracy
        accuracy_at_youden_threshold = accuracy_score(y_true, y_pred_youden)

        # --- 4. Calculate TPR at fixed FPR (unchanged) ---
        target_fpr = 0.05
        eligible = tpr[fpr <= target_fpr]
        tpr_at_fpr_5 = float(np.max(eligible)) if eligible.size else np.nan

        # --- 5. Return all metrics ---
        return {
            "roc_auc": roc_auc,
            "best_f1_score": best_f1,
            "accuracy_at_best_f1": accuracy_at_best_f1,
            "optimal_threshold_f1": optimal_threshold_f1,
            "youden_j_score": youden_j_score,
            "optimal_threshold_youden": optimal_threshold_youden,
            "f1_at_youden_threshold": f1_at_youden_threshold,
            "accuracy_at_youden_threshold": accuracy_at_youden_threshold,
            "tpr_at_fpr_5": tpr_at_fpr_5
        }


    @staticmethod
    def evaluate_performance(y_true, y_scores):
        """
        A helper function to calculate a complete set of evaluation metrics,
        including AUC, optimal F1, Youden's J, TPR at fixed FPR,
        and F1 and Accuracy corresponding to Youden's J threshold.
        """
        # --- 0. Data preprocessing ---
        y_true = np.asarray(y_true)
        y_scores = pd.to_numeric(y_scores, errors="coerce")
        y_scores = np.asarray(y_scores, dtype=float)
        is_finite = np.isfinite(y_scores)
        y_true = y_true[is_finite]
        y_scores = y_scores[is_finite]

        if len(np.unique(y_true)) < 2:
            return {
                "roc_auc": np.nan,
                "best_f1_score": np.nan,
                "accuracy_at_best_f1": np.nan,
                "optimal_threshold_f1": np.nan,
                "youden_j_score": np.nan,
                "optimal_threshold_youden": np.nan,
                "f1_at_youden_threshold": np.nan,
                "accuracy_at_youden_threshold": np.nan,
                "tpr_at_fpr_5": np.nan,
                "error": "Only one class present."
            }

        # --- 1. Calculate basic ROC curve and AUC ---
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # --- 2. Calculate optimal F1 score (F1-based threshold) ---
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        fscore = (2 * precision * recall) / (precision + recall + 1e-6)
        best_f1_idx = np.argmax(fscore[:-1]) if len(fscore) > 1 else 0
        optimal_threshold_f1 = pr_thresholds[best_f1_idx]
        best_f1 = fscore[best_f1_idx]
        y_pred_f1 = (y_scores >= optimal_threshold_f1).astype(int)
        accuracy_at_best_f1 = accuracy_score(y_true, y_pred_f1)

        # --- 3. Calculate Youden's J Statistic ---
        youden_j_scores = tpr - fpr
        best_youden_idx = np.argmax(youden_j_scores)
        youden_j_score = youden_j_scores[best_youden_idx]
        optimal_threshold_youden = roc_thresholds[best_youden_idx]

        y_pred_youden = (y_scores >= optimal_threshold_youden).astype(int)
        tp_youden = np.sum((y_true == 1) & (y_pred_youden == 1))
        fp_youden = np.sum((y_true == 0) & (y_pred_youden == 1))
        fn_youden = np.sum((y_true == 1) & (y_pred_youden == 0))

        precision_youden = tp_youden / (tp_youden + fp_youden + 1e-6)
        recall_youden = tp_youden / (tp_youden + fn_youden + 1e-6)

        f1_at_youden_threshold = (2 * precision_youden * recall_youden) / (precision_youden + recall_youden + 1e-6)

        # 3. Calculate corresponding accuracy
        accuracy_at_youden_threshold = accuracy_score(y_true, y_pred_youden)

        # --- 4. Calculate TPR at fixed FPR (unchanged) ---
        target_fpr = 0.05
        eligible = tpr[fpr <= target_fpr]
        tpr_at_fpr_5 = float(np.max(eligible)) if eligible.size else np.nan

        # --- 5. Return all metrics ---
        return {
            "roc_auc": roc_auc,
            "best_f1_score": best_f1,
            "accuracy_at_best_f1": accuracy_at_best_f1,
            "optimal_threshold_f1": optimal_threshold_f1,
            "youden_j_score": youden_j_score,
            "optimal_threshold_youden": optimal_threshold_youden,
            "f1_at_youden_threshold": f1_at_youden_threshold,
            "accuracy_at_youden_threshold": accuracy_at_youden_threshold,
            "tpr_at_fpr_5": tpr_at_fpr_5
        }


    @staticmethod
    def evaluate_by_source(df_scores, method_name, direction):
        """Macro AUC averages only sources containing both membership classes."""
        breakdown = {}
        source_aucs = []
        for source, group in df_scores.groupby('data_source'):
            performance = evaluate_performance(
                group['ground_truth_label'].values, group[method_name].values * direction)
            breakdown[source] = performance
            if np.isfinite(performance['roc_auc']):
                source_aucs.append(performance['roc_auc'])
        return breakdown, float(np.mean(source_aucs)) if source_aucs else None


class LayerFeatures:
    """Build layer ordering, segment features and aligned ranks."""

    @staticmethod
    def segment_lengths_9_9_10(n_layers: int) -> Tuple[int, int, int]:
        """Early / mid / late run lengths in a 9:9:10 ratio (sums to n_layers)."""
        if n_layers <= 0:
            return (0, 0, 0)
        n_e = (9 * n_layers) // 28
        n_m = (9 * n_layers) // 28
        n_l = n_layers - n_e - n_m
        return (n_e, n_m, n_l)


    @staticmethod
    def _layer_sort_key(layer_name: str) -> Tuple[int, str]:
        if layer_name.startswith("L") and layer_name[1:].isdigit():
            return (int(layer_name[1:]), layer_name)
        legacy = {"early": 0, "mid": 1, "late": 2}
        return (legacy.get(layer_name, 9999), layer_name)


    @staticmethod
    def _ordered_rep_stiff_layers(layers: List[str]) -> List[str]:
        return [name for _, name in sorted((_layer_sort_key(ln), ln) for ln in layers)]


    @staticmethod
    def _numeric_layer_indices(layers: List[str]) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        for ln in layers:
            if ln.startswith("L") and ln[1:].isdigit():
                out.append((int(ln[1:]), ln))
        return sorted(out)


    @staticmethod
    def _legacy_segment_layer_features(
        df_scores: pd.DataFrame,
        ordered_layers: List[str],
    ) -> Dict[str, pd.Series]:
        """Map rep_stiff_*_{early,mid,late}_score columns to segment feature series."""
        if not set(ordered_layers) <= _LEGACY_SEGMENT_LAYERS:
            return {}
        features: Dict[str, pd.Series] = {}
        for seg in ordered_layers:
            for metric, col_prefix in (
                ("rsi", "rep_stiff_rsi"),
                ("rsm", "rep_stiff_rsm"),
                ("directional_collapse", "rep_stiff_directional_collapse"),
            ):
                col = f"{col_prefix}_{seg}_score"
                if col in df_scores.columns:
                    features[f"{metric}_{seg}"] = pd.to_numeric(df_scores[col], errors="coerce")
        return features


    @staticmethod
    def _compute_rep_stiff_combined_score_new2(
        df_scores: pd.DataFrame,
        ordered_layers: List[str],
    ) -> Optional[pd.Series]:
        """Early-vs-late slope combined score (numeric L* layers or legacy early/mid/late)."""
        parts: List[pd.Series] = []
        for col_prefix, weight in (
            ("rep_stiff_rsm", 1.0),
            ("rep_stiff_directional_collapse", 1.0),
            ("rep_stiff_rsi", -1.0),
        ):
            early_cols: List[str] = []
            late_cols: List[str] = []
            numeric = _numeric_layer_indices(ordered_layers)
            if len(numeric) >= 2:
                idx_to_name = {idx: name for idx, name in numeric}
                lo_idx, hi_idx = numeric[0][0], numeric[-1][0]
                n_e, n_m, _ = segment_lengths_9_9_10(hi_idx - lo_idx + 1)
                early_idxs = [i for i, _ in numeric if i < lo_idx + n_e]
                late_idxs = [i for i, _ in numeric if i >= lo_idx + n_e + n_m]
                if not early_idxs or not late_idxs:
                    continue
                early_cols = [f"{col_prefix}_{idx_to_name[i]}_score" for i in early_idxs]
                late_cols = [f"{col_prefix}_{idx_to_name[i]}_score" for i in late_idxs]
            elif "early" in ordered_layers and "late" in ordered_layers:
                early_cols = [f"{col_prefix}_early_score"]
                late_cols = [f"{col_prefix}_late_score"]
            else:
                continue
            if all(c in df_scores.columns for c in early_cols + late_cols):
                early_mean = df_scores[early_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                late_mean = df_scores[late_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                parts.append(weight * (late_mean - early_mean))
        if not parts:
            return None
        return sum(parts)


    @staticmethod
    def _build_segment_layer_features(
        df_scores: pd.DataFrame,
        ordered_layers: List[str],
    ) -> Dict[str, pd.Series]:
        """Map per-layer columns to rsi/rsm/dc_{early,mid,late} means for trend rules."""
        numeric = _numeric_layer_indices(ordered_layers)
        if not numeric:
            return _legacy_segment_layer_features(df_scores, ordered_layers)
        n_layers = numeric[-1][0] - numeric[0][0] + 1
        n_e, n_m, _n_l = segment_lengths_9_9_10(n_layers)
        buckets = {
            "early": {idx for idx, _ in numeric if idx < numeric[0][0] + n_e},
            "mid": {idx for idx, _ in numeric if numeric[0][0] + n_e <= idx < numeric[0][0] + n_e + n_m},
            "late": {idx for idx, _ in numeric if idx >= numeric[0][0] + n_e + n_m},
        }
        idx_to_name = {idx: name for idx, name in numeric}
        features: Dict[str, pd.Series] = {}
        for seg, idx_set in buckets.items():
            if not idx_set:
                continue
            for metric, col_prefix in (
                ("rsi", "rep_stiff_rsi"),
                ("rsm", "rep_stiff_rsm"),
                ("directional_collapse", "rep_stiff_directional_collapse"),
            ):
                cols = [f"{col_prefix}_{idx_to_name[i]}_score" for i in sorted(idx_set) if i in idx_to_name]
                cols = [c for c in cols if c in df_scores.columns]
                if not cols:
                    continue
                features[f"{metric}_{seg}"] = df_scores[cols].apply(
                    pd.to_numeric, errors="coerce"
                ).mean(axis=1)
        return features


    @staticmethod
    def _aligned_layer_rank_columns(
        df_scores: pd.DataFrame,
        ordered_layers: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Percentile ranks per metric/layer and per-layer aligned rigidity s_l."""
        rsm_cols: List[str] = []
        dc_cols: List[str] = []
        rsi_cols: List[str] = []
        s_cols: List[str] = []
        for layer_name in ordered_layers:
            rsm_col = f"__rs_rank_{layer_name}"
            dc_col = f"__dc_rank_{layer_name}"
            rsi_col = f"__rsi_rank_{layer_name}"
            base_rsm = f"rep_stiff_rsm_{layer_name}_score"
            base_dc = f"rep_stiff_directional_collapse_{layer_name}_score"
            base_rsi = f"rep_stiff_rsi_{layer_name}_score"
            if base_rsm in df_scores.columns:
                df_scores[rsm_col] = pd.to_numeric(df_scores[base_rsm], errors="coerce").rank(pct=True)
                rsm_cols.append(rsm_col)
            if base_dc in df_scores.columns:
                df_scores[dc_col] = pd.to_numeric(df_scores[base_dc], errors="coerce").rank(pct=True)
                dc_cols.append(dc_col)
            if base_rsi in df_scores.columns:
                df_scores[rsi_col] = pd.to_numeric(df_scores[base_rsi], errors="coerce").rank(pct=True)
                rsi_cols.append(rsi_col)
            if rsm_col in df_scores.columns and dc_col in df_scores.columns and rsi_col in df_scores.columns:
                scol = f"__s_align_{layer_name}"
                df_scores[scol] = (
                    df_scores[rsm_col] + df_scores[dc_col] + (1.0 - df_scores[rsi_col])
                ) / 3.0
                s_cols.append(scol)
        return s_cols, ordered_layers


class RepresentationScoreAggregator:
    """Compute the existing combined representation and LaRA scores."""

    @staticmethod
    def _compute_rep_stiff_derived_columns(
        df_scores: pd.DataFrame,
        args: argparse.Namespace,
        rep_stiff_layers: List[str],
        rep_stiff_scores_list: Optional[List[dict]],
    ) -> None:
        """Post-process per-layer RepStiff scores into combined / LaRA detector columns."""
        ordered_layers = _ordered_rep_stiff_layers(rep_stiff_layers)
        if not ordered_layers:
            return

        segment_features = _build_segment_layer_features(df_scores, ordered_layers)
        if segment_features and (args.rep_stiff_combined_weights or args.rep_stiff_combined_fixed):
            for key, series in segment_features.items():
                df_scores[key] = series
            if args.rep_stiff_combined_fixed:
                seg_keys = list(segment_features.keys())
                df_scores["rep_stiff_combined_trend_v1_score"] = [
                    RepStiffDetector.compute_fixed_trend_score(
                        {k: segment_features[k].iloc[i] for k in seg_keys}
                    )
                    for i in range(len(df_scores))
                ]
                df_scores["rep_stiff_combined_trend_v2_score"] = [
                    RepStiffDetector.compute_fixed_trend_v2_score(
                        {k: segment_features[k].iloc[i] for k in seg_keys}
                    )
                    for i in range(len(df_scores))
                ]
                df_scores["rep_stiff_combined_trend_v3_score"] = [
                    RepStiffDetector.compute_fixed_trend_v3_score(
                        {k: segment_features[k].iloc[i] for k in seg_keys}
                    )
                    for i in range(len(df_scores))
                ]
                df_scores["rep_stiff_combined_trend_v4_score"] = [
                    RepStiffDetector.compute_fixed_trend_v4_score(
                        {k: segment_features[k].iloc[i] for k in seg_keys}
                    )
                    for i in range(len(df_scores))
                ]

        if args.rep_stiff_combined_weights or args.rep_stiff_combined_fixed:
            rule_name = getattr(args, "rep_stiff_combined_rule", "trend_v1")
            combined = []
            for i in range(len(df_scores)):
                features = {
                    f"{metric}_{layer}": df_scores.iloc[i].get(f"{prefix}_{layer}_score")
                    for layer in ordered_layers
                    for metric, prefix in (
                        ("rsi", "rep_stiff_rsi"), ("rsm", "rep_stiff_rsm"),
                        ("directional_collapse", "rep_stiff_directional_collapse"),
                    )
                }
                features.update({key: values.iloc[i] for key, values in segment_features.items()})
                score = None
                if args.rep_stiff_combined_weights:
                    score = RepStiffDetector.compute_combined_score(features, args.rep_stiff_combined_weights)
                if score is None and args.rep_stiff_combined_fixed:
                    score = df_scores.iloc[i].get(f"rep_stiff_combined_{rule_name}_score")
                combined.append(score)
            df_scores["rep_stiff_combined_score"] = combined

        s_cols, required_layers = _aligned_layer_rank_columns(df_scores, ordered_layers)
        if len(s_cols) == len(required_layers) and len(required_layers) >= 1:
            s_mat = df_scores[s_cols].to_numpy(dtype=float)
            df_scores["rep_stiff_combined_v4"] = np.nanmean(s_mat, axis=1)
            df_scores["rep_stiff_combined_v6"] = df_scores["rep_stiff_combined_v4"]
            df_scores["rep_stiff_combined_new"] = df_scores["rep_stiff_combined_v4"]
            if len(required_layers) >= 2:
                depth = np.linspace(0.0, 1.0, len(required_layers))
                dd = depth - float(depth.mean())
                denom_d = float(np.sqrt(np.dot(dd, dd)))
                s_mean = s_mat.mean(axis=1, keepdims=True)
                ss = s_mat - s_mean
                denom_s = np.sqrt((ss * ss).sum(axis=1))
                num = (ss * dd).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_pearson = num / (denom_d * denom_s)
                r_pearson = np.where(denom_s > 1e-12, r_pearson, np.nan)
                df_scores["rep_stiff_combined_v5"] = r_pearson.astype(float)
            else:
                df_scores["rep_stiff_combined_v5"] = np.nan

        new2 = _compute_rep_stiff_combined_score_new2(df_scores, ordered_layers)
        if new2 is not None:
            df_scores["rep_stiff_combined_score_new2"] = new2
            df_scores["rep_stiff_combined_new2"] = new2
            df_scores["rep_stiff_combined_new3"] = new2

        if segment_features:
            rsi_e = segment_features.get("rsi_early")
            dc_e = segment_features.get("directional_collapse_early")
            rsm_e = segment_features.get("rsm_early")
            if rsi_e is not None and dc_e is not None:
                df_scores["rep_stiff_combined_rsi_dc_score"] = rsi_e * dc_e
            if rsi_e is not None and rsm_e is not None:
                df_scores["rep_stiff_combined_rsi_rsm_score"] = rsi_e + rsm_e
            if dc_e is not None and rsm_e is not None:
                df_scores["rep_stiff_combined_dc_rsm_score"] = dc_e + rsm_e

        alpha_v4 = float(getattr(args, "rep_stiff_combined_v4_alpha", 0.0))
        alpha_v4 = min(1.0, max(0.0, alpha_v4))
        beta_v6 = float(getattr(args, "rep_stiff_v6_mix_beta", 0.65))
        beta_v6 = min(1.0, max(0.0, beta_v6))
        if "self_critique_score" in df_scores.columns:
            r_sc = pd.to_numeric(df_scores["self_critique_score"], errors="coerce").rank(pct=True)
            if "rep_stiff_combined_v4" in df_scores.columns:
                r_v4 = pd.to_numeric(df_scores["rep_stiff_combined_v4"], errors="coerce").rank(pct=True)
                df_scores["self_critique_rep_stiff_v4_mix"] = (
                    alpha_v4 * r_sc + (1.0 - alpha_v4) * r_v4
                )
            if "rep_stiff_combined_v5" in df_scores.columns:
                r_v5 = pd.to_numeric(df_scores["rep_stiff_combined_v5"], errors="coerce").rank(pct=True)
                df_scores["self_critique_rep_stiff_v5_mix"] = beta_v6 * r_sc + (1.0 - beta_v6) * r_v5
            if "rep_stiff_combined_v6" in df_scores.columns:
                r_v6 = pd.to_numeric(df_scores["rep_stiff_combined_v6"], errors="coerce").rank(pct=True)
                df_scores["self_critique_rep_stiff_v6_mix"] = beta_v6 * r_sc + (1.0 - beta_v6) * r_v6

        # --- LaRA ---
        lara_eps = float(getattr(args, "rep_stiff_lara_eps", 1e-8))
        lara_metrics: List[Tuple[str, str, str]] = [
            ("rep_stiff_rsm", "rsm", "signed"),
            ("rep_stiff_directional_collapse", "directional_collapse", "absolute"),
            ("rep_stiff_rsi", "rsi", "negated"),
        ]
        lara_layers = list(ordered_layers)
        external_ref: Dict[str, Dict[str, Dict[str, float]]] = {}
        ref_path = getattr(args, "rep_stiff_lara_clean_ref", None)
        if ref_path:
            try:
                with open(ref_path, "r", encoding="utf-8") as fref:
                    external_ref = json.load(fref)
                print(f"[info] LaRA: loaded external clean-reference stats from {ref_path}")
            except Exception as exc:
                print(
                    f"[warn] LaRA: could not load --rep_stiff_lara_clean_ref={ref_path}: {exc}; "
                    "falling back to in-set clean rows."
                )
                external_ref = {}

        label_arr = pd.to_numeric(df_scores.get("ground_truth_label"), errors="coerce").to_numpy()
        in_set_clean_mask = label_arr == 0
        n_clean = int(np.sum(in_set_clean_mask))
        if not external_ref and n_clean < 2:
            print(
                f"[warn] LaRA: only {n_clean} clean rows in eval set; "
                "falling back to all rows for clean-reference statistics."
            )
            in_set_clean_mask = np.ones_like(in_set_clean_mask, dtype=bool)

        n_rows = len(df_scores)

        def _signed_log1p(arr: np.ndarray) -> np.ndarray:
            return np.sign(arr) * np.log1p(np.abs(arr))

        def _layer_index_subset(window: str) -> set:
            L_local = len(lara_layers)
            if L_local == 0 or window == "all":
                return set(range(L_local))
            numeric_local = _numeric_layer_indices(lara_layers)
            if not numeric_local:
                return set(range(L_local))
            lo = numeric_local[0][0]
            hi = numeric_local[-1][0]
            n_e, n_m, _ = segment_lengths_9_9_10(hi - lo + 1)
            early = {i for i, _ in numeric_local if i < lo + n_e}
            mid = {i for i, _ in numeric_local if lo + n_e <= i < lo + n_e + n_m}
            late = {i for i, _ in numeric_local if i >= lo + n_e + n_m}
            pos = {idx: li for li, (idx, _) in enumerate(numeric_local)}
            if window == "early_mid":
                keep = early | mid
            elif window == "mid_late":
                keep = mid | late
            elif window == "early":
                keep = early
            elif window == "mid":
                keep = mid
            elif window == "late":
                keep = late
            else:
                keep = set(range(L_local))
            return {pos[i] for i in keep if i in pos}

        def _compute_lara(
            log_metrics: set,
            drop_delta: bool,
            use_mad: bool = False,
            layer_window: str = "all",
            metric_weights: Optional[Dict[str, float]] = None,
            active_metrics: Optional[List[Tuple[str, str, str]]] = None,
        ) -> np.ndarray:
            metrics = active_metrics if active_metrics is not None else lara_metrics
            M = len(metrics)
            L = len(lara_layers)
            zhat_stack = np.full((n_rows, M, L), np.nan, dtype=float)
            have_all = True
            layer_keep = _layer_index_subset(layer_window)

            for li, layer_name in enumerate(lara_layers):
                if li not in layer_keep:
                    continue
                for mi, (col_prefix, metric_short, alignment) in enumerate(metrics):
                    col = f"{col_prefix}_{layer_name}_score"
                    if col not in df_scores.columns:
                        have_all = False
                        break
                    raw = pd.to_numeric(df_scores[col], errors="coerce").to_numpy(dtype=float)
                    raw_for_stats = _signed_log1p(raw) if metric_short in log_metrics else raw

                    mu = sigma = None
                    ref_layer = external_ref.get(layer_name, {})
                    ref_metric = ref_layer.get(metric_short, {}) if isinstance(ref_layer, dict) else {}
                    if isinstance(ref_metric, dict) and "mean" in ref_metric and "std" in ref_metric:
                        try:
                            mu = float(ref_metric["mean"])
                            sigma = float(ref_metric["std"])
                        except (TypeError, ValueError):
                            mu = sigma = None

                    if mu is None or sigma is None or not (np.isfinite(mu) and np.isfinite(sigma)):
                        clean_vals = raw_for_stats[in_set_clean_mask]
                        clean_vals = clean_vals[np.isfinite(clean_vals)]
                        if clean_vals.size < 2:
                            mu, sigma = np.nan, np.nan
                        elif use_mad:
                            med = float(np.median(clean_vals))
                            mad = float(np.median(np.abs(clean_vals - med)))
                            mu = med
                            sigma = 1.4826 * mad
                        else:
                            mu = float(np.mean(clean_vals))
                            sigma = float(np.std(clean_vals, ddof=0))

                    if not (np.isfinite(mu) and np.isfinite(sigma)):
                        continue
                    z = (raw_for_stats - mu) / (sigma + lara_eps)
                    if alignment == "signed":
                        zhat = z
                    elif alignment == "absolute":
                        zhat = np.abs(z)
                    elif alignment == "negated":
                        zhat = -z
                    else:
                        zhat = z
                    zhat_stack[:, mi, li] = zhat
                if not have_all:
                    break

            if not have_all or L < 1 or not layer_keep:
                return np.full(n_rows, np.nan, dtype=float)

            abs_zhat = np.abs(zhat_stack)
            if drop_delta:
                summand = abs_zhat
            else:
                if L >= 2:
                    deltas = np.diff(zhat_stack, axis=2)
                    deltas = np.concatenate(
                        [deltas, np.zeros((deltas.shape[0], deltas.shape[1], 1), dtype=float)],
                        axis=2,
                    )
                else:
                    deltas = np.zeros_like(zhat_stack)
                summand = abs_zhat + np.abs(deltas)

            if metric_weights is not None:
                w = np.array(
                    [float(metric_weights.get(short, 1.0)) for _, short, _ in metrics],
                    dtype=float,
                )
                w_mean = float(np.mean(w))
                if w_mean > 0:
                    w = w / w_mean
                summand = summand * w[None, :, None]

            with np.errstate(invalid="ignore"):
                return np.nanmean(summand.reshape(n_rows, -1), axis=1)

        df_scores["rep_stiff_lara"] = _compute_lara(log_metrics=set(), drop_delta=False)
        if "self_critique_score" in df_scores.columns:
            fusion = SelfCritiqueLaRADetector(getattr(args, "rep_stiff_lara_mix_beta", 0.65))
            df_scores[fusion.get_name()] = fusion.combine_scores(
                df_scores["self_critique_score"], df_scores["rep_stiff_lara"])

        if rep_stiff_scores_list is not None and len(rep_stiff_scores_list) == len(df_scores):
            export_map = {
                "rep_stiff_combined_score": "combined_score",
                "rep_stiff_combined_v4": "combined_v4",
                "rep_stiff_combined_v5": "combined_v5",
                "rep_stiff_combined_v6": "combined_v6",
                "rep_stiff_combined_new": "combined_new",
                "rep_stiff_combined_new2": "combined_new2",
                "rep_stiff_combined_new3": "combined_new3",
                "rep_stiff_combined_score_new2": "combined_new2",
                "rep_stiff_lara": "lara",
                "self_critique_rep_stiff_v4_mix": "self_critique_rep_stiff_v4_mix",
                "self_critique_rep_stiff_v5_mix": "self_critique_rep_stiff_v5_mix",
                "self_critique_rep_stiff_v6_mix": "self_critique_rep_stiff_v6_mix",
                "self_critique_rep_stiff_lara_mix": "self_critique_rep_stiff_lara_mix",
                "rep_stiff_combined_rsi_dc_score": "combined_rsi_dc_score",
                "rep_stiff_combined_rsi_rsm_score": "combined_rsi_rsm_score",
                "rep_stiff_combined_dc_rsm_score": "combined_dc_rsm_score",
                "rep_stiff_combined_trend_v1_score": "combined_trend_v1_score",
                "rep_stiff_combined_trend_v2_score": "combined_trend_v2_score",
                "rep_stiff_combined_trend_v3_score": "combined_trend_v3_score",
                "rep_stiff_combined_trend_v4_score": "combined_trend_v4_score",
            }
            for col, key in export_map.items():
                if col in df_scores.columns:
                    vals = df_scores[col].tolist()
                    for i, v in enumerate(vals):
                        rep_stiff_scores_list[i][key] = v


class DetectorEvaluationApplication:
    """Own the command-line configuration and execute the existing workflow."""

    def __init__(self):
        self.parser = self.create_parser()

    @staticmethod
    def create_parser():
        parser = argparse.ArgumentParser(description="Calculate performance for all modular detection methods.")
        parser.add_argument("--input_file", type=str, required=True, help="JSONL file generated by generate_full_data.py.")
        parser.add_argument("--output_summary_json", type=str, required=True, help="JSON filename to save performance comparison of all methods.")
        parser.add_argument("--output_plot", type=str, required=True, help="Image name for the detector performance analysis plot.")
        parser.add_argument("--detectors", nargs="+", default=None,
                            choices=["all", "ppl", "mink", "mink_plus_plus", "recall", "cdd",
                                     "self_critique", "self_critique_ablation", "lara", "self_critique_lara"],
                            help="Run selected detectors; lara includes RSM, RSI, DC and combined scores.")
        parser.add_argument("--resume", action="store_true", help="Resume raw per-sample/layer scores after interruption")
        parser.add_argument("--mink_ratio", type=float, default=0.2, help="Percentage k for Min-K% method. The default setting in original paper")
        parser.add_argument("--rep_stiff_model_name", type=str, default=None,
                            help="Model name/path for RepStiff embeddings.")
        parser.add_argument("--rep_stiff_max_workers", type=int, default=None,
                            help="Max concurrent OpenRouter calls for RepStiff.")
        parser.add_argument("--rep_stiff_layers", type=str, default="early,mid,late",
                            help="Comma-separated RepStiff layers (e.g., early,mid,late).")
        parser.add_argument("--rep_stiff_scores_json", type=str, default=None,
                            help="Output JSON file for per-sample RepStiff scores.")
        parser.add_argument("--rep_stiff_output_dir", type=str, default=None,
                            help="Output directory for RepStiff cached JSON artifacts.")
        parser.add_argument("--rep_stiff_combined_weights", type=str, default=None,
                            help="JSON file with combined RepStiff weights/bias.")
        parser.add_argument("--rep_stiff_combined_fixed", action="store_true",
                            help="Use fixed layerwise trend coefficients for RepStiff combined score.")
        parser.add_argument("--rep_stiff_combined_rule", type=str, default="trend_v1",
                            choices=["trend_v1", "trend_v2", "trend_v3", "trend_v4"],
                            help="Fixed rule for RepStiff combined score.")
        parser.add_argument("--rep_stiff_incomplete_blank_strategy", type=str, default="important",
                            choices=[
                                "important", "info_rem", "guided", "info", "info_type", "guidance",
                                "num_replace", "var_rename", "distractor", "distractor_insert",
                            ],
                            help="RepStiff incomplete-question perturbation strategy.")
        parser.add_argument("--rep_stiff_incomplete_num_blanks", type=int, default=1,
                            help="Number of [BLANK] tokens per incomplete question (important mode).")
        parser.add_argument("--rep_stiff_combined_v4_alpha", type=float, default=0.0,
                            help="Weight on rank(self_critique) in self_critique_rep_stiff_v4_mix.")
        parser.add_argument("--rep_stiff_v6_mix_beta", type=float, default=0.65,
                            help="Weight on rank(self_critique) in v5/v6 mix scores.")
        parser.add_argument("--rep_stiff_lara_eps", type=float, default=1e-8,
                            help="Epsilon for LaRA clean-reference standardization.")
        parser.add_argument("--rep_stiff_lara_clean_ref", type=str, default=None,
                            help="Optional JSON of external clean-reference stats for LaRA.")
        parser.add_argument("--rep_stiff_lara_mix_beta", type=float, default=0.65,
                            help="Weight on rank(self_critique) in LaRA mix scores.")
        parser.add_argument("--output_summary_json_subset", type=str, default=None,
                            help="JSON output path for subset evaluation.")
        parser.add_argument("--output_plot_subset", type=str, default=None,
                            help="ROC plot output path for subset evaluation.")
        return parser

    def run(self):
        parser = self.parser
        args = parser.parse_args()
        # Import model/API dependencies only for representation-based detectors.
        available = ["ppl", "mink", "mink_plus_plus", "recall", "cdd",
                     "self_critique", "self_critique_ablation", "lara", "self_critique_lara"]
        selected = (set(available) - {"self_critique_ablation"} if args.detectors is None
                    else set(available) if "all" in args.detectors else set(args.detectors))
        if "self_critique_lara" in selected:
            selected.update(("self_critique", "lara"))
        if "lara" in selected:
            global RepStiffDetector
            from detectors.rep_stiff import RepStiffDetector
        else:
            args.rep_stiff_combined_fixed = False
            args.rep_stiff_combined_weights = None
        print(f"Initializing detectors: {', '.join(sorted(selected))}")
        class _ScoreOnlyDetector:
            def __init__(self, name, direction=1):
                self._name = name
                self._direction = direction
            def get_name(self):
                return self._name
            def get_direction(self):
                return self._direction
            def calculate_score(self, data_item):
                return data_item.get(self._name)
        rep_stiff_kwargs = {}
        if args.rep_stiff_model_name:
            rep_stiff_kwargs["model_name"] = args.rep_stiff_model_name
        if args.rep_stiff_max_workers is not None:
            rep_stiff_kwargs["max_openrouter_workers"] = args.rep_stiff_max_workers
        if args.rep_stiff_output_dir:
            rep_stiff_kwargs["output_dir"] = args.rep_stiff_output_dir
        rep_stiff_kwargs["incomplete_blank_strategy"] = args.rep_stiff_incomplete_blank_strategy
        rep_stiff_kwargs["incomplete_num_blanks"] = args.rep_stiff_incomplete_num_blanks

        rep_stiff_layers = [l.strip() for l in args.rep_stiff_layers.split(",") if l.strip()]
        if "lara" not in selected:
            rep_stiff_layers = []
        elif not rep_stiff_layers:
            rep_stiff_layers = ["mid"]

        rep_stiff_detectors = {}
        for layer_name in rep_stiff_layers:
            rep_stiff_detectors[layer_name] = RepStiffDetector(
                layer_name=layer_name,
                **rep_stiff_kwargs,
            )

        factories = {
            "ppl": PPLDetector,
            "mink": lambda: MinkDetector(mink_ratio=args.mink_ratio, use_plus_plus=False),
            "mink_plus_plus": lambda: MinkDetector(mink_ratio=args.mink_ratio, use_plus_plus=True),
            "recall": RecallDetector,
            "cdd": lambda: CDDDetector(alpha=0.05),
            "self_critique": SelfCritiqueDetector,
            "self_critique_ablation": SelfCritiqueAblationDetector,
        }
        detectors = [factory() for name, factory in factories.items() if name in selected]
        for layer_name in rep_stiff_layers:
            detectors.extend([
                _ScoreOnlyDetector(f"rep_stiff_rsi_{layer_name}_score"),
                _ScoreOnlyDetector(f"rep_stiff_rsm_{layer_name}_score"),
                _ScoreOnlyDetector(f"rep_stiff_directional_collapse_{layer_name}_score"),
            ])
        if args.rep_stiff_combined_weights or args.rep_stiff_combined_fixed:
            detectors.append(_ScoreOnlyDetector("rep_stiff_combined_score"))
        if args.rep_stiff_combined_fixed:
            detectors.extend([
                _ScoreOnlyDetector("rep_stiff_combined_trend_v1_score"),
                _ScoreOnlyDetector("rep_stiff_combined_trend_v2_score"),
                _ScoreOnlyDetector("rep_stiff_combined_trend_v3_score"),
                _ScoreOnlyDetector("rep_stiff_combined_trend_v4_score"),
            ])
        if rep_stiff_layers:
            detectors.extend([
                _ScoreOnlyDetector("rep_stiff_combined_v4"),
                _ScoreOnlyDetector("rep_stiff_combined_v5"),
                _ScoreOnlyDetector("rep_stiff_combined_v6"),
                _ScoreOnlyDetector("self_critique_rep_stiff_v4_mix"),
                _ScoreOnlyDetector("self_critique_rep_stiff_v5_mix"),
                _ScoreOnlyDetector("self_critique_rep_stiff_v6_mix"),
                _ScoreOnlyDetector("rep_stiff_combined_score_new2"),
                _ScoreOnlyDetector("rep_stiff_lara"),
                SelfCritiqueLaRADetector(args.rep_stiff_lara_mix_beta),
                _ScoreOnlyDetector("rep_stiff_combined_rsi_dc_score"),
                _ScoreOnlyDetector("rep_stiff_combined_rsi_rsm_score"),
                _ScoreOnlyDetector("rep_stiff_combined_dc_rsm_score"),
                _ScoreOnlyDetector("rep_stiff_combined_new"),
                _ScoreOnlyDetector("rep_stiff_combined_new3"),
            ])

        # --- 2. Read data and calculate all scores ---
        print(f"Reading data from {args.input_file} and calculating all scores...")
        all_scores_list = []
        rep_stiff_scores_list = []
        progress = None
        if args.resume:
            from eval_scripts.collection.checkpoints import EvaluationProgress
            progress = EvaluationProgress(args)
        with open(args.input_file, 'r') as f:
            for line in tqdm(f, desc="Processing samples"):
                data_item = json.loads(line)
                sample_key = progress.key(data_item) if progress else None
                cached = progress.load(sample_key) if progress else None
                if cached and cached.get("complete"):
                    all_scores_list.append(cached["scores"])
                    if rep_stiff_layers:
                        rep_stiff_scores_list.append(cached["representation"])
                    continue
                scores = {
                    "ground_truth_label": data_item['ground_truth_label'],
                    "data_source": data_item.get('data_source', 'unknown'),
                    "original_user_content": data_item.get('original_user_content')
                }
                if cached:
                    scores.update(cached["scores"])
                for layer_name, detector in rep_stiff_detectors.items():
                    layer_columns = [f"rep_stiff_{metric}_{layer_name}_score"
                                     for metric in ("rsi", "rsm", "directional_collapse")]
                    if all(column in scores for column in layer_columns):
                        continue
                    rep_stiff_scores, _paths = detector.calculate_scores(data_item)
                    scores[f"rep_stiff_rsi_{layer_name}_score"] = rep_stiff_scores.get("rsi_score")
                    scores[f"rep_stiff_rsm_{layer_name}_score"] = rep_stiff_scores.get("rsm_score")
                    scores[f"rep_stiff_directional_collapse_{layer_name}_score"] = rep_stiff_scores.get("directional_collapse_score")
                    if progress:
                        progress.save(sample_key, scores)

                if args.rep_stiff_combined_weights or args.rep_stiff_combined_fixed:
                    layer_features = {}
                    for layer_name in rep_stiff_layers:
                        layer_features[f"rsi_{layer_name}"] = scores.get(f"rep_stiff_rsi_{layer_name}_score")
                        layer_features[f"rsm_{layer_name}"] = scores.get(f"rep_stiff_rsm_{layer_name}_score")
                        layer_features[f"directional_collapse_{layer_name}"] = scores.get(
                            f"rep_stiff_directional_collapse_{layer_name}_score"
                        )
                    combined_score = None
                    if args.rep_stiff_combined_weights:
                        combined_score = RepStiffDetector.compute_combined_score(
                            layer_features,
                            args.rep_stiff_combined_weights,
                        )
                    if combined_score is None and args.rep_stiff_combined_fixed:
                        if args.rep_stiff_combined_rule == "trend_v2":
                            combined_score = RepStiffDetector.compute_fixed_trend_v2_score(layer_features)
                        elif args.rep_stiff_combined_rule == "trend_v3":
                            combined_score = RepStiffDetector.compute_fixed_trend_v3_score(layer_features)
                        elif args.rep_stiff_combined_rule == "trend_v4":
                            combined_score = RepStiffDetector.compute_fixed_trend_v4_score(layer_features)
                        else:
                            combined_score = RepStiffDetector.compute_fixed_trend_score(layer_features)
                    scores["rep_stiff_combined_score"] = combined_score
                    if args.rep_stiff_combined_fixed:
                        scores["rep_stiff_combined_trend_v1_score"] = RepStiffDetector.compute_fixed_trend_score(
                            layer_features
                        )
                        scores["rep_stiff_combined_trend_v2_score"] = RepStiffDetector.compute_fixed_trend_v2_score(
                            layer_features
                        )
                        scores["rep_stiff_combined_trend_v3_score"] = RepStiffDetector.compute_fixed_trend_v3_score(
                            layer_features
                        )
                        scores["rep_stiff_combined_trend_v4_score"] = RepStiffDetector.compute_fixed_trend_v4_score(
                            layer_features
                        )

                for detector in detectors:
                    if isinstance(detector, (_ScoreOnlyDetector, SelfCritiqueLaRADetector)):
                        # Scores already populated from RepStiff; skip recompute.
                        continue
                    scores[detector.get_name()] = detector.calculate_score(data_item)

                rep_stiff_entry = {
                    "ground_truth_label": data_item['ground_truth_label'],
                    "data_source": data_item.get('data_source', 'unknown'),
                    "original_user_content": data_item.get('original_user_content'),
                }
                for layer_name in rep_stiff_layers:
                    rep_stiff_entry[f"rsm_{layer_name}"] = scores.get(f"rep_stiff_rsm_{layer_name}_score")
                    rep_stiff_entry[f"directional_collapse_{layer_name}"] = scores.get(
                        f"rep_stiff_directional_collapse_{layer_name}_score"
                    )
                    rep_stiff_entry[f"rsi_{layer_name}"] = scores.get(f"rep_stiff_rsi_{layer_name}_score")
                if args.rep_stiff_combined_weights or args.rep_stiff_combined_fixed:
                    rep_stiff_entry["combined_score"] = scores.get("rep_stiff_combined_score")
                if args.rep_stiff_combined_fixed:
                    rep_stiff_entry["combined_trend_v1_score"] = scores.get("rep_stiff_combined_trend_v1_score")
                    rep_stiff_entry["combined_trend_v2_score"] = scores.get("rep_stiff_combined_trend_v2_score")
                    rep_stiff_entry["combined_trend_v3_score"] = scores.get("rep_stiff_combined_trend_v3_score")
                    rep_stiff_entry["combined_trend_v4_score"] = scores.get("rep_stiff_combined_trend_v4_score")
                if rep_stiff_layers:
                    rep_stiff_scores_list.append(rep_stiff_entry)
                all_scores_list.append(scores)
                if progress:
                    progress.save(sample_key, scores, rep_stiff_entry, complete=True)

        if not all_scores_list:
            raise ValueError("Evaluation input contains no samples")
        df_scores = pd.DataFrame(all_scores_list)

        if rep_stiff_layers:
            _compute_rep_stiff_derived_columns(
                df_scores, args, rep_stiff_layers, rep_stiff_scores_list
            )

        output_dir = os.path.dirname(args.output_summary_json) or "."
        all_scores_jsonl = os.path.join(output_dir, "all_method_scores.jsonl")
        df_scores.to_json(all_scores_jsonl, orient="records", lines=True)
        print(f"\nPer-sample scores for all methods saved to: {all_scores_jsonl}")

        method_directions_json = os.path.join(output_dir, "method_directions.json")
        method_directions = {
            detector.get_name(): detector.get_direction()
            for detector in detectors
            if detector.get_name() in df_scores.columns
        }
        with open(method_directions_json, "w") as f:
            json.dump(method_directions, f, indent=2)
        print(f"Method score directions saved to: {method_directions_json}")

        # --- 3. Evaluate all methods ---
        final_evaluation = {}
        print("\nCalculating and saving evaluation results...")
        for detector in detectors:
            method_name = detector.get_name()
            direction = detector.get_direction()

            if method_name not in df_scores.columns:
                print(f"[warn] Skipping {method_name}: score column not computed for this run")
                continue

            df_method = df_scores.dropna(subset=[method_name])
            y_true = df_method['ground_truth_label'].values
            y_scores = df_method[method_name].values * direction

            overall_perf = evaluate_performance_pop(y_true, y_scores)

            breakdown, mean_auc = evaluate_by_source(df_method, method_name, direction)

            final_evaluation[method_name] = {
                "overall_performance": overall_perf,
                "mean_auc": mean_auc,
                "breakdown_by_source": breakdown
            }

        with open(args.output_summary_json, 'w') as f:
            json.dump(final_evaluation, f, indent=4)
        print(f"\nEvaluation summary of all methods saved to: {args.output_summary_json}")

        if rep_stiff_scores_list:
            rep_stiff_out = args.rep_stiff_scores_json
            if rep_stiff_out is None:
                rep_stiff_out = os.path.join(
                    os.path.dirname(args.output_summary_json),
                    "rep_stiff_scores.json",
                )
            with open(rep_stiff_out, 'w') as f:
                json.dump(rep_stiff_scores_list, f, indent=2)
            print(f"\nRepStiff per-sample scores saved to: {rep_stiff_out}")


# Compatibility functions preserve the existing importable API.
def evaluate_performance_pop(y_true, y_scores):
    """
    A helper function to calculate a complete set of evaluation metrics,
    including AUC, optimal F1, Youden's J, TPR at fixed FPR,
    and F1 and Accuracy corresponding to Youden's J threshold.
    """
    return EvaluationMetrics.evaluate_performance_pop(y_true, y_scores)


def evaluate_performance(y_true, y_scores):
    """
    A helper function to calculate a complete set of evaluation metrics,
    including AUC, optimal F1, Youden's J, TPR at fixed FPR,
    and F1 and Accuracy corresponding to Youden's J threshold.
    """
    return EvaluationMetrics.evaluate_performance(y_true, y_scores)


def segment_lengths_9_9_10(n_layers: int) -> Tuple[int, int, int]:
    """Early / mid / late run lengths in a 9:9:10 ratio (sums to n_layers)."""
    return LayerFeatures.segment_lengths_9_9_10(n_layers)


def _layer_sort_key(layer_name: str) -> Tuple[int, str]:
    return LayerFeatures._layer_sort_key(layer_name)


def _ordered_rep_stiff_layers(layers: List[str]) -> List[str]:
    return LayerFeatures._ordered_rep_stiff_layers(layers)


def _numeric_layer_indices(layers: List[str]) -> List[Tuple[int, str]]:
    return LayerFeatures._numeric_layer_indices(layers)


_LEGACY_SEGMENT_LAYERS = frozenset({"early", "mid", "late"})


def _legacy_segment_layer_features(df_scores: pd.DataFrame, ordered_layers: List[str]) -> Dict[str, pd.Series]:
    """Map rep_stiff_*_{early,mid,late}_score columns to segment feature series."""
    return LayerFeatures._legacy_segment_layer_features(df_scores, ordered_layers)


def evaluate_by_source(df_scores, method_name, direction):
    """Macro AUC averages only sources containing both membership classes."""
    return EvaluationMetrics.evaluate_by_source(df_scores, method_name, direction)


def _compute_rep_stiff_combined_score_new2(df_scores: pd.DataFrame, ordered_layers: List[str]) -> Optional[pd.Series]:
    """Early-vs-late slope combined score (numeric L* layers or legacy early/mid/late)."""
    return LayerFeatures._compute_rep_stiff_combined_score_new2(df_scores, ordered_layers)


def _build_segment_layer_features(df_scores: pd.DataFrame, ordered_layers: List[str]) -> Dict[str, pd.Series]:
    """Map per-layer columns to rsi/rsm/dc_{early,mid,late} means for trend rules."""
    return LayerFeatures._build_segment_layer_features(df_scores, ordered_layers)


def _aligned_layer_rank_columns(df_scores: pd.DataFrame, ordered_layers: List[str]) -> Tuple[List[str], List[str]]:
    """Percentile ranks per metric/layer and per-layer aligned rigidity s_l."""
    return LayerFeatures._aligned_layer_rank_columns(df_scores, ordered_layers)


def _compute_rep_stiff_derived_columns(df_scores: pd.DataFrame, args: argparse.Namespace, rep_stiff_layers: List[str], rep_stiff_scores_list: Optional[List[dict]]) -> None:
    """Post-process per-layer RepStiff scores into combined / LaRA detector columns."""
    return RepresentationScoreAggregator._compute_rep_stiff_derived_columns(df_scores, args, rep_stiff_layers, rep_stiff_scores_list)


def main():
    return DetectorEvaluationApplication().run()

if __name__ == '__main__':
    main()
