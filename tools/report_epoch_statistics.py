#!/usr/bin/env python3
"""Report bootstrap CIs and paired significance tests for EURUS epoch runs."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


METRICS = (
    "roc_auc",
    "best_f1_score",
    "accuracy_at_best_f1",
    "youden_j_score",
    "f1_at_youden_threshold",
    "accuracy_at_youden_threshold",
    "tpr_at_fpr_5",
)

ID_COLUMNS = {"ground_truth_label", "data_source", "original_user_content"}


class ClassificationMetrics:
    """Compute the existing classification and ROC metrics."""

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


    @staticmethod
    def auc(x: np.ndarray, y: np.ndarray) -> float:
        # Explicit trapezoids work with both NumPy 1.x and 2.x.
        return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


    @staticmethod
    def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true).astype(int)
        y_scores = np.asarray(y_scores, dtype=float)
        order = np.argsort(y_scores, kind="mergesort")[::-1]
        y_true = y_true[order]
        y_scores = y_scores[order]

        distinct_indices = np.where(np.diff(y_scores))[0]
        threshold_indices = np.r_[distinct_indices, y_true.size - 1]
        tps = np.cumsum(y_true == 1)[threshold_indices]
        fps = 1 + threshold_indices - tps

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[np.inf, y_scores[threshold_indices]]

        positives = np.sum(y_true == 1)
        negatives = np.sum(y_true == 0)
        tpr = tps / positives
        fpr = fps / negatives
        return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


    @staticmethod
    def precision_recall_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true).astype(int)
        y_scores = np.asarray(y_scores, dtype=float)
        order = np.argsort(y_scores, kind="mergesort")[::-1]
        y_true = y_true[order]
        y_scores = y_scores[order]

        distinct_indices = np.where(np.diff(y_scores))[0]
        threshold_indices = np.r_[distinct_indices, y_true.size - 1]
        tps = np.cumsum(y_true == 1)[threshold_indices]
        fps = 1 + threshold_indices - tps

        precision = tps / np.maximum(tps + fps, 1)
        recall = tps / np.sum(y_true == 1)
        thresholds = y_scores[threshold_indices]

        precision = np.r_[precision, 1.0]
        recall = np.r_[recall, 0.0]
        return precision.astype(float), recall.astype(float), thresholds.astype(float)


    @staticmethod
    def evaluate_performance(y_true: Iterable[float], y_scores: Iterable[float]) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        y_scores = pd.to_numeric(pd.Series(y_scores), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y_scores)
        y_true = y_true[finite]
        y_scores = y_scores[finite]

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return {metric: np.nan for metric in METRICS}

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        fscore = (2 * precision * recall) / (precision + recall + 1e-6)
        best_f1_idx = np.argmax(fscore[:-1]) if len(fscore) > 1 else 0
        optimal_threshold_f1 = pr_thresholds[best_f1_idx]
        best_f1 = fscore[best_f1_idx]
        y_pred_f1 = (y_scores >= optimal_threshold_f1).astype(int)
        accuracy_at_best_f1 = accuracy_score(y_true, y_pred_f1)

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
        f1_at_youden_threshold = (
            2 * precision_youden * recall_youden
        ) / (precision_youden + recall_youden + 1e-6)
        accuracy_at_youden_threshold = accuracy_score(y_true, y_pred_youden)

        target_fpr = 0.05
        eligible = tpr[fpr <= target_fpr]
        tpr_at_fpr_5 = float(np.max(eligible)) if eligible.size else np.nan

        return {
            "roc_auc": float(roc_auc),
            "best_f1_score": float(best_f1),
            "accuracy_at_best_f1": float(accuracy_at_best_f1),
            "youden_j_score": float(youden_j_score),
            "f1_at_youden_threshold": float(f1_at_youden_threshold),
            "accuracy_at_youden_threshold": float(accuracy_at_youden_threshold),
            "tpr_at_fpr_5": float(tpr_at_fpr_5),
        }


class StatisticalComparisons:
    """Calculate bootstrap summaries and paired permutation tests."""

    @staticmethod
    def bootstrap_summary(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        rng: np.random.Generator,
        n_bootstrap: int,
        ci: float,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        observed = evaluate_performance(y_true, y_scores)
        samples: Dict[str, List[float]] = {metric: [] for metric in METRICS}
        n = len(y_true)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            perf = evaluate_performance(y_true[idx], y_scores[idx])
            for metric, value in perf.items():
                if np.isfinite(value):
                    samples[metric].append(value)

        alpha = (1.0 - ci) / 2.0
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for metric in METRICS:
            values = np.asarray(samples[metric], dtype=float)
            if len(values) == 0:
                out[metric] = {
                    "estimate": observed[metric],
                    "bootstrap_mean": None,
                    "bootstrap_median": None,
                    "bootstrap_std": None,
                    "ci_low": None,
                    "ci_high": None,
                    "n_bootstrap_valid": 0,
                }
                continue
            out[metric] = {
                "estimate": observed[metric],
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": float(np.quantile(values, alpha)),
                "ci_high": float(np.quantile(values, 1.0 - alpha)),
                "n_bootstrap_valid": int(len(values)),
            }
        return out


    @staticmethod
    def paired_permutation_test(
        y_true: np.ndarray,
        baseline_scores: np.ndarray,
        comparison_scores: np.ndarray,
        rng: np.random.Generator,
        n_permutations: int,
        metric: str,
    ) -> Dict[str, Optional[float]]:
        baseline_perf = evaluate_performance(y_true, baseline_scores)
        comparison_perf = evaluate_performance(y_true, comparison_scores)
        observed_diff = comparison_perf[metric] - baseline_perf[metric]
        if not np.isfinite(observed_diff):
            return {
                "baseline_estimate": baseline_perf[metric],
                "comparison_estimate": comparison_perf[metric],
                "observed_difference": None,
                "p_value_two_sided": None,
                "n_permutations": 0,
            }

        extreme = 0
        valid = 0
        n = len(y_true)
        for _ in range(n_permutations):
            swap = rng.random(n) < 0.5
            perm_baseline = np.where(swap, comparison_scores, baseline_scores)
            perm_comparison = np.where(swap, baseline_scores, comparison_scores)
            perm_diff = (
                evaluate_performance(y_true, perm_comparison)[metric]
                - evaluate_performance(y_true, perm_baseline)[metric]
            )
            if not np.isfinite(perm_diff):
                continue
            valid += 1
            if abs(perm_diff) >= abs(observed_diff):
                extreme += 1

        return {
            "baseline_estimate": baseline_perf[metric],
            "comparison_estimate": comparison_perf[metric],
            "observed_difference": float(observed_diff),
            "p_value_two_sided": float((extreme + 1) / (valid + 1)) if valid else None,
            "n_permutations": int(valid),
        }


class EpochRunRepository:
    """Locate, load and align epoch evaluation results."""

    @staticmethod
    def model_run_dir(results_root: str, model: str, dataset_tag: str, run_subdir: str) -> str:
        return os.path.join(results_root, model, dataset_tag, run_subdir)


    @staticmethod
    def parse_model_specs(args: argparse.Namespace) -> List[Dict[str, str]]:
        if args.model_specs:
            specs = []
            for spec in args.model_specs:
                if ":" not in spec:
                    raise ValueError(
                        f"Invalid --model-specs entry {spec!r}. Expected MODEL_NAME:DATASET_TAG."
                    )
                model, dataset_tag = spec.split(":", 1)
                if not model or not dataset_tag:
                    raise ValueError(
                        f"Invalid --model-specs entry {spec!r}. Expected MODEL_NAME:DATASET_TAG."
                    )
                specs.append({"model": model, "dataset_tag": dataset_tag})
            return specs

        if not args.models:
            raise ValueError("Provide either --models or --model-specs.")
        return [{"model": model, "dataset_tag": args.dataset_tag} for model in args.models]


    @staticmethod
    def load_model_run(run_dir: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        scores_path = os.path.join(run_dir, "all_method_scores.jsonl")
        directions_path = os.path.join(run_dir, "method_directions.json")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(
                f"Missing {scores_path}. Re-run the workflow after this patch so per-sample scores are exported."
            )
        df = pd.read_json(scores_path, lines=True)
        if os.path.exists(directions_path):
            with open(directions_path, "r") as f:
                directions = json.load(f)
        else:
            directions = {}
        return df, {k: int(v) for k, v in directions.items()}


    @staticmethod
    def method_columns(df: pd.DataFrame, directions: Dict[str, int]) -> List[str]:
        if directions:
            return [col for col in directions if col in df.columns]
        cols = []
        for col in df.columns:
            if col in ID_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        return cols


    @staticmethod
    def clean_method_arrays(
        df: pd.DataFrame,
        method: str,
        direction: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        subset = df[["ground_truth_label", method]].copy()
        subset[method] = pd.to_numeric(subset[method], errors="coerce")
        subset = subset.dropna(subset=[method])
        y_true = subset["ground_truth_label"].to_numpy(dtype=int)
        y_scores = subset[method].to_numpy(dtype=float) * direction
        return y_true, y_scores


    @staticmethod
    def attach_row_key(df: pd.DataFrame) -> pd.DataFrame:
        keyed = df.copy()
        content = keyed.get("original_user_content", pd.Series(index=keyed.index, dtype=object))
        keyed["_base_key"] = (
            keyed.get("data_source", "unknown").astype(str)
            + "||"
            + keyed["ground_truth_label"].astype(str)
            + "||"
            + content.fillna("").astype(str)
        )
        keyed["_dup_idx"] = keyed.groupby("_base_key").cumcount()
        keyed["_row_key"] = keyed["_base_key"] + "||" + keyed["_dup_idx"].astype(str)
        return keyed


    @staticmethod
    def matched_arrays(
        baseline_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        method: str,
        baseline_direction: int,
        comparison_direction: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = attach_row_key(baseline_df)[["_row_key", "ground_truth_label", method]].rename(
            columns={method: "baseline_score"}
        )
        comp = attach_row_key(comparison_df)[["_row_key", "ground_truth_label", method]].rename(
            columns={method: "comparison_score", "ground_truth_label": "comparison_label"}
        )
        merged = base.merge(comp, on="_row_key", how="inner")
        merged = merged[merged["ground_truth_label"] == merged["comparison_label"]]
        merged["baseline_score"] = pd.to_numeric(merged["baseline_score"], errors="coerce")
        merged["comparison_score"] = pd.to_numeric(merged["comparison_score"], errors="coerce")
        merged = merged.dropna(subset=["baseline_score", "comparison_score"])
        y_true = merged["ground_truth_label"].to_numpy(dtype=int)
        baseline_scores = merged["baseline_score"].to_numpy(dtype=float) * baseline_direction
        comparison_scores = merged["comparison_score"].to_numpy(dtype=float) * comparison_direction
        return y_true, baseline_scores, comparison_scores


class MarkdownReportWriter:
    """Render the existing epoch comparison report."""

    @staticmethod
    def write_markdown_summary(report: Dict, output_markdown: str) -> None:
        lines = [
            "# EURUS Epoch Statistical Report",
            "",
            "Bootstrap rows show observed ROC-AUC with percentile confidence intervals.",
            "Significance rows compare each epoch model against the first model via paired permutation tests.",
            "",
            "## Bootstrap ROC-AUC",
            "",
            "| Model | Method | ROC-AUC | CI Low | CI High | Bootstrap Std |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for model, model_stats in report["bootstrap"].items():
            for method, stats in model_stats.items():
                roc = stats.get("roc_auc", {})
                if roc.get("estimate") is None or not np.isfinite(roc.get("estimate", np.nan)):
                    continue
                lines.append(
                    f"| {model} | {method} | {roc['estimate']:.4f} | "
                    f"{roc['ci_low']:.4f} | {roc['ci_high']:.4f} | {roc['bootstrap_std']:.4f} |"
                )

        lines.extend([
            "",
            "## Paired Significance Tests",
            "",
            "| Comparison | Method | Metric | Difference | p-value | Matched n |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ])
        for comparison, comparison_stats in report["significance_tests"].items():
            for method, method_stats in comparison_stats.items():
                matched_n = method_stats.get("matched_n", 0)
                for metric, stats in method_stats.get("metrics", {}).items():
                    diff = stats.get("observed_difference")
                    p_value = stats.get("p_value_two_sided")
                    if diff is None or p_value is None:
                        continue
                    lines.append(
                        f"| {comparison} | {method} | {metric} | {diff:.4f} | {p_value:.4g} | {matched_n} |"
                    )

        with open(output_markdown, "w") as f:
            f.write("\n".join(lines) + "\n")


class EpochStatisticsApplication:
    """Own the command-line configuration and execute the existing workflow."""

    def __init__(self):
        self.parser = self.create_parser()

    @staticmethod
    def create_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--results-root", required=True)
        parser.add_argument("--models", nargs="+", default=None)
        parser.add_argument(
            "--model-specs",
            nargs="+",
            default=None,
            help="Optional MODEL_NAME:DATASET_TAG pairs when runs do not share one dataset tag.",
        )
        parser.add_argument("--dataset-tag", default="eurus_member")
        parser.add_argument("--run-subdir", default="_all__all_samples/blank_k_sweep/k1")
        parser.add_argument("--n-bootstrap", type=int, default=10000)
        parser.add_argument("--n-permutations", type=int, default=10000)
        parser.add_argument("--ci", type=float, default=0.95)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--metrics", nargs="+", default=["roc_auc", "tpr_at_fpr_5", "best_f1_score"])
        parser.add_argument(
            "--methods",
            nargs="+",
            default=None,
            help="Optional method score columns to include. Defaults to all common numeric score columns.",
        )
        parser.add_argument("--output-json", required=True)
        parser.add_argument("--output-markdown", default=None)
        return parser

    def run(self):
        parser = self.parser
        args = parser.parse_args()
        rng = np.random.default_rng(args.seed)
        model_specs = parse_model_specs(args)
        model_names = [spec["model"] for spec in model_specs]
        runs = {}
        for spec in model_specs:
            model = spec["model"]
            run_dir = model_run_dir(args.results_root, model, spec["dataset_tag"], args.run_subdir)
            df, directions = load_model_run(run_dir)
            runs[model] = {
                "run_dir": run_dir,
                "dataset_tag": spec["dataset_tag"],
                "df": df,
                "directions": directions,
            }

        report = {
            "config": {
                "models": model_specs,
                "run_subdir": args.run_subdir,
                "n_bootstrap": args.n_bootstrap,
                "n_permutations": args.n_permutations,
                "ci": args.ci,
                "seed": args.seed,
                "significance_metrics": args.metrics,
            },
            "bootstrap": {},
            "significance_tests": {},
        }

        common_methods = None
        for model, run in runs.items():
            methods = set(method_columns(run["df"], run["directions"]))
            common_methods = methods if common_methods is None else common_methods & methods
        common_methods = sorted(common_methods or [])
        if args.methods:
            requested_methods = set(args.methods)
            missing_methods = sorted(requested_methods - set(common_methods))
            if missing_methods:
                print(f"[warn] Skipping requested methods not present in every run: {missing_methods}")
            common_methods = [method for method in args.methods if method in common_methods]

        for model, run in runs.items():
            report["bootstrap"][model] = {}
            for method in common_methods:
                direction = run["directions"].get(method, 1)
                y_true, y_scores = clean_method_arrays(run["df"], method, direction)
                if len(y_true) == 0:
                    continue
                report["bootstrap"][model][method] = bootstrap_summary(
                    y_true, y_scores, rng, args.n_bootstrap, args.ci
                )

        baseline_model = model_names[0]
        for comparison_model in model_names[1:]:
            comparison_key = f"{comparison_model} vs {baseline_model}"
            report["significance_tests"][comparison_key] = {}
            for method in common_methods:
                y_true, baseline_scores, comparison_scores = matched_arrays(
                    runs[baseline_model]["df"],
                    runs[comparison_model]["df"],
                    method,
                    runs[baseline_model]["directions"].get(method, 1),
                    runs[comparison_model]["directions"].get(method, 1),
                )
                if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                    continue
                metric_tests = {}
                for metric in args.metrics:
                    metric_tests[metric] = paired_permutation_test(
                        y_true,
                        baseline_scores,
                        comparison_scores,
                        rng,
                        args.n_permutations,
                        metric,
                    )
                report["significance_tests"][comparison_key][method] = {
                    "matched_n": int(len(y_true)),
                    "metrics": metric_tests,
                }

        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Statistical report saved to: {args.output_json}")

        output_markdown = args.output_markdown
        if output_markdown is None:
            root, _ = os.path.splitext(args.output_json)
            output_markdown = f"{root}.md"
        write_markdown_summary(report, output_markdown)
        print(f"Markdown summary saved to: {output_markdown}")


# Compatibility functions preserve the existing importable API.
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ClassificationMetrics.accuracy_score(y_true, y_pred)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return ClassificationMetrics.auc(x, y)


def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ClassificationMetrics.roc_curve(y_true, y_scores)


def precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ClassificationMetrics.precision_recall_curve(y_true, y_scores)


def evaluate_performance(y_true: Iterable[float], y_scores: Iterable[float]) -> Dict[str, float]:
    return ClassificationMetrics.evaluate_performance(y_true, y_scores)


def bootstrap_summary(y_true: np.ndarray, y_scores: np.ndarray, rng: np.random.Generator, n_bootstrap: int, ci: float) -> Dict[str, Dict[str, Optional[float]]]:
    return StatisticalComparisons.bootstrap_summary(y_true, y_scores, rng, n_bootstrap, ci)


def paired_permutation_test(y_true: np.ndarray, baseline_scores: np.ndarray, comparison_scores: np.ndarray, rng: np.random.Generator, n_permutations: int, metric: str) -> Dict[str, Optional[float]]:
    return StatisticalComparisons.paired_permutation_test(y_true, baseline_scores, comparison_scores, rng, n_permutations, metric)


def model_run_dir(results_root: str, model: str, dataset_tag: str, run_subdir: str) -> str:
    return EpochRunRepository.model_run_dir(results_root, model, dataset_tag, run_subdir)


def parse_model_specs(args: argparse.Namespace) -> List[Dict[str, str]]:
    return EpochRunRepository.parse_model_specs(args)


def load_model_run(run_dir: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    return EpochRunRepository.load_model_run(run_dir)


def method_columns(df: pd.DataFrame, directions: Dict[str, int]) -> List[str]:
    return EpochRunRepository.method_columns(df, directions)


def clean_method_arrays(df: pd.DataFrame, method: str, direction: int) -> Tuple[np.ndarray, np.ndarray]:
    return EpochRunRepository.clean_method_arrays(df, method, direction)


def attach_row_key(df: pd.DataFrame) -> pd.DataFrame:
    return EpochRunRepository.attach_row_key(df)


def matched_arrays(baseline_df: pd.DataFrame, comparison_df: pd.DataFrame, method: str, baseline_direction: int, comparison_direction: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return EpochRunRepository.matched_arrays(baseline_df, comparison_df, method, baseline_direction, comparison_direction)


def write_markdown_summary(report: Dict, output_markdown: str) -> None:
    return MarkdownReportWriter.write_markdown_summary(report, output_markdown)


def main() -> None:
    return EpochStatisticsApplication().run()


if __name__ == "__main__":
    main()
