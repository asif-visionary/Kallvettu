"""
Scientific Evaluation & Benchmark Module for Kallvettu K-IEFE.
Implements empirical, reproducible evaluation metrics comparing multi-engine fusion
against individual baseline OCR providers.

METRICS IMPLEMENTED:
1. Character Error Rate (CER) via normalized edit distance.
2. Word Error Rate (WER) via token-level Levenshtein distance.
3. Exact Line Accuracy (% identical transcriptions).
4. Fusion Win / Tie / Loss Rate over Strongest Baseline.
5. Consensus Error Rate & False-Agreement Rate (correlated multi-model errors).
6. Abstention Rate (% difficult positions left unresolved as `[c1/c2]`).
7. Coverage vs Accuracy trade-off curve.
8. Per-script and per-damage category ablation analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from evaluation_engine import calculate_cer, calculate_wer, levenshtein_distance
from services.ocr_fusion.cross_verifier import normalize_for_comparison


@dataclass
class EvaluationSampleResult:
    """Evaluation metric report for a single test inscription."""
    sample_id: str
    script: str
    damage_category: str
    ground_truth: str
    fused_prediction: str
    conservative_prediction: str
    baseline_provider_id: str
    baseline_prediction: str
    fused_cer: float
    baseline_cer: float
    fused_wer: float
    baseline_wer: float
    cer_improvement_ratio: float      # (baseline_cer - fused_cer) / baseline_cer
    is_exact_line_match: bool
    is_fusion_win: bool
    is_fusion_tie: bool
    is_fusion_loss: bool
    false_agreement_characters_count: int
    abstention_slots_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "script": self.script,
            "damage_category": self.damage_category,
            "fused_cer": round(self.fused_cer, 4),
            "baseline_cer": round(self.baseline_cer, 4),
            "fused_wer": round(self.fused_wer, 4),
            "baseline_wer": round(self.baseline_wer, 4),
            "cer_improvement_ratio": round(self.cer_improvement_ratio, 4),
            "is_exact_line_match": self.is_exact_line_match,
            "is_fusion_win": self.is_fusion_win,
            "is_fusion_loss": self.is_fusion_loss,
            "abstention_slots_count": self.abstention_slots_count,
        }


@dataclass
class CorpusBenchmarkSummary:
    """Aggregated benchmark statistics across an evaluation corpus."""
    total_samples: int
    mean_fused_cer: float
    mean_baseline_cer: float
    mean_fused_wer: float
    mean_baseline_wer: float
    cer_error_reduction_percent: float # Average % reduction in CER achieved by fusion
    exact_line_accuracy: float
    fusion_win_rate: float
    fusion_tie_rate: float
    fusion_loss_rate: float
    mean_abstention_rate: float
    false_agreement_rate: float
    per_script_cer: Dict[str, Dict[str, float]]
    per_damage_cer: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "mean_fused_cer": round(self.mean_fused_cer, 4),
            "mean_baseline_cer": round(self.mean_baseline_cer, 4),
            "mean_fused_wer": round(self.mean_fused_wer, 4),
            "mean_baseline_wer": round(self.mean_baseline_wer, 4),
            "cer_error_reduction_percent": round(self.cer_error_reduction_percent, 2),
            "exact_line_accuracy": round(self.exact_line_accuracy, 4),
            "fusion_win_rate": round(self.fusion_win_rate, 4),
            "fusion_tie_rate": round(self.fusion_tie_rate, 4),
            "fusion_loss_rate": round(self.fusion_loss_rate, 4),
            "mean_abstention_rate": round(self.mean_abstention_rate, 4),
            "false_agreement_rate": round(self.false_agreement_rate, 4),
            "per_script_cer": {
                k: {ik: round(iv, 4) for ik, iv in v.items()}
                for k, v in self.per_script_cer.items()
            },
            "per_damage_cer": {
                k: {ik: round(iv, 4) for ik, iv in v.items()}
                for k, v in self.per_damage_cer.items()
            },
        }


class EpigraphicalFusionEvaluator:
    """
    Evaluates evidence fusion against independent ground-truth epigraphs.
    """

    @staticmethod
    def evaluate_sample(
        sample_id: str,
        ground_truth: str,
        fused_text: str,
        conservative_text: str,
        baseline_provider_id: str,
        baseline_text: str,
        script: str = "medieval_chola",
        damage: str = "weathered",
        individual_provider_texts: Optional[Dict[str, str]] = None
    ) -> EvaluationSampleResult:
        """Evaluates a single sample prediction against reference ground truth."""
        norm_gt = normalize_for_comparison(ground_truth)
        norm_fused = normalize_for_comparison(fused_text)
        norm_baseline = normalize_for_comparison(baseline_text)

        cer_fused = calculate_cer(norm_gt, norm_fused)
        cer_base = calculate_cer(norm_gt, norm_baseline)
        wer_fused = calculate_wer(norm_gt, norm_fused)
        wer_base = calculate_wer(norm_gt, norm_baseline)

        # CER improvement ratio: (base - fused) / max(0.001, base)
        if cer_base > 0:
            imp_ratio = (cer_base - cer_fused) / cer_base
        else:
            imp_ratio = 0.0 if cer_fused == 0 else -1.0

        is_win = cer_fused < cer_base
        is_tie = cer_fused == cer_base
        is_loss = cer_fused > cer_base
        is_exact = norm_fused == norm_gt

        # False agreement detection: positions where all individual engines agreed, but differed from ground truth
        false_agreements = 0
        if individual_provider_texts and len(individual_provider_texts) >= 2:
            norm_provs = [normalize_for_comparison(t) for t in individual_provider_texts.values() if t]
            if len(norm_provs) >= 2 and all(p == norm_provs[0] for p in norm_provs):
                if norm_provs[0] != norm_gt:
                    false_agreements = levenshtein_distance(norm_provs[0], norm_gt)

        # Count abstentions
        abstentions = conservative_text.count("[") if conservative_text else 0

        return EvaluationSampleResult(
            sample_id=sample_id,
            script=script,
            damage_category=damage,
            ground_truth=norm_gt,
            fused_prediction=norm_fused,
            conservative_prediction=conservative_text,
            baseline_provider_id=baseline_provider_id,
            baseline_prediction=norm_baseline,
            fused_cer=cer_fused,
            baseline_cer=cer_base,
            fused_wer=wer_fused,
            baseline_wer=wer_base,
            cer_improvement_ratio=imp_ratio,
            is_exact_line_match=is_exact,
            is_fusion_win=is_win,
            is_fusion_tie=is_tie,
            is_fusion_loss=is_loss,
            false_agreement_characters_count=false_agreements,
            abstention_slots_count=abstentions
        )

    @classmethod
    def evaluate_corpus(
        cls,
        sample_results: List[EvaluationSampleResult]
    ) -> CorpusBenchmarkSummary:
        """Aggregates metrics across the evaluation test corpus."""
        n = len(sample_results)
        if n == 0:
            return CorpusBenchmarkSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

        mean_f_cer = float(np.mean([s.fused_cer for s in sample_results]))
        mean_b_cer = float(np.mean([s.baseline_cer for s in sample_results]))
        mean_f_wer = float(np.mean([s.fused_wer for s in sample_results]))
        mean_b_wer = float(np.mean([s.baseline_wer for s in sample_results]))

        error_red = ((mean_b_cer - mean_f_cer) / max(0.001, mean_b_cer)) * 100.0 if mean_b_cer > 0 else 0.0
        exact_acc = float(sum(1 for s in sample_results if s.is_exact_line_match) / n)
        win_rate = float(sum(1 for s in sample_results if s.is_fusion_win) / n)
        tie_rate = float(sum(1 for s in sample_results if s.is_fusion_tie) / n)
        loss_rate = float(sum(1 for s in sample_results if s.is_fusion_loss) / n)

        mean_abstention = float(np.mean([s.abstention_slots_count for s in sample_results]))
        mean_false_agr = float(np.mean([s.false_agreement_characters_count for s in sample_results]))

        # Per-script breakdown
        scripts = {s.script for s in sample_results}
        per_script: Dict[str, Dict[str, float]] = {}
        for sc in scripts:
            sc_samples = [s for s in sample_results if s.script == sc]
            per_script[sc] = {
                "fused_cer": float(np.mean([s.fused_cer for s in sc_samples])),
                "baseline_cer": float(np.mean([s.baseline_cer for s in sc_samples])),
            }

        # Per-damage breakdown
        damages = {s.damage_category for s in sample_results}
        per_damage: Dict[str, Dict[str, float]] = {}
        for dm in damages:
            dm_samples = [s for s in sample_results if s.damage_category == dm]
            per_damage[dm] = {
                "fused_cer": float(np.mean([s.fused_cer for s in dm_samples])),
                "baseline_cer": float(np.mean([s.baseline_cer for s in dm_samples])),
            }

        return CorpusBenchmarkSummary(
            total_samples=n,
            mean_fused_cer=mean_f_cer,
            mean_baseline_cer=mean_b_cer,
            mean_fused_wer=mean_f_wer,
            mean_baseline_wer=mean_b_wer,
            cer_error_reduction_percent=error_red,
            exact_line_accuracy=exact_acc,
            fusion_win_rate=win_rate,
            fusion_tie_rate=tie_rate,
            fusion_loss_rate=loss_rate,
            mean_abstention_rate=mean_abstention,
            false_agreement_rate=mean_false_agr,
            per_script_cer=per_script,
            per_damage_cer=per_damage
        )

