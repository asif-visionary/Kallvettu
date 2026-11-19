"""
Consensus Entropy & Disagreement Distribution Analysis for Kallvettu (K-IEFE).
Calculates information-theoretic Shannon entropy over candidate character distributions.

CRITICAL SCIENTIFIC PRINCIPLE:
Candidate-distribution entropy measures the dispersion of agreement among OCR models.
It does NOT represent the probability of historical correctness.
Low entropy (unanimous agreement) does NOT guarantee truth, as multiple neural models
can exhibit shared systematic failure modes or identical thresholding biases on weathered granite.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from services.ocr_fusion.cross_verifier import CharacterAlignmentUnit, CrossVerificationReport


class DisagreementSeverity:
    """Disagreement categories based on candidate distribution entropy."""
    UNANIMOUS = "unanimous"              # H = 0.0 (All active providers agree on candidate)
    STRONG_CONSENSUS = "strong_consensus"# 0.0 < H <= 0.5 (One candidate dominates)
    DIVIDED = "divided"                  # 0.5 < H <= 1.0 (Competing alternative candidates)
    HIGH_DISPUTE = "high_dispute"        # H > 1.0 (Severe multi-way disagreement or divergence)
    SINGLE_PROVIDER = "single_provider"  # Only one provider evaluated (insufficient evidence)
    NOT_ASSESSABLE = "not_assessable"    # Missing data or omitted slot (cannot compute)


@dataclass
class PositionEntropyResult:
    """Consensus and entropy evaluation for a specific character slot."""
    position_index: int
    entropy: float                       # Shannon entropy in bits (H >= 0.0)
    severity: str                         # DisagreementSeverity
    distinct_candidates: List[str]
    candidate_distribution: Dict[str, float]  # Probability distribution across candidates
    is_unanimous: bool
    is_disputed: bool
    dominant_candidate: str
    dominant_margin: float               # P(c_1) - P(c_2)
    participating_providers_count: int
    normalized_entropy: Optional[float] = None # H / log2(K) if K > 1 else (0.0 if K == 1 else None)
    is_assessable: bool = True           # False if single provider or missing data


@dataclass
class ConsensusReport:
    """Global consensus summary across the full inscription."""
    mean_entropy: float
    max_entropy: float
    total_slots_evaluated: int
    unanimous_slots_count: int
    disputed_slots_count: int
    unanimous_ratio: float
    disputed_ratio: float
    high_dispute_positions: List[int]
    position_entropy_details: List[PositionEntropyResult]
    severity_distribution: Dict[str, int]
    epigraphical_disclaimer: str = (
        "Consensus entropy measures model dispersion, not historical veracity. "
        "Unanimous agreement does not eliminate correlated recognition errors caused by shared "
        "preprocessing, image artifacts or surface degradation. Physical inscription evidence "
        "and epigraphist review remain essential."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_entropy": round(self.mean_entropy, 4),
            "max_entropy": round(self.max_entropy, 4),
            "total_slots_evaluated": self.total_slots_evaluated,
            "unanimous_slots_count": self.unanimous_slots_count,
            "disputed_slots_count": self.disputed_slots_count,
            "unanimous_ratio": round(self.unanimous_ratio, 4),
            "disputed_ratio": round(self.disputed_ratio, 4),
            "high_dispute_positions": self.high_dispute_positions,
            "severity_distribution": self.severity_distribution,
            "epigraphical_disclaimer": self.epigraphical_disclaimer,
            "position_entropy_details": [
                {
                    "position_index": p.position_index,
                    "entropy": round(p.entropy, 4),
                    "normalized_entropy": round(p.normalized_entropy, 4) if p.normalized_entropy is not None else None,
                    "severity": p.severity,
                    "is_assessable": p.is_assessable,
                    "dominant_candidate": p.dominant_candidate,
                    "distinct_candidates": p.distinct_candidates,
                    "candidate_distribution": {k: round(v, 4) for k, v in p.candidate_distribution.items()},
                    "participating_providers_count": p.participating_providers_count,
                }
                for p in self.position_entropy_details
            ],
        }


class ConsensusEntropyCalculator:
    """
    Computes Shannon entropy across candidate readings to identify areas of OCR discordance.
    """

    @staticmethod
    def calculate_position_entropy(
        alignment_unit: CharacterAlignmentUnit,
        provider_weights: Optional[Dict[str, float]] = None
    ) -> PositionEntropyResult:
        """
        Calculates Shannon entropy and normalized entropy for an aligned character slot:
        H(X) = - sum( P(c) * log2(P(c)) )
        H_norm(X) = H(X) / log2(K) when K > 1
        """
        raw_map = alignment_unit.normalized_by_provider
        # Filter out gap characters '-' and empty strings
        valid_candidates = {pid: char for pid, char in raw_map.items() if char and char != "-"}

        if not valid_candidates:
            return PositionEntropyResult(
                position_index=alignment_unit.position_index,
                entropy=0.0,
                severity=DisagreementSeverity.NOT_ASSESSABLE,
                distinct_candidates=[],
                candidate_distribution={},
                is_unanimous=False,
                is_disputed=False,
                dominant_candidate="",
                dominant_margin=0.0,
                participating_providers_count=0,
                normalized_entropy=None,
                is_assessable=False,
            )

        if len(valid_candidates) == 1:
            cand = list(valid_candidates.values())[0]
            return PositionEntropyResult(
                position_index=alignment_unit.position_index,
                entropy=0.0,
                severity=DisagreementSeverity.SINGLE_PROVIDER,
                distinct_candidates=[cand],
                candidate_distribution={cand: 1.0},
                is_unanimous=False,
                is_disputed=False,
                dominant_candidate=cand,
                dominant_margin=1.0,
                participating_providers_count=1,
                normalized_entropy=0.0,
                is_assessable=False,
            )

        # Weight distribution
        weights = provider_weights or {}
        candidate_weights: Dict[str, float] = {}
        total_weight = 0.0

        for pid, char in valid_candidates.items():
            w = weights.get(pid, 1.0)
            candidate_weights[char] = candidate_weights.get(char, 0.0) + w
            total_weight += w

        if total_weight <= 0:
            total_weight = 1.0

        # Normalized probability distribution
        probs: Dict[str, float] = {
            char: w / total_weight for char, w in candidate_weights.items()
        }

        # Shannon Entropy
        entropy = 0.0
        for p in probs.values():
            if p > 0.0:
                entropy -= p * math.log2(p)

        # Candidate ranking
        sorted_cands = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        dominant_cand, dominant_prob = sorted_cands[0]
        second_prob = sorted_cands[1][1] if len(sorted_cands) > 1 else 0.0
        margin = dominant_prob - second_prob

        # Categorize severity and compute normalized entropy
        distinct = list(candidate_weights.keys())
        k = len(distinct)
        if k > 1:
            norm_entropy = float(entropy / math.log2(k))
            norm_entropy = max(0.0, min(1.0, norm_entropy))
        elif k == 1:
            norm_entropy = 0.0
        else:
            norm_entropy = None

        if k == 1:
            severity = DisagreementSeverity.UNANIMOUS
            is_disputed = False
        elif entropy <= 0.5:
            severity = DisagreementSeverity.STRONG_CONSENSUS
            is_disputed = True
        elif entropy <= 1.0:
            severity = DisagreementSeverity.DIVIDED
            is_disputed = True
        else:
            severity = DisagreementSeverity.HIGH_DISPUTE
            is_disputed = True

        return PositionEntropyResult(
            position_index=alignment_unit.position_index,
            entropy=float(entropy),
            severity=severity,
            distinct_candidates=distinct,
            candidate_distribution=probs,
            is_unanimous=(k == 1),
            is_disputed=is_disputed,
            dominant_candidate=dominant_cand,
            dominant_margin=float(margin),
            participating_providers_count=len(valid_candidates),
            normalized_entropy=norm_entropy,
            is_assessable=True,
        )

    @classmethod
    def evaluate_report(
        cls,
        cross_verification_report: CrossVerificationReport,
        provider_weights: Optional[Dict[str, float]] = None
    ) -> ConsensusReport:
        """
        Calculates consensus entropy across all character slots in an alignment report.
        """
        slots = cross_verification_report.character_alignments
        if not slots:
            return ConsensusReport(
                mean_entropy=0.0,
                max_entropy=0.0,
                total_slots_evaluated=0,
                unanimous_slots_count=0,
                disputed_slots_count=0,
                unanimous_ratio=1.0,
                disputed_ratio=0.0,
                high_dispute_positions=[],
                position_entropy_details=[],
                severity_distribution={
                    DisagreementSeverity.UNANIMOUS: 0,
                    DisagreementSeverity.STRONG_CONSENSUS: 0,
                    DisagreementSeverity.DIVIDED: 0,
                    DisagreementSeverity.HIGH_DISPUTE: 0,
                    DisagreementSeverity.SINGLE_PROVIDER: 0,
                    DisagreementSeverity.NOT_ASSESSABLE: 0,
                }
            )

        position_details: List[PositionEntropyResult] = []
        severity_counts = {
            DisagreementSeverity.UNANIMOUS: 0,
            DisagreementSeverity.STRONG_CONSENSUS: 0,
            DisagreementSeverity.DIVIDED: 0,
            DisagreementSeverity.HIGH_DISPUTE: 0,
            DisagreementSeverity.SINGLE_PROVIDER: 0,
            DisagreementSeverity.NOT_ASSESSABLE: 0,
        }
        high_dispute_positions = []
        entropies = []

        for unit in slots:
            pos_res = cls.calculate_position_entropy(unit, provider_weights=provider_weights)
            position_details.append(pos_res)
            entropies.append(pos_res.entropy)
            severity_counts[pos_res.severity] = severity_counts.get(pos_res.severity, 0) + 1

            if pos_res.severity in {DisagreementSeverity.DIVIDED, DisagreementSeverity.HIGH_DISPUTE}:
                high_dispute_positions.append(pos_res.position_index)

        total_slots = len(slots)
        disputed_count = sum(1 for p in position_details if p.is_disputed)
        unanimous_count = total_slots - disputed_count
        mean_ent = float(sum(entropies) / total_slots) if total_slots > 0 else 0.0
        max_ent = float(max(entropies)) if entropies else 0.0

        return ConsensusReport(
            mean_entropy=mean_ent,
            max_entropy=max_ent,
            total_slots_evaluated=total_slots,
            unanimous_slots_count=unanimous_count,
            disputed_slots_count=disputed_count,
            unanimous_ratio=float(unanimous_count / total_slots) if total_slots > 0 else 1.0,
            disputed_ratio=float(disputed_count / total_slots) if total_slots > 0 else 0.0,
            high_dispute_positions=high_dispute_positions,
            position_entropy_details=position_details,
            severity_distribution=severity_counts
        )

