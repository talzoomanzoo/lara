"""Dataset-level rank fusion of Self-Critique and LaRA scores."""
import math

import pandas as pd

from .base_detector import BaseDetector


class SelfCritiqueLaRADetector(BaseDetector):
    """Higher fused percentile ranks indicate stronger membership evidence.

    Call combine_scores on the evaluation cohort before reading individual
    scores. The fusion uses no membership labels and averages ranks for ties.
    """

    def __init__(self, beta=0.65):
        self.beta = float(beta)
        if not math.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError('Self-Critique + LaRA beta must be between 0 and 1')

    def get_name(self):
        # Preserve the existing output field for saved results and consumers.
        return 'self_critique_rep_stiff_lara_mix'

    def calculate_score(self, data_item):
        return data_item.get(self.get_name(), float('nan'))

    def combine_scores(self, self_critique, lara):
        sc_rank = pd.to_numeric(self_critique, errors='coerce').rank(pct=True)
        lara_rank = pd.to_numeric(lara, errors='coerce').rank(pct=True)
        return self.beta * sc_rank + (1.0 - self.beta) * lara_rank
