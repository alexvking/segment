import math
from collections import defaultdict
from .base_model import LanguageModel
from ..model import PenalizedUnigramModel

class BigramModel(LanguageModel):
    def __init__(self, unigram_counts, bigram_counts, lambda_val=0.2):
        """
        unigram_counts: Counter {word: count}
        bigram_counts: Counter {(w1, w2): count}
        lambda_val: Interpolation weight. P = lambda * P_bi + (1-lambda) * P_uni
        """
        self.unigram_model = PenalizedUnigramModel(unigram_counts)
        self.bigram_counts = bigram_counts
        
        # Precompute P(w2|w1) for known bigrams
        # P(w2|w1) = Count(w1, w2) / Count(w1)
        self.bigram_probs = {}
        
        # We need fast access to Count(w1)
        self.unigram_counts = unigram_counts
        
        self.lambda_val = lambda_val
        self.one_minus_lambda = 1.0 - lambda_val
        
    def score(self, word: str, prev_word: str = None) -> float:
        # Get Unigram Probability (Backoff)
        # Note: We need the actual probability, not log, for interpolation
        # But our unigram model returns LOG prob.
        log_uni = self.unigram_model.score(word)
        try:
            p_uni = math.exp(log_uni)
        except OverflowError:
            p_uni = 0.0
            
        p_bi = 0.0
        if prev_word is not None:
            # Check bigram existence
            pair = (prev_word, word)
            count_pair = self.bigram_counts.get(pair, 0)
            
            if count_pair > 0:
                count_prev = self.unigram_counts.get(prev_word, 0)
                if count_prev > 0:
                    p_bi = count_pair / count_prev
        
        # Linear Interpolation
        p_interpolated = (self.lambda_val * p_bi) + (self.one_minus_lambda * p_uni)
        
        if p_interpolated > 0:
            return math.log(p_interpolated)
        else:
            # Should not happen because p_uni is rarely 0 due to smoothing/penalty
            # But if p_uni underflows or is very small...
            return log_uni # Fallback to log-unigram
