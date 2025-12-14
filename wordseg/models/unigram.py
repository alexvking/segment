import math
from .base_model import LanguageModel

class PenalizedUnigramModel(LanguageModel):
    def __init__(self, counts):
        """
        counts: Counter object of {word: count}
        """
        self.counts = counts
        self.total_tokens = sum(counts.values())
        self.V = len(counts)
        self.log_N = math.log(self.total_tokens) if self.total_tokens > 0 else 0
        self.log_10 = math.log(10)
        
        # Precompute log probabilities for known words
        self.known_log_probs = {}
        for w, c in counts.items():
            self.known_log_probs[w] = math.log(c) - self.log_N

    def score(self, word: str, prev_word: str = None) -> float:
        """
        Returns log probability of word.
        Ignores prev_word (unigram assumption).
        """
        if word in self.known_log_probs:
            return self.known_log_probs[word]
        else:
            # Heuristic Penalty:
            # log(1e-40) - len * log(10) - log(N)
            length = len(word)
            term1 = -40 * self.log_10
            term2 = self.log_N
            term3 = length * self.log_10
            
            return term1 - term2 - term3
