import math

class UnigramModel:
    def __init__(self, counts):
        """
        counts: Counter object of {word: count}
        """
        self.counts = counts
        self.total_tokens = sum(counts.values())
        self.V = len(counts)
        # Precompute log(N) for speed
        self.log_N = math.log(self.total_tokens) if self.total_tokens > 0 else 0
        self.log_10 = math.log(10)
        
        # Precompute log probabilities for known words
        # P(w) = Count(w) / N
        # log(P(w)) = log(Count(w)) - log(N)
        self.known_log_probs = {}
        for w, c in counts.items():
            self.known_log_probs[w] = math.log(c) - self.log_N

    def score(self, word):
        """
        Returns log probability of word.
        """
        if word in self.known_log_probs:
            return self.known_log_probs[word]
        else:
            # Unknown logic from paper:
            # "change count of 1 to ... 10^-40"
            # "penalizing ... by dividing ... by 10^len(s)"
            # Interpretation:
            # Effective Count = 1e-40
            # P_base = 1e-40 / N
            # Penalty = 1 / 10^len
            # P_final = (1e-40 / N) / 10^len
            # log(P) = log(1e-40) - log(N) - len * log(10)
            
            length = len(word)
            # log(1e-40) = -40 * log(10)
            term1 = -40 * self.log_10
            term2 = self.log_N
            term3 = length * self.log_10
            
            return term1 - term2 - term3
