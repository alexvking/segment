import math
from abc import ABC, abstractmethod

class LanguageModel(ABC):
    @abstractmethod
    def score(self, word: str, prev_word: str = None) -> float:
        pass
