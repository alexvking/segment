from .base_model import LanguageModel

MAX_WORD_LEN = 20  # Optimization constant

def segment(text, model: LanguageModel):
    """
    Segments text using the provided model and DP.
    
    Optimized to O(N * MAX_WORD_LEN) which is effectively O(N).
    Supports context-aware models (Bigram) by tracking previous words in backpointers?
    
    Wait: Standard Viterbi for Bigrams requires storing optimal score for *each possible previous word* at index i.
    Because Score(w_current) depends on w_prev.
    
    If we just store "best score at i", we assume the best path to i is independent of what the word ending at i is.
    But it's NOT independent for the NEXT step.
    
    However, the paper describes a simpler 1D DP:
    seg(s) = max( quality(w) + seg(rest) )
    This is effectively a Unigram assumption even for their "Bigram" model unless they tracked state.
    
    If we strictly follow the paper's simpler DP structure but plug in a Bigram Score:
    Score(w_new) = BigramProb(w_new | w_last_of_prefix)
    
    But "w_last_of_prefix" depends on the segmentation of the prefix.
    
    Simplification:
    The "backptr" approach usually finds the best split k < i.
    The word ending at i is text[k:i].
    The word before that is determined by backptr[k].
    
    So when calculating score for transition k -> i:
    Current Word = text[k:i]
    Previous Word = text[ backptr[k] : k ]
    
    This is a "Greedy Viterbi" or "1-Best Viterbi". It is not guaranteed to be globally optimal for Bigrams because a suboptimal path to 'k' might end with a word that makes the transition to 'i' much better.
    
    To be truly optimal for Bigrams, we'd need O(N * V) state? No, O(N) but state is "Best score ending with word W". Too huge.
    
    Given the performance constrains and the paper's likely implementation:
    We will use the "1-Best" approximation: Assume the best segmentation up to k is the only history we care about.
    This keeps it O(N).
    """
    n = len(text)
    if n == 0:
        return []

    # dp[i] = max score for text[:i]
    dp = [-float('inf')] * (n + 1)
    dp[0] = 0.0
    backptr = [0] * (n + 1)
    
    # Store the actual words chosen to avoid re-slicing constantly?
    # Or just re-slice since we need it for the previous_word lookup
    
    for i in range(1, n + 1):
        best_score = -float('inf')
        best_k = -1
        
        # Optimization: Only look back MAX_WORD_LEN
        start_k = max(0, i - MAX_WORD_LEN)
        
        for k in range(start_k, i):
            word = text[k:i]
            
            # Find previous word for Bigram context
            prev_word = None
            if k > 0:
                pk = backptr[k]
                prev_word = text[pk:k]
            
            # Score
            current_score = dp[k] + model.score(word, prev_word)
            
            if current_score > best_score:
                best_score = current_score
                best_k = k
        
        dp[i] = best_score
        backptr[i] = best_k
        
    # Backtrack
    result = []
    curr = n
    while curr > 0:
        prev = backptr[curr]
        word = text[prev:curr]
        result.append(word)
        curr = prev
        
    return result[::-1]
