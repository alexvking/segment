from .model import UnigramModel

def segment(text, model: UnigramModel):
    """
    Segments text using the provided model and DP.
    Returns a list of strings.
    """
    n = len(text)
    if n == 0:
        return []

    # dp[i] stores the max score for text[:i]
    # backptr[i] stores the split index 'k' that gave that max score
    # So the last word is text[backptr[i]:i]
    
    # Initialize
    dp = [-float('inf')] * (n + 1)
    dp[0] = 0.0
    backptr = [0] * (n + 1)
    
    for i in range(1, n + 1):
        best_score = -float('inf')
        best_k = -1
        
        # Optimization: Heuristic? 
        # No, exact DP requires checking all k.
        # However, words are rarely > 20 chars. 
        # We could limit lookback to max_word_length in corpus + buffer?
        # The paper doesn't strictly limit, but for performance on huge strings it helps.
        # But for correctness with 'unknown' penalty that depends on length, we should check all.
        # Let's check all for now, N is small (~100 chars for a sentence).
        
        for k in range(i):
            word = text[k:i]
            # score = score of prefix + score of this word
            current_score = dp[k] + model.score(word)
            
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
        
    return result[::-1] # Reverse to get correct order
