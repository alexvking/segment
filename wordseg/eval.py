def get_intervals(words):
    """
    Converts a list of words into a set of (start, end) intervals.
    Assumes words are concatenated to form the underlying string.
    """
    intervals = set()
    current_idx = 0
    for w in words:
        start = current_idx
        end = start + len(w)
        intervals.add((start, end))
        current_idx = end
    return intervals

def compute_metrics(truth_sentences, pred_sentences):
    """
    Computes aggregated Precision, Recall, F-Score, and Sentence Accuracy.
    truth_sentences: list of lists of words (ground truth)
    pred_sentences: list of lists of words (segmented)
    """
    correct_sentences = 0
    total_sentences = len(truth_sentences)
    
    total_correct_words = 0
    total_pred_words = 0
    total_truth_words = 0
    
    for truth, pred in zip(truth_sentences, pred_sentences):
        # Sentence Accuracy
        if truth == pred:
            correct_sentences += 1
            
        # Word Level Metrics
        truth_intervals = get_intervals(truth)
        pred_intervals = get_intervals(pred)
        
        # Sanity check: Total length must match
        # if max(t[1] for t in truth_intervals) != max(p[1] for p in pred_intervals):
        #     Warning: Length mismatch?
        
        common = truth_intervals.intersection(pred_intervals)
        
        total_correct_words += len(common)
        total_pred_words += len(pred_intervals)
        total_truth_words += len(truth_intervals)
        
    precision = total_correct_words / total_pred_words if total_pred_words > 0 else 0
    recall = total_correct_words / total_truth_words if total_truth_words > 0 else 0
    
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0
        
    term_accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0
    
    return {
        "sentence_accuracy": term_accuracy,
        "precision": precision,
        "recall": recall,
        "f_score": f_score
    }
