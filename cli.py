import argparse
import time
import multiprocessing
import sys
from collections import Counter
from wordseg.corpus import CorpusBuilder, clean_sentence
from wordseg.model import PenalizedUnigramModel
from wordseg.models.bigram import BigramModel
from wordseg.worker import segment_chunk
from wordseg.eval import compute_metrics
from wordseg.algo import segment

def run_evaluation(model, name, test_data_input, ground_truth):
    print(f"Running Evaluation for {name}...")
    t_start = time.time()
    
    num_workers = multiprocessing.cpu_count()
    chunk_size = max(1, len(test_data_input) // num_workers)
    
    chunks = []
    for i in range(0, len(test_data_input), chunk_size):
        chunk_texts = test_data_input[i:i + chunk_size]
        chunks.append((chunk_texts, model))
        
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_nested = pool.map(segment_chunk, chunks)
        
    predictions = [item for sublist in results_nested for item in sublist]
    t_end = time.time()
    
    metrics = compute_metrics(ground_truth, predictions)
    metrics['time'] = t_end - t_start
    return metrics

def print_comparison(m1, name1, m2, name2):
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {name1:<15} | {name2:<15}")
    print("-" * 60)
    
    keys = [('sentence_accuracy', 'Sentence Acc'), 
            ('precision', 'Precision'), 
            ('recall', 'Recall'), 
            ('f_score', 'F-Score'),
            ('time', 'Time (s)')]
            
    for key, label in keys:
        val1 = m1[key]
        val2 = m2[key]
        
        # Format
        if key == 'time':
            s1 = f"{val1:.2f}s"
            s2 = f"{val2:.2f}s"
        elif key == 'f_score':
             s1 = f"{val1*100:.2f}"
             s2 = f"{val2*100:.2f}"
        else:
             s1 = f"{val1*100:.2f}%"
             s2 = f"{val2*100:.2f}%"
             
        print(f"{label:<20} | {s1:<15} | {s2:<15}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Word Segmentation Engine v2")
    parser.add_argument("--size", type=int, default=10000, help="Training size")
    parser.add_argument("--test-size", type=int, default=1000, help="Test size")
    parser.add_argument("--corpus", type=str, default="brown", choices=["brown", "webtext"])
    
    args = parser.parse_args()
    
    # 1. Corpus
    builder = CorpusBuilder(args.corpus)
    train_sents, test_sents = builder.get_split(args.size, args.test_size)
    print(f"Loaded Data: {len(train_sents)} train, {len(test_sents)} test.")
    
    # 2. Build Models
    print("Building Unigram Vocabulary...")
    uni_counts = builder.build_vocab_parallel(train_sents)
    
    print("Building Bigram Vocabulary...")
    bi_counts = builder.build_bigram_vocab_parallel(train_sents)
    
    print("Initializing Models...")
    uni_model = PenalizedUnigramModel(uni_counts)
    bi_model = BigramModel(uni_counts, bi_counts, lambda_val=0.2)
    
    # 3. Prepare Test Data
    ground_truth = [clean_sentence(s) for s in test_sents if clean_sentence(s)]
    input_strings = ["".join(s) for s in ground_truth]
    
    # 4. Evaluate
    m1 = run_evaluation(uni_model, "Penalized Unigram", input_strings, ground_truth)
    m2 = run_evaluation(bi_model, "Bigram (Interpolated)", input_strings, ground_truth)
    
    print_comparison(m1, "Unigram", m2, "Bigram")
    
    # REPL
    print("Entering REPL. Models: [u]nigram, [b]igram.")
    while True:
        try:
            line = input("segment> ").strip()
            if line == 'q': break
            if not line: continue
            
            clean = line.replace(" ","")
            
            # Uni
            t0 = time.time()
            res_u = segment(clean, uni_model)
            t1 = time.time()
            print(f"[Unigram] ({ (t1-t0)*1000:.2f}ms): {' '.join(res_u)}")
            
            # Bi
            t0 = time.time()
            res_b = segment(clean, bi_model)
            t1 = time.time()
            print(f"[Bigram ] ({ (t1-t0)*1000:.2f}ms): {' '.join(res_b)}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
