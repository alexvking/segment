import argparse
import time
import multiprocessing
import sys
from wordseg.corpus import CorpusBuilder, clean_sentence
from wordseg.model import UnigramModel
from wordseg.worker import segment_chunk
from wordseg.eval import compute_metrics
from wordseg.algo import segment

def main():
    parser = argparse.ArgumentParser(description="Probabilistic Word Segmentation Engine")
    parser.add_argument("--size", type=int, default=10000, help="Number of sentences to use for training")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of sentences to use for testing")
    parser.add_argument("--corpus", type=str, default="brown", choices=["brown", "webtext"], help="Corpus source")
    
    args = parser.parse_args()
    
    print(f"Initializing Corpus: {args.corpus}...")
    start_time = time.time()
    builder = CorpusBuilder(args.corpus)
    
    # Get Train/Test split
    # We ask for size + test_size total
    required_total = args.size + args.test_size
    train_sents, test_sents = builder.get_split(args.size, args.test_size)
    print(f"Loaded {len(train_sents)} training sentences and {len(test_sents)} test sentences in {time.time() - start_time:.2f}s")
    
    # 1. Build Model
    print("Building Vocabulary (Parallel)...")
    t0 = time.time()
    # We pass the raw sentences (list of words) to build_vocab_parallel
    # It handles cleaning inside.
    vocab_counts = builder.build_vocab_parallel(train_sents)
    t1 = time.time()
    print(f"Vocabulary built in {t1 - t0:.2f}s. Unique Tokens: {len(vocab_counts)}")
    
    print("Training Model...")
    model = UnigramModel(vocab_counts)
    
    # 2. Evaluation
    print("Running Evaluation on Test Set...")
    # Prepare test data:
    # We need the "ground truth" (cleaned list of words)
    # And the "input string" (concatenated cleaned words)
    
    ground_truth = [clean_sentence(s) for s in test_sents]
    # Filter empty sentences
    ground_truth = [s for s in ground_truth if s]
    
    input_strings = ["".join(s) for s in ground_truth]
    
    # Parallel Segment
    t_eval_start = time.time()
    num_workers = multiprocessing.cpu_count()
    chunk_size = max(1, len(input_strings) // num_workers)
    
    chunks = []
    for i in range(0, len(input_strings), chunk_size):
        chunk_texts = input_strings[i:i + chunk_size]
        chunks.append((chunk_texts, model))
        
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_nested = pool.map(segment_chunk, chunks)
        
    # Flatten
    predictions = [item for sublist in results_nested for item in sublist]
    t_eval_end = time.time()
    
    metrics = compute_metrics(ground_truth, predictions)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Training Data: {args.size} sentences")
    print(f"Test Data:     {len(ground_truth)} sentences")
    print(f"Vocab Size:    {len(vocab_counts)}")
    print("-" * 40)
    print(f"Sentence Acc:  {metrics['sentence_accuracy']*100:.2f}%")
    print(f"Precision:     {metrics['precision']*100:.2f}%")
    print(f"Recall:        {metrics['recall']*100:.2f}%")
    print(f"F-Score:       {metrics['f_score']*100:.2f}")
    print("-" * 40)
    print(f"Evaluation Time: {t_eval_end - t_eval_start:.2f}s")
    print("="*40 + "\n")
    
    # REPL
    print("Entering REPL. Type a string to segment (or 'q' to quit).")
    while True:
        try:
            text = input("segment> ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue
                
            # Strip spaces from input just in case user typed them
            clean_text = text.replace(" ", "")
            
            t0 = time.time()
            result = segment(clean_text, model)
            t1 = time.time()
            
            print(f"Input:  '{clean_text}'")
            print(f"Result: '{' '.join(result)}'")
            print(f"Time:   {(t1-t0)*1000:.2f}ms")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Ensure correct start method for multiprocessing on macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()
