# Word Segmentation Engine v2 (Evolution)

This release implements a significant evolution of the word segmentation engine, addressing theoretical shortcomings of the original academic paper.

## Key Changes in v2

### 1. Corrected Terminology
- **Renamed**: The "Laplace Smoothing" model has been renamed to **`PenalizedUnigramModel`**.
- **Reasoning**: The original implementation was not true Laplace smoothing but a heuristic length-penalty model. The new name accurately reflects the underlying logic ($\text{Score} \approx \log(10^{-40}) - \text{len} \cdot \log(10)$).

### 2. Fair Model Comparison (Bigram Support)
- **New Model**: Added **`BigramModel`** with Linear Interpolation.
- **Fairness**: We applied the *same* unknown-word penalty heuristics to the Bigram model that gave the Unigram model its advantage.
- **Result**: Under fair conditions, the Bigram model **outperforms** the Unigram model, restoring consistency with NLP theory.

### 3. Algorithmic Optimization
- **Linear Time Complexity**: The Dynamic Programming algorithm was optimized from $O(N^2)$ to $O(N \cdot K)$ (K=20 constants).
- **Performance**: Sub-millisecond (0.28ms) segmentation latency.

## Benchmark Results

### Full Corpus (52k Training / 5k Test)
Using the maximum available data from the Brown corpus (~57k total sentences):

| Metric | Penalized Unigram | Bigram (Interpolated) | Change |
|--------|-------------------|-----------------------|--------|
| **Sentence Accuracy** | 67.34% | **69.46%** | **+2.12%**|
| **F-Score** | 94.64 | **94.97** | +0.33% |
| **Precision** | 93.08% | **93.33%** | +0.25% |
| **Recall** | 96.26% | **96.67%** | +0.41% |

### 10k Training Benchmark
| Metric | Penalized Unigram | Bigram (Interpolated) |
|--------|-------------------|-----------------------|
| F-Score | 85.86 | 86.06 |

**Conclusion**: The Bigram model scales better, offering a decisive 2%+ improvement in perfect sentence accuracy on larger datasets.

## Usage

### Run Max Benchmark
```bash
# Uses ~57k sentences (Dataset Limit)
python3 cli.py --size 52000 --test-size 5000
```

### REPL
```
segment> thetabledownthere
[Unigram] (0.39ms): the table down there
[Bigram ] (0.28ms): the table down there
```
