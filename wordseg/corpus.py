import nltk
from nltk.corpus import brown, webtext
from collections import Counter
import multiprocessing
import functools
import string

def ensure_nltk_data():
    """Ensure necessary NLTK corpora are downloaded."""
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown', quiet=True)
    try:
        nltk.data.find('corpora/webtext')
    except LookupError:
        nltk.download('webtext', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def clean_sentence(words):
    """
    Takes a list of words (strings) and cleans them.
    Returns a list of cleaned words.
    """
    clean = []
    for w in words:
        # Lowercase
        w = w.lower()
        # Remove punctuation-only tokens
        if all(c in string.punctuation for c in w):
            continue
        clean.append(w)
    return clean

def count_tokens_in_chunk(sentences):
    """
    Worker function to count tokens in a chunk of sentences.
    """
    c = Counter()
    for sent in sentences:
        c.update(clean_sentence(sent))
    return c

class CorpusBuilder:
    def __init__(self, corpus_name='brown'):
        ensure_nltk_data()
        if corpus_name == 'brown':
            self.sentences = list(brown.sents())
        elif corpus_name == 'webtext':
            self.sentences = list(webtext.sents())
        else:
            raise ValueError(f"Unknown corpus: {corpus_name}")
            
    def get_split(self, train_size, test_size=1000):
        """
        Returns (train_sentences, test_sentences).
        Reshuffle could be added here, but for now we follow simple slicing
        to ensure reproducibility if order is deterministic.
        """
        # Simple deterministic Shuffle or just take from end?
        # Paper notes issues with partitioning. Let's do a deterministic shuffle
        # based on a fixed seed to be safe.
        import random
        random.seed(42)
        
        # Working with indices to avoid copying huge lists yet
        indices = list(range(len(self.sentences)))
        random.shuffle(indices)
        
        if train_size + test_size > len(self.sentences):
            raise ValueError(f"Requested size {train_size + test_size} exceeds corpus size {len(self.sentences)}")
            
        train_idx = indices[:train_size]
        test_idx = indices[train_size:train_size+test_size]
        
        # Materialize
        train_data = [self.sentences[i] for i in train_idx]
        test_data = [self.sentences[i] for i in test_idx]
        
        return train_data, test_data

    def build_vocab_parallel(self, sentences, num_workers=None):
        """
        Builds a unigram vocabulary Counter from sentences using methods optimized for M2 (multiprocessing).
        """
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            
        chunk_size = max(1, len(sentences) // num_workers)
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            counters = pool.map(count_tokens_in_chunk, chunks)
            
        # Reduce
        final_counter = Counter()
        for c in counters:
            final_counter.update(c)
            
        return final_counter
