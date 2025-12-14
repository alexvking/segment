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
        w = w.lower()
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

def count_bigrams_in_chunk(sentences):
    """
    Worker function to count bigrams.
    """
    c = Counter()
    for sent in sentences:
        words = clean_sentence(sent)
        if not words:
            continue
        # Generate bigrams
        # We can add a START token if we want strict bigram modeling at start of sent
        # but for simple segmentation, raw bigrams are okay.
        # Let's add explicit START token logic? 
        # The paper doesn't explicitly mention it, but it's standard.
        # Let's keep it simple: just raw bigrams for now.
        for i in range(len(words)-1):
            c[(words[i], words[i+1])] += 1
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
        import random
        random.seed(42)
        indices = list(range(len(self.sentences)))
        random.shuffle(indices)
        
        if train_size + test_size > len(self.sentences):
            raise ValueError(f"Requested size {train_size + test_size} exceeds corpus size {len(self.sentences)}")
            
        train_idx = indices[:train_size]
        test_idx = indices[train_size:train_size+test_size]
        return [self.sentences[i] for i in train_idx], [self.sentences[i] for i in test_idx]

    def build_vocab_parallel(self, sentences, num_workers=None):
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        chunk_size = max(1, len(sentences) // num_workers)
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            counters = pool.map(count_tokens_in_chunk, chunks)
            
        final_counter = Counter()
        for c in counters:
            final_counter.update(c)
        return final_counter

    def build_bigram_vocab_parallel(self, sentences, num_workers=None):
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        chunk_size = max(1, len(sentences) // num_workers)
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            counters = pool.map(count_bigrams_in_chunk, chunks)
            
        final_counter = Counter()
        for c in counters:
            final_counter.update(c)
        return final_counter
