from .algo import segment

def segment_chunk(chunk_data):
    """
    chunk_data: tuple (list_of_strings_to_segment, model)
    Returns list of segmented lists.
    """
    texts, model = chunk_data
    results = []
    for text in texts:
        # text is expected to be a concatenated string
        results.append(segment(text, model))
    return results
