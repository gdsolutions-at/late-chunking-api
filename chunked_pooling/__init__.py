def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)

    token_offsets = inputs['offset_mapping'][0].tolist()
    token_ids = inputs['input_ids'][0].tolist()

    # Decode each token to check for sentence-ending punctuation
    sentence_endings = {'.', '?', '!'}

    chunk_positions = []  # List of (end_token_idx, end_char_offset)

    for i in range(1, len(token_ids)):  # Skip CLS at index 0
        start_offset, end_offset = token_offsets[i]

        # Skip special tokens (offset 0,0)
        if start_offset == 0 and end_offset == 0:
            continue

        # Get the actual text for this token
        token_text = input_text[start_offset:end_offset]

        # Check if token ends with sentence-ending punctuation
        if token_text and token_text[-1] in sentence_endings:
            # Check if it's followed by space or end of text
            is_last_content_token = (i + 1 >= len(token_ids)) or (token_offsets[i + 1][0] == 0 and token_offsets[i + 1][1] == 0)

            if is_last_content_token:
                # Last content token
                chunk_positions.append((i + 1, end_offset))
            else:
                next_start, _ = token_offsets[i + 1]
                # Check if there's a gap (whitespace) between this token and the next
                if next_start > end_offset:
                    chunk_positions.append((i + 1, end_offset))

    # Ensure we capture the last chunk if text doesn't end with punctuation
    # Find last content token (non-special token)
    last_content_idx = len(token_ids) - 1
    while last_content_idx > 0:
        s, e = token_offsets[last_content_idx]
        if s != 0 or e != 0:  # Not a special token
            break
        last_content_idx -= 1

    if last_content_idx > 0:
        _, last_end = token_offsets[last_content_idx]
        # If no chunks found, or last chunk doesn't include the end
        if not chunk_positions or chunk_positions[-1][0] <= last_content_idx:
            chunk_positions.append((last_content_idx + 1, last_end))

    # Build chunks and span_annotations
    chunks = []
    span_annotations = []

    start_token_idx = 1  # Skip CLS
    start_char_offset = 0

    for end_token_idx, end_char_offset in chunk_positions:
        chunk_text = input_text[start_char_offset:end_char_offset].strip()
        if chunk_text:
            chunks.append(chunk_text)
            span_annotations.append((start_token_idx, end_token_idx))

        start_token_idx = end_token_idx
        start_char_offset = end_char_offset

    return chunks, span_annotations


# Cache chunker instance to avoid loading the embedding model on every request
_chunker_cache: dict = {}

def chunk_semantically(input_text: str, tokenizer, embedding_model_name: str, max_tokens: int | None = None) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Split the input text semantically using the Chunker class
    :param input_text: The text snippet to split semantically
    :param tokenizer: The tokenizer to use
    :param embedding_model_name: The name of the embedding model for semantic splitting
    :param max_tokens: Optional maximum token size per chunk. If a chunk exceeds this, it will be split further.
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    from .chunking import Chunker

    # Reuse cached chunker instance to avoid memory leak
    if 'semantic' not in _chunker_cache:
        _chunker_cache['semantic'] = Chunker(chunking_strategy='semantic')
    chunker = _chunker_cache['semantic']

    # Get token spans from the semantic chunker
    span_annotations = chunker.chunk_semantically(
        input_text,
        tokenizer,
        embedding_model_name=embedding_model_name
    )

    # If max_tokens is specified, split chunks that are too large
    if max_tokens:
        final_spans = []
        for start, end in span_annotations:
            chunk_size = end - start
            if chunk_size > max_tokens:
                # Split this chunk into smaller pieces
                for sub_start in range(start, end, max_tokens):
                    sub_end = min(sub_start + max_tokens, end)
                    final_spans.append((sub_start, sub_end))
            else:
                final_spans.append((start, end))
        span_annotations = final_spans

    # Convert spans to text chunks
    tokens = tokenizer.encode_plus(input_text, add_special_tokens=False)
    input_ids = tokens['input_ids']
    chunks = [
        tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
        for start, end in span_annotations
    ]

    # Adjust spans to account for [CLS] token in model output
    span_annotations_adjusted = [(start + 1, end + 1) for start, end in span_annotations]

    return chunks, span_annotations_adjusted


def chunked_pooling(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.float().detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs
