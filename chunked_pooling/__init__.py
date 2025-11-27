def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


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

    # Create chunker instance with semantic strategy
    chunker = Chunker(chunking_strategy='semantic')

    # Get token spans from the semantic chunker
    span_annotations: list[tuple[int, int]] = chunker.chunk_semantically(
        input_text,
        tokenizer,
        embedding_model_name=embedding_model_name
    )

    # If max_tokens is specified, split chunks that are too large
    if max_tokens:
        final_spans: list[tuple[int, int]] = []
        for start, end in span_annotations:
            chunk_size = end - start
            if chunk_size > max_tokens:
                # Split this chunk into smaller pieces
                for sub_start in range(start, end, max_tokens):
                    sub_end = min(sub_start + max_tokens, end)
                    if sub_end > sub_start:  # Only add non-empty spans
                        final_spans.append((sub_start, sub_end))
            else:
                final_spans.append((start, end))
        span_annotations = final_spans

    # Convert spans to text chunks
    tokens = tokenizer.encode_plus(input_text, add_special_tokens=False)
    input_ids = tokens['input_ids']
    chunks: list[str] = [
        tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
        for start, end in span_annotations
    ]

    # Adjust spans to account for [CLS] token in model output
    span_annotations_adjusted: list[tuple[int, int]] = [(start + 1, end + 1) for start, end in span_annotations]

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
