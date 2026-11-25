import re


# ==========================================
# Function 1: Split Text into Chunks
# ==========================================

def split_text(text, chunk_size=100, overlap=20):

    # Step 1: Handle empty text
    if not text or not text.strip():
        return []

    # Step 2: Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []

    current_chunk = []
    current_size = 0

    # Step 3: Process each sentence
    for sentence in sentences:

        sentence_words = sentence.split()
        sentence_size = len(sentence_words)

        if not sentence_words:
            continue

        # Step 4: Check whether sentence fits
        if current_size + sentence_size <= chunk_size:

            current_chunk.append(sentence)
            current_size += sentence_size

        else:

            # Save existing chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Step 5: Preserve overlapping sentences
            overlap_sentences = []
            overlap_size = 0

            for previous_sentence in reversed(current_chunk):

                previous_size = len(previous_sentence.split())

                if overlap_size + previous_size <= overlap:

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_size += previous_size

                else:
                    break

            current_chunk = overlap_sentences + [sentence]

            current_size = overlap_size + sentence_size

    # Step 6: Save remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ==========================================
# Function 2: Chunk Documents with Metadata
# ==========================================

def chunk_documents(documents, chunk_size=100, overlap=20):

    all_chunks = []

    for document in documents:

        chunks = split_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:

            all_chunks.append({

                "text": chunk,

                "source": document["source"],

                "page": document["page"]

            })

    return all_chunks