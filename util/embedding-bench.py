#!/usr/bin/env python3

import argparse
import time
import os
import numpy as np
from numpy.linalg import norm
from pymilvus import MilvusClient
from pymilvus import model
from google import genai
from google.genai import types
from google_api_utils import get_google_api_key, GoogleAPIKeyError

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"
BATCH_SIZE = 10
GEMINI_MODEL_NAME = "gemini-embedding-001" # Gemini model to use

def gen_embeddings_gemini_batch(contents:list) -> float:
    """
    Generate embeddings for each of the text strings
    TODO: this function doesn't work right

    Args:
        contents (list): list of strings

    Returns:
        float: floating point seconds elapsed
    """

    # get api key
    try:
        api_key = get_google_api_key()
    except GoogleAPIKeyError as e:
        print(f"Error: could not ackquire API key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    start_time = time.perf_counter()
    # TODO: can't figure out how to format this correctly in current API
    # https://developers.googleblog.com/en/gemini-batch-api-now-supports-embeddings-and-openai-compatibility/
    # only shows how to do it if you upload content to operate on rather
    # than using inlined requests
    inline_requests_list = [
        {"key": "request_1", "request": {"output_dimensionality": 512, "content": {"parts": [{"text": "Explain GenAI"}]}}},
        {"key": "request_2", "request": {"output_dimensionality": 512, "content": {"parts": [{"text": "Explain quantum computing"}]}}}
    ]

    # breakpoint();

    # Create the batch job with the inline requests.
    print("Creating inline batch job...")
    batch_job_inline = client.batches.create_embeddings(
        model=GEMINI_MODEL_NAME,
        src={'inlined_requests': inline_requests_list},
        config={'display_name': 'my-batch-job-inline-example'}
    )
    print(f"Created inline batch job: {batch_job_inline.name}")
    print("-" * 20)

    # Monitor the job until completion.
    job_name = batch_job_inline.name
    print(f"Polling status for job: {job_name}")

    while True:
        batch_job_inline = client.batches.get(name=job_name)
        if batch_job_inline.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
            break
        print(f"Job not finished. Current state: {batch_job_inline.state.name}. Waiting 30 seconds...")
        time.sleep(30)

    print(f"Job finished with state: {batch_job_inline.state.name}")
    if batch_job_inline.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job_inline.error}")

    end_time = time.perf_counter()

    return end_time-start_time


def gen_embeddings_gemini(contents:list) -> float:
    """
    Generate embeddings for each of the text strings

    Args:
        contents (list): list of strings

    Returns:
        float: floating point seconds elapsed
    """

    # get api key
    try:
        api_key = get_google_api_key()
    except GoogleAPIKeyError as e:
        print(f"Error: could not ackquire API key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    start_time = time.perf_counter()
    result = client.models.embed_content(model=GEMINI_MODEL_NAME,
                                         contents=contents,
                                         config=types.EmbedContentConfig(output_dimensionality=768))


    # load the resulting embeddings into a numpy 2D matrix, one row per
    # embedding
    embedding_values_np = np.empty((BATCH_SIZE, 768))
    for i, embedding_obj in enumerate(result.embeddings):
        embedding_values_np[i,:] = embedding_obj.values

    # sanity check to make sure vectors look right
    # print(f"Embedding length (first vector): {len(embedding_values_np[0,:])}")
    # print(f"Norm of embedding (first vector): {np.linalg.norm(embedding_values_np[0,:]):.6f}")
    # print(f"Embedding length (last vector): {len(embedding_values_np[BATCH_SIZE-1,:])}")
    # print(f"Norm of embedding (last vector): {np.linalg.norm(embedding_values_np[BATCH_SIZE-1,:]):.6f}")

    # Calculate the norm for each row
    # The axis=-1 argument ensures the norm is calculated along the last
    # axis (rows for a 2D array) np.newaxis is used to reshape the norms
    # into a column vector for broadcasting
    row_norms = np.linalg.norm(embedding_values_np, axis=-1)[:, np.newaxis]
    # Normalize each row by dividing by its corresponding norm
    normed_embedding = embedding_values_np / row_norms

    # sanity check to make sure vectors look right
    # print(f"Normed embedding length (first vector): {len(normed_embedding[0,:])}")
    # print(f"Norm of normed embedding (first vector): {np.linalg.norm(normed_embedding[0,:]):.6f}") # Should be very close to 1
    # print(f"Normed embedding length (last vector): {len(normed_embedding[BATCH_SIZE-1,:])}")
    # print(f"Norm of normed embedding (last vector): {np.linalg.norm(normed_embedding[BATCH_SIZE-1,:]):.6f}") # Should be very close to 1

    end_time = time.perf_counter()

    # TODO: should we convert back to a 2d list after the numpy steps?  that
    # might be more representative of what we have to do at db insert time
    return end_time-start_time


def gen_embeddings_milvus_default(contents:list) -> float:
    """
    Generate embeddings for each of the text strings

    Args:
        contents (list): list of strings

    Returns:
        float: floating point seconds elapsed
    """

    # This will download a small embedding model
    # "paraphrase-albert-small-v2" (~50MB).
    embedding_fn = model.DefaultEmbeddingFunction()

    start_time = time.perf_counter()

    vectors = embedding_fn.encode_documents(contents)
    if vectors[0].shape[0] != 768:
        raise ValueError(
            f"Embedding dimension mismatch for {filepath}. "
            f"Expected 768 but got {vectors[0].shape[0]}"
        )

    end_time = time.perf_counter()

    return end_time-start_time


def read_batch(batch_size:int) -> list:
    """
    Read a batch of files and return a list containing their contents

    Args:
        batch_size (int): number of files to read

    Returns:
        list: list with each element being the complete contents of a file
    """

    # Iterate through vault looking for .md files.
    # For each directory in the tree rooted at the directory top (including
    # top itself), os.walk yields a 3-tuple (dirpath, subdirnames, filenames).
    contents = []
    file_count = 0
    for root, _, files in os.walk(OBSIDIAN_VAULT_PATH):
        if file_count >= batch_size:
            break
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        print(f"File: {filepath}")
                        contents.append(f.read())
                        file_count += 1
                        # TODO: temporary short circuit while prototyping;
                        # it is too expensive to calculate emdeddings for
                        # all notes
                        if file_count >= batch_size:
                            break
                except IOError as e:
                    print(f"Error reading file '{filepath}': {e}. Skipping.")
    return contents

def main():
    """
    Main routine
    """
    # Ensure the vault path exists
    if not os.path.exists(OBSIDIAN_VAULT_PATH):
        print(f"Error: Obsidian vault path not found: {OBSIDIAN_VAULT_PATH}")
        print("Please ensure the path is correct or update OBSIDIAN_VAULT_PATH.")
        return

    contents = read_batch(BATCH_SIZE)

    print(f"Elapsed time (seconds) for encoding {BATCH_SIZE} files")
    print(f"-------------------------")

    elapsed = gen_embeddings_milvus_default(contents)
    print(f"Milvus DefaultEmbeddingFunction: {elapsed}")

    elapsed = gen_embeddings_gemini(contents)
    print(f"Gemini: {elapsed}")

    # elapsed = gen_embeddings_gemini_batch(contents)
    # print(f"Gemini: {elapsed}")

    # print(contents)

if __name__ == "__main__":
    main()
