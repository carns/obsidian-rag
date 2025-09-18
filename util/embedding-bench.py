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

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"
BATCH_SIZE = 10
GEMINI_MODEL_NAME = "gemini-embedding-001" # Gemini model to use
GEMINI_API_KEY_FILE = "~/.config/gemini.token" # file to read token from
GEMINI_API_KEY_ENVVAR = "GOOGLE_API_KEY" # environment variable to get token from

class GoogleAPIKeyError(Exception):
    """Custom exception raised when the Google API key cannot be found."""
    pass

def get_google_api_key() -> str:
    """
    Find Google API key to use.

    Tries multiple methods, starting with environment variable then a configuration file

    Raises:
        GoogleAPIKeyError: If the Google API key cannot be found.

    Returns:
        str: The found Google API key.
    """

    # try environment variable first
    google_api_key = os.getenv(GEMINI_API_KEY_ENVVAR)
    if google_api_key:
        return google_api_key

    # try file next, continue if not found
    try:
        file_path = os.path.expanduser(GEMINI_API_KEY_FILE)
        with open(file_path, 'r') as f:
            google_api_key = f.read().strip()
            return google_api_key
    except FileNotFoundError:
        # File doesn't exist, which is expected if not configured this way.
        pass
    # don't catch other exceptions; those would be unexpected

    # NOTE: I would like to also support getting an API key from gnome keyring.  That would be an exercise for later.

    # raise an exception if we get this far without finding an API key
    if google_api_key is None:
        raise GoogleAPIKeyError(
                f"could not find Google API key in ${GEMINI_API_KEY_ENVVAR} or '{GEMINI_API_KEY_FILE}'"
        )

    return google_api_key

def gen_embeddings_gemini_batch(contents:list) -> float:
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

    # TODO: fill this in, see google docs for batch mode

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

    # TODO: this isn't right. Need to do row-wise normalization, and keep
    # the dimensions at 10x768.  I'm not sure it makes sense to do this in
    # one big matrix like this.
    # TODO maybe convert in and out of numpy one array/list at a time; it
    # depends on what format we want the final result to be in
    embedding_values_np = np.empty((10,768))
    for i, embedding_obj in enumerate(result.embeddings):
        embedding_values_np[i,:] = embedding_obj.values
    normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

    print(f"Normed embedding length: {len(normed_embedding)}")
    print(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}") # Should be very close to 1

    breakpoint()

    end_time = time.perf_counter()

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

    # elapsed = gen_embeddings_milvus_default(contents)
    # print(f"Milvus DefaultEmbeddingFunction: {elapsed}")

    elapsed = gen_embeddings_gemini(contents)
    print(f"Gemini: {elapsed}")

    # print(contents)

if __name__ == "__main__":
    main()
