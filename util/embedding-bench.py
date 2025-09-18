#!/usr/bin/env python3

import argparse
import time
import os
from pymilvus import MilvusClient
from pymilvus import model

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"
BATCH_SIZE = 10

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

    # print(contents)

if __name__ == "__main__":
    main()
