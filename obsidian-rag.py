#!/usr/bin/env python3

import argparse
import os
from pymilvus import MilvusClient
from pymilvus import model

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"

def regenerate_index():
    """
    Placeholder function for regenerating the index.
    """
    print(f"[{OBSIDIAN_VAULT_PATH}] Opening database...")
    client = MilvusClient(OBSIDIAN_VAULT_DB)
    if client.has_collection(collection_name="notes"):
        client.drop_collection(collection_name="notes")
    client.create_collection(
        collection_name="notes",
        dimension=768,
    )

    print(f"[{OBSIDIAN_VAULT_PATH}] Regenerating index...")

    # TODO: try a more sophisticated model. per Seth recommendation, start
    # with https://huggingface.co/sentence-transformers

    # TODO: can we query the dimensionality of the model first, and set the
    # collection dimensions based on that, instead of hard coding?

    # This will download a small embedding model
    # "paraphrase-albert-small-v2" (~50MB).
    embedding_fn = model.DefaultEmbeddingFunction()

    # Iterate through vault looking for .md files.
    # For each directory in the tree rooted at the directory top (including
    # top itself), os.walk yields a 3-tuple (dirpath, subdirnames, filenames).
    file_count = 0
    for root, _, files in os.walk(OBSIDIAN_VAULT_PATH):
        if file_count >= 5:
            break
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        print(f"File: {filepath}")
                        # note that we are not chunking; generate an
                        # embedding for the entire contents of the .md file
                        vectors = embedding_fn.encode_documents([content])
                        if vectors[0].shape[0] != 768:
                            raise ValueError(
                                f"Embedding dimension mismatch for {filepath}. "
                                f"Expected 768 but got {vectors[0].shape[0]}"
                            )
                        # storing the path to the file in the db, not the
                        # contents of the file
                        # TODO: restructure this to insert vectors as a
                        # batch.  Right now doing one embedding and one
                        # insertion at a time to get an understanding of the
                        # process.
                        client.insert(collection_name="notes", data={"id": file_count, "vector": vectors[0], "path": filepath})
                        file_count += 1
                        # TODO: temporary short circuit while prototyping;
                        # it is too expensive to calculate emdeddings for
                        # all notes
                        if file_count >= 5:
                            break
                except IOError as e:
                    print(f"Error reading file '{filepath}': {e}. Skipping.")

    # In a real scenario, you'd add your indexing logic here
    # Example: scan files, build a search index, etc.
    print("Index regenerated successfully.")

def query_vault(query_string):
    """
    Placeholder function for querying the vault.
    """
    print(f"[{OBSIDIAN_VAULT_PATH}] Querying for: '{query_string}'")
    # In a real scenario, you'd add your query logic here
    # Example: search your index, grep files, etc.
    if query_string:
        print(f"Search results for '{query_string}':")
        # Simulate some results
        print("  - Found 'your_note_about_query.md'")
        print("  - Found 'another_relevant_document.md'")
    else:
        print("No query string provided.")


def main():
    """
    Main routine
    """

    parser = argparse.ArgumentParser(
        description="Manage and query an Obsidian vault.",
        epilog=f"Vault path: {OBSIDIAN_VAULT_PATH}"
    )

    # Option for regenerating the index
    parser.add_argument(
        '-i', '--index',
        action='store_true',  # This makes it a boolean flag (True if present, False otherwise)
        help='(Re)generate the index for the Obsidian vault.'
    )

    # Option for specifying a query string
    parser.add_argument(
        '-q', '--query',
        type=str,  # Specifies that the argument expects a string value
        metavar='QUERY_STRING', # How the argument will be displayed in help message
        help='Specify a string to query the Obsidian vault.'
    )

    args = parser.parse_args()

    # Ensure the vault path exists (optional, but good practice)
    if not os.path.exists(OBSIDIAN_VAULT_PATH):
        print(f"Error: Obsidian vault path not found: {OBSIDIAN_VAULT_PATH}")
        print("Please ensure the path is correct or update OBSIDIAN_VAULT_PATH.")
        return

    # Check which options were provided and call the corresponding functions
    if args.index:
        regenerate_index()
    elif args.query is not None: # `is not None` because an empty string '' is also a valid query
        query_vault(args.query)
    else:
        # If no specific options were given, print help or default behavior
        print("No specific action requested. Use -i to index or -q to query.")
        parser.print_help()

if __name__ == "__main__":
    main()
