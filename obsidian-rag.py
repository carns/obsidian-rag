#!/usr/bin/env python3

import argparse
import os
from pymilvus import MilvusClient, DataType
from google import genai
from google.genai import types
import numpy as np
from numpy.linalg import norm
import random
import time

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"
VECTOR_DIMENSIONS = 768
# number of files to read and generate embeddings for at a time
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

def regenerate_index(api_key:str, vault_db:str, vault_path:str):
    """
    Regenerate vector DB index

    Args:
        api_key (str): api_key for Gemini
        vault_path (str): path to Obsidian vault
        vault_db (str): path to DB

    """

    print(f"[{vault_path}] Opening database...")
    mclient = MilvusClient(vault_db)

    schema = MilvusClient.create_schema()

    # specify schema, and ask Milvus to automatically generate IDs
    schema.add_field(field_name="id", datatype=DataType.INT64,
                     is_primary=True, auto_id=True,)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR,
                     dim=VECTOR_DIMENSIONS)
    schema.add_field(field_name="path", datatype=DataType.VARCHAR,
                     max_length=512)

    if mclient.has_collection(collection_name="notes"):
        mclient.drop_collection(collection_name="notes")
    mclient.create_collection(
        collection_name="notes",
        dimension=VECTOR_DIMENSIONS,
        schema=schema,
    )

    gclient = genai.Client(api_key=api_key)

    print(f"[{vault_path}] Regenerating index...")

    # Iterate through vault looking for .md files.
    # For each directory in the tree rooted at the directory top (including
    # top itself), os.walk yields a 3-tuple (dirpath, subdirnames, filenames).
    # TODO: two passes; one to count files, and one to actually process them.
    # Then we can show percentage progress
    # TODO: also we should time this and report the elapsed time
    total_file_count = 0
    content_list = []
    file_list = []
    for root, _, files in os.walk(vault_path):
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        content_list.append(f.read())
                        # content_list.append(f"{total_file_count} is a magic number")
                        if len(content_list[-1]) == 0:
                            # Gemini won't generate embeddings for empty
                            # strings; skip this file
                            content_list.pop()
                            print(f"File: {filepath} SKIPPING (empty)")
                            continue
                        print(f"File: {filepath} length {len(content_list[-1])}")
                        file_list.append(filepath)
                        total_file_count += 1
                        if len(content_list) == BATCH_SIZE:
                            insert_into_db(gclient=gclient, mclient=mclient, content_list=content_list,
                                           file_list=file_list)
                            content_list.clear()
                            file_list.clear()
                except IOError as e:
                    print(f"Error reading file '{filepath}': {e}. Skipping.")

    # insert any leftovers
    if len(content_list) > 0:
        insert_into_db(gclient=gclient, mclient=mclient, content_list=content_list,
                       file_list=file_list)
        content_list.clear()
        file_list.clear()

    print(f"Index regenerated successfully from {total_file_count} notes files.")

# TODO: We should have a concurrent pipeline going so that we concurrently
# calculate embeddings and insert them into the database
def insert_into_db(gclient: genai.Client, mclient: MilvusClient, content_list: list, file_list: list):
    """
    Inserts a list of elements into the vector db

    Args:
        gclient (genai.Client): reference to a Gemini client
        mclient (MilvusClient): reference to a Milvus client
        content_list (list): list of text contents to be indexed
        file_list (list): list of file names corresponding to content_list

    """

    max_retries = 5
    initial_delay = 30 # seconds

    for i in range(max_retries):
        try:
            result = gclient.models.embed_content(model=GEMINI_MODEL_NAME,
                                                 contents=content_list,
                                                 config=types.EmbedContentConfig(output_dimensionality=VECTOR_DIMENSIONS))
            # Process the successful response
            # print("API call successful!")
            break # Exit the loop on success
        except Exception as e:
            if e.code == 429:
                print(f"Rate limit exceeded (attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = initial_delay * (2 ** i) + random.uniform(0, 0.5) # Add jitter
                    print(f"Waiting for {delay:.2f} seconds before retrying...")
                    time.sleep(delay)
                    continue
                else:
                    print("Max retries reached. Aborting.")
                    raise
            else:
                print(f"An unexpected error occurred: {e}")
                raise

    # load the resulting embeddings into a numpy 2D matrix, one row per
    # embedding
    embedding_values_np = np.empty((len(content_list), 768))
    for i, embedding_obj in enumerate(result.embeddings):
        embedding_values_np[i,:] = embedding_obj.values

    # Calculate the norm for each row
    # The axis=-1 argument ensures the norm is calculated along the last
    # axis (rows for a 2D array) np.newaxis is used to reshape the norms
    # into a column vector for broadcasting
    row_norms = np.linalg.norm(embedding_values_np, axis=-1)[:, np.newaxis]
    # Normalize each row by dividing by its corresponding norm
    normed_embedding = embedding_values_np / row_norms

    # constuct data to insert into Milvus.  Convert each numpy array row (each
    # vector) into a list and associate with the corresponding path
    # TODO: does Milvus support any other formats for the vector?
    data = []
    for vector, file in zip(normed_embedding, file_list):
        data.append({"vector":vector.tolist(), "path":file})

    mclient.insert(collection_name="notes", data=data)


def query_vault(query: str, api_key: str, vault_db: str, vault_path: str):
    """
    Placeholder function for querying the vault.

    Args:
        query (str): Query string
        api_key (str): API key for Gemini
        vault_db (str): path to DB
        vault_path (str): path to Obsidian vault
    """

    print(f"[{vault_path}] Opening database...")

    client = MilvusClient(vault_db)
    if not client.has_collection(collection_name="notes"):
        raise ValueError(
            f"'notes' collection not found in {vault_db}"
        )


    print(f"[{vault_path}] Querying for: '{query}'")
    # In a real scenario, you'd add your query logic here
    # Example: search your index, grep files, etc.
    if query:
        print(f"Search results for '{query}':")
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

    # get api key for Gemini
    try:
        api_key = get_google_api_key()
    except GoogleAPIKeyError as e:
        print(f"Error: could not ackquire API key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    # Check which options were provided and call the corresponding functions
    if args.index:
        regenerate_index(api_key=api_key, vault_path=OBSIDIAN_VAULT_PATH,
                         vault_db=OBSIDIAN_VAULT_DB)
    elif args.query is not None: # `is not None` because an empty string '' is also a valid query
        query_vault(query=args.query, api_key=api_key,
                    vault_path=OBSIDIAN_VAULT_PATH,
                    vault_db=OBSIDIAN_VAULT_DB)
    else:
        # If no specific options were given, print help or default behavior
        print("No specific action requested. Use -i to index or -q to query.")
        parser.print_help()

if __name__ == "__main__":
    main()
