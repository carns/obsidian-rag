#!/usr/bin/env python3

import argparse
import os
from pymilvus import MilvusClient
from google import genai
from google.genai import types

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"
VECTOR_DIMENSIONS = 768
# number of files to read and generate embeddings for at a time
BATCH_SIZE = 10
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

    # TODO: make this an argument
    print(f"[{vault_path}] Opening database...")
    client = MilvusClient(vault_db)
    if client.has_collection(collection_name="notes"):
        client.drop_collection(collection_name="notes")
    client.create_collection(
        collection_name="notes",
        dimension=VECTOR_DIMENSIONS,
    )

    client = genai.Client(api_key=api_key)

    print(f"[{vault_path}] Regenerating index...")

    # Iterate through vault looking for .md files.
    # For each directory in the tree rooted at the directory top (including
    # top itself), os.walk yields a 3-tuple (dirpath, subdirnames, filenames).
    total_file_count = 0
    content_list = []
    file_list = []
    for root, _, files in os.walk(vault_path):
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        print(f"File: {filepath}")
                        content_list.append(f.read())
                        file_list.append(filepath)
                        total_file_count += 1
                        if len(content_list) == BATCH_SIZE:
                            insert_into_db(content_list=content_list,
                                           file_list=file_list)
                            content_list = []
                            file_list = []
                except IOError as e:
                    print(f"Error reading file '{filepath}': {e}. Skipping.")

    # insert any leftovers
    if len(content_list) > 0:
        insert_into_db(content_list=content_list,
                       file_list=file_list)
        content_list = []
        file_list = []

    print(f"Index regenerated successfully from {total_file_count} notes files.")

def insert_into_db(content_list: list, file_list: list):
    """
    Inserts a list of elements into the vector db

    Args:
        content_list (list): list of text contents to be indexed
        file_list (list): list of file names corresponding to content_list

    """

    print(content_list)

#                        client.insert(collection_name="notes", data={"id": total_file_count, "vector": vectors[0], "path": filepath})

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
