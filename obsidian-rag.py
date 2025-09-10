#!/usr/bin/env python3

import argparse
import os
from pymilvus import MilvusClient

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"

def regenerate_index():
    """
    Placeholder function for regenerating the index.
    """
    print(f"[{OBSIDIAN_VAULT_PATH}] Regenerating index...")
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
