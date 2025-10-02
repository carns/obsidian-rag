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
from google_api_utils import get_google_api_key, GoogleAPIKeyError
from tqdm import tqdm
import time

# --- Module-level Constants ---
OBSIDIAN_VAULT_PATH = "/home/carns/Documents/carns-obsidian"
OBSIDIAN_VAULT_DB = OBSIDIAN_VAULT_PATH + "/milvus_index.db"
VECTOR_DIMENSIONS = 768
# number of files to read and generate embeddings for at a time
BATCH_SIZE = 64
GEMINI_MODEL_NAME = "gemini-embedding-001" # Gemini model to use

# base class for any embedding methods we want to support
class embedder:
    def __init__(self, num_dimensions: int):
        self.num_dimensions = num_dimensions

    def generate_embeddings(self, data: list) -> np.ndarray:
        return np.ndarray([])

# TODO: have an api_key argument here, or just call the utility function to
# get it?
# embedding function that uses a hosted Gemini model
class gemini_embedder(embedder):
    def __init__(self, num_dimensions: int, api_key: str, model_name: str):
        """
        Initializes the instance, setting up the Google Generative AI client and
        model configuration.

        This constructor calls the parent class's __init__ method with
        `num_dimensions` and then instantiates a `genai.Client` using the
        provided `api_key`.  The `model_name` is stored for subsequent
        operations with the Generative AI service.

        Args:
            num_dimensions (int): The number of dimensions to initialize the base
                                  class with. This typically relates to the
                                  vector space or embedding size managed by the
                                  parent class.
            api_key (str): Your Google Generative AI API key for authentication.
                           This key is used to create an authenticated client.
            model_name (str): The name of the Generative AI model to use for
                              operations (e.g., 'gemini-pro', 'embedding-001').

        Attributes:
            model_name (str): The name of the Generative AI model configured
                              for this instance.
            gclient (genai.Client): An authenticated client object for interacting
                                    with the Google Generative AI service.
        """

        super().__init__(num_dimensions)
        self.model_name = model_name
        self.gclient = genai.Client(api_key=api_key)

    def generate_embeddings(self, content_list: list) -> np.ndarray:
        """
        Generate normalized embeddings for a list of input contents 

        Args:
            content_list (list): list of strings for which embeddings will be calculated

        Returns:
            np.ndarray: a 2d array where each row corresponds to an embedding
            vector for the corresponding content_list element
        """

        max_retries = 5
        initial_delay = 30 # seconds

        for i in range(max_retries):
            try:
                result = self.gclient.models.embed_content(model=self.model_name,
                                                     contents=content_list,
                                                     config=types.EmbedContentConfig(output_dimensionality=self.num_dimensions))
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
        embedding_values_np = np.empty((len(content_list),
                                        self.num_dimensions))
        for i, embedding_obj in enumerate(result.embeddings):
            embedding_values_np[i,:] = embedding_obj.values

        # Calculate the norm for each row
        # The axis=-1 argument ensures the norm is calculated along the last
        # axis (rows for a 2D array) np.newaxis is used to reshape the norms
        # into a column vector for broadcasting
        row_norms = np.linalg.norm(embedding_values_np, axis=-1)[:, np.newaxis]
        # Normalize each row by dividing by its corresponding norm
        normed_embedding = embedding_values_np / row_norms

        return(normed_embedding)


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

    # specify schema, and ask Milvus to automatically generate IDs
    schema = MilvusClient.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64,
                     is_primary=True, auto_id=True,)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR,
                     dim=VECTOR_DIMENSIONS)
    schema.add_field(field_name="path", datatype=DataType.VARCHAR,
                     max_length=512)

    # generate index for vector field
    index_params = mclient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    if mclient.has_collection(collection_name="notes"):
        mclient.drop_collection(collection_name="notes")
    mclient.create_collection(
        collection_name="notes",
        dimension=VECTOR_DIMENSIONS,
        schema=schema,
        index_params=index_params
    )

    my_embedder = gemini_embedder(num_dimensions=VECTOR_DIMENSIONS,
                                  api_key=api_key,
                                  model_name=GEMINI_MODEL_NAME)

    # walk vault and see how many files there are
    print(f"[{vault_path}] Looking for markdown files...")
    total_file_count = 0
    for root, _, files in os.walk(vault_path):
        for filename in files:
            if filename.endswith(".md"):
                total_file_count += 1;
    if total_file_count == 0:
        print(f"[{vault_path}] No files found!")
        sys.exit(1);

    print(f"[{vault_path}] Regenerating index from {total_file_count} files...")
    start_time = time.perf_counter()
    # Iterate through vault and process files 
    # For each directory in the tree rooted at the directory top (including
    # top itself), os.walk yields a 3-tuple (dirpath, subdirnames, filenames).
    # TODO: also we should time this and report the elapsed time
    #   we can use tqdm for the progress bar: https://github.com/tqdm/tqdm?tab=readme-ov-file#manual
    with tqdm(total=total_file_count, desc="Obsidian notes", unit="note") as pbar:
        content_list = []
        file_list = []
        for root, _, files in os.walk(vault_path):
            for filename in files:
                if filename.endswith(".md"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r') as f:
                            content_list.append(f.read())
                            if len(content_list[-1]) == 0:
                                # Gemini won't generate embeddings for empty
                                # strings; skip this file
                                content_list.pop()
                                print(f"File: {filepath} SKIPPING (empty)")
                                continue
                            # print(".", end='', flush=True)
                            # print(f"File: {filepath} length {len(content_list[-1])}")
                            file_list.append(filepath)
                            if len(content_list) == BATCH_SIZE:
                                insert_into_db(embedder=my_embedder, mclient=mclient, content_list=content_list,
                                               file_list=file_list)
                                pbar.update(len(content_list))
                                content_list.clear()
                                file_list.clear()
                    except IOError as e:
                        print(f"Error reading file '{filepath}': {e}. Skipping.")
        # print("")

        # insert any leftovers
        if len(content_list) > 0:
            insert_into_db(embedder=my_embedder, mclient=mclient, content_list=content_list,
                           file_list=file_list)
            pbar.update(len(content_list))
            content_list.clear()
            file_list.clear()

    end_time = time.perf_counter()
    print(f"[{vault_path}] Index regenerated in {end_time-start_time} seconds.")


# TODO: We should have a concurrent pipeline going so that we concurrently
# calculate embeddings and insert them into the database
def insert_into_db(embedder: embedder, mclient: MilvusClient, content_list: list, file_list: list):
    """
    Inserts a list of elements into the vector db

    Args:
        embedder (embedder): embedder object
        mclient (MilvusClient): reference to a Milvus client
        content_list (list): list of text contents to be indexed
        file_list (list): list of file names corresponding to content_list

    """

    embedding_matrix = embedder.generate_embeddings(content_list);

    # constuct data to insert into Milvus.  Convert each numpy array row (each
    # vector) into a list and associate with the corresponding path

    # note that Milvus natively supports Numpy arrays for dense vectors
    data = []
    for vector, file in zip(embedding_matrix, file_list):
        data.append({"vector":vector, "path":file})

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

    my_embedder = gemini_embedder(num_dimensions=VECTOR_DIMENSIONS,
                                  api_key=api_key,
                                  model_name=GEMINI_MODEL_NAME);

    # generate embedding for the query
    embedding = my_embedder.generate_embeddings([query]);

    print(f"[{vault_path}] Querying for: '{query}'")

    # note that Milvus natively supports Numpy arrays for dense vectors
    res = client.search(
        collection_name="notes",  # target collection
        data=embedding,  # query vector
        limit=5,  # number of returned entities
        output_fields=["path"],  # specifies fields to be returned
    )

    print(f"# <distance>\t<file>")
    for hit in res[0]:
        print(f"{hit['distance']}\t{hit['entity']['path']}")

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
