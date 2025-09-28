# obsidian-rag

This is prototype code that will index a (local) Obsidian vault using a vector database and enable natural language queries over that database.  It uses Milvus as the vector database (in its "lite" configuration with a local database) and Gemini as the embedding generator.

Example usage:
* To generate a new index: `./obsidian-rag.py --query`
* To query the database: `./obsidian-rag.py --query "Some query string"`

Example output from a query:
```
> ./obsidian-rag.py --query "performance of io_uring with Mochi"
[/home/carns/Documents/carns-obsidian] Opening database...
[/home/carns/Documents/carns-obsidian] Querying for: 'performance of io_uring with Mochi'
# <distance>	<file>
0.7298460602760315	/home/carns/Documents/carns-obsidian/work/mochi/liburing-ideas.md
0.702826201915741	/home/carns/Documents/carns-obsidian/work/mochi/mochi-quarterly-2023-07-27.md
0.6995981931686401	/home/carns/Documents/carns-obsidian/work/mochi/pdsw-2025-near-store-sketch.md
0.6982080936431885	/home/carns/Documents/carns-obsidian/work/mochi/mochi-meeting-notes-2021-12-1.md
0.6954102516174316	/home/carns/Documents/carns-obsidian/work/mochi/mochi-rpc-benchmarking-2024-12.md
```

Modify the constants at the top of `obsidian-rag.py` to specify the path to your Obsidian vault and the parameters for your embedding model (including how to obtain your Gemini API key).

There is also a simple benchmark in the `util` directory that can be used to benchmark different embedding models.
