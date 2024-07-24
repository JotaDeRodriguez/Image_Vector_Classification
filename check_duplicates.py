from chromadb.utils import embedding_functions
import chromadb
from chromadb import Settings, PersistentClient



# Assuming you have already created a client
db_path = r"Y:\ChromaDB"
client = PersistentClient(path=db_path, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection("image_embeddings")

# Get all collections
collections = client.list_collections()

for collection in collections:
    # Get all embeddings in the collection
    results = collection.get(include=['embeddings'])

    embeddings = results['embeddings']

    # Create a set of embeddings (as tuples, since lists are not hashable)
    unique_embeddings = set(tuple(emb) for emb in embeddings)

    # Compare the length of the original list with the set
    if len(embeddings) != len(unique_embeddings):
        print(f"Duplicates found in collection: {collection.name}")
        print(f"Total embeddings: {len(embeddings)}")
        print(f"Unique embeddings: {len(unique_embeddings)}")
        print(f"Number of duplicates: {len(embeddings) - len(unique_embeddings)}")
    else:
        print(f"No duplicates found in collection: {collection.name}")