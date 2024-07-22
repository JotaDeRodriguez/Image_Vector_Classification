from chromadb import Settings, PersistentClient
import chromadb

# Initialize PersistentClient
db_path = r"Y:\ChromaDB"
client = PersistentClient(path=db_path, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection("image_embeddings")



# Set to hold all unique category values
category_values = set()

# Process images in batches
count = collection.count()
batch_size = 30

for offset in range(0, count, batch_size):
    batch = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
    # Add each metadata's category value to the set
    category_values.update(metadata.get('category', 'Unknown') for metadata in batch["metadatas"])

print("List of all unique category values:", list(category_values))

