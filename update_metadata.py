import pandas as pd
import numpy as np

from chromadb import Settings, PersistentClient
import chromadb

# Initialize PersistentClient
db_path = r"Y:\ChromaDB"
client = PersistentClient(path=db_path, settings=Settings(allow_reset=False))
collection = client.get_collection(name="image_embeddings")

peek = collection.peek()

