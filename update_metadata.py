from chromadb import Settings, PersistentClient
import chromadb
from ultralytics import YOLO
from Yolo_classify_image import yolo_classify_image

# Initialize YOLO model
model_path = r"C:\Users\juandavid.rodriguez\Downloads\last.pt"
model = YOLO(model_path)

# Initialize PersistentClient
db_path = r"Y:\ChromaDB"
client = PersistentClient(path=db_path, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection("image_embeddings")


def update_metadata_category(metadata):
    image_path = metadata.get("image_path")
    result = yolo_classify_image(image_path=image_path, model=model, confidence_threshold=0.65)
    metadata["category"] = result
    return metadata


def add_text_metadata_field(metadata, description):
    """
    Adds a 'text' field to metadata.
    """
    metadata['description'] = description
    return metadata


# Process images in batches
count = collection.count()
batch_size = 100

description = ""

for offset in range(0, count, batch_size):
    batch = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
    # Apply update only if category is 'Argilla'
    updated_metadata = [update_metadata_category(metadata) if metadata.get('category') == 'Argilla' else metadata for metadata in batch["metadatas"]]
    collection.update(ids=batch["ids"], metadatas=updated_metadata)


print("Processing complete.")
