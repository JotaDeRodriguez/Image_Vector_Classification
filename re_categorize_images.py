from ImageEmbeddingPipeline import ImageEmbeddingPipeline
import asyncio

config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection": "image_embeddings",
    "image_folder": r"Y:\Image_Pool\ChromaDB",
    "max_workers": 2,
    "clip_model": "openai/clip-vit-base-patch32"
}

pipeline = ImageEmbeddingPipeline(config)
pipeline.categorize_uncategorized_images()
