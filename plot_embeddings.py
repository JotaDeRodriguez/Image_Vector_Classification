from ImageEmbeddingPipeline import ImageEmbeddingPipeline


config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection": "image_embeddings_600_random_images",
    "image_folder": r"Y:\Image_Pool\ChromaDB",
    "max_workers": 4,
    "clip_model": "openai/clip-vit-base-patch32"
}

pipeline = ImageEmbeddingPipeline(config)

pipeline.print_collection_info()

pipeline.plot_embeddings_2d_interactive(perplexity=100)

