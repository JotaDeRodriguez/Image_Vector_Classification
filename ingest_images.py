from ImageEmbeddingPipeline import ImageEmbeddingPipeline
import asyncio

async def main():
    config = {
        "chroma_path": r"Y:\ChromaDB",
        "collection": "image_embeddings",
        "image_folder": r"Y:\Image_Pool\150_random_images",
        "max_workers": 4,
        "clip_model": "openai/clip-vit-base-patch32"
    }

    pipeline = ImageEmbeddingPipeline(config)

    # Print initial collection info
    pipeline.print_collection_info()

    await pipeline.process_images()

    # Print collection info after processing
    pipeline.print_collection_info()

if __name__ == "__main__":
    asyncio.run(main())
