from ImageEmbeddingPipeline import ImageEmbeddingPipeline


config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection": "image_embeddings",
    "image_folder": r"Y:\Image_Pool\ChromaDB",
    "max_workers": 4,
    "clip_model": "openai/clip-vit-base-patch32"
}

pipeline = ImageEmbeddingPipeline(config)


# # Query by image
# query_image_path = r"Y:\American Tower\20230502_114624.jpg"
# results = pipeline.query_similar_images(query_image_path, n_results=5)
# if results:
#     print("Query results:")
#     for i, (id, distance, metadata) in enumerate(
#             zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
#         print(f"{i + 1}. ID: {id}, Distance: {distance}, Category: " + metadata["category"] + ", Path: " + metadata["image_path"])
# else:
#     print("No results found or error occurred during query.")


# # Query by text
# query_text = "Tall Antenna Tower"
#
# results = pipeline.query_similar_images_by_text(query_text, n_results=10)
#
# if results:
#     print("Query results:")
#     for i, (id, distance, metadata) in enumerate(
#             zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
#         print(f"{i + 1}. ID: {id}, Distance: {distance}, Category: " + metadata["category"] + ", Path: " + metadata["image_path"])
# else:
#     print("No results found or error occurred during query.")

pipeline.print_collection_info()
