from ImageEmbeddingPipeline import ImageEmbeddingPipeline
import cv2


config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection": "image_embeddings",
    "image_folder": r"Y:\Image_Pool\ChromaDB",
    "max_workers": 4,
    "clip_model": "openai/clip-vit-base-patch32"
}

# Initialize the pipeline
pipeline = ImageEmbeddingPipeline(config)

# Filter the dataset to only include 'uncategorized' images
# pipeline.filter_by_category('uncategorized')

# # Query by image
# query_image_path = r"Y:\Imagenes_Clasificadas_Overhauled\Nuevas_Categorias\Caseta\89364_220131_123710959.jpg"
# results = pipeline.query_similar_images(query_image_path, n_results=10)
#
# if results:
#     print("Query results:")
#     for i, (id, metadata) in enumerate(
#             zip(results['ids'][0], results['metadatas'][0])):
#         print(f"{i + 1}. ID: {id}, Category: " + metadata["category"] + ", Path: " + metadata["image_path"])
#
#         # Display the image
#         image_path = metadata["image_path"]
#         image = cv2.imread(image_path)
#         if image is not None:
#             # Resize image to make it smaller for display
#             height, width = image.shape[:2]
#             new_height = 500  # desired height
#             new_width = int((new_height / height) * width)
#             resized_image = cv2.resize(image, (new_width, new_height))
#
#             # Display the image in a window
#             cv2.imshow(f"Image {i + 1}", resized_image)
#         else:
#             print(f"Could not load image at path: {image_path}")
#
#     # Wait for a key press to close the windows
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No results found or error occurred during query.")



# # Query by text
# query_text = "Photo of a white utility shed"
#
# results = pipeline.query_similar_images_by_text(query_text, n_results=20)
#
# if results:
#     print("Query results:")
#     for i, (id, metadata) in enumerate(
#             zip(results['ids'][0], results['metadatas'][0])):
#         print(f"{i + 1}. ID: {id}, Category: " + metadata["category"] + ", Path: " + metadata["image_path"])
#
#         # Display the image
#         image_path = metadata["image_path"]
#         image = cv2.imread(image_path)
#         if image is not None:
#             # Resize image to make it smaller for display
#             height, width = image.shape[:2]
#             new_height = 500  # desired height
#             new_width = int((new_height / height) * width)
#             resized_image = cv2.resize(image, (new_width, new_height))
#
#             # Display the image in a window
#             cv2.imshow(f"Image {i + 1}", resized_image)
#         else:
#             print(f"Could not load image at path: {image_path}")
#
#     # Wait for a key press to close the windows
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# else:
#     print("No results found or error occurred during query.")

pipeline.print_collection_info()
