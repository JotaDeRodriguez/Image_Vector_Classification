import streamlit as st
import os
from PIL import Image
import io
from ChromaClient import ImageEmbeddingPipeline

# Initialize the ImageEmbeddingPipeline
config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection_name": "image_embeddings_test",
    "image_folder": r"Y:\Image_Pool\ChromaDB",
    "max_workers": 4,
    "clip_model": "openai/clip-vit-base-patch32"
}

pipeline = ImageEmbeddingPipeline(config)

st.title("Image Similarity Search")

# Sidebar for query type selection
query_type = st.sidebar.radio("Select Query Type", ["Image", "Text"])

if query_type == "Image":
    uploaded_file = st.file_uploader("Choose a query image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Query Image", use_column_width=True)

        # Save the uploaded image temporarily
        with open("temp_query_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Perform the query
        results = pipeline.query_similar_images("temp_query_image.jpg", n_results=5)

        # Remove the temporary file
        os.remove("temp_query_image.jpg")

        if results:
            st.subheader("Similar Images:")
            for i, (id, distance, metadata) in enumerate(
                    zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
                st.write(f"Image {i + 1}")
                st.write(f"Distance: {distance:.4f}")
                st.write(f"Category: {metadata['category']}")
                image_path = metadata['image_path']
                image = Image.open(image_path)
                st.image(image, caption=f"Image {i + 1}", use_column_width=True)
                st.write("---")
        else:
            st.write("No similar images found.")

elif query_type == "Text":
    query_text = st.text_input("Enter query text")
    if st.button("Search"):
        results = pipeline.query_similar_images_by_text(query_text, n_results=5)

        if results:
            st.subheader("Similar Images:")
            for i, (id, distance, metadata) in enumerate(
                    zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
                st.write(f"Image {i + 1}")
                st.write(f"Distance: {distance:.4f}")
                st.write(f"Category: {metadata['category']}")
                image_path = metadata['image_path']
                image = Image.open(image_path)
                st.image(image, caption=f"Image {i + 1}", use_column_width=True)
                st.write("---")
        else:
            st.write("No similar images found.")

# Display some stats about the database
st.sidebar.subheader("Database Stats")
total_images = pipeline.collection.count()
st.sidebar.write(f"Total images in database: {total_images}")

# Add a button to show the 2D embedding plot
if st.sidebar.button("Show 2D Embedding Plot"):
    plot_buf = pipeline.plot_embeddings_2d()
    st.image(plot_buf, caption="2D visualization of image embeddings", use_column_width=True)