import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import chromadb
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import filetype
import hashlib
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import io

class ImageEmbeddingPipeline:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_chroma()
        self.setup_clip_model()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_chroma(self):
        self.client = chromadb.PersistentClient(path=self.config['chroma_path'])
        self.collection = self.client.get_or_create_collection(self.config['collection_name'])

    def setup_clip_model(self):
        self.model = CLIPModel.from_pretrained(self.config['clip_model'])
        self.processor = CLIPProcessor.from_pretrained(self.config['clip_model'])

    async def process_images(self):
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            for root, _, files in os.walk(self.config['image_folder']):
                for file in files:
                    if filetype.is_image(os.path.join(root, file)):
                        futures.append(executor.submit(self.process_single_image, os.path.join(root, file)))

            for future in tqdm(futures, desc="Processing images"):
                await asyncio.wrap_future(future)

    def process_single_image(self, image_path):
        try:
            image_hash = self.get_image_hash(image_path)
            existing_item = self.collection.get(ids=[image_hash])

            if not existing_item['ids']:
                embedding = self.get_image_embedding(image_path)
                if embedding is not None:
                    category = self.get_category(image_path)
                    self.add_image_to_collection(image_path, image_hash, embedding, category)
                    self.logger.info(f"Added {image_path} to collection")
                else:
                    self.logger.warning(f"Failed to get embedding for {image_path}")
            else:
                self.logger.info(f"Image {image_path} already in database")
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")

    def get_image_hash(self, image_path):
        with open(image_path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def get_image_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features.squeeze().numpy()
        except FileNotFoundError:
            self.logger.error(f"File not found: {image_path}")
        except Exception as e:
            self.logger.error(f"Error getting embedding for {image_path}: {str(e)}")
        return None

    def get_category(self, image_path):
        # TODO: Implement category detection using CLIP's text capabilities
        return "uncategorized"

    def get_text_description(self, image_path):
        # TODO: Fill out text description
        return ""

    def add_image_to_collection(self, image_path, image_hash, embedding, category):
        metadata = {
            "image_path": image_path,
            "category": category,
            "image_hash": image_hash
        }
        self.collection.add(
            embeddings=[embedding.tolist()],
            ids=[image_hash],
            metadatas=[metadata]
        )
        self.logger.info(f"Added to collection: {image_hash}")

    def query_similar_images(self, query_image_path, n_results=5):
        query_embedding = self.get_image_embedding(query_image_path)
        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            self.logger.info(f"Query results: {results}")
            return results
        self.logger.warning(f"Failed to get embedding for query image: {query_image_path}")
        return None

    def print_collection_info(self):
        count = self.collection.count()
        self.logger.info(f"Total items in collection: {count}")

    def query_similar_images_by_text(self, query_text, n_results=5):
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_embedding = text_features.squeeze().numpy()

        results = self.collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n_results
        )
        self.logger.info(f"Text query results: {results}")
        return results

    def get_all_embeddings(self):
        all_results = self.collection.get(include=['embeddings', 'metadatas'])
        embeddings = np.array(all_results['embeddings'])
        metadatas = all_results['metadatas']
        return embeddings, metadatas

    def plot_embeddings_2d(self):
        embeddings, metadatas = self.get_all_embeddings()

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(embeddings)), cmap='viridis')
        plt.colorbar(scatter)
        plt.title('2D visualization of image embeddings')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        return buf


async def main():
    config = {
        "chroma_path": r"Y:\ChromaDB",
        "collection_name": "image_embeddings_test",
        "image_folder": r"Y:\Image_Pool\ChromaDB",
        "max_workers": 4,
        "clip_model": "openai/clip-vit-base-patch32"
    }

    pipeline = ImageEmbeddingPipeline(config)

    # Print initial collection info
    pipeline.print_collection_info()

    await pipeline.process_images()

    # Print collection info after processing
    pipeline.print_collection_info()

    # Example query
    query_image_path = r"Y:\Imagenes_Clasificadas_Overhauled\Categorias Final\torre\2022-03-23 15.32.53.jpg"
    results = pipeline.query_similar_images(query_image_path, n_results=5)
    if results:
        print("Query results:")
        for i, (id, distance, metadata) in enumerate(
                zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
            print(f"{i + 1}. ID: {id}, Distance: {distance}, Metadata: {metadata}")
    else:
        print("No results found or error occurred during query.")


if __name__ == "__main__":
    asyncio.run(main())