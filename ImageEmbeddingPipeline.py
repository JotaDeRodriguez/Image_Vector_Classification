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
import plotly.express as px


from Yolo_classify_image import yolo_classify_image
from ultralytics import YOLO

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
        self.collection = self.client.get_or_create_collection(self.config['collection'])

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
                    # category = os.path.basename(os.path.dirname(image_path))
                    category = self.get_category(image_path)
                    self.add_image_to_collection(image_path, image_hash, embedding, category)
                    self.logger.info(f"Added {image_path} to collection. Category: {category}")
                else:
                    self.logger.warning(f"Failed to get embedding for {image_path}")
            else:
                pass
                # self.logger.info(f"Image {image_path} already in database")
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

        model = YOLO(r"C:\Users\juandavid.rodriguez\Downloads\last.pt")

        result = yolo_classify_image(image_path=image_path,
                                     model=model,
                                     confidence_threshold=0.75
                                     )

        return result

    def get_text_description(self, image_path):

        return ""

    def add_image_to_collection(self, image_path, image_hash, embedding, category):
        metadata = {
            "image_path": image_path,
            "category": category,
            "image_hash": image_hash,
            "description": "",
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
                n_results=n_results,
                include=['metadatas']
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

    def plot_embeddings_2d_interactive(self):
        embeddings, metadatas = self.get_all_embeddings()

        # Extract categories and filenames from metadatas
        categories = [meta['category'] for meta in metadatas]
        filenames = [meta['image_path'] for meta in metadatas]  # Adjust this if 'image_path' is not the correct key

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings-1)))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create an interactive plot
        fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], color=categories,
                         hover_data={'filename': filenames, 'category': categories},
                         title="2D Visualization of Image Embeddings by Category")

        # Update hover template to include category and filename
        fig.update_traces(marker=dict(size=5),
                          selector=dict(mode='markers'),
                          hovertemplate="<b>Filename:</b> %{customdata[0]}<br><b>Category:</b> %{customdata[1]}<extra></extra>")

        fig.show()
        return embeddings_2d

    def get_all_categories(self):
        _, metadatas = self.get_all_embeddings()
        categories = set(metadata['category'] for metadata in metadatas)
        return list(categories)

    def get_images_by_category(self, category):
        _, metadatas = self.get_all_embeddings()
        return [metadata['image_path'] for metadata in metadatas if metadata['category'] == category]

    def update_tags(self, image_hash, new_tags):
        metadata = self.collection.get(ids=[image_hash])['metadatas'][0]
        metadata['tags'] = new_tags
        self.collection.update(ids=[image_hash], metadatas=[metadata])
        self.logger.info(f"Updated tags for {image_hash}")
