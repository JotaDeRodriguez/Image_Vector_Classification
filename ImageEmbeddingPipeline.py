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
import pandas as pd
from PIL import Image



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
        def get_simple_category(category):

            grouped_categories = {
                "signage": [
                    "Señalizacion - Prohibido el Paso",
                    "Señalizacion - Antena RF",
                    "Señalizacion - Salida",
                    "Señalizacion - Vigilancia",
                    "Señalizacion - Caida a distinto nivel",
                    "Señalizacion Uso EPP",
                    "Señalizacion - Riesgo electrico",
                    "Señalizacion Grupal"
                ],
                "rru": [
                    "RRU",
                    "RRUs"
                ],
                "rru_nokia": [
                    "RRU NOKIA"
                ],

                "telecom_equipment_outdoors": [
                    "Parabola",
                    "Panel",
                    "Fotos_Cima_torre_binoculares"

                ],

                "equipment": [
                    "Rectificadores",
                    "Gabinete TX",
                    "Equipo de Comunicaciones",
                    "Equipos de TX",
                    "AC-Purificador",
                    "Tablero Electrico abierto",
                    "Baterias",
                    "Energia-Disyuntores",
                    "Tab.Electrico",
                    "Gabinetes cerrados",
                    "Gabinete Rectificador",
                    "Equipos en pared",
                    "Gabinete en pared",
                    "Caja FO CTO",
                    "Remota",
                    "Detector de humo",
                    "Caja de energia adosada en pared",
                    "RBS 6201 Ericsson",
                    "Rack mural"

                ],

                "shelter_interior": [
                    "Sala Equipos - Luminarias Indoor",
                    "interior_sala_equipos",
                    "Interior Sala de equipos",
                    "rejiband",
                    "Sala Equipos - Combinadores",
                    "AC-Ventiladores",
                    "AC-Interior",
                    "Sala Equipos - Panoramicas Indoor",
                    "Bastidor vacio",
                    "AC-Rejillas interior",
                    "Rack mural"
                ],

                "tower": [
                    "torre_cima",
                    "torre"

                ],

                "extinguisher": [
                    "Extintor"
                ],

                "energy_displays": [
                    "Circutor - Contador Albertis Telecom",
                    "Energia-Display-Equipos",
                    "Central de alarmas"

                ],

                "measurements": [
                    "Cinta Metrica",
                    "Vernier",
                    "Medidor de campo electromagnético",
                    "Coordenadas",
                    "Pinza Amperimetrica -Voltimetro",
                    "Brujula",
                    "Inclinacion perfiles",
                    "Analizador de baterias"
                ],

                "gamesystem": [
                    "Etiqueta Gamesystem"
                ],

                "rooftop_views": [
                    "RT.Mastil Slim",
                    "RT.Mastil",
                    "RT_Mastil_Otros",
                    "Empalizada",
                    "RT.interno",
                    "RT. Mastil Camuflado",
                    "Rooftop_Vista_General"
                ],

                "greenfield_views": [
                    "Greenfield_Vista_General",
                    "torre_base",
                    "torre_general_100_mts",
                    "Greenfield_Vallado",
                    "Greenfield_torre Transfomrador",
                    "GF.interno",
                    "Greenfield_Acceso",
                    "GF Bastidor abierto",
                    "GF-Bastidores Cerrado",
                    "Tablero Outdoor Cerrado",
                    "Greenfield_Noise"
                ],

                "shelter_exterior": [
                    "Caseta",
                    "AC-Casetas",
                    "Casetas",
                    "Sala Equipos - Luminaria Outdoor",
                    "AC-Puerta Gabinete",
                    "Sala Equipos - Puertas",
                    "AC-Rejillas Exterior"

                ],
                "buildings": [
                    "Edificios",
                    "Greenfield  Vista Urbana"
                ],

                "profiles": [
                    "Balizaje",
                    "Perfiles",
                    "Anclaje Inferior - Linea de vida",
                    "Anclaje Superior - Linea de vida",

                ],

                "cabling": [
                    "Interior Mastil Camuflado",
                    "Sala Equipos - Pasamuros",
                    "Bocas de conexion",
                    "Cableado externo",
                ],

                "panoramic": [
                    "Greenfield_Panoramica",
                    "Panoramicas outdoor",
                ],
                "panoramic_rooftop": [
                    "RT.Panoramicas",

                ],
                "keys": [
                    "GF_LLaves",
                    "RT_Llaves",
                ],

                "documentation": [
                    "CNX_Responsabilidades",
                    "Intructivos",
                    "Etiquetas-Series",
                    "Etiquetado Gabinetes y equipos",
                    "Diagramas unifilares Electricos",
                    "Digrama de conexiones"
                ]}
            simple_category = "uncategorized"

            for grouped_category, items in grouped_categories.items():
                if category in items:
                    simple_category = grouped_category
                    break

            return simple_category

        metadata = {
            "image_path": image_path,
            "category": category,
            "image_hash": image_hash,
            "description": "",
            "simple_category": get_simple_category(category)
        }
        self.collection.add(
            embeddings=[embedding.tolist()],
            ids=[image_hash],
            metadatas=[metadata]
        )
        self.logger.info(f"Added to collection: {image_hash}")


    def filter_by_category(self, category):
        """
        Filter the collection to only include images of a specific category.
        This method creates a new filtered collection.
        """
        filtered_collection_name = f"{self.config['collection']}_{category}"
        self.filtered_collection = self.client.get_or_create_collection(filtered_collection_name)

        # Get all items from the main collection
        all_items = self.collection.get(include=['embeddings', 'metadatas'])

        # Filter items by category
        filtered_ids = []
        filtered_embeddings = []
        filtered_metadatas = []

        for id, embedding, metadata in zip(all_items['ids'], all_items['embeddings'], all_items['metadatas']):
            if metadata['category'] == category:
                filtered_ids.append(id)
                filtered_embeddings.append(embedding)
                filtered_metadatas.append(metadata)

        # Add filtered items to the new collection
        if filtered_ids:
            self.filtered_collection.add(
                ids=filtered_ids,
                embeddings=filtered_embeddings,
                metadatas=filtered_metadatas
            )

        self.logger.info(f"Created filtered collection '{filtered_collection_name}' with {len(filtered_ids)} items")

    def categorize_uncategorized_images(self):
        # Get all embeddings and metadata
        embeddings, metadatas = self.get_all_embeddings()

        # Separate categorized and uncategorized images
        categorized_indices = [i for i, m in enumerate(metadatas) if m['simple_category'] != 'uncategorized']
        uncategorized_indices = [i for i, m in enumerate(metadatas) if m['simple_category'] == 'uncategorized']

        categorized_embeddings = embeddings[categorized_indices]
        uncategorized_embeddings = embeddings[uncategorized_indices]

        # If there are no uncategorized images, we're done
        if not uncategorized_indices:
            self.logger.info("No uncategorized images found.")
            return

        # Find nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(categorized_embeddings)

        distances, indices = nn.kneighbors(uncategorized_embeddings)

        # Assign categories based on nearest neighbors
        for i, nearest_index in enumerate(indices.flatten()):
            uncategorized_metadata = metadatas[uncategorized_indices[i]]
            nearest_neighbor_metadata = metadatas[categorized_indices[nearest_index]]

            # Update the category
            new_category = nearest_neighbor_metadata['simple_category']
            uncategorized_metadata['simple_category'] = new_category
            print(f"Assigned {i} to {new_category}")

            # Update the item in the collection
            self.collection.update(
                ids=[uncategorized_metadata['image_hash']],
                metadatas=[uncategorized_metadata]
            )

            self.logger.info(f"Assigned category '{new_category}' to image {uncategorized_metadata['image_path']}")

        self.logger.info(f"Categorization complete. {len(uncategorized_indices)} images were categorized.")


    def query_similar_images(self, query_image_path, n_results=5):
        query_embedding = self.get_image_embedding(query_image_path)
        if query_embedding is not None:
            # Use the filtered collection if it exists, otherwise use the main collection
            collection_to_query = self.filtered_collection if hasattr(self, 'filtered_collection') else self.collection
            results = collection_to_query.query(
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

        # Use the filtered collection if it exists, otherwise use the main collection
        collection_to_query = self.filtered_collection if hasattr(self, 'filtered_collection') else self.collection
        results = collection_to_query.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n_results,
            include=['metadatas']
        )
        self.logger.info(f"Query results: {results}")
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
        categories = [meta['simple_category'] for meta in metadatas] # change to 'category'
        filenames = [meta['image_path'] for meta in metadatas]
        orig_categories = [meta['category'] for meta in metadatas]

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings-1)))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create an interactive plot
        fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], color=categories,
                         hover_data={'filename': filenames, 'category': categories, 'original_category': orig_categories},
                         title="2D Visualization of Image Embeddings by Category")

        # Update hover template to include category and filename
        fig.update_traces(marker=dict(size=5),
                          selector=dict(mode='markers'),
                          hovertemplate="<b>Filename:</b> %{customdata[0]}<br><b>Category:</b> %{customdata[1]}<br><b>Original Category:</b> %{customdata[2]}<extra></extra>")

        fig.show()
        return embeddings_2d

    def plot_embeddings_3d_interactive(self):
        embeddings, metadatas = self.get_all_embeddings()

        # Extract categories and filenames from metadatas
        categories = [meta['simple_category'] for meta in metadatas]  # change to 'category'
        filenames = [meta['image_path'] for meta in metadatas]
        orig_categories = [meta['category'] for meta in metadatas]

        # Perform t-SNE with 3 components
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings - 1)))
        embeddings_3d = tsne.fit_transform(embeddings)

        # Create an interactive 3D plot
        fig = px.scatter_3d(x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
                            color=categories,
                            hover_data={'filename': filenames, 'category': categories,
                                        'original_category': orig_categories},
                            title="3D Visualization of Image Embeddings by Category")

        # Update hover template to include category and filename
        fig.update_traces(marker=dict(size=3),
                          selector=dict(mode='markers'),
                          hovertemplate="<b>Filename:</b> %{customdata[0]}<br><b>Category:</b> %{customdata[1]}<extra></extra>")

        fig.show()
        return embeddings_3d

    def add_metadata_column(self, column_name, value_function):
        """
        Add a new metadata column to the collection and populate it using a custom function.
        If the column already exists, it will not be added again.

        :param column_name: The name of the new metadata column
        :param value_function: A function that takes the existing metadata as input and returns the new value
        """
        # Get all items from the collection
        all_items = self.collection.get(include=['metadatas'])

        # Check if the column already exists
        if any(column_name in metadata for metadata in all_items['metadatas']):
            self.logger.info(f"Column '{column_name}' already exists. Skipping addition.")
            return

        # Update each item with the new metadata
        for i, metadata in enumerate(all_items['metadatas']):
            new_value = value_function(metadata)
            metadata[column_name] = new_value

            # Update the item in the collection
            self.collection.update(
                ids=[all_items['ids'][i]],
                metadatas=[metadata]
            )

        self.logger.info(f"Added new metadata column: {column_name}")

    def update_metadata_column(self, column_name, value_function):
        """
        Update an existing metadata column in the collection or add it if it doesn't exist.
        The column is populated using a custom function.

        :param column_name: The name of the metadata column to update or add
        :param value_function: A function that takes the existing metadata as input and returns the new value
        """
        # Get all items from the collection
        all_items = self.collection.get(include=['metadatas'])

        # Update each item's metadata
        for i, metadata in enumerate(all_items['metadatas']):
            new_value = value_function(metadata)
            print(f"got {new_value} value for {i}")
            metadata[column_name] = new_value

            print("Updating_Collection")
            # Update the item in the collection
            self.collection.update(
                ids=[all_items['ids'][i]],
                metadatas=[metadata]
            )

        self.logger.info(f"Updated metadata column: {column_name}")


    def visualize_collection(self, max_rows=10, vector_truncate=5):
        # Get all data from the collection
        data = self.collection.get(include=['metadatas', 'embeddings'])

        # Create a DataFrame
        df = pd.DataFrame({
            'ID': data['ids'],
            'Vector': [str(np.array(v[:vector_truncate])) + '...' for v in data['embeddings']],
        })

        # Add metadata columns
        if data['metadatas']:
            for key in data['metadatas'][0].keys():
                df[key] = [item.get(key, '') for item in data['metadatas']]

        # Truncate IDs
        df['ID'] = df['ID'].apply(lambda x: x[:8] + '...' if len(x) > 8 else x)

        # Limit the number of rows
        df = df.head(max_rows)

        return df

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
