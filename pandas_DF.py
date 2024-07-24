from ImageEmbeddingPipeline import ImageEmbeddingPipeline

config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection": "image_embeddings",
    "image_folder": r"Y:\Image_Pool\ChromaDB",
    "max_workers": 2,
    "clip_model": "openai/clip-vit-base-patch32"
}

# Initialize the pipeline
pipeline = ImageEmbeddingPipeline(config)

# # Visualize the collection
# df = pipeline.visualize_collection(max_rows=15, vector_truncate=3)
# print(df)
#
# # If you want to save the DataFrame to a CSV file
# # df.to_csv('collection_visualization.csv', index=False)
#
# # If you want to display the DataFrame in a more readable format in the console
# print(df.to_string())


def populate_category(metadata):
    current_category = metadata['category']

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
            "Fotos_Cima_torre_binoculares",
        ],

        "equipment": [
            "Gabinete TX",
            "Equipo de Comunicaciones",
            "Equipos de TX",
            "AC-Purificador",
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
            "Central de alarmas",
            "Rack mural"
        ],

        "batteries": [
            "Baterias"
        ],

        "energy": [
            "Rectificadores",
            "Tablero Electrico abierto",
            "Energia-Disyuntores",
            "Tab.Electrico",
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
        ],

        "measurements": [
            "Cinta Metrica",
            "Vernier",
            "Brujula",
        ],

        "measurements_displays": [
            "Medidor de campo electromagnético",
            "Coordenadas",
            "Inclinacion perfiles",
            "Analizador de baterias",
            "Pinza Amperimetrica -Voltimetro"
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
            "Rooftop_Vista_General",
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
            "Greenfield_Noise",
        ],

        "shelter_exterior": [
            "Caseta",
            "AC-Casetas",
            "Casetas",
            "Sala Equipos - Luminaria Outdoor",
            "AC-Puerta Gabinete",
            "Sala Equipos - Puertas",
            "AC-Rejillas Exterior",
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
            "Digrama de conexiones",
        ]
    }

    grouped_category = "uncategorized"
    for category, items in grouped_categories.items():
        if current_category in items:
            grouped_category = category
            break

    return grouped_category
#
#
# Add a new metadata column
pipeline.update_metadata_column('simple_category', populate_category)

# Visualize the updated collection
df = pipeline.visualize_collection(max_rows=15, vector_truncate=3)
print(df.to_string())
