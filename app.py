from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
import os
from ImageEmbeddingPipeline import ImageEmbeddingPipeline

app = Flask(__name__)

# Initialize the ImageEmbeddingPipeline
config = {
    "chroma_path": r"Y:\ChromaDB",
    "collection": "image_embeddings",
    "image_folder": r"Y:\Image_Pool\300_random_images",
    "max_workers": 4,
    "clip_model": "openai/clip-vit-base-patch32"
}
pipeline = ImageEmbeddingPipeline(config)


@app.route('/')
def index():
    _, metadatas = pipeline.get_all_embeddings()
    categories = {}
    image_hashes = {}
    for metadata in metadatas:
        category = metadata['category']
        image_path = metadata['image_path']
        image_hash = pipeline.get_image_hash(image_path)
        if category not in categories:
            categories[category] = []
        categories[category].append(image_path)
        image_hashes[image_path] = image_hash

    return render_template('index.html', categories=categories, image_hashes=image_hashes)


@app.route('/image/<path:image_path>')
def serve_image(image_path):
    return send_file(image_path, mimetype='image/jpeg')


@app.route('/similar_images', methods=['POST'])
def find_similar_images():
    image_path = request.form['image_path']
    n_results = int(request.form.get('n_results', 5))
    results = pipeline.query_similar_images(image_path, n_results)
    return jsonify(results)


@app.route('/tag_similar', methods=['POST'])
def tag_similar_images():
    image_path = request.form['image_path']
    new_tag = request.form['new_tag']
    n_results = int(request.form.get('n_results', 5))

    results = pipeline.query_similar_images(image_path, n_results)

    for id in results['ids'][0]:
        metadata = pipeline.collection.get(ids=[id])['metadatas'][0]
        current_tags = metadata.get('tags', [])
        if new_tag not in current_tags:
            current_tags.append(new_tag)
        pipeline.collection.update(ids=[id], metadatas=[{"tags": current_tags}])

    return redirect(url_for('index'))

@app.route('/update_tags', methods=['POST'])
def update_tags():
    image_hash = request.form['image_hash']
    new_tags = request.form['new_tags'].split(',')
    new_tags = [tag.strip() for tag in new_tags if tag.strip()]

    pipeline.collection.update(ids=[image_hash], metadatas=[{"tags": new_tags}])

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)