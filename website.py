from flask import Flask, request, jsonify, send_file
import os
import random
from chatty_core import (
    create_scene_content,
    create_simplified_image_prompt,
    generate_image,
    download_image,
    MAX_PROMPT_LENGTH_TELEGRAM
)

app = Flask(__name__)

@app.route("/")
def home():
    return "Chatty Meme Generator API is running!"

@app.route("/create_meme", methods=["POST"])
def create_meme():
    data = request.get_json()
    scene = data.get("scene", "").strip()
    if not scene:
        scene = "Chatty in an awesome scene"
    try:
        # Build the detailed scene description for Chatty
        scene_content = create_scene_content(scene)
        # Create a simplified DALL·E prompt from the scene
        prompt = create_simplified_image_prompt(scene_content)
        # Generate the image URL using chatty_core
        generated_url = generate_image(prompt, max_length=MAX_PROMPT_LENGTH_TELEGRAM)
        if not generated_url:
            return jsonify({"error": "Image generation failed."}), 500
        # Download the image locally
        image_path = download_image(generated_url, prompt)
        if not image_path:
            return jsonify({"error": "Image download failed."}), 500
        # Return a relative URL for the generated image
        image_url = "/meme_image/" + os.path.basename(image_path)
        return jsonify({"image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/meme_image/<filename>")
def serve_meme(filename):
    image_dir = os.getenv("IMAGE_DIR", "generated_images")
    image_path = os.path.join(image_dir, filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype="image/png")
    return "Image not found", 404

if __name__ == "__main__":
    app.run(debug=True)
