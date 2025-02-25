from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS for cross-origin requests
import os
import random
from chatty_core import (
    create_scene_content,
    generate_image,
    download_image,
    MAX_PROMPT_LENGTH_TELEGRAM
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins (for debugging)
# If you want to restrict to only Shopify, replace "*" with your actual Shopify domain:
# CORS(app, resources={r"/*": {"origins": "https://your-shopify-store.myshopify.com"}})

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
        # **Improved AI prompt to ensure Chatty is properly placed in the requested scene**
        prompt = (
            f"Create a highly detailed digital painting of Chatty, a retro CRT robot character, fully immersed in a {scene}. "
            f"Ensure Chatty is actively engaging with the environment in a way that makes the setting clear. "
            f"For example, if Chatty is in a medical setting, depict a hospital background, medical equipment, "
            f"and Chatty wearing a lab coat or interacting with patients. "
            f"If Chatty is at a construction site, show Chatty in a hard hat, surrounded by construction vehicles "
            f"and working with blueprints or tools. "
            f"The setting should be visually rich and immersive, making it immediately recognizable."
        )

        # Generate the image URL using chatty_core
        generated_url = generate_image(prompt, max_length=MAX_PROMPT_LENGTH_TELEGRAM)
        if not generated_url:
            return jsonify({"error": "Image generation failed."}), 500

        # Download the image locally
        image_path = download_image(generated_url, prompt)
        if not image_path:
            return jsonify({"error": "Image download failed."}), 500

        # Ensure the correct absolute path for serving the image
        image_filename = os.path.basename(image_path)
        return jsonify({"image_url": f"/meme_image/{image_filename}"})

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

