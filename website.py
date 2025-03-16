from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS for cross-origin requests
import os
import random
# Replaced create_scene_content with generate_scene_from_theme
from chatty_core import (
    generate_scene_from_theme,
    create_simplified_image_prompt,
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
        # Build the scene description for Chatty (dynamic approach)
        scene_content = generate_scene_from_theme(scene)

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

        # Return the filename for the client to retrieve
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
