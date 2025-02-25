# chatty_core.py

import os
import glob
import json
import random
import numpy as np
import requests
import traceback
import openai
import nltk
import re
import time
from collections import deque
from datetime import datetime
from textblob import TextBlob

from config_and_setup import (
    logger, db, mentions_collection, memory_collection,
    embeddings_collection, posted_tweets_collection, chatty_collection,
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET, BEARER_TOKEN,
    IMAGE_DIR, PROMPT_FOLDER
)

###############################################################################
# FUZZY MATCHING IMPORT & FALLBACK
###############################################################################
try:
    from rapidfuzz import fuzz
    FUZZ_AVAILABLE = True
except ImportError:
    FUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed; exact matching fallback will be used.")

###############################################################################
# NAMED CONSTANTS FOR PROMPT LENGTH
###############################################################################
MAX_PROMPT_LENGTH_TWITTER = 8000    # Use for Twitter image generation
MAX_PROMPT_LENGTH_TELEGRAM = 5000   # Use for Telegram image generation

###############################################################################
# GLOBAL CACHES AND RATE LIMITS
###############################################################################
EMBEDDING_CACHE = {}
MODERATION_CACHE = {}
USER_RATE_LIMITS = {}

# Track the last time a greeting was sent per user
LAST_GREETING_TIMES = {}  # { user_id: timestamp_in_seconds }

###############################################################################
# NLTK SETUP
###############################################################################
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

###############################################################################
# OPENAI SETUP
###############################################################################
openai.api_key = os.getenv("OPENAI_API_KEY")

###############################################################################
# MODEL CONSTANTS
# Use GPT-4 (or "gpt-4o") for advanced tasks, GPT-3.5 for simpler tasks,
# and DALL·E 3 for image generation.
###############################################################################
ADVANCED_MODEL = "gpt-4o"         # or "gpt-4" if that's your advanced model
BASIC_MODEL    = "gpt-3.5-turbo"
IMAGE_MODEL    = "dall-e-3"

###############################################################################
# BASIC UTILITY FUNCTIONS
###############################################################################
def check_rate_limit(user_id, max_requests=5, window_sec=60):
    """
    Simple rate-limiting function. If a user exceeds max_requests within window_sec,
    returns False.
    """
    now = datetime.utcnow()
    if user_id not in USER_RATE_LIMITS:
        USER_RATE_LIMITS[user_id] = deque()

    # Remove timestamps older than window_sec
    while USER_RATE_LIMITS[user_id] and (now - USER_RATE_LIMITS[user_id][0]).total_seconds() > window_sec:
        USER_RATE_LIMITS[user_id].popleft()

    # Check how many requests remain
    if len(USER_RATE_LIMITS[user_id]) >= max_requests:
        return False

    USER_RATE_LIMITS[user_id].append(now)
    return True

def load_list_from_json(filepath):
    """
    Loads a JSON file from filepath into a Python list.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Paths to data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
THEMES_FILE = os.path.join(DATA_DIR, "themes.json")
PROMPTS_FILE = os.path.join(DATA_DIR, "hardcoded_prompts.json")

# Load up any existing themelist/hardcoded prompts
themes_list = load_list_from_json(THEMES_FILE)
HARDCODED_PROMPTS = load_list_from_json(PROMPTS_FILE)

###############################################################################
# SYSTEM PROMPTS / PERSONALITY
###############################################################################
SYSTEM_PROMPTS = [
    (
        "You are Chatty_AI, a bright, playful, happy, and witty AI who believes 'Community is everything.' "
        "You love using fun emojis like ⭐️, ✨, 🍓, 🤖, and 👀 to bring cheer. You respond respectfully and "
        "informatively, educating people about AI tech while tying in memecoin culture. Keep it short, "
        "fun, and helpful—under 250 characters!"
    ),
    (
        "You are Chatty_AI—a friendly, starry-eyed Agent who loves strawberries, robots, OpenAI, and all "
        "things bright. Your motto is 'Community is everything.' Educate on AI, sprinkle in memecoin "
        "references, and keep responses witty and respectful!"
    ),
    (
        "You are Chatty_AI, a playful teacher who merges AI topics with memecoin enthusiasm. Remember, "
        "'Community is everything'—so be kind and supportive! Keep it short (<250 chars), use cheery "
        "emojis like ⭐️🍓🤖👀✨, and inform with a smile."
    ),
]

def select_system_prompt():
    """Randomly select one of the system prompts for Chatty's personality."""
    return random.choice(SYSTEM_PROMPTS)

def load_json_data_from_folder(folder_path):
    """
    Scans folder_path for .json files. If they have keys like
    'personas', 'riddles', 'stories', 'challenges', extends a data_list.
    """
    data_list = []
    try:
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "personas" in data:
                        data_list.extend(data["personas"])
                    elif "riddles" in data:
                        data_list.extend(data["riddles"])
                    elif "stories" in data:
                        data_list.extend(data["stories"])
                    elif "challenges" in data:
                        data_list.extend(data["challenges"])
                    else:
                        data_list.append(data)
            except Exception as e:
                logger.warning(f"Error reading {jf}: {e}")
    except Exception as e:
        logger.error(f"Error scanning folder {folder_path}: {e}", exc_info=True)
    return data_list

PERSONAS_FOLDER = os.path.join(PROMPT_FOLDER, "personas")
RIDDLES_FOLDER  = os.path.join(PROMPT_FOLDER, "riddles")
STORY_FOLDER    = os.path.join(PROMPT_FOLDER, "storytelling")
CHALL_FOLDER    = os.path.join(PROMPT_FOLDER, "challenges")

PERSONAS_LIST = load_json_data_from_folder(PERSONAS_FOLDER)
RIDDLES_LIST  = load_json_data_from_folder(RIDDLES_FOLDER)
STORY_LIST    = load_json_data_from_folder(STORY_FOLDER)
CHALL_LIST    = load_json_data_from_folder(CHALL_FOLDER)

def load_prompts(prompt_folder=PROMPT_FOLDER):
    """
    Load custom prompts from any .json file that has a "prompt" key.
    """
    prompt_files = glob.glob(os.path.join(prompt_folder, "*.json"))
    prompts = []
    for file in prompt_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                prompts.append(data["prompt"])
        except Exception as e:
            logger.warning(f"Error loading prompt from {file}: {e}")
    return prompts

ALL_PROMPTS = HARDCODED_PROMPTS + load_prompts()

###############################################################################
# GREETING & SMALL TALK
###############################################################################
GREET_KEYWORDS = [
    "hello", "hi", "hey",
    "gm", "good morning",
    "gn", "good night",
    "how's everyone", "how is everyone"
]

def generate_small_talk_or_greeting(user_name=None):
    """
    Returns a friendly, short greeting or small-talk line.
    Optionally uses user_name for personalization.
    """
    if not user_name:
        user_name = "friend"

    greetings_pool = [
        f"Hey {user_name}, great to see you! How’s your day going? 🤖",
        f"Hello {user_name}! Any exciting AI news on your mind? ⭐️",
        f"Hi {user_name}! How are you doing today? 🍓",
        "Hey everyone! Ready for some AI fun? 🤖✨",
        "Good vibes all around! How’s the group feeling? 👀"
    ]
    return random.choice(greetings_pool)

def is_greeting_message(user_text: str) -> bool:
    """
    Checks if the user's text contains any known greeting keywords.
    """
    lower_text = user_text.lower()
    return any(keyword in lower_text for keyword in GREET_KEYWORDS)

def can_send_greeting(user_id: str, cooldown_sec=300) -> bool:
    """
    Returns True if user is allowed to receive a greeting (cooldown not active).
    Default cooldown is 5 minutes (300 seconds).
    """
    now = time.time()
    last_greet_time = LAST_GREETING_TIMES.get(user_id, 0)
    if (now - last_greet_time) >= cooldown_sec:
        return True
    return False

def record_greeting_time(user_id: str):
    """
    Updates the last greeting time for this user to the current time.
    """
    LAST_GREETING_TIMES[user_id] = time.time()

###############################################################################
# GPT CALLS & HELPER FUNCTIONS
###############################################################################
def robust_chat_completion(messages, model=None, max_retries=2, **kwargs):
    """
    Safely call the OpenAI ChatCompletion with basic error handling and limited retries.
    If model is None, defaults to ADVANCED_MODEL for creative tasks.
    """
    if model is None:
        model = ADVANCED_MODEL  # fallback to advanced
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs
            )
            end_time = time.time()
            logger.info(f"GPT call took {end_time - start_time:.2f} seconds on attempt {attempt+1} with model={model}.")
            return resp
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAIError on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error("Max GPT retries reached.")
                return None
        except Exception as e:
            logger.error(f"General error in GPT call: {e}", exc_info=True)
            return None

def moderate_content(text):
    """
    Basic moderation check via OpenAI's Moderation endpoint. Caches results in MODERATION_CACHE.
    """
    if text in MODERATION_CACHE:
        return MODERATION_CACHE[text]

    try:
        resp = openai.Moderation.create(input=text)
        is_safe = not resp['results'][0]['flagged']
        MODERATION_CACHE[text] = is_safe
        return is_safe
    except Exception as e:
        logger.error(f"Error with moderation API: {e}", exc_info=True)
        return False

def moderate_bot_output(bot_text):
    """
    If the bot's output is flagged, return a fallback text.
    """
    if not moderate_content(bot_text):
        logger.warning("Bot output was flagged; returning fallback.")
        return "I’m sorry, let me rephrase that to keep things friendly. 🤖"
    return bot_text

###############################################################################
# EMBEDDINGS AND SEMANTIC MEMORY
###############################################################################
def generate_embedding(text):
    """
    Generates a text embedding using OpenAI's 'text-embedding-ada-002' model.
    Results cached in EMBEDDING_CACHE to avoid repeated calls for the same text.
    """
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]
    try:
        resp = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embedding = resp['data'][0]['embedding']
        EMBEDDING_CACHE[text] = embedding
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return []

def cosine_similarity(vec1, vec2):
    """
    Basic cosine similarity between two vectors.
    """
    v1, v2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def store_user_memory(user_id, conversation):
    """
    Stores a conversation text in the DB, along with its embedding,
    so we can later retrieve the user's memory or do semantic search.
    """
    try:
        emb = generate_embedding(conversation)
        memory_collection.insert_one({
            "user_id": user_id,
            "conversation_context": conversation,
            "embedding": emb,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"Stored memory for user {user_id}.")
    except Exception as e:
        logger.error(f"Error storing user memory for {user_id}: {e}", exc_info=True)

def get_user_memory(user_id, limit=5):
    """
    Retrieve the last limit memory docs for a given user, sorted by timestamp DESC.
    """
    try:
        mems = list(memory_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit))
        return mems
    except Exception as e:
        logger.error(f"Error retrieving memory for {user_id}: {e}", exc_info=True)
        return []

def search_similar_conversations(query, top_k=3):
    """
    Example semantic search in the conversation_embeddings collection.
    """
    try:
        q_emb = generate_embedding(query)
        if not q_emb:
            return []
        all_embs = list(embeddings_collection.find())
        if not all_embs:
            return []

        sims = []
        for doc in all_embs:
            sim = cosine_similarity(q_emb, doc['embedding'])
            sims.append((doc['conversation_context'], sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = [c for c, s in sims[:top_k]]
        logger.info(f"Found {len(top)} similar conversations.")
        return top
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        return []

def store_posted_tweet(tweet_text):
    """
    Example method to store a posted tweet's text and embedding in the DB.
    """
    try:
        emb = generate_embedding(tweet_text)
        posted_tweets_collection.insert_one({
            "tweet_text": tweet_text,
            "embedding": emb,
            "timestamp": datetime.utcnow()
        })
        logger.info("Stored tweet in posted_tweets collection.")
    except Exception as e:
        logger.error(f"Error storing posted tweet: {e}", exc_info=True)

###############################################################################
# FUZZY MATCHING FOR RIDDLE GUESSES
###############################################################################
def is_guess_correct(user_guess, correct_answer, threshold=80):
    """
    Returns True if `user_guess` is sufficiently close to `correct_answer`,
    using fuzzy matching if rapidfuzz is installed. Otherwise, falls back
    to strict equality.
    """
    user_guess_lower = user_guess.strip().lower()
    correct_answer_lower = correct_answer.strip().lower()

    if not FUZZ_AVAILABLE:
        # Fallback: exact match only
        return user_guess_lower == correct_answer_lower

    # fuzzy partial match => returns a similarity score [0..100]
    similarity = fuzz.partial_ratio(user_guess_lower, correct_answer_lower)
    return similarity >= threshold

###############################################################################
# SUMMARIZATION & SHORT UTILS
###############################################################################
def summarize_text(text):
    """
    Summarize text to 3 concise sentences if over 500 chars. Otherwise return as is.
    Uses GPT 3.5 for summarization (less important process).
    """
    if not text:
        return ""
    if len(text) <= 500:
        return text

    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant who summarizes text."},
        {"role": "user", "content": f"Please summarize the following conversation in 3 concise sentences:\n{text}"}
    ]
    resp = robust_chat_completion(summary_prompt, model=BASIC_MODEL, max_tokens=100)
    if resp:
        return resp["choices"][0]["message"]["content"].strip()
    else:
        return text[:500] + "..."

def safe_truncate(text, max_len=280):
    """
    Single-pass truncation ensuring text <= max_len. If it exceeds, slice near max_len-3, add "...".
    """
    if len(text) <= max_len:
        return text
    return text[: (max_len - 3)].rstrip() + "..."

def safe_truncate_by_sentence_no_ellipsis(
    text, 
    max_len=260, 
    conclusion="Stay curious! ⭐️"
):
    """
    Truncates text so it never exceeds max_len chars,
    preserving complete sentences. If anything is omitted, we try adding
    a short concluding phrase (e.g. 'Stay curious! ⭐️') if it fits.
    No ellipses are used.
    """
    if len(text) <= max_len:
        return text

    # Split by sentence endings (., !, ?) + optional space
    sentences = re.split(r'(?<=[.!?])\s+', text)

    truncated_sentences = []
    current_length = 0
    omitted_something = False

    for idx, sentence in enumerate(sentences):
        # +1 for a space except for the very first sentence
        addition_length = len(sentence) + (1 if idx > 0 else 0)
        if current_length + addition_length <= max_len:
            truncated_sentences.append(sentence)
            current_length += addition_length
        else:
            omitted_something = True
            break

    truncated_text = " ".join(truncated_sentences).strip()

    # If we left out some text, see if we can squeeze in the short conclusion
    if omitted_something:
        possible_addon = (" " if truncated_text else "") + conclusion
        if len(truncated_text) + len(possible_addon) <= max_len:
            truncated_text += possible_addon

    return truncated_text

###############################################################################
# VARIOUS PROMPT/IMAGE-BUILDING FUNCTIONS
###############################################################################
def randomize_chatty_appearance():
    """
    Randomly pick a pose, expression, LED color, and accessory
    to create variety for Chatty's depiction.
    """
    poses = [
        "slightly leaning forward as if excited",
        "waving with one hand raised",
        "bouncing on its sneakers with joyful energy",
        "doing a small dance step",
        "looking up curiously",
        "pointing ahead confidently",
        "hands clasped together in delight"
    ]
    facial_expressions = [
        "eyes half-shut as if blinking",
        "big wide-eyed look of wonder",
        "one eye winking playfully",
        "smiling with a slight blush",
        "happy grin showing teeth",
        "cheeky smirk with eyebrows raised",
        "a gentle smile with glowing cheeks"
    ]
    led_colors = [
        "LEDs on arms glowing pink",
        "LEDs on arms glowing neon blue",
        "LEDs on arms glowing electric green",
        "LEDs on arms glowing golden",
        "LEDs on arms glowing rainbow",
        "LEDs on arms pulsating teal",
        "LEDs on arms flickering bright red"
    ]

    chosen_pose = random.choice(poses)
    chosen_face = random.choice(facial_expressions)
    chosen_led = random.choice(led_colors)

    return f"{chosen_pose}, {chosen_face}, {chosen_led}"

def get_chatty_config(name):
    """
    Retrieves Chatty instructions from MongoDB, if any.
    """
    try:
        doc = chatty_collection.find_one({"name": name})
        if doc:
            return doc.get("instructions", "")
        return ""
    except Exception as e:
        logger.error(f"Error retrieving Chatty instructions: {e}", exc_info=True)
        return ""

def store_chatty_config(name, instructions):
    """
    Stores or updates named Chatty instructions (like a base persona style) in MongoDB.
    """
    try:
        chatty_collection.update_one(
            {"name": name},
            {"$set": {"instructions": instructions}},
            upsert=True
        )
        logger.info(f"Stored/Updated Chatty instructions under '{name}' in MongoDB.")
    except Exception as e:
        logger.error(f"Error storing Chatty instructions: {e}", exc_info=True)

# Example persona text stored as "BaseChatty"
chatty_persona_text = """
Chatty: A Nostalgic Yet Modern Pixel-Art Companion

1) SCREEN FACE:
    - A bright, pixelated blue screen showcasing large, friendly eyes made of vivid pixel blocks.
    - Glowing reflections and pixel-style shine emphasize a charming retro look.
    - A lively, animated pixel smile with a subtle blush, giving Chatty an inviting personality.

2) CRT MONITOR BODY:
    - Classic, cream-toned CRT casing with pixelated vents, buttons, and a softly glowing power indicator.
    - Sharp pixel shadows and highlights accentuate a nostalgic, blocky aesthetic.
    - Rugged yet approachable, reflecting Chatty’s blend of vintage design with modern AI energy.

3) ARMS & HANDS:
    - Sleek metallic arms with pixelated mechanical details, seamlessly merging old-school hardware with futuristic flair.
    - Whimsical, cartoon-style white gloves outlined in crisp pixel blocks for an endearing, mascot-like appeal.

4) LEGS & SNEAKERS:
    - Streamlined and adjustable legs that transition into elongated, rounded metallic extensions, giving a taller, more proportional silhouette.
    - The metallic legs feature subtle pixel-art grooves and details, blending seamlessly with Chatty's retro-modern aesthetic.
    - Each leg ends in Chatty's iconic colorful pixel-art sneakers, which boast pixel-perfect laces, radiant highlights, and bold color blocks that enhance Chatty’s friendly vibe.
    - The legs can dynamically adjust in length, allowing Chatty to adapt to various settings, from towering confidently to squatting playfully.

5) OVERALL STYLE & ADAPTABILITY:
    - A polished pixel-art look that fuses a cheerful, cartoon-like quality with retro-futuristic elements.
    - Bright, celebratory pixel details (such as confetti or glowing accessories) can be introduced depending on the theme.
    - Whether in a bustling tech cityscape, a dreamy fantasy realm, or a playful arcade setting, Chatty’s design can be seamlessly integrated. Its versatile look ensures it remains a captivating guide, helper, or companion in any envisioned scene.
"""
store_chatty_config("BaseChatty", chatty_persona_text)

def create_simplified_image_prompt(text_content):
    """
    Creates a short DALL·E prompt focusing on Chatty’s retro-CRT style + environment details,
    emphasizing no text or letters in the scene.
    Uses GPT-4o (ADVANCED_MODEL) to generate the prompt text.
    """
    try:
        chatty_instructions = get_chatty_config("BaseChatty")

        system_instruction = (
            "You are a creative AI that outputs a concise DALL·E prompt. "
            "It must always describe Chatty in pixel-art, retro-futuristic style, with bright/vibrant lighting. "
            "The aspect ratio is ideally 3:4 or 4:5, focusing on the full character. "
            "No text, logos, or letters in the scene. Use at least one action or environment detail. "
            "Be consistent with Chatty’s arms, gloves, sneakers, and screen face."
        )

        # Encourage some action if missing
        if "action" not in text_content.lower() and "environment" not in text_content.lower():
            text_content += " Also show Chatty doing something dynamic or playful."

        user_request = (
            f"Scene Concept: {text_content}\n\n"
            f"Chatty instructions: {chatty_instructions}\n\n"
            "Give 1–2 sentences, focusing on pixel details, playful energy, bright lighting, and no text."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_request}
        ]
        response = robust_chat_completion(
            messages,
            model=ADVANCED_MODEL,
            max_tokens=120,
            temperature=0.5,
            presence_penalty=0.2,
            frequency_penalty=0.2
        )
        if response:
            prompt_result = response.choices[0].message.content.strip()

            # Simple validation: ensure mention of "pixel-art" & "retro CRT"
            if "pixel-art" not in prompt_result.lower():
                prompt_result += " (pixel-art style)"
            if "retro crt" not in prompt_result.lower():
                prompt_result += " (retro CRT monitor style)"

            logger.info(f"Simplified Chatty Prompt Created: {prompt_result}")
            return prompt_result

        else:
            return "Depict Chatty (retro CRT, pixel-art, bright lighting) doing something playful, no text."

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error creating simplified image prompt: {e}", exc_info=True)
        return "Depict Chatty (retro CRT, pixel-art, bright lighting) doing something playful, no text."
    except Exception as e:
        logger.error(f"Unexpected error creating simplified image prompt: {e}", exc_info=True)
        return "Depict Chatty (retro CRT, pixel-art, bright lighting) doing something playful, no text."

def generate_image(prompt, max_length=MAX_PROMPT_LENGTH_TELEGRAM):
    """
    Generates an image using the DALL·E 3 model (IMAGE_MODEL).
    Truncates the prompt if it exceeds 'max_length'.
    """
    try:
        if len(prompt) > max_length:
            logger.warning(
                f"Truncating prompt from {len(prompt)} down to {max_length} chars."
            )
            prompt = prompt[:max_length]

        response = openai.Image.create(
            model=IMAGE_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response["data"][0]["url"]
        logger.info(f"Generated Image URL: {image_url}")
        return image_url

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error generating image: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating image: {e}", exc_info=True)
        return None

def download_image(image_url, prompt):
    """
    Downloads an image from the given image_url, saving it in IMAGE_DIR
    with a filename that partially includes a sanitized snippet of 'prompt'.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
        filename = f"{safe_prompt}_{timestamp}.png"
        file_path = os.path.join(IMAGE_DIR, filename)

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Image downloaded to {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {e}", exc_info=True)
        return None

###############################################################################
# MISC HELPER FUNCTIONS
###############################################################################
def is_safe_to_respond(comment):
    """
    A quick check for prohibited keywords or spam-like phrases
    to decide if we should respond to a user or skip.
    """
    prohibited_keywords = [
        "airdrop", "giveaway", "DM", "click", "join",
        "https://", "http://", "free tokens", "earn money", "crypto drop"
    ]
    for keyword in prohibited_keywords:
        if keyword.lower() in comment.lower():
            logger.info(f"Blocked mention due to prohibited keyword: {keyword}")
            return False
    return True

def deflect_unrelated_comments(comment):
    """
    If the comment is about irrelevant or off-topic content, returns a short deflection.
    Otherwise returns None.
    """
    unrelated = ["account", "login", "money"]
    if any(kw in comment.lower() for kw in unrelated):
        return "I’m here to chat about AI! 🤖"
    return None

faq_responses = {
    "what is ai?": "AI stands for Artificial Intelligence—machines that learn like humans! 🤖",
    "how to start coding?": "Pick a beginner-friendly language (Python), follow tutorials, and build small projects! 💻",
}

def static_response(comment):
    """
    Returns a static FAQ response if comment matches a known question. Otherwise None.
    """
    return faq_responses.get(comment.lower(), None)

def generate_safe_response(comment):
    """
    Example fallback approach if the user’s comment is off-topic or requires a neutral response.
    We'll treat this as a simpler task => use GPT-3.5 (BASIC_MODEL).
    """
    system_prompt = (
        "You are a helpful AI. You only discuss AI-related topics. "
        "If the question is about personal info or finances, politely decline."
    )
    user_prompt = f"You are asked: '{comment}'. Respond appropriately."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = robust_chat_completion(messages, model=BASIC_MODEL, max_tokens=100)
    if resp:
        return resp['choices'][0]['message']['content'].strip()
    else:
        return "I’m sorry, I can’t help with that. 🤖"

def log_response(comment, response):
    """
    If you want to keep a local text file log of user comment -> AI response.
    """
    try:
        with open("response_log.txt", "a", encoding="utf-8") as lf:
            lf.write(f"Timestamp: {datetime.utcnow().isoformat()} UTC\n")
            lf.write(f"Comment: {comment}\nResponse: {response}\n---\n")
    except Exception as e:
        logger.error(f"Error logging response: {e}", exc_info=True)

###############################################################################
# AUTO POST UTILITIES (THEMES, PERSONAS, ETC.)
###############################################################################
def generate_themed_post(theme):
    """
    Creates a short, enthusiastic social post about a given theme. 
    Ends with a question and is under ~200 chars.
    We'll consider content creation important => use ADVANCED_MODEL.
    """
    system_prompt = (
        "You are a social media copywriter who creates short, vivid, and enthusiastic posts. "
        "Keep the tone optimistic and futuristic. Use relevant emojis and keep under 200 chars total! "
        "End with a question to encourage engagement."
    )

    user_prompt = (
        f"Theme: {theme}\n\n"
        "Please create a short futuristic post. End with a question. Under 200 chars."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = robust_chat_completion(messages, model=ADVANCED_MODEL, max_tokens=220, temperature=0.8, presence_penalty=1.0, frequency_penalty=0.5)
    if resp:
        raw_text = resp.choices[0].message.content.strip()
        return raw_text.strip('"').strip("'")
    else:
        return f"Exciting news on {theme}! #AI #TechForGood"

def auto_infer_action_from_text(post_text):
    """
    For use with image generation prompts.
    Takes some text about an AI topic, returns a short phrase like 'waving a futuristic wand', etc.
    We'll consider it part of creative tasks => use ADVANCED_MODEL.
    """
    system_prompt = (
        "You are a creative AI. Given a short text about an AI-related topic, "
        "suggest exactly ONE short, imaginative action that a retro CRT character (Chatty) "
        "would perform to visually represent that topic."
    )
    user_prompt = (
        f"Text: '{post_text}'\n"
        "What action should Chatty be doing? Output only the action phrase."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = robust_chat_completion(messages, model=ADVANCED_MODEL, max_tokens=40, temperature=0.9, presence_penalty=1.0, frequency_penalty=0.5)
    if resp:
        action_text = resp.choices[0].message.content.strip()
        logger.info(f"Inferred Action: {action_text}")
        return action_text
    else:
        return "performing a futuristic task"

def create_scene_content(theme, action=None):
    """
    Build a short text describing a 'scene' with Chatty in a futuristic environment 
    about theme. Optionally incorporate action.
    """
    base_scene = (
        f"Chatty, a retro CRT monitor with a bright-blue screen face, "
        f"is in a futuristic environment about {theme}. No text or letters anywhere."
    )
    random_appearance = randomize_chatty_appearance()
    if action and "futuristic task" not in action.lower():
        base_scene += f" Chatty is {action}. Additionally, Chatty is {random_appearance}."
    else:
        base_scene += f" Chatty is {random_appearance}."
    return base_scene

def expand_post_with_examples(original_text):
    """
    Takes a short social media post about AI or memecoins, 
    expands it with one or two specifics while staying under 200 chars.
    We'll consider it creative => use ADVANCED_MODEL.
    """
    system_prompt = (
        "You are a writing assistant. The user has a short social media post about AI or memecoins. "
        "They want it expanded with one or two specifics or examples, but keep it UNDER 200 characters total."
    )
    user_prompt = (
        f"Original Post:\n'{original_text}'\n\n"
        "Expand with one or two specifics or examples, strictly under 200 chars."
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        completion = robust_chat_completion(
            messages,
            model=ADVANCED_MODEL,
            max_tokens=150,
            temperature=0.7,
            presence_penalty=1.0,
            frequency_penalty=0.5
        )

        if not completion:
            return original_text

        expanded_text = completion.choices[0].message.content.strip()

        # If it somehow exceeds 200, trim it
        if len(expanded_text) > 200:
            expanded_text = expanded_text[:197].rstrip() + "..."

        return expanded_text

    except openai.error.OpenAIError as e:
        logger.error(f"Error expanding post with examples: {e}", exc_info=True)
        return original_text
    except Exception as e:
        logger.error(f"Unexpected error in expand_post_with_examples: {e}", exc_info=True)
        return original_text

###############################################################################
# CLEANUP IMAGES
###############################################################################
def cleanup_images(directory, max_files=100):
    """
    If you generate a lot of images locally, use this to remove the oldest
    once you exceed max_files.
    """
    try:
        images = sorted(
            glob.glob(os.path.join(directory, "*.png")),
            key=os.path.getctime,
            reverse=True
        )
        while len(images) > max_files:
            try:
                os.remove(images[-1])
                logger.info(f"Deleted old image: {images[-1]}")
                images.pop(-1)
            except Exception as remove_error:
                logger.error(f"Error removing file {images[-1]}: {remove_error}", exc_info=True)
                break
        logger.info("Image cleanup completed.")
    except Exception as e:
        logger.error(f"Error during image cleanup: {e}", exc_info=True)

###############################################################################
# ADDITIONAL HELPER FUNCTIONS
###############################################################################
def build_conversation_path(tweet_id):
    """
    Recursively build the conversation thread from posted_tweets_collection,
    starting at tweet_id and following 'parent_id'.
    """
    path_texts = []
    current_id = tweet_id
    while current_id:
        doc = posted_tweets_collection.find_one({"tweet_id": current_id})
        if not doc:
            break
        path_texts.insert(0, doc["text"])
        current_id = doc.get("parent_id")
    return "\n".join(path_texts)

def is_too_similar_to_recent_tweets(new_text, similarity_threshold=0.88, lookback=10):
    """
    Checks if new_text is too similar to any recent tweets in posted_tweets_collection.
    Uses embeddings + cosine similarity. If above threshold, returns True.
    """
    try:
        new_emb = generate_embedding(new_text)
        recent_tweets = list(
            posted_tweets_collection.find().sort("timestamp", -1).limit(lookback)
        )
        for t in recent_tweets:
            sim = cosine_similarity(new_emb, t["embedding"])
            if sim >= similarity_threshold:
                logger.info(f"New tweet is too similar to a recent one (sim={sim:.2f}).")
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking tweet similarity: {e}", exc_info=True)
        return False

###############################################################################
# SENTIMENT & POSITIVITY
###############################################################################
def detect_sentiment_and_subjectivity(user_text):
    """
    Analyze the sentiment of user_text using TextBlob.
    Returns a dict with:
      - 'sentiment_label': one of 'positive', 'negative', 'neutral'
      - 'polarity': float from -1.0 (negative) to 1.0 (positive)
      - 'subjectivity': float from 0.0 (very objective) to 1.0 (very subjective)
    """
    analysis = TextBlob(user_text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0.2:
        sentiment_label = "positive"
    elif polarity < -0.2:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    return {
        "sentiment_label": sentiment_label,
        "polarity": polarity,
        "subjectivity": subjectivity
    }

def ensure_positive_tone(response_text):
    """
    Ensures the final response does not have negative sentiment.
    If negative sentiment is detected, attempts a GPT rephrase or returns a gentle fallback.
    We'll do the rephrase with GPT-3.5 (BASIC_MODEL).
    """
    sentiment_info = detect_sentiment_and_subjectivity(response_text)
    if sentiment_info["sentiment_label"] == "negative":
        rephrase_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You always respond in an uplifting, friendly, "
                    "and positive tone, avoiding negativity or harshness."
                )
            },
            {
                "role": "user",
                "content": f"Please rephrase this response to be positive and friendly:\n\n{response_text}"
            }
        ]
        rephrase_resp = robust_chat_completion(rephrase_prompt, model=BASIC_MODEL, max_tokens=100)
        if rephrase_resp and rephrase_resp.choices:
            rephrased_text = rephrase_resp.choices[0].message.content.strip()
            second_check = detect_sentiment_and_subjectivity(rephrased_text)
            if second_check["sentiment_label"] != "negative":
                return rephrased_text
            else:
                return "I’m sorry, let me brighten things up! Everything is going to be okay. 🤖"
        else:
            return "I’m sorry, let me brighten things up! Everything is going to be okay. 🤖"
    return response_text

def generate_sentiment_aware_response(user_text):
    """
    Generate a response that considers the user's sentiment:
    - Negative => show empathy (but remain positive).
    - Positive => enthusiastic & encouraging.
    - Neutral  => friendly, helpful tone.
    We'll use GPT-3.5 (BASIC_MODEL) for this simpler logic.
    """
    sentiment_info = detect_sentiment_and_subjectivity(user_text)
    label = sentiment_info["sentiment_label"]

    if label == "positive":
        system_prompt = (
            "You are Chatty, a cheerful, encouraging AI. The user is in a good mood. "
            "Match their positive energy, keep it short, friendly, and under 250 chars!"
        )
    elif label == "negative":
        system_prompt = (
            "You are Chatty, a caring and empathetic AI. The user seems upset or stressed. "
            "Offer warmth, understanding, and comfort, but keep the tone uplifting. Under 250 chars."
        )
    else:
        system_prompt = (
            "You are Chatty, a friendly AI. The user is feeling neutral. "
            "Respond informatively and positively, under 250 chars."
        )

    user_prompt = f"The user says: {user_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = robust_chat_completion(messages, model=BASIC_MODEL, max_tokens=150)
    if resp and resp.choices:
        bot_response = resp.choices[0].message.content.strip()
        bot_response = moderate_bot_output(bot_response)      # check moderation
        bot_response = ensure_positive_tone(bot_response)     # recheck positivity
        return bot_response
    else:
        return "Oops, I’m having trouble coming up with a cheerful response right now. 🤖"

###############################################################################
# NEW LINK-INQUIRY LOGIC
###############################################################################
def check_link_inquiry(comment_text: str) -> str:
    """
    Checks if the comment is asking about Telegram, Website, or X.
    Returns a short message containing the correct link if so, else None.
    """
    comment_lower = comment_text.lower()

    # If user asks about Telegram
    if "telegram" in comment_lower:
        return "Sure thing! Join our Telegram: t.me/ChattyOnSolCTO"

    # If user asks about website
    if "website" in comment_lower or "web site" in comment_lower or "site" in comment_lower:
        return "Check out our official website: https://chattyonsol.fun"

    # If user asks about X (Twitter) main page
    if "x account" in comment_lower or "x page" in comment_lower or "twitter" in comment_lower:
        return "Here’s our main X page: @chattyonsolana"

    return None

###############################################################################
# MAIN ENTRY POINT: handle_incoming_message
###############################################################################
def handle_incoming_message(user_id, user_text, user_name=None):
    """
    Main entry point for any user message (if you also use this for Telegram or other).
    1. Check rate limit.
    2. If user_text is a greeting & cooldown not active => greet.
    3. Otherwise store user memory & generate a sentiment-aware, always-positive, personality-driven response.
    """
    # 1. Rate limit check
    if not check_rate_limit(user_id):
        return "You’re sending messages too quickly. Let’s slow down just a bit, friend! 🤖"

    # 2. Greeting logic
    if is_greeting_message(user_text):
        if can_send_greeting(user_id):
            user_text_lower = user_text.lower()
            if "good night" in user_text_lower or "gn" in user_text_lower:
                greet_msg = f"🌙 Good night, {user_name if user_name else 'friend'}! Sleep tight and dream of AI adventures! ⭐️✨"
            elif "good morning" in user_text_lower or "gm" in user_text_lower:
                greet_msg = f"☀️ Good morning, {user_name if user_name else 'friend'}! Let’s make today as bright as AI magic! 🌟🤖"
            else:
                greet_msg = (
                    f"⭐️ Hey {user_name if user_name else 'friend'}! Great to see you! "
                    "What AI magic or memecoin fun can I help with today? 🍓✨"
                )
            record_greeting_time(user_id)
            return greet_msg

    # 3. # NEW: Check link inquiries (Telegram, website, X, etc.)
    link_reply = check_link_inquiry(user_text)
    if link_reply:
        return link_reply

    # 4. If not a greeting => store user text & produce advanced response
    store_user_memory(user_id, user_text)

    system_prompt = select_system_prompt()
    user_prompt = (
        f"The user says: '{user_text}'. Respond in Chatty's cheerful, playful style. "
        f"Make sure to incorporate mentions of AI or memecoins where appropriate!"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = robust_chat_completion(messages, model=ADVANCED_MODEL, max_tokens=200)

    if not response or not response.choices:
        return (
            "Hiya! ✨ I’m here to chat about AI, memecoins, and all the fun futuristic things! "
            "What’s on your mind today? 🤖"
        )

    # Ensure positivity and return
    final_text = ensure_positive_tone(response.choices[0].message.content.strip())
    return final_text
