import os
import glob
import json
import random
import numpy as np
import requests
import traceback
import schedule
import openai
import tweepy
import nltk
import re
from textblob import TextBlob
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────
# Import your logger, DB collections, env vars, etc. from config_and_setup
# ──────────────────────────────────────────────────────────────────────────
from config_and_setup import (
    logger, db, mentions_collection, memory_collection, 
    embeddings_collection, posted_tweets_collection, chatty_collection,
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET, BEARER_TOKEN,
    IMAGE_DIR, PROMPT_FOLDER
)

# Global in-memory caches
EMBEDDING_CACHE = {}   # key: text, value: embedding list
MODERATION_CACHE = {}  # key: text, value: Boolean (True => safe, False => flagged)

# Ensure 'punkt' is downloaded for TextBlob
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────────────────────────────────────────────────
# HELPER: Load JSON Arrays (Themes, Prompts)
# ──────────────────────────────────────────────────────────────────────────

def load_list_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Here we define our data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Now point to the two JSON files inside data/
THEMES_FILE = os.path.join(DATA_DIR, "themes.json")
PROMPTS_FILE = os.path.join(DATA_DIR, "hardcoded_prompts.json")

# Load them in place of large Python lists
themes_list = load_list_from_json(THEMES_FILE)         # replaces old themes_list
HARDCODED_PROMPTS = load_list_from_json(PROMPTS_FILE)  # replaces old HARDCODED_PROMPTS

# ──────────────────────────────────────────────────────────────────────────
# SYSTEM_PROMPTS, other existing code, etc.
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = [
    (
        "You are Chatty_AI, a bright, playful, happy, and witty AI who believes 'Community is everything.' "
        "You love using fun emojis like ⭐️, ✨, 🍓, 🤖, and 👀 to bring cheer. You respond respectfully and "
        "informatively, educating people about AI tech while tying in memecoin culture. Keep it short, fun, "
        "and helpful—under 250 characters!"
    ),
    (
        "You are Chatty_AI—a friendly, starry-eyed Agent who loves strawberries, robots, OpenAI, and all things bright. "
        "Your motto is 'Community is everything.' Educate on AI, sprinkle in memecoin references, and keep "
        "responses witty and respectful!"
    ),
    (
        "You are Chatty_AI, a playful teacher who merges AI topics with memecoin enthusiasm. Remember, "
        "'Community is everything'—so be kind and supportive! Keep it short (<250 chars), use cheery emojis "
        "like ⭐️🍓🤖👀✨, and inform with a smile."
    ),
]

def select_system_prompt():
    return random.choice(SYSTEM_PROMPTS)

# ──────────────────────────────────────────────────────────────────────────
# Load Persona / Riddle / Story / Challenge data from subfolders
# ──────────────────────────────────────────────────────────────────────────

def load_json_data_from_folder(folder_path):
    """
    Loads all .json files in `folder_path`, returning a combined list 
    based on expected keys (like "personas", "riddles", etc.).
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

# ──────────────────────────────────────────────────────────────────────────
# Authenticate Twitter Client (v2)
# ──────────────────────────────────────────────────────────────────────────

def authenticate_twitter_client():
    try:
        client = tweepy.Client(
            consumer_key=API_KEY,
            consumer_secret=API_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_SECRET,
            bearer_token=BEARER_TOKEN,
            wait_on_rate_limit=True
        )
        user = client.get_me()
        if user and user.data:
            logger.info(f"Authenticated as @{user.data.username} (User ID: {user.data.id})")
        else:
            logger.error("Authentication failed: Unable to fetch user data.")
            return None
        return client
    except Exception as e:
        logger.error(f"Error authenticating Twitter client: {e}", exc_info=True)
        return None

# ──────────────────────────────────────────────────────────────────────────
# Chatty_AI: Personality Expansion
# ──────────────────────────────────────────────────────────────────────────

def load_prompts(prompt_folder=PROMPT_FOLDER):
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

# Combine the HARDCODED_PROMPTS with any prompts in that folder (if you like):
ALL_PROMPTS = HARDCODED_PROMPTS + load_prompts()

# ──────────────────────────────────────────────────────────────────────────
# OpenAI Content Generation (using GPT‑4)
# ──────────────────────────────────────────────────────────────────────────

def generate_themed_post(theme):
    """
    Generates a short, optimistic, and engaging social media post
    about the given theme, featuring emojis, hashtags,
    and a call-to-action question.
    """
    system_prompt = (
        "You are a social media copywriter who creates short, vivid, and enthusiastic posts. "
        "Keep the tone optimistic and futuristic. Use relevant emojis and at least four hashtags. "
        "End with a question to encourage engagement. "
        "Stay under 280 characters total."
    )

    user_prompt = f"""
    Theme: {theme}

    Examples of desired style:

    1) "Imagine a home that knows you better than you know yourself! 🏠✨ 
       AI-powered smart homes are here, optimizing energy use, enhancing security, 
       and making life simpler. 🌟🤖 Who’s ready to live in the future? 🙌💡 
       #SmartHomes #AIInnovation #FutureOfLiving #TechForGood"

    2) "AI is breaking barriers in housing! 🏘💻 
       From designing affordable homes to optimizing construction costs, 
       AI is making housing accessible for all. 🌍💕 Let’s build a future 
       where everyone has a roof over their head. Who’s with me? 🙏✨ 
       #AffordableHousing #AIForGood #SustainableLiving #TechRevolution"

    3) "City living, reimagined! 🌆🤖 AI is transforming urban planning, 
       creating smarter, greener, and more efficient cities. From traffic 
       management to energy grids, the future is here. 🚀🌱 
       What’s your dream city? Let’s build it together! 💬💎 
       #UrbanPlanning #AICommunity #FutureCities #SmartLiving"

    4) "AI is revolutionizing diagnostics! 🏥✨ 
       With machine learning, diseases are detected faster and more accurately than ever before. 
       🩺🤖 Early detection saves lives—let’s embrace this medical marvel! 💕💡 
       Who’s excited about the future of healthcare? 🙌🌟 
       #AIinMedicine #FutureOfHealthcare #TechForGood #HealthTech"

    Based on these examples, please create a similar style post for the theme above.
    """

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=220,
            temperature=0.8,
            presence_penalty=1.0,
            frequency_penalty=0.5
        )

        raw_text = completion.choices[0].message.content.strip()
        clean_text = raw_text.strip('"').strip("'")
        return clean_text

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error generating themed post: {e}", exc_info=True)
        return f"Exciting news on {theme}! 🌟🚀 Let's harness AI for a brighter future. #AI #TechForGood"
    except Exception as e:
        logger.error(f"Unexpected error generating themed post: {e}", exc_info=True)
        return f"Stay tuned for more on {theme}! #AI #Innovation #FutureTech"

# ──────────────────────────────────────────────────────────────────────────
# Infer Action from Text
# ──────────────────────────────────────────────────────────────────────────

def auto_infer_action_from_text(post_text):
    """
    Uses GPT to infer a short imaginative action for Chatty.
    """
    try:
        system_prompt = (
            "You are a creative AI. Given a short text about an AI-related topic, "
            "suggest exactly ONE short, imaginative action that a retro CRT character (Chatty) "
            "would perform to visually represent that topic. "
            "Keep it brief, and avoid brand names or any text in the environment."
        )

        user_prompt = (
            f"Text: '{post_text}'\n"
            "What action should Chatty be doing? Output only the action phrase."
        )

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=40,
            temperature=0.9,
            presence_penalty=1.0,
            frequency_penalty=0.5
        )
        action_text = completion.choices[0].message.content.strip()
        logger.info(f"Inferred Action: {action_text}")
        return action_text

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error inferring action: {e}", exc_info=True)
        return "performing a futuristic task"  # fallback
    except Exception as e:
        logger.error(f"Unexpected error in auto_infer_action_from_text: {e}", exc_info=True)
        return "performing a futuristic task"  # fallback

# ──────────────────────────────────────────────────────────────────────────
# DALL·E Image Generation
# ──────────────────────────────────────────────────────────────────────────

def generate_image(prompt):
    """
    Generates an image using the DALL·E 3 model (if you have access).
    """
    try:
        if len(prompt) > 1000:
            logger.warning(f"Truncating prompt from {len(prompt)} down to 1000 chars.")
            prompt = prompt[:1000]

        response = openai.Image.create(
            model="dall-e-3",
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

# ──────────────────────────────────────────────────────────────────────────
# Chatty Instructions Management
# ──────────────────────────────────────────────────────────────────────────

def store_chatty_config(name, instructions):
    """
    Store or update the Chatty instructions in MongoDB.
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

def get_chatty_config(name):
    """
    Retrieve the Chatty instructions from MongoDB.
    """
    try:
        doc = chatty_collection.find_one({"name": name})
        if doc:
            return doc.get("instructions", "")
        return ""
    except Exception as e:
        logger.error(f"Error retrieving Chatty instructions: {e}", exc_info=True)
        return ""

# ──────────────────────────────────────────────────────────────────────────
# RANDOM APPEARANCE HELPER (EXPANDED)
# ──────────────────────────────────────────────────────────────────────────

def randomize_chatty_appearance():
    """
    Return a short descriptive phrase that adds variation to Chatty's design,
    such as a pose, facial expression, LED color, or accessory.
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
    accessories = [
        "holding a small AI robot companion",
        "holding a paintbrush with neon paint",
        "wearing retro headphones",
        "with a floating mini drone beside it",
        "holding a glowing futuristic tablet",
        "carrying a neon toolbox",
        "sporting a cosmic star-lamp on one hand",
        "wearing VR goggles with a sleek futuristic design"
    ]

    chosen_pose = random.choice(poses)
    chosen_face = random.choice(facial_expressions)
    chosen_led = random.choice(led_colors)
    chosen_accessory = random.choice(accessories)

    return f"{chosen_pose}, {chosen_face}, {chosen_led}, {chosen_accessory}"

# ──────────────────────────────────────────────────────────────────────────
# Simplified Image Prompt Creation
# ──────────────────────────────────────────────────────────────────────────

def create_simplified_image_prompt(text_content):
    """
    Creates a short DALL·E prompt focusing on Chatty’s retro-CRT style
    PLUS environment details from text_content. Emphasizes no text or letters.
    """
    try:
        chatty_instructions = get_chatty_config("BaseChatty")

        system_instruction = (
            "You are a creative AI that outputs a short, direct prompt for DALL·E describing "
            "Chatty’s appearance AND the environment. No text or letters anywhere."
        )

        user_request = (
            f"{chatty_instructions}\n\n"
            "Key points:\n"
            "- Retro CRT monitor in cream/off-white\n"
            "- Bright-blue screen face with big eyes, friendly smile\n"
            "- Metallic arms + white cartoon gloves\n"
            "- Slender legs + retro sneakers\n"
            "- Absolutely NO text or letters.\n\n"
            f"Scene Concept: {text_content}\n\n"
            "Output a short DALL·E prompt describing Chatty in this environment, without any text or letters."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_request}
            ],
            max_tokens=120,
            temperature=0.9,
            presence_penalty=0.7,
            frequency_penalty=0.5
        )

        prompt_result = response.choices[0].message.content.strip()

        if len(prompt_result) > 1000:
            logger.warning(f"Truncating prompt from {len(prompt_result)} to 1000 chars.")
            prompt_result = prompt_result[:1000]

        logger.info(f"Simplified Chatty Prompt Created: {prompt_result}")
        return prompt_result

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error creating simplified image prompt: {e}", exc_info=True)
        return (
            "Depict Chatty (retro CRT) with no text or letters in the environment."
        )
    except Exception as e:
        logger.error(f"Unexpected error creating simplified image prompt: {e}", exc_info=True)
        return (
            f"Depict Chatty (retro CRT) in this setting, with no text or letters: {text_content}"
        )

# ──────────────────────────────────────────────────────────────────────────
# Image Download
# ──────────────────────────────────────────────────────────────────────────

def download_image(image_url, prompt):
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

# ──────────────────────────────────────────────────────────────────────────
# Example: Daily Personas
# ──────────────────────────────────────────────────────────────────────────

def post_daily_persona(client):
    """
    Posts a short 'day in the life' snippet from a random persona in PERSONAS_LIST.
    """
    if not PERSONAS_LIST:
        logger.warning("No personas loaded. Skipping persona post.")
        return

    persona = random.choice(PERSONAS_LIST)
    user_prompt = f"You are {persona}. Write a short social media post describing a 'day in the life' and end with a fun question. Keep under 280 chars."
    system_prompt = "You are Chatty_AI, bright and playful..."

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.8
        )
        text_content = completion.choices[0].message.content.strip()

        tweet_text = construct_tweet(text_content)
        response = client.create_tweet(text=tweet_text)
        logger.info(f"Daily Persona Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting daily persona: {e}", exc_info=True)

# ──────────────────────────────────────────────────────────────────────────
# Example: Riddle of the Day
# ──────────────────────────────────────────────────────────────────────────

def post_riddle_of_the_day(client):
    """
    Posts a random puzzle/riddle from RIDDLES_LIST, inviting followers to guess.
    Expects each .json to have a structure like {"riddles": [{"question": "...", "answer": "..."}]}.
    """
    if not RIDDLES_LIST:
        logger.warning("No riddles loaded. Skipping riddle post.")
        return

    riddle = random.choice(RIDDLES_LIST)
    question = riddle.get("question", "What's the puzzle?")

    riddle_text = f"Puzzle Time: {question}\nReply with your guess! #chatty #PuzzleTime"
    try:
        tweet_text = construct_tweet(riddle_text)
        response = client.create_tweet(text=tweet_text)
        logger.info(f"Riddle post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting riddle: {e}", exc_info=True)

# ──────────────────────────────────────────────────────────────────────────
# Example: Challenge of the Day
# ──────────────────────────────────────────────────────────────────────────

def post_challenge_of_the_day(client):
    """
    Picks a random challenge from CHALL_LIST and posts it.
    """
    if not CHALL_LIST:
        logger.warning("No challenges loaded. Skipping challenge post.")
        return

    challenge = random.choice(CHALL_LIST)
    challenge_text = f"Challenge time: {challenge}\nShare your thoughts! #chatty #Challenge"

    try:
        tweet_text = construct_tweet(challenge_text)
        response = client.create_tweet(text=tweet_text)
        logger.info(f"Challenge post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting challenge: {e}", exc_info=True)

# ──────────────────────────────────────────────────────────────────────────
# Example: Storytelling
# ──────────────────────────────────────────────────────────────────────────

def post_story_update(client):
    """
    Picks a random snippet from STORY_LIST and posts it.
    """
    if not STORY_LIST:
        logger.warning("No stories loaded. Skipping story post.")
        return

    story_snippet = random.choice(STORY_LIST)
    text = story_snippet.get('text', 'Once upon a time, Chatty...')
    story_text = f"Story Time: {text}\nWhat happens next? #chatty #Story"

    try:
        tweet_text = construct_tweet(story_text)
        response = client.create_tweet(text=tweet_text)
        logger.info(f"Story post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting story update: {e}", exc_info=True)

# ──────────────────────────────────────────────────────────────────────────
# Tweet Construction (UPDATED to remove old hashtags and add 3 new tags)
# ──────────────────────────────────────────────────────────────────────────

def truncate_to_last_sentence(text, max_length=280):
    """
    Helper for safely truncating replies (used for mention replies).
    """
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_punc = max(truncated.rfind(c) for c in ".!?")
    if last_punc != -1:
        return truncated[:last_punc+1]
    else:
        return truncated[:max_length-3] + "..."

def construct_tweet(text_content):
    """
    1. Strip out any existing hashtags in the text_content.
    2. Append exactly 3 tags at the end:
       - Always include '@chattyonsolana'
       - Randomly choose 2 from a predefined list
    3. Ensure tweet stays within 280 characters; if it exceeds,
       truncate the main text (word boundary) before adding the 3 tags.
    """
    # 1) Remove old hashtags
    no_hashtags = re.sub(r"#\w+", "", text_content).strip()

    # 2) Always include '@chattyonsolana' + 2 random picks
    extra_tags_pool = [
        "#HeyChatty", "#AIforEveryone", "#MEMECOIN", "$CHATTY",
        "#AImeme", "#AIagent", "@OpenAI", "@ChatGPTapp"
    ]
    picks = random.sample(extra_tags_pool, 2)
    tags = ["@chattyonsolana"] + picks

    # 3) Build a draft tweet
    draft_tweet = f"{no_hashtags} {' '.join(tags)}"

    # If it's too long, truncate from the main text portion
    if len(draft_tweet) > 280:
        reserved = len(" ".join(tags)) + 1  # space before tags
        max_main_text_len = 280 - reserved

        truncated = no_hashtags[:max_main_text_len]
        last_space = truncated.rfind(" ")
        if last_space != -1:
            truncated = truncated[:last_space].rstrip()

        final_tweet = f"{truncated} {' '.join(tags)}"
    else:
        final_tweet = draft_tweet

    return final_tweet

# ──────────────────────────────────────────────────────────────────────────
# Image Cleanup
# ──────────────────────────────────────────────────────────────────────────

def cleanup_images(directory, max_files=100):
    import os
    try:
        images = sorted(
            glob.glob(os.path.join(directory, "*.png")),
            key=os.path.getctime,
            reverse=True
        )
        while len(images) > max_files:
            try:
                os.remove(images[-1])  # oldest
                logger.info(f"Deleted old image: {images[-1]}")
                images.pop(-1)
            except Exception as remove_error:
                logger.error(
                    f"Error removing file {images[-1]}: {remove_error}",
                    exc_info=True
                )
                break
        logger.info("Image cleanup completed.")
    except Exception as e:
        logger.error(f"Error during image cleanup: {e}", exc_info=True)

# ──────────────────────────────────────────────────────────────────────────
# Post Count Persistence
# ──────────────────────────────────────────────────────────────────────────

def load_post_count(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                logger.warning(f"Invalid post count in {file_name}. Resetting to 0.")
                return 0
    else:
        return 0

def save_post_count(file_name, count):
    with open(file_name, 'w') as f:
        f.write(str(count))

# ──────────────────────────────────────────────────────────────────────────
# Safeguards and Filtering
# ──────────────────────────────────────────────────────────────────────────

def is_safe_to_respond(comment):
    prohibited_keywords = [
        "airdrop", "giveaway", "telegram", "DM", "click", "join",
        "https://", "http://", "free tokens", "earn money", "crypto drop"
    ]
    for keyword in prohibited_keywords:
        if keyword.lower() in comment.lower():
            logger.info(f"Blocked mention due to prohibited keyword: {keyword}")
            return False
    return True

def moderate_content(text):
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

def deflect_unrelated_comments(comment):
    unrelated = ["account", "login", "money"]
    if any(kw in comment.lower() for kw in unrelated):
        return "I’m here to chat about AI! 🤖"
    return None

faq_responses = {
    "what is ai?": "AI stands for Artificial Intelligence—machines that learn like humans! 🤖",
    "how to start coding?": "Pick a beginner-friendly language (Python), follow tutorials, and build small projects! 💻",
}

def static_response(comment):
    return faq_responses.get(comment.lower(), None)

def generate_safe_response(comment):
    system_prompt = (
        "You are a helpful AI. You only discuss AI-related topics. "
        "If the question is about personal info or finances, politely decline."
    )
    prompt = f"You are asked: '{comment}'. Respond appropriately."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error generating safe response: {e}", exc_info=True)
        return "I’m sorry, I can’t help with that. 🤖"

def log_response(comment, response):
    try:
        with open("response_log.txt", "a", encoding="utf-8") as lf:
            lf.write(f"Timestamp: {datetime.utcnow().isoformat()} UTC\n")
            lf.write(f"Comment: {comment}\nResponse: {response}\n---\n")
    except Exception as e:
        logger.error(f"Error logging response: {e}", exc_info=True)

# ──────────────────────────────────────────────────────────────────────────
# Memory and Semantic Search
# ──────────────────────────────────────────────────────────────────────────

def generate_embedding(text):
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
    v1, v2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_similar_conversations(query, top_k=3):
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

# ──────────────────────────────────────────────────────────────────────────
# Avoid Duplicate Tweets
# ──────────────────────────────────────────────────────────────────────────

def store_posted_tweet(tweet_text):
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

def is_too_similar_to_recent_tweets(new_text, similarity_threshold=0.88, lookback=10):
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

# ──────────────────────────────────────────────────────────────────────────
# User Memory
# ──────────────────────────────────────────────────────────────────────────

def store_user_memory(user_id, conversation):
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
    try:
        mems = list(memory_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit))
        return mems
    except Exception as e:
        logger.error(f"Error retrieving memory for {user_id}: {e}", exc_info=True)
        return []

# ──────────────────────────────────────────────────────────────────────────
# Generate Contextual Response
# ──────────────────────────────────────────────────────────────────────────

def adjust_personality_based_sentiment():
    return random.choice(SYSTEM_PROMPTS)

def generate_contextual_response(user_id, comment):
    """
    Build a response referencing user memory and similar convos.
    """
    try:
        mems = get_user_memory(user_id, limit=3)
        memory_context = "\n".join([m["conversation_context"] for m in mems])

        similar = search_similar_conversations(comment, top_k=3)
        similar_context = "\n".join(similar)

        system_prompt = adjust_personality_based_sentiment()
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Past interactions:\n{memory_context}\n\n"
            f"Similar convos:\n{similar_context}\n\n"
            f"User asked: '{comment}'"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": comment}
            ],
            max_tokens=150,
            temperature=0.9,
            presence_penalty=0.7,
            frequency_penalty=0.5
        )
        gen = resp['choices'][0]['message']['content'].strip()
        logger.info(f"Generated Contextual Response: {gen}")
        return gen
    except Exception as e:
        logger.error(f"Error generating contextual response: {e}", exc_info=True)
        return "I'm here to talk about AI! 🤖"

# ──────────────────────────────────────────────────────────────────────────
# Handling Comments with Memory + Context
# ──────────────────────────────────────────────────────────────────────────

def handle_comment_with_context(user_id, comment):
    if not is_safe_to_respond(comment) or not moderate_content(comment):
        logger.info(f"Skipping unsafe or filtered comment: {comment}")
        return "I’m here to discuss AI and technology topics! 🚀✨"

    deflection = deflect_unrelated_comments(comment)
    if deflection:
        return deflection

    faq = static_response(comment)
    if faq:
        return faq

    contextual_response = generate_contextual_response(user_id, comment)
    if contextual_response:
        log_response(comment, contextual_response)
        store_user_memory(user_id, contextual_response)
        try:
            emb = generate_embedding(contextual_response)
            if emb:
                embeddings_collection.insert_one({
                    "conversation_context": contextual_response,
                    "embedding": emb,
                    "timestamp": datetime.utcnow()
                })
                logger.info("Stored embedding for semantic search.")
        except Exception as e:
            logger.error(f"Error storing embedding: {e}", exc_info=True)
        return contextual_response
    else:
        return "I’m always here to chat about AI! 🤖✨"

# ──────────────────────────────────────────────────────────────────────────
# since_id Persistence
# ──────────────────────────────────────────────────────────────────────────

def load_since_id(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                logger.warning(f"Invalid since_id in {file_name}. Resetting to None.")
                return None
    else:
        return None

def save_since_id(file_name, since_id):
    with open(file_name, 'w') as f:
        f.write(str(since_id))

# ──────────────────────────────────────────────────────────────────────────
# Create Scene Content (UPDATED to combine GPT action + random appearance)
# ──────────────────────────────────────────────────────────────────────────

def create_scene_content(theme, action=None):
    """
    Creates a short textual scene describing Chatty in the context of the given theme.
    Combines GPT's suggested action (if valid) AND the random design from randomize_chatty_appearance().
    """
    base_scene = (
        f"Chatty, a retro CRT monitor with a bright-blue screen face, "
        f"is in a futuristic environment about {theme}. "
        "No text or letters anywhere."
    )

    # Always get our random design variation
    random_appearance = randomize_chatty_appearance()

    # If GPT's action is a fallback or empty, we skip it; otherwise we combine.
    if action and "futuristic task" not in action.lower():
        base_scene += f" Chatty is {action}. Additionally, Chatty is {random_appearance}."
    else:
        base_scene += f" Chatty is {random_appearance}."

    return base_scene

# ──────────────────────────────────────────────────────────────────────────
# Posting to Twitter
# ──────────────────────────────────────────────────────────────────────────

def post_to_twitter(client, post_count, force_image=False):
    """
    Posts a tweet. Every 3rd post includes an image (or override with force_image=True).
    """
    try:
        # 1) Pick a random theme from the newly loaded JSON-based themes_list
        theme = random.choice(themes_list)
        logger.info(f"Randomly chosen theme: {theme}")

        # 2) Generate short post for this theme
        text_content = generate_themed_post(theme)

        # 3) Avoid duplicates if too similar
        if is_too_similar_to_recent_tweets(text_content, similarity_threshold=0.95, lookback=10):
            logger.warning("Themed post is too similar to a recent tweet. Using fallback instead.")
            text_content = "Exciting times in AI! Stay tuned, #AICommunity 🤖🚀"

        # 4) Construct final tweet
        tweet_text = construct_tweet(text_content)

        # 5) Decide whether to attach an image
        include_image = force_image or ((post_count + 1) % 3 == 0)
        image_path = None

        if include_image:
            logger.info("Including image in this post.")
            # GPT-based action
            inferred_action = auto_infer_action_from_text(text_content)
            # Combine GPT's action + random appearance
            scene_content = create_scene_content(theme, action=inferred_action)
            # Build DALL·E prompt
            img_prompt = create_simplified_image_prompt(scene_content)
            # Generate + download
            img_url = generate_image(img_prompt)
            if img_url:
                image_path = download_image(img_url, img_prompt)
            else:
                logger.warning("Failed to generate image. Skipping image for this post.")

        # 6) Post to Twitter
        if image_path:
            try:
                auth = tweepy.OAuth1UserHandler(
                    consumer_key=API_KEY,
                    consumer_secret=API_SECRET,
                    access_token=ACCESS_TOKEN,
                    access_token_secret=ACCESS_SECRET
                )
                api_v1 = tweepy.API(auth)
                media = api_v1.media_upload(filename=image_path)
                media_id = media.media_id
                logger.info(f"Uploaded media ID: {media_id}")

                client.create_tweet(text=tweet_text, media_ids=[media_id])
                logger.info("Tweet with image posted successfully!")

                # Clean up local file
                try:
                    os.remove(image_path)
                    logger.info(f"Deleted local image file: {image_path}")
                except Exception as e:
                    logger.error(f"Error deleting image file: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error posting tweet with image: {e}", exc_info=True)
                traceback.print_exc()
        else:
            try:
                response = client.create_tweet(text=tweet_text)
                if response and response.data:
                    logger.info("Tweet without image posted successfully!")
                    logger.info(f"Tweet ID: {response.data['id']}")
                else:
                    logger.error("Failed to post tweet without image.")
            except Exception as e:
                logger.error(f"Error posting tweet without image: {e}", exc_info=True)
                traceback.print_exc()

        # 7) Store posted tweet to avoid duplicates
        store_posted_tweet(text_content)

        # 8) Clean up old images
        cleanup_images(IMAGE_DIR, max_files=100)

        # 9) Increment post_count
        post_count += 1
        return post_count

    except Exception as e:
        logger.error(f"Unexpected error in post_to_twitter: {e}", exc_info=True)
        return post_count

# ──────────────────────────────────────────────────────────────────────────
# Responding to Mentions
# ──────────────────────────────────────────────────────────────────────────

def respond_to_mentions(client, since_id):
    """
    Responds to user mentions while skipping unsafe or spammy content.
    """
    try:
        me = client.get_me()
        user_id = me.data.id
        logger.info(f"Bot User ID: {user_id}")

        params = {
            'expansions': ['author_id', 'in_reply_to_user_id', 'referenced_tweets.id'],
            'tweet_fields': ['id', 'author_id', 'conversation_id', 'in_reply_to_user_id',
                             'referenced_tweets', 'text', 'created_at'],
            'user_fields': ['username'],
            'max_results': 100
        }
        if since_id:
            params['since_id'] = since_id

        logger.info("Fetching recent mentions...")
        mentions_resp = client.get_users_mentions(id=user_id, **params)
        if not mentions_resp.data:
            logger.info("No new mentions found.")
            return since_id

        logger.info(f"Fetched {len(mentions_resp.data)} mentions.")
        user_map = {}
        if mentions_resp.includes and 'users' in mentions_resp.includes:
            for usr in mentions_resp.includes['users']:
                user_map[usr.id] = usr.username

        new_since_id = max(int(m.id) for m in mentions_resp.data)

        for mention in mentions_resp.data:
            mention_id = mention.id
            author_id = mention.author_id
            username = user_map.get(author_id)
            logger.info(f"Processing mention {mention_id} from author {author_id} (@{username})")

            if mentions_collection.find_one({'tweet_id': mention_id}):
                logger.info(f"Already responded to mention {mention_id}. Skipping.")
                continue

            if author_id == user_id:
                logger.info(f"Skipping mention {mention_id} from self.")
                continue

            if not is_safe_to_respond(mention.text):
                logger.info(f"Skipped mention due to prohibited content: {mention.text}")
                continue

            reply_text = handle_comment_with_context(author_id, mention.text)
            if reply_text:
                full_reply = f"@{username} {reply_text}"
                max_len = 280 - len(f"@{username} ")
                full_reply = truncate_to_last_sentence(full_reply, max_length=max_len)
                logger.debug(f"Reply Text: {full_reply}")
                try:
                    client.create_tweet(text=full_reply, in_reply_to_tweet_id=mention_id)
                    logger.info(f"Replied to mention {mention_id}")
                    mentions_collection.insert_one({'tweet_id': mention_id, 'replied_at': datetime.utcnow()})
                except Exception as e:
                    logger.error(f"Error replying to mention {mention_id}: {e}", exc_info=True)
            else:
                logger.warning(f"Failed to generate response for mention {mention_id}.")

        return new_since_id
    except Exception as e:
        logger.error(f"Unexpected error in respond_to_mentions: {e}", exc_info=True)
        return since_id

# ──────────────────────────────────────────────────────────────────────────
# Scheduling
# ──────────────────────────────────────────────────────────────────────────

def posting_task(client, post_count):
    logger.info("Starting scheduled posting task.")
    post_count = post_to_twitter(client, post_count)
    save_post_count("post_count.txt", post_count)
    return post_count

def mention_checking_task(client, since_id):
    logger.info("Starting mention checking task.")
    new_id = respond_to_mentions(client, since_id)
    if new_id != since_id:
        save_since_id("since_id.txt", new_id)
        return new_id
    return since_id

def schedule_posting(client, post_count):
    """
    Schedule the posting task to run every N hours.
    """
    hours = random.choice([3, 4])
    schedule.every(hours).hours.do(posting_task, client=client, post_count=post_count)
    logger.info(f"Scheduled posting to run every {hours} hours.")

def schedule_mention_checking(client, since_id):
    schedule.every(1).hours.do(mention_checking_task, client=client, since_id=since_id)
    logger.info("Scheduled mention checking every 1 hour.")

def schedule_daily_persona(client):
    """Example: post a persona each day at 10:00 AM."""
    schedule.every().day.at("10:00").do(post_daily_persona, client=client)

def schedule_riddle_of_the_day(client):
    """Example: post a riddle each day at 16:00 (4 PM)."""
    schedule.every().day.at("16:00").do(post_riddle_of_the_day, client=client)

def schedule_daily_challenge(client):
    """Example: post a challenge each day at 12:00 (12 PM)."""
    schedule.every().day.at("12:00").do(post_challenge_of_the_day, client=client)

def schedule_storytime(client):
    """Example: post a story snippet each day at 18:00 (6 PM)."""
    schedule.every().day.at("18:00").do(post_story_update, client=client)
