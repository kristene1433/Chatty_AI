import os
import time
import random
from datetime import datetime
from dotenv import load_dotenv
import openai
import tweepy
import requests
import traceback
import glob
import json
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import logging
from logging.handlers import RotatingFileHandler
import signal
import sys
import schedule
from textblob import TextBlob
import nltk
import numpy as np
import re  # for regex if partial matches are desired

# Global in-memory caches
EMBEDDING_CACHE = {}   # key: text, value: embedding list
MODERATION_CACHE = {}  # key: text, value: Boolean (True => safe, False => flagged)


# Ensure 'punkt' is downloaded for TextBlob
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ──────────────────────────────────────────────────────────────────────────
# Configuration and Setup
# ──────────────────────────────────────────────────────────────────────────

load_dotenv()  # Load environment variables from .env if present

LOG_FILE = os.getenv("LOG_FILE", "application.log")
handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        handler,
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    logger.info("Shutting down gracefully...")
    try:
        mongo_client.close()
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

required_env_vars = [
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_OAUTH1_ACCESS_TOKEN",
    "TWITTER_OAUTH1_ACCESS_SECRET",
    "TWITTER_BEARER_TOKEN",
    "OPENAI_API_KEY",
    "MONGODB_URI",
    "LOG_FILE"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Masked logging of env var presence
for var in required_env_vars:
    val = os.getenv(var)
    if val:
        if var in ["TWITTER_API_SECRET", "TWITTER_OAUTH1_ACCESS_SECRET"]:
            logger.info(f"{var} is loaded.")
        else:
            masked = "*" * (len(val) - 4) + val[-4:] if len(val) > 4 else "*" * len(val)
            logger.info(f"{var} is loaded. Value: {masked}")
    else:
        logger.warning(f"{var} is NOT set.")

API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_OAUTH1_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_OAUTH1_ACCESS_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")

openai.api_key = os.getenv("OPENAI_API_KEY")

IMAGE_DIR = "generated_images"
PROMPT_FOLDER = "prompts"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PROMPT_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# MongoDB Setup
# ──────────────────────────────────────────────────────────────────────────

try:
    mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    db = mongo_client["TwitterBotProject"]

    mentions_collection = db["processed_mentions"]
    mentions_collection.create_index("tweet_id", unique=True)

    memory_collection = db["user_memory"]
    memory_collection.create_index([("user_id", 1), ("timestamp", -1)])

    embeddings_collection = db["conversation_embeddings"]
    embeddings_collection.create_index("embedding")

    posted_tweets_collection = db["posted_tweets"]
    posted_tweets_collection.create_index("timestamp")

    chatty_collection = db["chatty_instructions"]
    chatty_collection.create_index("name", unique=True)
    logger.info("chatty_instructions collection is ready.")

    logger.info("Connected to MongoDB Atlas successfully.")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {e}")
    traceback.print_exc()
    sys.exit(1)

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
            sys.exit(1)
        return client
    except Exception as e:
        logger.error(f"Error authenticating Twitter client: {e}", exc_info=True)
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────
# Chatty_AI: Personality Expansion
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
# Loading Extra Prompts from Files
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

# ──────────────────────────────────────────────────────────────────────────
# 25+ Prompts for Variety
# ──────────────────────────────────────────────────────────────────────────

HARDCODED_PROMPTS = [
    "What’s one surprising way AI is used in everyday life?",
    "Describe how machine learning can predict weather patterns.",
    "Name a cool AI application in the automotive industry.",
    "What’s a fun fact about AI in video game development?",
    "Explain how AI might improve medical diagnostics.",
    "Share a futuristic vision for AI in smart homes.",
    "What is quantum computing’s relationship to AI?",
    "Talk about AI-driven creativity in music or art.",
    "Highlight a cutting-edge breakthrough in robotics.",
    "Why do some people fear AI, and how is it addressed?",
    "Discuss how AI can help solve environmental challenges.",
    "What’s an interesting fact about neural networks?",
    "Describe how AI might transform online education.",
    "Mention a memorable milestone in AI history.",
    "How is AI used in agriculture to optimize crop yields?",
    "Talk about an AI tool that boosts productivity.",
    "What’s one important ethical concern around AI?",
    "Discuss how AI impacts cyber-security.",
    "Explain the concept of reinforcement learning in AI.",
    "Share something new about GPT-style language models.",
    "Describe an AI-driven gadget you’d love to see invented.",
    "What is the role of big data in fueling AI progress?",
    "Discuss a humorous AI scenario that could happen one day.",
    "How can AI help with personalized fitness or health?",
    "Talk about a dream future collaboration between humans and AI.",
    "How can AI revolutionize mental health care and therapy?",
    "Describe how AI can make public transportation systems more efficient.",
    "What role does AI play in renewable energy optimization?",
    "How is AI transforming the fashion industry?",
    "What’s an example of AI improving accessibility for people with disabilities?",
    "How might AI enhance disaster preparedness and response efforts?",
    "What are some surprising uses of AI in sports performance and coaching?",
    "Explain how AI can create hyper-personalized shopping experiences.",
    "How does AI help conserve endangered species and wildlife?",
    "What’s a fascinating way AI is being used in space exploration?",
    "Discuss how AI could eliminate language barriers in global communication.",
    "What role does AI play in combating misinformation online?",
    "How can AI improve workplace safety in hazardous industries?",
    "What’s a surprising way AI is used in the film and entertainment industry?",
    "How does AI support personalized learning in education?",
    "Explain how AI could enhance customer service in e-commerce.",
    "What are some ways AI is transforming urban planning and smart cities?",
    "How is AI helping to fight climate change?",
    "What role does AI play in advancing autonomous delivery systems?",
    "Describe how AI could transform the way we design and build homes.",
    "How can AI empower small businesses to compete with larger corporations?",
    "What’s a unique way AI is used in tracking global health trends?",
    "How might AI redefine the way we experience art and culture?",
    "Explain how AI can improve the safety and efficiency of supply chains.",
    "What is the potential for AI in personalized medicine and treatments?"
]

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

    # This user_prompt can be whatever guidance you’re currently providing
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

        # 1) Get the raw text returned by GPT
        raw_text = completion.choices[0].message.content.strip()
        
        # 2) Strip out any leading/trailing double-quotes or single-quotes
        clean_text = raw_text.strip('"').strip("'")
        
        return clean_text

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error generating themed post: {e}", exc_info=True)
        return f"Exciting news on {theme}! 🌟🚀 Let's harness AI for a brighter future. #AI #TechForGood"
    except Exception as e:
        logger.error(f"Unexpected error generating themed post: {e}", exc_info=True)
        return f"Stay tuned for more on {theme}! #AI #Innovation #FutureTech"


# ──────────────────────────────────────────────────────────────────────────
# STEP 1 (REPLACEMENT): Infer Chatty's Action from the Post Text
# ──────────────────────────────────────────────────────────────────────────

def auto_infer_action_from_text(post_text):
    """
    Analyzes the post_text and returns a short phrase describing
    an action Chatty could be doing to visually represent the theme.
    e.g., “holding a movie camera,” “analyzing medical charts,” etc.
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
            "What action should Chatty be doing to show this theme visually? "
            "Output only the action phrase, nothing else."
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
# DALL·E Image Generation (Truncation & Fallback)
# ──────────────────────────────────────────────────────────────────────────

def generate_image(prompt):
    """
    Generates an image using the DALL·E 3 model (if you have access).
    Truncates prompt if >1000 chars, sets n=1, size=1024x1024.
    """
    try:
        if len(prompt) > 1000:
            logger.warning(f"Truncating prompt from {len(prompt)} down to 1000 chars.")
            prompt = prompt[:1000]

        response = openai.Image.create(
            model="dall-e-3",  # explicitly requesting DALL·E 3
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
# Simplified Image Prompt Creation (Dynamic Scene)
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
            "Chatty’s appearance AND the environment from the user content. "
            "Under no circumstances should text, letters, signage, or written words appear in the image."
        )

        user_request = (
            f"{chatty_instructions}\n\n"
            "Key points to always include:\n"
            "- Retro CRT monitor in cream/off-white\n"
            "- Bright-blue screen face with big eyes, friendly smile\n"
            "- Metallic arms + white cartoon gloves\n"
            "- Slender legs + retro sneakers\n"
            "- Absolutely NO text, letters, or signage anywhere.\n\n"
            f"Scene Concept: {text_content}\n\n"
            "Your goal: Output a short, direct DALL·E prompt describing Chatty within this environment, "
            "without any text or letters visible. If text/letters appear, remove them."
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
# Themed Prompt (Optional) 
# ──────────────────────────────────────────────────────────────────────────

def get_theme_for_today():
    """
    50% chance: day-of-week prompt
    50% chance: random from ALL_PROMPTS
    """
    if random.random() < 0.5:
        day_of_week = datetime.utcnow().strftime("%A")
        day_prompt_mapping = {
            "Monday": "monday_ai.json",
            "Wednesday": "wednesday_tech.json",
            "Friday": "friday_inspiration.json"
        }
        if day_of_week in day_prompt_mapping:
            try:
                with open(os.path.join(PROMPT_FOLDER, day_prompt_mapping[day_of_week]), "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
                    prompt = prompt_data["prompt"]
                    logger.info(f"Selected prompt for {day_of_week}: {prompt}")
                    return prompt
            except Exception as e:
                logger.warning(f"Error loading prompt for {day_of_week}: {e}")

    if ALL_PROMPTS:
        prompt = random.choice(ALL_PROMPTS)
        logger.info(f"Selected random prompt: {prompt}")
        return prompt
    else:
        logger.warning("No prompts available. Using default prompt.")
        return "Share an interesting fact about AI."

# ──────────────────────────────────────────────────────────────────────────
# Tweet Construction (hashtags/emojis)
# ──────────────────────────────────────────────────────────────────────────

def get_contextual_emojis(content):
    ai_emojis = {
        "ai": "🤖",
        "future": "🔮",
        "innovation": "🚀",
        "code": "💻",
        "robot": "🤖"
    }
    selected = [emoji for key, emoji in ai_emojis.items() if key in content.lower()]
    return " ".join(selected) if selected else random.choice(["🤖", "💻", "🚀", "🔮"])

def get_relevant_hashtags(content):
    hashtag_map = {
        "ai": "#ArtificialIntelligence",
        "future": "#FutureTech",
        "coding": "#100DaysOfCode",
        "innovation": "#TechTrends"
    }
    tags = [ht for kw, ht in hashtag_map.items() if kw in content.lower()]
    while len(tags) < 2:
        tags.append(random.choice(["#AICommunity", "#Innovation", "#TechInsights"]))
    return " ".join(random.sample(tags, len(tags)))  # shuffle or just keep as-is

def truncate_to_last_sentence(text, max_length=280):
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_punc = max(truncated.rfind(c) for c in ".!?")
    if last_punc != -1:
        return truncated[:last_punc+1]
    else:
        return truncated[:max_length-3] + "..."

def construct_tweet(text_content):
    # 1) Pick an emoji (or emojis) relevant to the content.
    emojis = get_contextual_emojis(text_content)
    
    # 2) Gather relevant hashtags.
    raw_hashtags = get_relevant_hashtags(text_content)

    # 3) Make #chatty the first hashtag, then add the others.
    #    This ensures #chatty appears at the end, but before any other hashtags.
    hashtags = f"#chatty {raw_hashtags}"

    # 4) Build the core tweet (main text + emojis).
    tweet_core = f"{text_content} {emojis}"

    # 5) Check how many characters remain for hashtags (with a space in front).
    remaining_length = 280 - len(tweet_core) - 1  # minus 1 for the space

    # 6) Truncate hashtags if needed, then combine.
    hashtags = hashtags[:remaining_length]
    tweet = f"{tweet_core} {hashtags}"

    # 7) Finally, if we still exceed 280, truncate the entire text to the last sentence.
    if len(tweet) > 280:
        tweet = truncate_to_last_sentence(tweet, max_length=280)

    return tweet



# ──────────────────────────────────────────────────────────────────────────
# Image Cleanup
# ──────────────────────────────────────────────────────────────────────────

def cleanup_images(directory, max_files=100):
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

def contains_prohibited_phrases(text):
    prohibited_phrases = ["airdrop", "giveaway", "telegram", "click", "join", "https://"]
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in prohibited_phrases)

    
def moderate_content(text):
    # 1) Check cache first
    if text in MODERATION_CACHE:
        # Return True/False (cached result)
        return MODERATION_CACHE[text]

    try:
        resp = openai.Moderation.create(input=text)
        is_safe = not resp['results'][0]['flagged']
        # 2) Cache the moderation result
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
# Memory Layer with MongoDB
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
# Sentiment / Community
# ──────────────────────────────────────────────────────────────────────────

def fetch_community_posts(hashtag, count=50):
    try:
        tweets = client.search_recent_tweets(query=f"#{hashtag} -is:retweet lang:en", max_results=100)
        return [t.text for t in tweets.data] if tweets.data else []
    except Exception as e:
        logger.error(f"Error fetching posts for #{hashtag}: {e}", exc_info=True)
        return []

def analyze_community_sentiment(posts):
    try:
        sentiments = [TextBlob(p).sentiment.polarity for p in posts]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
        logger.info(f"Average community sentiment: {avg_sent}")
        return avg_sent
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
        return 0

def adjust_personality_based_sentiment():
    posts = fetch_community_posts("AICommunity", count=50)
    avg_sent = analyze_community_sentiment(posts)
    if avg_sent > 0.5:
        selected_prompt = SYSTEM_PROMPTS[0]
        logger.info("Adjusted personality: Upbeat (Chatty style 1)")
    elif avg_sent < -0.5:
        selected_prompt = SYSTEM_PROMPTS[1]
        logger.info("Adjusted personality: Joyful (Chatty style 2)")
    else:
        selected_prompt = random.choice(SYSTEM_PROMPTS)
        logger.info("Adjusted personality: Random from Chatty prompts")
    return selected_prompt

# ──────────────────────────────────────────────────────────────────────────
# Embeddings + Semantic Search
# ──────────────────────────────────────────────────────────────────────────


def generate_embedding(text):
    # 1) Check if we've already computed an embedding for this exact text
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]

    try:
        resp = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embedding = resp['data'][0]['embedding']
        # 2) Cache the result for this text
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
# Generate Response with Context
# ──────────────────────────────────────────────────────────────────────────

def generate_contextual_response(user_id, comment):
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
    """
    Handles user mentions by generating context-aware responses,
    with filtering for spam or unsafe content.
    """
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
# STEP 2: Update create_scene_content() to accept an optional "action"
# ──────────────────────────────────────────────────────────────────────────

def create_scene_content(theme, action=None):
    """
    Creates a short textual scene that describes Chatty (the retro CRT AI) 
    in the context of the given theme. Also includes an optional action parameter
    so Chatty is actually doing something in the scene.
    """
    # Base environment for the theme
    base_scene = (
        f"Chatty, a retro CRT monitor with a bright-blue screen face, "
        f"is in a vibrant, futuristic environment about {theme}. "
        "No text, letters, or signage anywhere."
    )

    # If an action is provided, incorporate it
    if action:
        base_scene += f" Chatty is {action}."

    return base_scene

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: post_to_twitter() - Now uses auto_infer_action_from_text()
# ──────────────────────────────────────────────────────────────────────────

def post_to_twitter(client, post_count, force_image=False):
    """
    Posts a tweet where:
      - The TEXT is based on the 'themed' style from generate_themed_post(theme).
      - The ACTION is inferred from that text, so Chatty performs a relevant task.
      - The IMAGE is generated with that theme + action.
    """
    themes_list = [
        "AI in Earth Observation Satellites",
        "AI into Housing Solutions",
        "AI in Medicine",
        "AI in Education",
        "AI for Mental Health",
        "AI in Transportation",
        "AI in Agriculture",
        "AI in Sports Performance",
        "AI for Disaster Prevention",
        "AI in Fashion",
        "AI in Energy Optimization",
        "AI in Robotics",
        "AI in Finance",
        "AI for Urban Development",
        "AI for Cybersecurity",
        "AI in Space Exploration",
        "AI for Climate Change",
        "AI for Smart Homes",
        "AI for Accessibility",
        "AI for Art and Creativity",
        "AI for Language Translation",
        "AI for Environmental Conservation",
        "AI in Entertainment",
        "AI in Retail",
        "AI for Personal Assistants",
        "AI for Food and Beverage",
        "AI in Manufacturing",
        "AI for Wildlife Protection",
        "AI in Genomics",
        "AI for Drone Swarm Coordination",
        "AI in Film Post-Production",
        "AI for Hospitality Services",
        "AI in Emotional Recognition",
        "AI for Music Composition",
        "AI in Sustainable Fisheries",
        "AI for Public Health Monitoring",
        "AI in Home Security Systems",
        "AI for Personalized Nutrition",
        "AI in Voice Recognition",
        "AI in Telehealth",
        "AI for Music Recommendation",
        "AI in Virtual Reality",
        "AI for Agricultural Robotics",
        "AI for Traffic Management",
        "AI in Insurance",
        "AI for Fraud Detection",
        "AI in Real Estate Valuation",
        "AI for Genetic Research",
        "AI in 3D Printing",
        "AI for E-Learning Platforms",
        "AI in Drone Delivery Services",
        "AI for Marine Conservation",
        "AI in Aviation",
        "AI for Voice Cloning",
        "AI in Food Delivery Optimization",
        "AI for Earthquake Prediction",
        "AI in Online Dating",
        "AI for Remote Workforce Management",
        "AI in Astronomy",
        "AI for Search and Rescue",
        "AI in Automotive Safety",
        "AI for Social Good Campaigns",
        "AI in Cultural Preservation",
        "AI for Political Polling",
        "AI in Talent Recruiting",
        "AI for Water Resource Management",
        "AI in Crowd Control",
        "AI for Psychotherapy Chatbots",
        "AI in Hardware Optimization",
        "AI for Autonomous Underwater Vehicles",
        "AI in Battery Technology",
        "AI for Emotional Support Bots",
        "AI in Weather Forecasting",
        "AI for Brain-Computer Interfaces",
        "AI in Pharmaceuticals",
        "AI for Restaurant Menu Insights",
        "AI in Nonprofit Fundraising",
        "AI for Workforce Reskilling",
        "AI in eSports Analytics",
        "AI for Journalism Assistance",
        "AI in Supply Chain Resilience",
        "AI for Consumer Behavior Analysis",
        "AI in Ethical Decision-Making",
        "AI for Demographic Predictions",
        "AI in Customer Service Chatbots",
        "AI for Digital Marketing Automation",
        "AI in HR Analytics",
        "AI for Personalized Fitness",
        "AI in Brain Research",
        "AI for Cultural Heritage Restoration",
        "AI in Prosthetics Design",
        "AI for Firefighting Drones",
        "AI in Children's Education",
        "AI for Marine Robotics",
        "AI in Hospital Administration",
        "AI for Elderly Care",
        "AI in Literary Analysis",
        "AI for Rare Disease Diagnosis",
        "AI in Carbon Capture",
        "AI for Personalized Travel Recommendations"
    ]

    try:
        # 1) Pick a random theme each time
        theme = random.choice(themes_list)
        logger.info(f"Randomly chosen theme: {theme}")

        # 2) Generate a short, upbeat post for this theme
        text_content = generate_themed_post(theme)

        # 3) Avoid duplicates if it's too similar to recent tweets
        if is_too_similar_to_recent_tweets(text_content, similarity_threshold=0.95, lookback=10):
            logger.warning("Themed post is too similar to a recent tweet. Using fallback instead.")
            text_content = "Exciting times in AI! Stay tuned, #AICommunity 🤖🚀"

        # 4) Construct the final tweet (shortening if needed)
        tweet_text = construct_tweet(text_content)

        # 5) Decide whether to attach an image
        include_image = force_image or ((post_count + 1) % 3 == 0)
        image_path = None

        if include_image:
            logger.info("Including image in this post.")

            # (A) Infer the best action from the post text
            inferred_action = auto_infer_action_from_text(text_content)

            # (B) Create a scene referencing the theme + that action
            scene_content = create_scene_content(theme, action=inferred_action)

            # (C) Convert scene to a DALL·E prompt
            img_prompt = create_simplified_image_prompt(scene_content)

            # (D) Generate the DALL·E image
            img_url = generate_image(img_prompt)
            if img_url:
                image_path = download_image(img_url, img_prompt)
            else:
                logger.warning("Failed to generate image. Skipping image for this post.")

        # 6) Post to Twitter (with or without image)
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

def schedule_posting(client):
    hours = random.choice([3, 4])
    schedule.every(hours).hours.do(posting_task, client=client)
    logger.info(f"Scheduled posting to run every {hours} hours.")

def posting_task(client):
    global post_count
    logger.info("Starting scheduled posting task.")
    post_count = post_to_twitter(client, post_count)
    save_post_count("post_count.txt", post_count)

def schedule_mention_checking():
    schedule.every(1).hours.do(mention_checking_task)
    logger.info("Scheduled mention checking every 1 hour.")

def mention_checking_task():
    global since_id
    logger.info("Starting mention checking task.")
    new_id = respond_to_mentions(client, since_id)
    if new_id != since_id:
        save_since_id("since_id.txt", new_id)
        since_id = new_id

def initialize_scheduling(client):
    schedule_posting(client)
    schedule_mention_checking()

# ──────────────────────────────────────────────────────────────────────────
# Main Execution Flow
# ──────────────────────────────────────────────────────────────────────────

post_count = 0
since_id = None
client = None

if __name__ == "__main__":
    client = authenticate_twitter_client()
    logger.info("AI bot is running...")

    post_count = load_post_count("post_count.txt")
    since_id = load_since_id("since_id.txt")

    # Store a bullet-point Chatty persona in "BaseChatty", used for images
    chatty_persona_text = """
Chatty is a delightful, retro-styled character blending classic computers with modern AI features:

1) SCREEN FACE:
   - Vibrant bright blue screen with large, friendly eyes.
   - Subtle shine or reflections for a lifelike effect.
   - Cheerful, dynamic smile; slight blush or highlights on cheeks.

2) CRT MONITOR BODY:
   - Retro CRT monitor casing (cream/off-white/beige) with ventilation grilles, buttons, power light.
   - Soft shadows and reflections for a 3D, photorealistic look.

3) ARMS & HANDS:
   - Flexible, polished metallic arms, cartoon-style white gloves.

4) LEGS & SNEAKERS:
   - Rounded, slender legs with colorful, retro-inspired sneakers (detailed shoelaces, reflections).

5) OVERALL STYLE:
   - Clean, polished cartoon vibe in photorealistic settings, with consistent lighting and shadows.
"""
    store_chatty_config("BaseChatty", chatty_persona_text)

    logger.info("Posting an immediate tweet WITH an image.")
    post_count = post_to_twitter(client, post_count, force_image=True)
    save_post_count("post_count.txt", post_count)

    initialize_scheduling(client)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            traceback.print_exc()
            time.sleep(60)

