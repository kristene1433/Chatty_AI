import os
import sys
import time
import random
import signal
import logging
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables from .env, if present
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────
# Configuration and Setup
# ──────────────────────────────────────────────────────────────────────────

# Create a rotating log handler
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

# Required environment variables
required_env_vars = [
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_OAUTH1_ACCESS_TOKEN",
    "TWITTER_OAUTH1_ACCESS_SECRET",
    "TWITTER_BEARER_TOKEN",
    "TELEGRAM_BOT_TOKEN",
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

# Read environment variables
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_OAUTH1_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_OAUTH1_ACCESS_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Directories for images and custom prompt files
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
# Signal Handling
# ──────────────────────────────────────────────────────────────────────────

def signal_handler(sig, frame):
    logger.info("Shutting down gracefully...")
    try:
        mongo_client.close()
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
