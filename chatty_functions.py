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
import time
from collections import deque
from textblob import TextBlob
from datetime import datetime

from config_and_setup import (
    logger, db, mentions_collection, memory_collection,
    embeddings_collection, posted_tweets_collection, chatty_collection,
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET, BEARER_TOKEN,
    IMAGE_DIR, PROMPT_FOLDER
)

EMBEDDING_CACHE = {}
MODERATION_CACHE = {}
USER_RATE_LIMITS = {}

# NEW: We'll track recently used themes here (adjust maxlen as needed).
RECENT_THEMES = deque(maxlen=10)

def check_rate_limit(user_id, max_requests=5, window_sec=60):
    now = datetime.utcnow()
    if user_id not in USER_RATE_LIMITS:
        USER_RATE_LIMITS[user_id] = deque()

    while USER_RATE_LIMITS[user_id] and (now - USER_RATE_LIMITS[user_id][0]).total_seconds() > window_sec:
        USER_RATE_LIMITS[user_id].popleft()

    if len(USER_RATE_LIMITS[user_id]) >= max_requests:
        return False

    USER_RATE_LIMITS[user_id].append(now)
    return True

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_list_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
THEMES_FILE = os.path.join(DATA_DIR, "themes.json")
PROMPTS_FILE = os.path.join(DATA_DIR, "hardcoded_prompts.json")

themes_list = load_list_from_json(THEMES_FILE)
HARDCODED_PROMPTS = load_list_from_json(PROMPTS_FILE)

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
    return random.choice(SYSTEM_PROMPTS)

def load_json_data_from_folder(folder_path):
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

ALL_PROMPTS = HARDCODED_PROMPTS + load_prompts()

def robust_chat_completion(messages, model="gpt-4", max_retries=2, **kwargs):
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs
            )
            end_time = time.time()
            logger.info(f"GPT call took {end_time - start_time:.2f} seconds on attempt {attempt+1}.")
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

def summarize_text(text):
    if not text:
        return ""
    if len(text) <= 500:
        return text

    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant who summarizes text."},
        {"role": "user", "content": f"Please summarize the following conversation in 3 concise sentences:\n{text}"}
    ]
    resp = robust_chat_completion(summary_prompt, model="gpt-3.5-turbo", max_tokens=100)
    if resp:
        return resp["choices"][0]["message"]["content"].strip()
    else:
        return text[:500] + "..."

def build_conversation_path(tweet_id):
    path_texts = []
    current_id = tweet_id
    while current_id:
        doc = posted_tweets_collection.find_one({"tweet_id": current_id})
        if not doc:
            break
        path_texts.insert(0, doc["text"])
        current_id = doc.get("parent_id")
    return "\n".join(path_texts)

def generate_themed_post(theme):
    system_prompt = (
        "You are a social media copywriter who creates short, vivid, and enthusiastic posts. "
        "Keep the tone optimistic and futuristic. Use relevant emojis and stay under 200 characters total! "
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
    resp = robust_chat_completion(messages, max_tokens=220, temperature=0.8, presence_penalty=1.0, frequency_penalty=0.5)
    if resp:
        raw_text = resp.choices[0].message.content.strip()
        return raw_text.strip('"').strip("'")
    else:
        return f"Exciting news on {theme}! #AI #TechForGood"

def auto_infer_action_from_text(post_text):
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
    resp = robust_chat_completion(messages, max_tokens=40, temperature=0.9, presence_penalty=1.0, frequency_penalty=0.5)
    if resp:
        action_text = resp.choices[0].message.content.strip()
        logger.info(f"Inferred Action: {action_text}")
        return action_text
    else:
        return "performing a futuristic task"

MAX_PROMPT_LENGTH = 3000  # NEW: Define a max prompt length constant here.

def generate_image(prompt):
    """
    Generates an image using the DALL·E 3 model (if you have access).
    """
    try:
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(
                f"Truncating prompt from {len(prompt)} down to {MAX_PROMPT_LENGTH} chars."
            )
            prompt = prompt[:MAX_PROMPT_LENGTH]

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

def store_chatty_config(name, instructions):
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
    try:
        doc = chatty_collection.find_one({"name": name})
        if doc:
            return doc.get("instructions", "")
        return ""
    except Exception as e:
        logger.error(f"Error retrieving Chatty instructions: {e}", exc_info=True)
        return ""

def randomize_chatty_appearance():
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

def create_simplified_image_prompt(text_content):
    """
    Creates a short DALL·E prompt focusing on Chatty’s retro-CRT style
    PLUS environment details from text_content. Emphasizes no text or letters.
    """
    try:
        chatty_instructions = get_chatty_config("BaseChatty")

        system_instruction = (
            "You are a creative AI that outputs a short, direct prompt for DALL·E. "
            "Describe Chatty’s retro CRT appearance and the environment, in 1–2 sentences. "
            "Absolutely no text or letters in the scene."
        )

        user_request = (
            f"Scene Concept: {text_content}\n\n"
            f"Chatty instructions: {chatty_instructions}\n\n"
            "Be concise—no more than 2 sentences. Emphasize Chatty’s environment and mood. No text or letters!"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_request}
        ]
        response = robust_chat_completion(messages, model="gpt-4", max_tokens=120, 
                                          temperature=0.9, presence_penalty=0.7, frequency_penalty=0.5)
        if response:
            prompt_result = response.choices[0].message.content.strip()

            if len(prompt_result) > MAX_PROMPT_LENGTH:
                logger.warning(
                    f"Truncating prompt from {len(prompt_result)} to {MAX_PROMPT_LENGTH} chars."
                )
                prompt_result = prompt_result[:MAX_PROMPT_LENGTH]

            logger.info(f"Simplified Chatty Prompt Created: {prompt_result}")
            return prompt_result
        else:
            return "Depict Chatty (retro CRT) with no text or letters in the environment."

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error creating simplified image prompt: {e}", exc_info=True)
        return "Depict Chatty (retro CRT) with no text or letters in the environment."
    except Exception as e:
        logger.error(f"Unexpected error creating simplified image prompt: {e}", exc_info=True)
        return f"Depict Chatty (retro CRT) in this setting, with no text or letters: {text_content}"

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

def post_daily_persona(client):
    if not PERSONAS_LIST:
        logger.warning("No personas loaded. Skipping persona post.")
        return

    persona = random.choice(PERSONAS_LIST)
    user_prompt = (
        f"You are {persona}. Write a short social media post describing a 'day in the life' "
        "and end with a fun question. Keep under 280 chars."
    )
    system_prompt = "You are Chatty_AI, bright and playful..."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        completion = robust_chat_completion(messages, model="gpt-4", max_tokens=200, temperature=0.8)
        if completion:
            text_content = completion.choices[0].message.content.strip()
        else:
            text_content = "I'm living a bright day as a persona—what's your next move? #chatty"

        tweet_text = construct_tweet(text_content)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Daily Persona Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting daily persona: {e}", exc_info=True)

def post_riddle_of_the_day(client):
    if not RIDDLES_LIST:
        logger.warning("No riddles loaded. Skipping riddle post.")
        return

    riddle = random.choice(RIDDLES_LIST)
    question = riddle.get("question", "What's the puzzle?")
    riddle_text = f"Puzzle Time: {question}\nReply with your guess! #chatty #PuzzleTime"
    try:
        tweet_text = construct_tweet(riddle_text)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Riddle post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting riddle: {e}", exc_info=True)

def post_challenge_of_the_day(client):
    if not CHALL_LIST:
        logger.warning("No challenges loaded. Skipping challenge post.")
        return

    challenge = random.choice(CHALL_LIST)
    challenge_text = f"Challenge time: {challenge}\nShare your thoughts! #chatty #Challenge"
    try:
        tweet_text = construct_tweet(challenge_text)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Challenge post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting challenge: {e}", exc_info=True)

def post_story_update(client):
    if not STORY_LIST:
        logger.warning("No stories loaded. Skipping story post.")
        return

    story_snippet = random.choice(STORY_LIST)
    text = story_snippet.get('text', 'Once upon a time, Chatty...')
    story_text = f"Story Time: {text}\nWhat happens next? #chatty #Story"

    try:
        tweet_text = construct_tweet(story_text)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Story post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting story update: {e}", exc_info=True)

def safe_truncate(text, max_len=280):
    if len(text) <= max_len:
        return text
    return text[: (max_len - 3)].rstrip() + "..."

def construct_tweet(text_content):
    """
    Builds the final tweet text by appending mention(s) or hashtags.
    """
    text_content = text_content.strip().strip('"').strip("'")
    # Remove existing hashtags if you want
    no_hashtags = re.sub(r"#\w+", "", text_content).strip()

    extra_tags_pool = [
        "#HeyChatty", "#AIforEveryone", "#MEMECOIN", "$CHATTY",
        "#AImeme", "#AIagent", "@ChatGPTapp"
    ]
    pick = random.choice(extra_tags_pool)
    tags = ["@chattyonsolana", pick]

    return f"{no_hashtags} {' '.join(tags)}"

def cleanup_images(directory, max_files=100):
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

def chatty_info_check(comment):
    """
    Checks if the user is asking for info about Chatty and returns a standard
    response directing them to @chattyonsolana if so. Otherwise returns None.
    """
    triggers = [
        "what is chatty",
        "tell me about chatty",
        "info about chatty",
        "how does chatty work",
        "where can i find chatty info",
        "where can i find info about chatty"
    ]
    text_lower = comment.lower()

    for phrase in triggers:
        if phrase in text_lower:
            return (
                "For all the details on Chatty, head to our main page @chattyonsolana! "
                "Everything you need is right there. ⭐️🤖"
            )
    return None

def generate_safe_response(comment):
    system_prompt = (
        "You are a helpful AI. You only discuss AI-related topics. "
        "If the question is about personal info or finances, politely decline."
    )
    user_prompt = f"You are asked: '{comment}'. Respond appropriately."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = robust_chat_completion(messages, model="gpt-4", max_tokens=100)
    if resp:
        return resp['choices'][0]['message']['content'].strip()
    else:
        return "I’m sorry, I can’t help with that. 🤖"

def log_response(comment, response):
    try:
        with open("response_log.txt", "a", encoding="utf-8") as lf:
            lf.write(f"Timestamp: {datetime.utcnow().isoformat()} UTC\n")
            lf.write(f"Comment: {comment}\nResponse: {response}\n---\n")
    except Exception as e:
        logger.error(f"Error logging response: {e}", exc_info=True)

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

def moderate_bot_output(bot_text):
    if not moderate_content(bot_text):
        logger.warning("Bot output was flagged; returning fallback.")
        return "I’m sorry, let me rephrase that to keep things friendly. 🤖"
    return bot_text

def create_scene_content(theme, action=None):
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
    system_prompt = (
        "You are a writing assistant. The user has a short social media post about AI or memecoins. "
        "They want it expanded with more specifics, but keep it UNDER 230 characters total."
    )
    user_prompt = (
        f"Original Post:\n'{original_text}'\n\n"
        "Expand with one or two specifics or examples, strictly under 230 chars."
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        completion = robust_chat_completion(
            messages,
            model="gpt-4",
            max_tokens=150,
            temperature=0.7,
            presence_penalty=1.0,
            frequency_penalty=0.5
        )

        if not completion:
            return original_text

        expanded_text = completion.choices[0].message.content.strip()
        if len(expanded_text) > 230:
            expanded_text = expanded_text[:227].rstrip() + "..."
        return expanded_text

    except openai.error.OpenAIError as e:
        logger.error(f"Error expanding post with examples: {e}", exc_info=True)
        return original_text
    except Exception as e:
        logger.error(f"Unexpected error in expand_post_with_examples: {e}", exc_info=True)
        return original_text

# NEW: function to pick a theme not used recently.
def get_new_theme(themes_list):
    """Pick a theme excluding what's in RECENT_THEMES. If all are excluded, allow them again."""
    available_themes = [t for t in themes_list if t not in RECENT_THEMES]
    if not available_themes:
        # If we've excluded everything, just allow them all again
        available_themes = themes_list
    
    theme = random.choice(available_themes)
    RECENT_THEMES.append(theme)
    return theme

def post_to_twitter(client, post_count, force_image=False):
    try:
        # Instead of random.choice, call our new function:
        theme = get_new_theme(themes_list)
        logger.info(f"Randomly chosen theme: {theme}")

        text_content = generate_themed_post(theme)
        expanded_text = expand_post_with_examples(text_content)

        if is_too_similar_to_recent_tweets(expanded_text, similarity_threshold=0.95, lookback=10):
            logger.warning("Expanded post is too similar to a recent tweet. Using fallback instead.")
            expanded_text = "Exciting times in AI! Stay tuned, #AICommunity 🤖🚀"

        tweet_text = construct_tweet(expanded_text)
        tweet_text = safe_truncate(tweet_text, 280)

        logger.info("Including image in this post (every post).")
        inferred_action = auto_infer_action_from_text(expanded_text)
        scene_content = create_scene_content(theme, action=inferred_action)
        img_prompt = create_simplified_image_prompt(scene_content)
        img_url = generate_image(img_prompt)
        image_path = None

        if img_url:
            image_path = download_image(img_url, img_prompt)
        else:
            logger.warning("Failed to generate image. Skipping image for this post.")

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

        store_posted_tweet(expanded_text)
        cleanup_images(IMAGE_DIR, max_files=100)
        post_count += 1
        return post_count

    except Exception as e:
        logger.error(f"Unexpected error in post_to_twitter: {e}", exc_info=True)
        return post_count

def handle_comment_with_context(user_id, comment, tweet_id=None, parent_id=None):
    """
    Generates a GPT-based reply to a user mention or comment,
    respecting the rate limit and moderation checks.
    """
    if not check_rate_limit(user_id):
        return "Please wait a bit before sending more requests. 🤖"

    if not is_safe_to_respond(comment) or not moderate_content(comment):
        logger.info(f"Skipping unsafe/filtered comment: {comment}")
        return "I’m here to discuss AI and technology topics! 🚀✨"

    deflection = deflect_unrelated_comments(comment)
    if deflection:
        return deflection

    faq = static_response(comment)
    if faq:
        return faq

    chatty_info = chatty_info_check(comment)
    if chatty_info:
        return chatty_info

    full_convo = ""
    if parent_id:
        full_convo = build_conversation_path(parent_id)
    short_summary = summarize_text(full_convo)

    system_msg = select_system_prompt()
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                f"Conversation so far (summary):\n{short_summary}\n\n"
                f"User says: {comment}"
            )
        }
    ]
    resp = robust_chat_completion(messages, model="gpt-4", max_tokens=150,
                                  temperature=0.9, presence_penalty=0.7, frequency_penalty=0.5)
    if not resp:
        bot_reply = "I’m sorry, I'm having trouble processing right now."
    else:
        bot_reply = resp["choices"][0]["message"]["content"].strip()

    bot_reply = moderate_bot_output(bot_reply)
    log_response(comment, bot_reply)

    if tweet_id:
        posted_tweets_collection.update_one(
            {"tweet_id": tweet_id},
            {"$set": {
                "text": comment,
                "parent_id": parent_id
            }},
            upsert=True
        )

    store_user_memory(user_id, bot_reply)

    try:
        emb = generate_embedding(bot_reply)
        if emb:
            embeddings_collection.insert_one({
                "conversation_context": bot_reply,
                "embedding": emb,
                "timestamp": datetime.utcnow()
            })
            logger.info("Stored embedding for semantic search.")
    except Exception as e:
        logger.error(f"Error storing embedding: {e}", exc_info=True)

    return bot_reply


def respond_to_mentions(client, since_id):
    """
    Fetch mentions and reply to them with handle_comment_with_context.
    """
    try:
        me = client.get_me()
        bot_user_id = me.data.id
        logger.info(f"Bot User ID: {bot_user_id}")

        params = {
            'expansions': ['author_id', 'in_reply_to_user_id', 'referenced_tweets.id'],
            'tweet_fields': [
                'id','author_id','conversation_id','in_reply_to_user_id',
                'referenced_tweets','text','created_at'
            ],
            'user_fields': ['username'],
            'max_results': 100
        }
        if since_id:
            params['since_id'] = since_id

        logger.info("Fetching recent mentions...")
        mentions_resp = client.get_users_mentions(id=bot_user_id, **params)
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
            username = user_map.get(author_id, "unknown_user")

            logger.info(f"Processing mention {mention_id} from author {author_id} (@{username})")

            # Skip if we've already replied to this mention
            if mentions_collection.find_one({'tweet_id': mention_id}):
                logger.info(f"Already responded to mention {mention_id}. Skipping.")
                continue

            # Skip if mention is from self (the bot)
            if author_id == bot_user_id:
                logger.info(f"Skipping mention {mention_id} from self.")
                continue

            # Block or skip unsafe/spammy mentions
            if not is_safe_to_respond(mention.text):
                logger.info(f"Skipped mention due to prohibited content: {mention.text}")
                continue

            parent_id = None
            if mention.referenced_tweets:
                parent_id = mention.referenced_tweets[0].id

            # Generate a reply using GPT or fallback logic
            reply_text = handle_comment_with_context(
                user_id=author_id,
                comment=mention.text,
                tweet_id=mention_id,
                parent_id=parent_id
            )

            if reply_text:
                # Include the author's username in the reply
                full_reply = f"@{username} {reply_text}"
                # Truncate to 240 chars so there's less risk of cutting off
                final_reply = safe_truncate(full_reply, max_len=240)

                logger.debug(f"Reply Text (final): {final_reply}")
                try:
                    client.create_tweet(text=final_reply, in_reply_to_tweet_id=mention_id)
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
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(since_id))
        logger.info(f"Updated since_id in {file_name} to {since_id}.")
    except Exception as e:
        logger.error(f"Error saving since_id to {file_name}: {e}", exc_info=True)

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
    hours = random.choice([3, 4])
    schedule.every(hours).hours.do(posting_task, client=client, post_count=post_count)
    logger.info(f"Scheduled posting to run every {hours} hours.")

def schedule_mention_checking(client, since_id):
    schedule.every(1).hours.do(mention_checking_task, client=client, since_id=since_id)
    logger.info("Scheduled mention checking every 1 hour.")

def schedule_daily_persona(client):
    schedule.every().day.at("10:00").do(post_daily_persona, client=client)

def schedule_riddle_of_the_day(client):
    schedule.every().day.at("16:00").do(post_riddle_of_the_day, client=client)

def schedule_daily_challenge(client):
    schedule.every().day.at("12:00").do(post_challenge_of_the_day, client=client)

def schedule_storytime(client):
    schedule.every().day.at("18:00").do(post_story_update, client=client)
