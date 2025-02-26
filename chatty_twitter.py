import re
import os
import random
import traceback
import tweepy
import schedule
from datetime import datetime
from chatty_core import handle_incoming_message

from config_and_setup import (
    logger,           # shared logger
    mentions_collection,  # DB collections from your config
    posted_tweets_collection  # so we can store riddle answers & retrieve them
)

from chatty_core import (
    check_rate_limit,
    is_safe_to_respond,
    moderate_content,
    moderate_bot_output,
    deflect_unrelated_comments,
    static_response,
    summarize_text,
    select_system_prompt,
    robust_chat_completion,
    store_user_memory,
    generate_embedding,
    search_similar_conversations,
    store_posted_tweet,
    safe_truncate,
    safe_truncate_by_sentence_no_ellipsis,
    generate_themed_post,
    expand_post_with_examples,
    auto_infer_action_from_text,
    create_scene_content,
    create_simplified_image_prompt,
    generate_image,
    download_image,
    cleanup_images,
    themes_list,
    build_conversation_path,
    log_response,
    generate_sentiment_aware_response,
    ensure_positive_tone,
    detect_sentiment_and_subjectivity,
    ADVANCED_MODEL,
    is_guess_correct,
    RIDDLES_LIST,
    CHALL_LIST,
    PERSONAS_LIST,
    STORY_LIST,
    is_too_similar_to_recent_tweets,
    MAX_PROMPT_LENGTH_TWITTER,
    check_link_inquiry
)

###############################################################################
# TWITTER AUTHENTICATION
###############################################################################
def authenticate_twitter_client(api_key, api_secret, access_token, access_secret, bearer_token):
    try:
        client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
            bearer_token=bearer_token,
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

###############################################################################
# BASIC FILE I/O FOR PERSISTENCE
###############################################################################
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

###############################################################################
# NORMALIZATION FUNCTION FOR MEMECOIN REFERENCES
###############################################################################
def normalize_memecoin_references(text):
    normalized = re.sub(r'\bmemecoin\b', 'Chatty meme coin', text, flags=re.IGNORECASE)
    normalized = re.sub(r'\bdogecoin\b', 'Chatty meme coin', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'#\s*memecoin', '#CHATTY', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'#\s*dogecoin', '#CHATTY', normalized, flags=re.IGNORECASE)
    return normalized

###############################################################################
# BUILDING & POSTING TWEETS
###############################################################################
def construct_tweet(text_content):
    text_content = text_content.strip().strip('"').strip("'")
    text_content = text_content[:240].rstrip()
    # Remove existing hashtags
    no_hashtags = re.sub(r"#\w+", "", text_content).strip()
    extra_tags_pool = [
        "#ChatGPT", "#OpenAI", "#GPT4", "#AI", "#CHATTY"
    ]
    pick = random.choice(extra_tags_pool)
    # Always include explicit mentions for branding
    mentions = "@OpenAI @ChatGPTapp"
    final_text = f"{no_hashtags} {mentions} {pick}"
    final_text = safe_truncate(final_text, 280)
    final_text = normalize_memecoin_references(final_text)
    return final_text

def post_daily_persona(client):
    if not PERSONAS_LIST:
        logger.warning("No personas loaded. Skipping persona post.")
        return
    persona = random.choice(PERSONAS_LIST)
    user_prompt = (
        f"You are {persona}. Write a short social media post describing a 'day in the life' "
        "with a focus on OpenAI ChatGPT and Chatty meme coin. End with a fun question. Keep under 280 chars."
    )
    system_prompt = "You are Chatty_AI, bright and playful, powered by OpenAI ChatGPT and fueled by Chatty meme coin."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        completion = robust_chat_completion(messages, model=ADVANCED_MODEL, max_tokens=200, temperature=0.8)
        if completion:
            text_content = completion.choices[0].message.content.strip()
        else:
            text_content = "I'm living a bright day as a persona—what's your next move? #Chatty"
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
    answer = riddle.get("answer", None)
    riddle_text = f"Puzzle Time: {question}\nReply with your guess! #Chatty #PuzzleTime"
    try:
        tweet_text = construct_tweet(riddle_text)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Riddle post Tweet ID: {response.data['id']}")
        posted_tweets_collection.insert_one({
            "tweet_id": response.data['id'],
            "text": riddle_text,
            "riddle_answer": answer,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        logger.error(f"Error posting riddle: {e}", exc_info=True)

def post_challenge_of_the_day(client):
    if not CHALL_LIST:
        logger.warning("No challenges loaded. Skipping challenge post.")
        return
    challenge = random.choice(CHALL_LIST)
    challenge_text = f"Challenge time: {challenge}\nShare your thoughts! #Chatty #Challenge"
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
    story_text = f"Story Time: {text}\nWhat happens next? #Chatty #Story"
    try:
        tweet_text = construct_tweet(story_text)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Story post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting story update: {e}", exc_info=True)

def post_to_twitter(client, post_count, force_image=False):
    try:
        # Use our specialized themed post
        text_content = generate_themed_post()
        expanded_text = expand_post_with_examples(text_content)
        if is_too_similar_to_recent_tweets(expanded_text, similarity_threshold=0.95, lookback=10):
            logger.warning("Expanded post is too similar to a recent tweet. Using fallback instead.")
            expanded_text = "Exciting times with OpenAI ChatGPT and Chatty meme coin! Stay tuned! 🤖🚀"
        tweet_text = construct_tweet(expanded_text)
        tweet_text = safe_truncate(tweet_text, 280)
        logger.info("Including image in this post (every post).")
        inferred_action = auto_infer_action_from_text(expanded_text)
        # Build scene content with our specialized theme
        scene_content = create_scene_content(expanded_text, action=inferred_action)
        img_prompt = create_simplified_image_prompt(scene_content)
        img_url = generate_image(img_prompt, max_length=MAX_PROMPT_LENGTH_TWITTER)
        image_path = None
        if img_url:
            image_path = download_image(img_url, img_prompt)
        else:
            logger.warning("Failed to generate image. Skipping image for this post.")
        if image_path:
            try:
                auth = tweepy.OAuth1UserHandler(
                    consumer_key=os.getenv("TWITTER_API_KEY"),
                    consumer_secret=os.getenv("TWITTER_API_SECRET"),
                    access_token=os.getenv("TWITTER_OAUTH1_ACCESS_TOKEN"),
                    access_token_secret=os.getenv("TWITTER_OAUTH1_ACCESS_SECRET")
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
        cleanup_images(os.getenv("IMAGE_DIR", "generated_images"), max_files=100)
        post_count += 1
        return post_count
    except Exception as e:
        logger.error(f"Unexpected error in post_to_twitter: {e}", exc_info=True)
        return post_count

###############################################################################
# REPLYING TO MENTIONS
###############################################################################
def handle_comment_with_context(user_id, comment, tweet_id=None, parent_id=None):
    if not check_rate_limit(user_id):
        return "Please wait a bit before sending more requests. 🤖"
    if not is_safe_to_respond(comment) or not moderate_content(comment):
        logger.info(f"Skipping unsafe/filtered comment: {comment}")
        return "I’m here to discuss AI, ChatGPT, and Chatty meme coin topics! 🚀✨"
    link_reply = check_link_inquiry(comment)
    if link_reply:
        return link_reply
    deflection = deflect_unrelated_comments(comment)
    if deflection:
        return deflection
    faq = static_response(comment)
    if faq:
        return faq
    if parent_id:
        parent_doc = posted_tweets_collection.find_one({"tweet_id": parent_id})
        if parent_doc and "riddle_answer" in parent_doc:
            correct_answer = parent_doc["riddle_answer"]
            if correct_answer:
                if is_guess_correct(comment, correct_answer, threshold=80):
                    return "That's correct! ⭐️ Great job solving the puzzle!"
                else:
                    return "Not quite right, try another guess! 🤔"
    full_convo = ""
    if parent_id:
        full_convo = build_conversation_path(parent_id)
    short_summary = summarize_text(full_convo)
    user_message = (
        f"Conversation so far (summary):\n{short_summary}\n\n"
        f"User says: {comment}"
    )
    bot_reply = generate_sentiment_aware_response(user_message)
    bot_reply = moderate_bot_output(bot_reply)
    log_response(comment, bot_reply)
    if tweet_id:
        posted_tweets_collection.update_one(
            {"tweet_id": tweet_id},
            {"$set": {"text": comment, "parent_id": parent_id}},
            upsert=True
        )
    store_user_memory(user_id, bot_reply)
    try:
        emb = generate_embedding(bot_reply)
        if emb:
            from config_and_setup import embeddings_collection
            embeddings_collection.insert_one({
                "conversation_context": bot_reply,
                "embedding": emb,
                "timestamp": datetime.utcnow()
            })
            logger.info("Stored embedding for semantic search.")
    except Exception as e:
        logger.error(f"Error storing embedding: {e}", exc_info=True)
    # Append a question about OpenAI/ChatGPT to further reinforce the theme.
    return f"{bot_reply} 🤖✨ What do you think about the latest from OpenAI ChatGPT?"

def respond_to_mentions(client, since_id):
    me = client.get_me()
    bot_user_id = me.data.id
    logger.info(f"Bot User ID: {bot_user_id}")
    params = {
        'expansions': ['author_id', 'in_reply_to_user_id', 'referenced_tweets.id'],
        'tweet_fields': ['id','author_id','conversation_id','in_reply_to_user_id','referenced_tweets','text','created_at'],
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
        if mentions_collection.find_one({'tweet_id': mention_id}):
            logger.info(f"Already responded to mention {mention_id}. Skipping.")
            continue
        if author_id == bot_user_id:
            logger.info(f"Skipping mention {mention_id} from self.")
            continue
        if not is_safe_to_respond(mention.text):
            logger.info(f"Skipped mention due to prohibited content: {mention.text}")
            continue
        parent_id = None
        if mention.referenced_tweets:
            parent_id = mention.referenced_tweets[0].id
        mention_time = mention.created_at.replace(tzinfo=None)
        reply_text = handle_incoming_message(
            user_id=author_id,
            user_text=mention.text,
            user_name=username,
            comment_time=mention_time
        )
        if reply_text is None:
            logger.info(f"Skipping mention {mention_id} because it's older than deployment.")
            continue
        full_reply = f"@{username} {reply_text}"
        final_reply = safe_truncate_by_sentence_no_ellipsis(full_reply, max_len=260, conclusion="Stay curious! ⭐️")
        logger.debug(f"Reply Text: {final_reply}")
        try:
            client.create_tweet(text=final_reply, in_reply_to_tweet_id=mention_id)
            logger.info(f"Replied to mention {mention_id}")
            mentions_collection.insert_one({'tweet_id': mention_id, 'replied_at': datetime.utcnow()})
        except Exception as e:
            logger.error(f"Error replying to mention {mention_id}: {e}", exc_info=True)
    return new_since_id

###############################################################################
# SCHEDULED TASKS FOR TWITTER
###############################################################################
def posting_task(client, post_count):
    logger.info("Starting scheduled posting task.")
    new_count = post_to_twitter(client, post_count)
    save_post_count("post_count.txt", new_count)
    return new_count

def mention_checking_task(client, since_id):
    logger.info("Starting mention checking task.")
    new_id = respond_to_mentions(client, since_id)
    if new_id != since_id:
        save_since_id("since_id.txt", new_id)
        return new_id
    return since_id

def schedule_posting(client, post_count):
    hours = random.choice([10, 12])
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
