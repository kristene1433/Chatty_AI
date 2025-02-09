# # twitter_chatty.py

import re
import os
import random
import traceback
import tweepy
import schedule
from datetime import datetime

from config_and_setup import (
    logger,  # shared logger
    mentions_collection,  # DB collections from your config
    posted_tweets_collection  # so we can store riddle answers & retrieve them
)

###############################################################################
# IMPORTS FROM CHATTY_CORE
###############################################################################
from chatty_core import (
    # Core utility & AI functions
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
    safe_truncate,  # original truncation
    safe_truncate_by_sentence_no_ellipsis,  # sentence-based truncation
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
    # Sentiment & positivity
    generate_sentiment_aware_response,
    ensure_positive_tone,
    detect_sentiment_and_subjectivity,
    # Model constants
    ADVANCED_MODEL,
    # NEW: Fuzzy matching function
    is_guess_correct,
    # Data for riddles, challenges, etc.
    RIDDLES_LIST,
    CHALL_LIST,
    PERSONAS_LIST,
    STORY_LIST,
    is_too_similar_to_recent_tweets,
    # NAMED CONSTANT for prompt length
    MAX_PROMPT_LENGTH_TWITTER,
    # NEW LINK-INQUIRY LOGIC
    check_link_inquiry
)

###############################################################################
# TWITTER AUTHENTICATION
###############################################################################
def authenticate_twitter_client(
    api_key, api_secret, access_token, access_secret, bearer_token
):
    """
    Creates and returns a tweepy.Client instance for Twitter v2 API.
    """
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
# BASIC FILE I/O FOR PERSISTENCE (POST COUNTS, SINCE_ID, ETC.)
###############################################################################
def load_post_count(file_name):
    """
    Reads an integer post_count from file_name, or returns 0 if invalid/missing.
    """
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
    """
    Writes the integer 'count' to file_name.
    """
    with open(file_name, 'w') as f:
        f.write(str(count))

def load_since_id(file_name):
    """
    Reads an integer 'since_id' from a file, or returns None if invalid/missing.
    Used to track the last mention the bot responded to.
    """
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
    """
    Writes the integer 'since_id' to a file for tracking mention replies.
    """
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
    """
    Replaces generic memecoin and dogecoin mentions with 'Chatty memecoin'
    and updates hashtag references accordingly.
    """
    # Replace any occurrence of "memecoin" with "Chatty memecoin"
    normalized = re.sub(r'\bmemecoin\b', 'Chatty memecoin', text, flags=re.IGNORECASE)
    # Replace any occurrence of "dogecoin" with "Chatty memecoin"
    normalized = re.sub(r'\bdogecoin\b', 'Chatty memecoin', normalized, flags=re.IGNORECASE)
    # Replace hashtag references such as "#memecoin" or "#dogecoin" with "#CHATTY"
    normalized = re.sub(r'#\s*memecoin', '#CHATTY', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'#\s*dogecoin', '#CHATTY', normalized, flags=re.IGNORECASE)
    return normalized

###############################################################################
# BUILDING & POSTING TWEETS
###############################################################################
def construct_tweet(text_content):
    """
    Builds the final tweet text by optionally appending hashtags,
    mention to your main handle, etc. Then returns the raw text.
    This function now also normalizes any generic memecoin references
    to use only "Chatty memecoin".
    """
    text_content = text_content.strip().strip('"').strip("'")

    # EARLY TRUNCATION to ~240 so we have space for extra tags
    text_content = text_content[:240].rstrip()

    # Optionally remove existing hashtags
    no_hashtags = re.sub(r"#\w+", "", text_content).strip()

    extra_tags_pool = [
        "#HeyChatty", "#AIforEveryone", "#MEMECOIN", "$CHATTY",
        "#AImeme", "#AIagent"
    ]
    pick = random.choice(extra_tags_pool)
    tags = ["@chattyonsolana", pick]

    final_text = f"{no_hashtags} {' '.join(tags)}"
    # Now do a final safe truncate to 280
    final_text = safe_truncate(final_text, 280)
    
    # Normalize memecoin references to ensure only "Chatty memecoin" is used
    final_text = normalize_memecoin_references(final_text)
    
    return final_text

def post_daily_persona(client):
    """
    Example function that picks a random persona from PERSONAS_LIST and posts it.
    Uses GPT-4 (ADVANCED_MODEL) for creative text.
    """
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
        # Use the advanced model
        completion = robust_chat_completion(
            messages,
            model=ADVANCED_MODEL,
            max_tokens=200,
            temperature=0.8
        )
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
    """
    Picks a random riddle from RIDDLES_LIST and posts it as a tweet.
    Stores the riddle answer in posted_tweets_collection so we can 
    later compare user guesses with is_guess_correct.
    """
    if not RIDDLES_LIST:
        logger.warning("No riddles loaded. Skipping riddle post.")
        return

    riddle = random.choice(RIDDLES_LIST)
    question = riddle.get("question", "What's the puzzle?")
    answer = riddle.get("answer", None)  # <--- For fuzzy matching
    riddle_text = f"Puzzle Time: {question}\nReply with your guess! #chatty #PuzzleTime"

    try:
        tweet_text = construct_tweet(riddle_text)
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Riddle post Tweet ID: {response.data['id']}")

        # Store riddle answer for later fuzzy matching when users reply.
        posted_tweets_collection.insert_one({
            "tweet_id": response.data['id'],
            "text": riddle_text,
            "riddle_answer": answer,
            "timestamp": datetime.utcnow()
        })

    except Exception as e:
        logger.error(f"Error posting riddle: {e}", exc_info=True)

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
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Challenge post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting challenge: {e}", exc_info=True)

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
        safe_text = safe_truncate(tweet_text, 280)
        response = client.create_tweet(text=safe_text)
        logger.info(f"Story post Tweet ID: {response.data['id']}")
    except Exception as e:
        logger.error(f"Error posting story update: {e}", exc_info=True)

def post_to_twitter(client, post_count, force_image=False):
    """
    Example "auto-post" function that picks a random theme,
    writes a short GPT-based post, optionally includes an image, etc.
    """
    try:
        theme = random.choice(themes_list)
        logger.info(f"Randomly chosen theme: {theme}")

        text_content = generate_themed_post(theme)
        expanded_text = expand_post_with_examples(text_content)

        # Check if it's too similar to recent tweets
        if is_too_similar_to_recent_tweets(expanded_text, similarity_threshold=0.95, lookback=10):
            logger.warning("Expanded post is too similar to a recent tweet. Using fallback instead.")
            expanded_text = "Exciting times in AI! Stay tuned, #AICommunity 🤖🚀"

        # Build final tweet text
        tweet_text = construct_tweet(expanded_text)
        tweet_text = safe_truncate(tweet_text, 280)

        logger.info("Including image in this post (every post).")
        inferred_action = auto_infer_action_from_text(expanded_text)
        scene_content = create_scene_content(theme, action=inferred_action)
        img_prompt = create_simplified_image_prompt(scene_content)

        # Use the 8000-character prompt limit for Twitter
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

                # Clean up local image file
                try:
                    os.remove(image_path)
                    logger.info(f"Deleted local image file: {image_path}")
                except Exception as e:
                    logger.error(f"Error deleting image file: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Error posting tweet with image: {e}", exc_info=True)
                traceback.print_exc()
        else:
            # No image fallback
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
    """
    Build a GPT-based reply to a mention or comment, referencing conversation history
    from posted tweets, but also checking if the mention is a guess at a riddle, etc.
    """

    # 1. Rate limit check
    if not check_rate_limit(user_id):
        return "Please wait a bit before sending more requests. 🤖"

    # 2. Filter out prohibited/spammy content
    if not is_safe_to_respond(comment) or not moderate_content(comment):
        logger.info(f"Skipping unsafe/filtered comment: {comment}")
        return "I’m here to discuss AI and technology topics! 🚀✨"

    # 3. # NEW LINK-INQUIRY LOGIC (check if user asked about Telegram, website, or X)
    link_reply = check_link_inquiry(comment)
    if link_reply:
        return link_reply

    # 4. Possibly deflect if off-topic
    deflection = deflect_unrelated_comments(comment)
    if deflection:
        return deflection

    # 5. Check static FAQ
    faq = static_response(comment)
    if faq:
        return faq

    # 6. If this mention is replying to a riddle tweet, do a fuzzy guess check
    if parent_id:
        parent_doc = posted_tweets_collection.find_one({"tweet_id": parent_id})
        if parent_doc and "riddle_answer" in parent_doc:
            correct_answer = parent_doc["riddle_answer"]
            if correct_answer:
                if is_guess_correct(comment, correct_answer, threshold=80):
                    return "That's correct! ⭐️ Great job solving the puzzle!"
                else:
                    return "Not quite right, try another guess! 🤔"
            # else continue

    # 7. Summarize conversation from any parent tweet
    full_convo = ""
    if parent_id:
        full_convo = build_conversation_path(parent_id)
    short_summary = summarize_text(full_convo)

    # 8. Combine summary + user message, then generate a sentiment-aware response
    user_message = (
        f"Conversation so far (summary):\n{short_summary}\n\n"
        f"User says: {comment}"
    )
    bot_reply = generate_sentiment_aware_response(user_message)

    # 9. Final moderation check on the bot reply
    bot_reply = moderate_bot_output(bot_reply)

    # 10. Log to file
    log_response(comment, bot_reply)

    # 11. If we have tweet_id, store the chain in posted_tweets_collection
    if tweet_id:
        posted_tweets_collection.update_one(
            {"tweet_id": tweet_id},
            {"$set": {
                "text": comment,
                "parent_id": parent_id
            }},
            upsert=True
        )

    # 12. Store user memory & embeddings
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

    return bot_reply

def respond_to_mentions(client, since_id):
    """
    Fetch new mentions since 'since_id', generate replies using handle_comment_with_context(),
    then post the replies. Returns the new 'since_id' after processing.

    We'll apply sentence-based truncation to keep replies from cutting off mid-sentence.
    """
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

        # Check if we've already responded to this mention
        if mentions_collection.find_one({'tweet_id': mention_id}):
            logger.info(f"Already responded to mention {mention_id}. Skipping.")
            continue

        # Skip if from self
        if author_id == bot_user_id:
            logger.info(f"Skipping mention {mention_id} from self.")
            continue

        # Another quick spam check
        if not is_safe_to_respond(mention.text):
            logger.info(f"Skipped mention due to prohibited content: {mention.text}")
            continue

        parent_id = None
        if mention.referenced_tweets:
            parent_id = mention.referenced_tweets[0].id

        reply_text = handle_comment_with_context(
            user_id=author_id,
            comment=mention.text,
            tweet_id=mention_id,
            parent_id=parent_id
        )
        if reply_text:
            # Combine with @username
            full_reply = f"@{username} {reply_text}"

            # Use sentence-based truncation for replies
            final_reply = safe_truncate_by_sentence_no_ellipsis(
                full_reply, 
                max_len=260, 
                conclusion="Stay curious! ⭐️"
            )

            logger.debug(f"Reply Text: {final_reply}")
            try:
                client.create_tweet(text=final_reply, in_reply_to_tweet_id=mention_id)
                logger.info(f"Replied to mention {mention_id}")
                mentions_collection.insert_one({'tweet_id': mention_id, 'replied_at': datetime.utcnow()})
            except Exception as e:
                logger.error(f"Error replying to mention {mention_id}: {e}", exc_info=True)
        else:
            logger.warning(f"Failed to generate response for mention {mention_id}.")

    return new_since_id

###############################################################################
# SCHEDULED TASKS FOR TWITTER
###############################################################################
def posting_task(client, post_count):
    """
    A scheduled job that calls post_to_twitter(), then saves the updated post_count.
    """
    logger.info("Starting scheduled posting task.")
    new_count = post_to_twitter(client, post_count)
    save_post_count("post_count.txt", new_count)
    return new_count

def mention_checking_task(client, since_id):
    """
    A scheduled job that checks mentions, replies as needed, then saves the new since_id.
    """
    logger.info("Starting mention checking task.")
    new_id = respond_to_mentions(client, since_id)
    if new_id != since_id:
        save_since_id("since_id.txt", new_id)
        return new_id
    return since_id

def schedule_posting(client, post_count):
    """
    Example: schedule random auto-posting every 3 or 4 hours.
    """
    hours = random.choice([10, 12])
    schedule.every(hours).hours.do(posting_task, client=client, post_count=post_count)
    logger.info(f"Scheduled posting to run every {hours} hours.")

def schedule_mention_checking(client, since_id):
    """
    Example: schedule mention-checking every hour.
    """
    schedule.every(1).hours.do(mention_checking_task, client=client, since_id=since_id)
    logger.info("Scheduled mention checking every 1 hour.")

def schedule_daily_persona(client):
    """
    Schedules daily persona post at 10:00 AM.
    """
    schedule.every().day.at("10:00").do(post_daily_persona, client=client)

def schedule_riddle_of_the_day(client):
    """
    Schedules riddle posting at 16:00 (4 PM).
    """
    schedule.every().day.at("16:00").do(post_riddle_of_the_day, client=client)

def schedule_daily_challenge(client):
    """
    Schedules challenge posting at 12:00 (noon).
    """
    schedule.every().day.at("12:00").do(post_challenge_of_the_day, client=client)

def schedule_storytime(client):
    """
    Schedules a story snippet at 18:00 (6 PM).
    """
    schedule.every().day.at("18:00").do(post_story_update, client=client)
