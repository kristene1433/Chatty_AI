# chatty_telegram.py

import os
import time
import random
import logging
import schedule
import requests
import threading

from datetime import datetime
from PIL import Image, ImageDraw, ImageFont  # For meme text overlay

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    ChatMemberHandler, ContextTypes
)

from config_and_setup import (
    logger, TELEGRAM_BOT_TOKEN, memory_collection
)

###############################################################################
# IMPORTS FROM CHATTY_CORE
###############################################################################
from chatty_core import (
    # GPT calls
    robust_chat_completion,
    ADVANCED_MODEL,

    # Spam & content checks
    is_safe_to_respond,
    moderate_content,

    # Sentiment & positivity
    generate_sentiment_aware_response,
    ensure_positive_tone,
    detect_sentiment_and_subjectivity,

    # Greeting & memory logic
    store_user_memory,
    get_user_memory,
    handle_incoming_message,  # Uses GPT-4 logic automatically

    # Image utilities
    download_image,
    generate_image,  # We'll specify max_length=MAX_PROMPT_LENGTH_TELEGRAM for Telegram

    # Named constants
    MAX_PROMPT_LENGTH_TELEGRAM,

    # Themes & scene-building
    themes_list,
    create_scene_content,
    create_simplified_image_prompt
)

###############################################################################
# START & IMAGE COMMANDS
###############################################################################
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Greets the user with a short message (async for v20)."""
    await update.message.reply_text(
        "Hello! I'm Chatty_AI on Telegram. "
        "Type /meme [text] to create a comedic image, "
        "/image for a random AI pic, or just say hi!"
    )

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates an AI image using Chatty's random scene logic for Telegram."""
    chat_id = update.effective_chat.id

    # 1) Pick a random theme
    theme = random.choice(themes_list)
    # 2) Build a short scene description
    text_scene = create_scene_content(theme)
    # 3) Convert that scene text into a short DALL·E prompt
    prompt = create_simplified_image_prompt(text_scene)

    # 4) Generate the DALL·E image with Telegram's 3000-char limit
    image_url = generate_image(prompt, max_length=MAX_PROMPT_LENGTH_TELEGRAM)
    if not image_url:
        await update.message.reply_text("Oops, couldn't generate an image right now.")
        return

    # 5) Download the image locally
    filename = download_image(image_url, prompt="telegram_chatty")
    if not filename:
        await update.message.reply_text("Failed to download the image.")
        return

    # 6) Send the image back to the user
    caption_text = f"Chatty exploring a '{theme}'!"
    with open(filename, "rb") as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f, caption=caption_text)

    # 7) Cleanup local file
    try:
        os.remove(filename)
    except Exception:
        pass

###############################################################################
# GROUP ADMIN (Welcome) & SPAM DETECTION
###############################################################################
async def welcome_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcomes new users who join the group."""
    if update.message and update.message.new_chat_members:
        for member in update.message.new_chat_members:
            welcome_text = (
                f"Welcome, {member.full_name}! ⭐️👋\n\n"
                "I’m Chatty_AI, here to share positivity and AI insights. "
                "Type /start or /help for commands!"
            )
            await update.message.reply_text(welcome_text)

async def advanced_spam_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Enhanced spam detection:
      1) Check for known blocked keywords.
      2) If borderline, ask GPT (gpt-3.5) "Is this spam?"
      3) If GPT says yes, delete.
    """
    if not update.message:
        return

    text = update.message.text or ""

    # 1) Basic blocked phrases
    blocked_phrases = ["click here", "join now", "free money", "t.me/", "http://", "https://"]
    if any(bp in text.lower() for bp in blocked_phrases):
        try:
            await update.message.delete()  # MUST await in v20
            logger.info(f"[SpamFilter] Deleted obvious spam message: {text}")
        except Exception as e:
            logger.warning(f"[SpamFilter] Could not delete spam message: {e}")
        return

    # 2) If not obviously spam, check via GPT
    if await is_suspicious_text_gpt_3_5(text):
        try:
            await update.message.delete()
            logger.info(f"[GPT SpamFilter] Deleted GPT-classified spam: {text}")
        except Exception as e:
            logger.warning(f"[GPT SpamFilter] Could not delete GPT-flagged spam: {e}")
        return

async def is_suspicious_text_gpt_3_5(user_text: str) -> bool:
    """
    Calls GPT-3.5 with a prompt like "Is this message spam?"
    Return True if GPT indicates spam, False otherwise.
    """
    # This function can be sync or async. We'll do a quick approach:
    messages = [
        {"role": "system", "content": "You are a helpful AI content moderator."},
        {"role": "user", "content": (
            "Determine if the following user message is spam or not. "
            "Answer with only 'SPAM' or 'NOT SPAM'.\n\n"
            f"Message: {user_text}"
        )}
    ]
    resp = robust_chat_completion(messages, model=ADVANCED_MODEL, max_tokens=10, temperature=0.0)
    if resp and "choices" in resp:
        classification = resp["choices"][0]["message"]["content"].strip().upper()
        logger.info(f"[GPT Spam] Classification: {classification}")
        return classification.startswith("SPAM")
    return False

###############################################################################
# LEGACY USER-SPECIFIC MEMORY + SENTIMENT-AWARE REPLY
###############################################################################
async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Captures user messages, stores them, references them,
    and replies using generate_sentiment_aware_response() (GPT-3.5).
    """
    if not update.message:
        return

    user_id = update.effective_user.id
    user_text = update.message.text or ""

    store_user_message(user_id, user_text)
    user_history = get_user_history(user_id, limit=5)
    context_text = build_memory_context(user_history)

    combined_text = (
        f"Short conversation memory:\n{context_text}\n"
        f"User says: {user_text}"
    )
    bot_reply = generate_sentiment_aware_response(combined_text)
    await update.message.reply_text(bot_reply)

def build_memory_context(user_history: list) -> str:
    if not user_history:
        return "No recent messages stored."
    context_lines = []
    for doc in user_history:
        txt = doc["message"]
        context_lines.append(f"- {txt}")
    return "\n".join(context_lines)

def store_user_message(user_id: int, message: str):
    doc = {
        "user_id": user_id,
        "message": message,
        "timestamp": datetime.utcnow()
    }
    try:
        memory_collection.insert_one(doc)
        logger.info(f"Stored message for user {user_id}: {message}")
    except Exception as e:
        logger.error(f"Error storing user message: {e}", exc_info=True)

def get_user_history(user_id: int, limit=5):
    try:
        docs = list(
            memory_collection.find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return list(reversed(docs))
    except Exception as e:
        logger.error(f"Error fetching user history: {e}", exc_info=True)
        return []

###############################################################################
# MAIN GROUP MESSAGE HANDLER
###############################################################################
async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles group messages and responds only to specific triggers.
    It uses handle_incoming_message (GPT-4 logic).
    """
    if not update.message:
        return

    user_id = str(update.effective_user.id)
    user_text = update.message.text or ""
    user_name = update.effective_user.first_name or "friend"

    triggers = ["gm", "good morning", "gn", "good night", "chatty", "welcome", "to the moon"]

    if any(trigger in user_text.lower() for trigger in triggers):
        reply_text = handle_incoming_message(user_id, user_text, user_name=user_name)
        # handle_incoming_message is synchronous. It's fine if it returns a string.
        # We'll just await the telegram call:
        if reply_text:
            await update.message.reply_text(reply_text)
    else:
        logger.info(f"Message ignored: {user_text}")

###############################################################################
# AI MEME COMMAND
###############################################################################
async def meme_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /meme [text] -> Overlays user text onto a random meme template in 'templates/'.
    """
    if not update.message:
        return

    user_input = update.message.text
    text_to_overlay = user_input.replace("/meme", "").strip()
    if not text_to_overlay:
        text_to_overlay = "Chatty Meme Time!"

    meme_template_path = pick_random_template("templates")
    if not meme_template_path:
        await update.message.reply_text("No meme templates found on server!")
        return

    meme_file = overlay_text_on_meme(meme_template_path, text_to_overlay)
    if not meme_file:
        await update.message.reply_text("Failed to create meme. Check logs.")
        return

    chat_id = update.effective_chat.id
    with open(meme_file, "rb") as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f, caption="Here's your meme!")
    try:
        os.remove(meme_file)
    except:
        pass

def pick_random_template(folder_path: str) -> str:
    valid_extensions = (".jpg", ".jpeg", ".png")
    try:
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_extensions)
        ]
        if not files:
            return None
        chosen = random.choice(files)
        return os.path.join(folder_path, chosen)
    except Exception as e:
        logger.error(f"Error picking random template: {e}", exc_info=True)
        return None

def overlay_text_on_meme(template_path: str, text: str) -> str:
    """
    Overlays `text` onto the meme using Pillow, then saves as JPEG.
    """
    try:
        base_img = Image.open(template_path).convert("RGBA")
        draw = ImageDraw.Draw(base_img)

        W, H = base_img.size
        dynamic_font_size = max(20, W // 8)
        font_path = "fonts/Impact.ttf"
        if not os.path.exists(font_path):
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, dynamic_font_size)

        # Measure text bounding box
        test_bbox = draw.textbbox((0, 0), text, font=font)
        text_width  = test_bbox[2] - test_bbox[0]
        text_height = test_bbox[3] - test_bbox[1]

        # Position near bottom center
        text_x = (W - text_width) // 2
        text_y = H - text_height - 30

        # Semi-transparent rectangle behind text
        box_margin = 10
        rect_x1 = text_x - box_margin
        rect_y1 = text_y - box_margin
        rect_x2 = text_x + text_width + box_margin
        rect_y2 = text_y + text_height + box_margin
        draw.rectangle(
            [(rect_x1, rect_y1), (rect_x2, rect_y2)],
            fill=(0, 0, 0, 150)
        )

        # Outline
        outline_range = 2
        for x_off in range(-outline_range, outline_range + 1):
            for y_off in range(-outline_range, outline_range + 1):
                draw.text((text_x + x_off, text_y + y_off), text, font=font, fill=(0, 0, 0))

        # Main text in white
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

        # Convert RGBA -> RGB
        rgb_img = base_img.convert("RGB")
        out_file = "generated_meme.jpg"
        rgb_img.save(out_file, "JPEG")
        return out_file
    except Exception as e:
        logger.error(f"Error overlaying text on meme: {e}", exc_info=True)
        return None

###############################################################################
# ERROR HANDLER (optional but recommended)
###############################################################################
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Logs errors that happen within handlers. Helps avoid unhandled exceptions.
    """
    logger.exception("An error occurred:", exc_info=context.error)

###############################################################################
# SCHEDULING (Optional)
###############################################################################
def schedule_telegram_tasks(application):
    """
    Example scheduled job posting an AI-generated image
    to a group chat at a specified time each day.
    """
    schedule.every().day.at("10:00").do(send_scheduled_image, application=application)
    logger.info("Scheduled daily Chatty image at 10:00 AM.")

def send_scheduled_image(application):
    """
    Example scheduled job posting an AI-generated image
    to a specific group chat at 10:00 AM every day.
    """
    chat_id = -1001234567890  # replace with your actual group chat ID

    scene = random.choice(["AI orchard", "cosmic forest", "robotic greenhouse"])
    prompt = create_simplified_image_prompt(scene)
    image_url = generate_image(prompt, max_length=MAX_PROMPT_LENGTH_TELEGRAM)
    if not image_url:
        return

    filename = download_image(image_url, prompt="telegram_scheduled.jpg")
    if not filename:
        return

    with open(filename, "rb") as f:
        # Note this is synchronous because schedule is sync. It's okay for small tasks.
        application.bot.send_photo(chat_id=chat_id, photo=f, caption=f"Chatty at a {scene}!")
    try:
        os.remove(filename)
    except Exception:
        pass

def run_scheduling_loop():
    """Runs schedule in a separate thread so it doesn't block run_polling()."""
    while True:
        schedule.run_pending()
        time.sleep(60)

###############################################################################
# MAIN BOT RUNNER
###############################################################################
def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("No TELEGRAM_BOT_TOKEN set. Exiting.")
        return

    # Build the async Application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(CommandHandler("meme", meme_command))

    # Group Admin: welcome new members
    application.add_handler(ChatMemberHandler(welcome_new_member, ChatMemberHandler.CHAT_MEMBER))

    # Add error handler
    application.add_error_handler(error_handler)

    # Advanced spam detection (runs first => group 0)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, advanced_spam_filter), 0)

    # Then handle group messages (group 1)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_group_message), 1)

    # (Optional) If you still want the older approach:
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message), 2)

    # Start the schedule in a separate thread
    schedule_telegram_tasks(application)
    threading.Thread(target=run_scheduling_loop, daemon=True).start()

    # Run the bot (async)
    logger.info(
        "Chatty_AI Telegram bot starting (python-telegram-bot v20+). "
        "Handlers are async, GPT calls are sync, and positivity rules!"
    )
    application.run_polling()
