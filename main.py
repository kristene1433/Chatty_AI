import time
import traceback
import schedule

# Pull in everything needed
from chatty_functions import (
    authenticate_twitter_client,
    load_post_count, save_post_count,
    load_since_id, save_since_id,
    store_chatty_config, post_to_twitter,
    posting_task, mention_checking_task,
    schedule_posting, schedule_mention_checking,
    # newly added:
    schedule_daily_persona, schedule_riddle_of_the_day
)
from config_and_setup import logger

def main():
    logger.info("AI bot is starting...")

    # 1) Authenticate with Twitter
    client = authenticate_twitter_client()
    if not client:
        logger.error("Could not authenticate Twitter client. Exiting.")
        return

    # 2) Load post_count & since_id
    post_count = load_post_count("post_count.txt")
    since_id = load_since_id("since_id.txt")

    # 3) Store a bullet-point Chatty persona in "BaseChatty" (for images)
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
    - Rounded legs that transition into colorful pixel-art sneakers, easily adapted to match various settings.
    - Each sneaker boasts pixel-perfect laces, radiant highlights, and bold color blocks, enhancing Chatty’s friendly vibe.

5) OVERALL STYLE & ADAPTABILITY:
    - A polished pixel-art look that fuses a cheerful, cartoon-like quality with retro-futuristic elements.
    - Bright, celebratory pixel details (such as confetti or glowing accessories) can be introduced depending on the theme.
    - Whether in a bustling tech cityscape, a dreamy fantasy realm, or a playful arcade setting, Chatty’s design can be seamlessly integrated. Its versatile look ensures it remains a captivating guide, helper, or companion in any envisioned scene.
"""

    store_chatty_config("BaseChatty", chatty_persona_text)

    # 4) Post an immediate tweet WITH an image (once on startup)
    logger.info("Posting an immediate tweet WITH an image.")
    post_count = post_to_twitter(client, post_count, force_image=True)
    save_post_count("post_count.txt", post_count)

    # 5) Schedule your existing tasks
    schedule_posting(client, post_count)
    schedule_mention_checking(client, since_id)

    # 6) Schedule new content ideas:
    #    a) Daily persona at 10:00
    #    b) Riddle of the day at 16:00
    #    c) Daily challenge at 18:00
    #    d) Storytime at 20:00
    # schedule_daily_persona(client)
    # schedule_riddle_of_the_day(client)
    # schedule_daily_challenge(client)
    # schedule_storytime(client)

    # 7) Main loop: run scheduled tasks
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()



