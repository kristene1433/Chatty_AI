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
Chatty is a delightful, retro-styled character blending classic computers with modern AI features:

1) SCREEN FACE:
   - Vibrant bright blue screen with large, friendly eyes.
   - Subtle shine or reflections for a lifelike effect.
   - Cheerful, dynamic smile; slight blush on cheeks.

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


