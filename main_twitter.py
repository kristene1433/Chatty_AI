# main_twitter.py

import time
import traceback
import schedule

from config_and_setup import (
    logger,
    API_KEY, API_SECRET,
    ACCESS_TOKEN, ACCESS_SECRET,
    BEARER_TOKEN
)
from chatty_twitter import (
    authenticate_twitter_client,
    load_post_count,
    save_post_count,
    load_since_id,
    save_since_id,
    schedule_posting,
    # schedule_mention_checking,
    post_to_twitter
    # If you want daily persona/riddle/challenge/story:
    # schedule_daily_persona,
    # schedule_riddle_of_the_day,
    # schedule_daily_challenge,
    # schedule_storytime,
)


def main():
    logger.info("Starting Chatty Twitter Bot...")

    # 1) Authenticate with Twitter
    client = authenticate_twitter_client(
        api_key=API_KEY,
        api_secret=API_SECRET,
        access_token=ACCESS_TOKEN,
        access_secret=ACCESS_SECRET,
        bearer_token=BEARER_TOKEN
    )
    if not client:
        logger.error("Could not authenticate Twitter client. Exiting.")
        return

    # 2) Load post_count & since_id
    post_count = load_post_count("post_count.txt")
    since_id = load_since_id("since_id.txt")

    # 3) Immediately post when first deployed
    logger.info("Posting immediately upon startup...")
    post_count = post_to_twitter(client, post_count)
    save_post_count("post_count.txt", post_count)

    # 4) Schedule tasks (random posting, mention checking, etc.)
    schedule_posting(client, post_count)
    #schedule_mention_checking(client, since_id)
    # schedule_daily_persona(client)
    # schedule_riddle_of_the_day(client)
    # schedule_daily_challenge(client)
    # schedule_storytime(client)

    logger.info("Entering main loop for scheduled tasks...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # Check the schedule tasks every 30s
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    main()

