# main_telegram.py

from chatty_telegram import run_telegram_bot
from config_and_setup import logger

def main():
    logger.info("Starting Chatty Telegram Bot with group admin & scheduled posts...")
    run_telegram_bot()

if __name__ == "__main__":
    main()
