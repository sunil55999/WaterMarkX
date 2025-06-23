import json
import logging
import os
import subprocess
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import requests
from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.errors import FloodWait, RPCError
import asyncio
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "api_id": "YOUR_API_ID",
    "api_hash": "YOUR_API_HASH",
    "bot_token": "YOUR_BOT_TOKEN",
    "source_chat_id": -1001234567890,  # Source channel/chat ID
    "target_chat_id": -1009876543210,  # Target channel ID
    "input_dir": "images/input",
    "mask_dir": "images/masks",
    "output_dir": "images/output",
    "mask_x": -300,  # Bottom-right corner (relative to width)
    "mask_y": -60,   # Bottom-right corner (relative to height)
    "mask_width": 300,
    "mask_height": 60,
    "allowed_chats": [-1001234567890],  # Optional: restrict to specific chats
    "max_retries": 3,
    "retry_delay": 5
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logger.warning(f"Created default config at {CONFIG_PATH}. Please update it.")
        raise SystemExit
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

config = load_config()

# Create directories
for dir_path in [config["input_dir"], config["mask_dir"], config["output_dir"]]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Initialize Pyrogram client
app = Client(
    "watermark_remover_bot",
    api_id=config["api_id"],
    api_hash=config["api_hash"],
    bot_token=config["bot_token"]
)

async def download_image(message: Message, retry_count=0):
    """Download an image from a message with retries."""
    try:
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{message.id}.jpg"
        input_path = os.path.join(config["input_dir"], file_name)
        await message.download(file_path=input_path)
        if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
            logger.info(f"Downloaded image to {input_path}")
            return input_path
        else:
            logger.error(f"Failed to download image: Empty or missing file {input_path}")
            return None
    except (FloodWait, RPCError) as e:
        if retry_count < config["max_retries"]:
            wait_time = config["retry_delay"] * (2 ** retry_count)
            logger.warning(f"Download error: {e}. Retrying in {wait_time}s (Attempt {retry_count+1}/{config['max_retries']})")
            await asyncio.sleep(wait_time)
            return await download_image(message, retry_count + 1)
        logger.error(f"Max retries reached for download: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected download error: {e}")
        return None

def create_mask(image_path: str, output_mask_path: str):
    """Create a binary mask for the watermark area."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        height, width = img.shape[:2]
        
        # Calculate mask position (relative to bottom-right)
        x = width + config["mask_x"] if config["mask_x"] < 0 else config["mask_x"]
        y = height + config["mask_y"] if config["mask_y"] < 0 else config["mask_y"]
        mask_width = config["mask_width"]
        mask_height = config["mask_height"]
        
        # Ensure mask stays within image bounds
        x = max(0, min(x, width - mask_width))
        y = max(0, min(y, height - mask_height))
        
        # Create white mask on black background
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y:y+mask_height, x:x+mask_width] = 255
        cv2.imwrite(output_mask_path, mask)
        logger.info(f"Created mask at {output_mask_path}")
        return output_mask_path
    except Exception as e:
        logger.error(f"Mask creation error: {e}")
        return None

def remove_watermark(input_path: str, mask_path: str, output_path: str, retry_count=0):
    """Run IOPaint to remove watermark using the mask."""
    try:
        cmd = [
            "iopaint", "run",
            "--model", "mig",
            "--device", "cpu",
            "--image", input_path,
            "--mask", mask_path,
            "--output", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Watermark removed, saved to {output_path}")
            return output_path
        else:
            logger.error(f"IOPaint failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("IOPaint timed out")
        return None
    except Exception as e:
        if retry_count < config["max_retries"]:
            wait_time = config["retry_delay"] * (2 ** retry_count)
            logger.warning(f"IOPaint error: {e}. Retrying in {wait_time}s (Attempt {retry_count+1}/{config['max_retries']})")
            time.sleep(wait_time)
            return remove_watermark(input_path, mask_path, output_path, retry_count + 1)
        logger.error(f"Max retries reached for IOPaint: {e}")
        return None

async def forward_image(output_path: str):
    """Forward the cleaned image to the target channel."""
    try:
        await app.send_photo(config["target_chat_id"], output_path)
        logger.info(f"Forwarded cleaned image to chat {config['target_chat_id']}")
    except Exception as e:
        logger.error(f"Failed to forward image: {e}")

@app.on_message(filters.photo & filters.chat(config["allowed_chats"]))
async def handle_image(client: Client, message: Message):
    """Handle incoming image messages."""
    logger.info(f"Received image from chat {message.chat.id}, message ID {message.id}")
    
    # Download image
    input_path = await download_image(message)
    if not input_path:
        return
    
    # Create mask
    mask_file = f"mask_{os.path.basename(input_path)}"
    mask_path = os.path.join(config["mask_dir"], mask_file)
    mask_path = create_mask(input_path, mask_path)
    if not mask_path:
        return
    
    # Remove watermark
    output_file = f"cleaned_{os.path.basename(input_path)}"
    output_path = os.path.join(config["output_dir"], output_file)
    output_path = remove_watermark(input_path, mask_path, output_path)
    if not output_path:
        return
    
    # Forward cleaned image
    await forward_image(output_path)

@app.on_message(filters.command("ping"))
async def ping_command(client: Client, message: Message):
    """Respond to /ping command."""
    await message.reply("Pong! Bot is running.")
    logger.info(f"Ping command received from user {message.from_user.id}")

async def main():
    """Start the bot."""
    try:
        await app.start()
        logger.info("Bot started successfully")
        await asyncio.Event().wait()  # Keep bot running
    except Exception as e:
        logger.error(f"Bot startup error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
