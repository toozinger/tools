# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:38:19 2024

@author: dowdt
"""
from flask import Flask


import openai
import requests
from PIL import Image
import io

import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import concurrent.futures
import re


flask_app = Flask(__name__)

# %% Setup and tokens
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize slack bot
slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=slack_bot_token)

# Initialize the Slack app
lunch_app_token = os.environ.get("LUNCH_APP_TOKEN")
app = App(token=slack_bot_token)

openAI_api_key = os.environ.get("OPENAI_API_KEY")

# %% Functions


def make_slack_title_safe(title):
    # Replace newlines and tabs with spaces
    title = re.sub(r'[\n\t]+', ' ', title)

    # Remove control characters
    title = re.sub(r'[\x00-\x1F\x7F]', '', title)

    # Remove any potential HTML or script tags
    title = re.sub(r'<[^>]*>', '', title)

    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title)

    # Trim the title to 1000 characters
    title = title[:1000]

    # Strip leading and trailing whitespace
    title = title.strip()

    return title


def slack_notify(message, channel_id=None, image_url=None, **kwargs):
    """Send a notification message to a Slack channel using a bot.

    Parameters
    ----------
        message (str): The text of the message to send.
        image_url (list of str): A list of image file paths (optional).

    """

    if channel_id is None:
        channel_id = "C07FZ0W13R7"

    logger.debug(f"Slack message: {message}, channel_id: {channel_id}\n"
                 f"image paths: {image_url}")

    # Send the main text part of the message
    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        if response["ok"]:
            logger.debug("Slack message sent successfully.")
            logger.info(f"Slack message sent:\n{message}")
        else:
            logger.warn("Failed to send message.")
    except Exception as e:
        logger.warning(f"Error sending message: {str(e)}")

    # Send each of the images in their own messages
    if image_url:
        try:
            # Download the image from the URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raises an HTTPError for bad requests

            # Get the filename from the URL
            filename = image_url.split('/')[-1]

            # Create a BytesIO object from the image content
            image_data = io.BytesIO(response.content)

            # Upload the image to Slack
            response = client.files_upload_v2(
                channel=channel_id,
                file=image_data,
                filename=filename,
                title=make_slack_title_safe(message)
            )

            if response["ok"]:
                logger.info("Slack message with image sent successfully")
            else:
                logger.warning("Failed to send slack message with image.")
        except requests.RequestException as e:
            logger.warning(f"Error downloading image from URL: {str(e)}")
        except SlackApiError as e:
            logger.warning(f"Error sending image: {e.response['error']}")
        except Exception as e:
            logger.warning(f"Error: {str(e)}")

# %% image generator


class ImageGenerator:
    def __init__(self):
        self.preprocessed_prompt = None
        openai.api_key = openAI_api_key

    def preprocess_prompt(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role":
                 "system",
                 "content":
                     "You are a helpful assistant that improves image generation prompts for dalle3."},
                {"role":
                 "user",
                 "content":
                     f"Improve this image generation prompt: {prompt}. "
                     f"ONLY output the improved prompt. "
                     f"If the given prompt does NOT describe an image, "
                     f"return the words: 'DENIED'"}
            ]
        )

        processed_response = response.choices[0].message['content'].strip()

        if "denied" not in processed_response.lower():

            logging.warning(f"Prompt input: '{prompt}'\n"
                            f"Modified prompt: '{processed_response}'")

            self.preprocessed_prompt = processed_response
            return True
        else:
            logging.warning("Failed to preprocess prompt")
            return False

    def generate_image(self):

        if self.preprocessed_prompt is None:
            logging.warning("Must pre-process prompt first")
            return

        logger.info("Processing image prompt...")
        res = openai.Image.create(
            model="dall-e-3",
            prompt=self.preprocessed_prompt,
            n=1,
        )
        image_url = res["data"][0]["url"]

        logger.info(f"Image generated, url: {image_url}")
        return image_url

    def display_image(self, url):
        response = requests.get(url)
        image_data = io.BytesIO(response.content)
        image = Image.open(image_data)
        image.show()

        logging.warning("Image displayed")

# %% /Lunch callback functions


def lunch_callback(user_input, channel_id):
    logger.info("Running lunch callback")

    generator = ImageGenerator()

    success = generator.preprocess_prompt(user_input)

    if not success:
        slack_notify("Unable to generate request. Request MUST be food based",
                     channel_id=channel_id)
        return

    image_url = generator.generate_image()
    # generator.display_image(image_url)

    slack_notify(f"{generator.preprocessed_prompt}",
                 channel_id=channel_id,
                 image_url=image_url)


@app.command("/lunch")
def handle_plot_command(ack, command, say):
    ack()

    logger.info("Received /lunch command")
    try:
        user_input = command['text']
        channel_id = command['channel_id']
        logger.info(f"Lunch request received. Processing")
        say(f"You requested image with prompt: '{user_input}'. "
            "Processing your request")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lunch_callback, user_input, channel_id)
            future.add_done_callback(
                lambda x: logger.info("Plotter callback completed")
            )
    except Exception as e:
        logger.error(f"Error handling command: {str(e)}")
        say("Sorry, an error occurred while processing your command.")


@flask_app.route('/')
def home():
    return "Slack Bot is running!"


if __name__ == "__main__":
    handler = SocketModeHandler(app, lunch_app_token)
    handler.start()
    flask_app.run(host='0.0.0.0', port=8080)
