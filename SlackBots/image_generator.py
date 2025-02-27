import openai
import requests
from PIL import Image
import io

import os
import logging
from dotenv import load_dotenv, find_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import concurrent.futures
import re
import replicate

# %% Setup and tokens
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the environment variables from the .env file
load_dotenv(find_dotenv(filename="slackbot.env"))

# Get the tokens from the loaded environment variables
slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
lunch_app_token = os.getenv("LUNCH_APP_TOKEN")
openAI_api_key = os.getenv("OPENAI_API_KEY")
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")

# Initialize slack bot
client = WebClient(token=slack_bot_token)

# Initialize the Slack app
app = App(token=slack_bot_token)

# Set OpenAI API key
openai.api_key = openAI_api_key

# Set Replicate API token in the environment
os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
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
        logger.info("Attempting to send slack message.")
        response = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        logger.info(f"Slack API Response: {response}")
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
            logger.info(f"Attempting to download image from URL: {image_url}")
            # Download the image from the URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raises an HTTPError for bad requests
            logger.info("Image downloaded successfully.")

            # Get the filename from the URL
            filename = image_url.split('/')[-1]
            logger.info(f"Extracted filename: {filename}")

            # Create a BytesIO object from the image content
            image_data = io.BytesIO(response.content)
            logger.info("Created BytesIO object from image content.")

            # Upload the image to Slack
            logger.info("Attempting to upload image to Slack.")
            response = client.files_upload_v2(
                channel=channel_id,
                file=image_data,
                filename=filename,
                title=make_slack_title_safe(message)
            )
            logger.info(f"Slack File Upload API Response: {response}")

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

        self.raw_prompt = None
        self.preprocessed_prompt = None
        openai.api_key = openAI_api_key
        logger.info("ImageGenerator initialized.")

    import openai

    def preprocess_prompt(self, prompt):
        logger.info(f"Preprocessing prompt: {prompt}")
        self.raw_prompt = prompt
        try:
            logger.info("Calling OpenAI ChatCompletion API...")
            client = openai.OpenAI()  # Create an OpenAI client
            response = client.chat.completions.create(  # Use client.chat.completions.create
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that improves image generation prompts."},
                    {"role": "user",
                     "content": f"Improve this image generation prompt. RETAIN all information regarding:"
                     f"{prompt}.\n"
                     f"ONLY output the improved prompt."}
                ]
            )
            logger.info(f"OpenAI API Response: {response}")
            processed_response = response.choices[0].message.content.strip()

            if "denied" not in processed_response.lower():
                logger.info(f"Prompt input: '{prompt}'\n"
                            # Changed logging level from warning to info
                            f"Modified prompt: '{processed_response}'")
                self.preprocessed_prompt = processed_response
                logger.info(f"Preprocessed prompt successfully: {
                            self.preprocessed_prompt}")
                return True
            else:
                logging.warning("Failed to preprocess prompt")
                return False
        except Exception as e:
            logger.error(f"Error in preprocess_prompt: {e}")
            return False

    def generate_image(self, model="flux"):

        if self.preprocessed_prompt is None:
            logging.warning("Must pre-process prompt first")
            return

        logger.info("Processing image prompt...")

        if "flux" in model:

            replicate_api = replicate.Client(
                api_token=os.environ["REPLICATE_API_TOKEN"])

            logger.info("Calling Replicate API...")
            try:
                output = replicate_api.run(
                    "black-forest-labs/flux-schnell",
                    input={
                        "prompt": self.preprocessed_prompt,
                        "num_outputs": 1,
                        "aspect_ratio": "16:9",
                        "output_format": "png",
                        "output_quality": 90
                    }
                )
                logger.info(f"Replicate API Output: {output}")

                # image_url = output[0]
                # logger.info(f"Image URL from Replicate: {image_url}")
                image_url = str(output[0])  # convert FileOutput to a string
                logger.info(f"Image URL from Replicate: {image_url}")

            except Exception as e:
                logger.error(f"Error in Replicate API call: {e}")
                return None

        elif "dalle" in model:

            logger.info("Calling OpenAI Image API (DALL-E)...")
            try:
                res = openai.images.generate(  # Use client.images.generate
                    model="dall-e-3",
                    prompt=self.preprocessed_prompt,
                    n=1,
                )
                logger.info(f"OpenAI Image API Response: {res}")
                image_url = res.data[0].url
                logger.info(f"Image URL from DALL-E: {image_url}")
            except Exception as e:
                logger.error(f"Error in OpenAI Image API call: {e}")
                return None

        logger.info(f"Image generated, url: {image_url}")
        return image_url

    def display_image(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            image.show()

            logging.warning("Image displayed")
        except Exception as e:
            logger.error(f"Error displaying image: {e}")

# %% /Lunch callback functions


def lunch_callback(user_input, channel_id):
    logger.info("Running lunch callback")

    generator = ImageGenerator()

    success = generator.preprocess_prompt(user_input)

    logger.info(f"Preprocess prompt returned: {success}")

    if not success:
        logger.info("Prompt preprocessing failed.")
        slack_notify("Unable to generate request",
                     channel_id=channel_id)
        return

    logger.info("Prompt preprocessing succeeded.")

    if "dalle" in user_input.lower():
        model = "dalle"
    elif "flux" in user_input.lower():
        model = "flux"
    else:
        model = "flux"

    image_url = generator.generate_image(model=model)

    if image_url:
        slack_notify(f"{generator.preprocessed_prompt}",
                     channel_id=channel_id,
                     image_url=image_url)
    else:
        slack_notify("Failed to generate image.", channel_id=channel_id)


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


# Usage example:
if __name__ == "__main__":

    handler = SocketModeHandler(app, lunch_app_token)
    logger.info("Starting the app...")
    handler.start()
