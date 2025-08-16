from PIL import ImageGrab, Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv # Import to load .env file
from google import genai       # Import the Google Generative AI SDK
import pyautogui
import re

# ------ CONFIGURATION ------
SCREEN_LEFT = 450   # Left coordinate of screenshot region
SCREEN_TOP = 250    # Top coordinate of screenshot region
SCREEN_WIDTH = 1750 # Width of screenshot region
SCREEN_HEIGHT = 1000# Height of screenshot region

def card_is_selected(card_image: Image.Image) -> bool:
    """
    Crop bottom 20% of card, determine if it's closer to blue or gray.
    Returns True if selected (blue), False if unselected (gray).
    """
    img = np.array(card_image.convert("RGB"))
    h = img.shape[0]
    y_start = int(h * 0.8)
    bar = img[y_start:, :, :]  # bottom 20%

    # Compute average color in RGB
    avg_color = bar.reshape(-1, 3).mean(axis=0) # shape: (3,)
    # Define typical "blue" and "gray" values
    # (Adjust these as needed for your screenshots)
    blue_rgb = np.array([120, 170, 245])    # approx blue (R,G,B)
    gray_rgb = np.array([32, 33, 36])       # approx gray (R,G,B)
    avg_color_rgb = avg_color

    blue_dist = np.linalg.norm(avg_color_rgb - blue_rgb)
    gray_dist = np.linalg.norm(avg_color_rgb - gray_rgb)

    # Debug output
    print(f"Avg color: {avg_color_rgb}, blue_dist: {blue_dist:.1f}, gray_dist: {gray_dist:.1f}")

    return blue_dist < gray_dist



def parse_album_count(text):
    """
    Parse the number of albums mentioned in the LLM-extracted text.
    Returns an integer count, or 0 if no albums are found.
    """
    text = text.lower()
    if "not in any album" in text:
        return 0

    # Regex to find phrases like "in X album(s)"
    match = re.search(r'in (\d+) album', text)
    if match:
        return int(match.group(1))

    # If parsing failed, raise an error
    raise ValueError(f"Could not determine album count from text: {text}")

# --- Image Utility Functions (Remain mostly unchanged) ---

def show_image_matplotlib(img_array, title=None, gray=False):
    """Displays an image using Matplotlib."""
    plt.figure(figsize=(16, 9))
    if gray:
        plt.imshow(img_array, cmap='gray')
    else:
        plt.imshow(img_array)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    # Minimize plot so it won't interfere with clicking
    try:
        plt.get_current_fig_manager().window.iconify()
    except Exception:
        pass

def take_screenshot(left, top, width, height):
    """Takes a screenshot of a specified region."""
    bbox = (left, top, left + width, top + height)
    img = ImageGrab.grab(bbox)
    return img

def find_card_bounding_boxes_canny(img, debug=False):
    """
    Finds potential card bounding boxes using Canny edge detection and contour analysis.
    Removes any rectangles fully contained within other rectangles.
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 130, 140, 1, L2gradient=False)
    
    # Morphological closing to connect nearby edge segments
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if debug:
        show_image_matplotlib(edges, title="Edges After Canny and Closing", gray=True)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        contour_vis = img_np.copy()
        cv2.drawContours(contour_vis, contours, -1, (255,0,0), 2)
        show_image_matplotlib(contour_vis, title="All Found Contours")

    rectangles = []
    min_area = 200 # Minimum area for a detected card
    img_area = img_np.shape[0] * img_np.shape[1]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # Filter for quadrilateral contours (likely cards)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect = w / float(h)
            
            # Apply size and aspect ratio filters
            if (min_area < area < 0.8 * img_area and
                0.45 < aspect < 0.95 and # Adjust these aspect ratio bounds if your cards are different
                w > 100 and h > 150):   # Minimum width and height for a card
                rectangles.append((x, y, w, h))

    # Remove rectangles fully contained within other rectangles
    def is_inside(r1, r2):
        # Returns True if r1 is fully inside r2
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return (x1 > x2 and 
                y1 > y2 and 
                x1 + w1 < x2 + w2 and 
                y1 + h1 < y2 + h2)

    filtered_rectangles = []
    for i, r1 in enumerate(rectangles):
        inside_any = False
        for j, r2 in enumerate(rectangles):
            if i != j and is_inside(r1, r2):
                inside_any = True
                break
        if not inside_any:
            filtered_rectangles.append(r1)

    if debug:
        vis = img_np.copy()
        for i_rect, (x, y, w, h) in enumerate(filtered_rectangles):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0,255,0), 3)
            cv2.putText(vis, f'{i_rect+1}', (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        show_image_matplotlib(vis, title="Detected Card Bounding Boxes (Filtered)")

    return sorted(filtered_rectangles, key=lambda r: (r[1], r[0])) # Sort by y then x for consistent ordering


# --- Updated Gemini LLM Integration Function ---
def extract_info_with_gemini(gemini_client, card_pil_img: Image.Image):
    """
    Sends a PIL Image of a card to Google Gemini LLM for information extraction.
    Tries the "lite" model first; if it fails to parse, retries with the non-lite model.
    If the album count still cannot be parsed, plots the cropped card and Gemini output.
    """
    import matplotlib.pyplot as plt

    # crop down for debugging
    w, h = card_pil_img.size
    card = card_pil_img.crop((0, h * 0.7, w, h)).convert("L")

    # Define the prompt for the LLM.
    prompt_text = (
        "Extract all of the words from the given image. Return only this text."
    )
    contents = [prompt_text, card_pil_img]

    models_to_try = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    last_response = None

    for model_name in models_to_try:
        print(f"Sending image to Gemini LLM for analysis with model '{model_name}'...")
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            last_response = response
            count = parse_album_count(response.text)
            # If we successfully parse count, return results immediately
            return response.text, count
        except Exception as e:
            print(f"Attempt with model '{model_name}' failed: {e}")

    # If both attempts fail parsing album count, plot and raise
    if last_response is not None:
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(12, 4))
        ax_img.imshow(card, cmap='gray')
        ax_img.set_title("Bottom 40% of Card")
        ax_img.axis("off")
        ax_text.axis("off")
        ax_text.set_title("Gemini LLM Output")
        ax_text.text(0, 0.5, last_response.text, wrap=True, fontsize=10)
        plt.tight_layout()
        plt.show()
  
    # Raise error since parsing failed on both models
    raise ValueError("Failed to parse album count from both lite and non-lite Gemini model outputs.")



def display_card_and_gemini_text(card_index: int, card_image: Image.Image, gemini_text: str):
    """
    Displays the card image and the Gemini-extracted text side-by-side using Matplotlib.
    """
    # Convert PIL Image to numpy array for matplotlib display if not already
    img_np = np.array(card_image)

    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 2]})
    
    # Display the card image
    ax_img.imshow(img_np)
    ax_img.axis("off") # Hide axes for the image
    ax_img.set_title(f"Card {card_index}")

    # Display Gemini LLM text
    ax_text.axis("off") # Hide axes for the text box
    ax_text.set_title("Gemini LLM Analysis")
    
    # Wrap text properly
    wrapped_text = "\n".join(gemini_text.splitlines())
    ax_text.text(0, 1, wrapped_text, fontsize=10, va='top', ha='left', wrap=True,
                 bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()
    # Minimize the plot so it doesn't block the screen
    try:
        plt.get_current_fig_manager().window.iconify()
    except Exception:
        pass

# --- Main Execution Block ---

if __name__ == "__main__":
    plt.close('all')
    # Load environment variables from .env file
    load_dotenv() 
    
    debug = False

    # Initialize Google Gemini client
    genai_client = None
    google_api_key = os.getenv("GOOGLE_API_KEY")
    genai_client = genai.Client(api_key=google_api_key) 
    
    # Try loading a screenshot from file, otherwise take a live one
    # pil_img = Image.open("screenshot.PNG") 
    # print("Loaded image from 'screenshot.PNG'.")

    pil_img = take_screenshot(SCREEN_LEFT, SCREEN_TOP, SCREEN_WIDTH, SCREEN_HEIGHT)
    print(f"Screenshot size: {pil_img.size}")

    # Find bounding boxes of cards using your Canny detection method
    bboxes_contours = find_card_bounding_boxes_canny(pil_img, debug=debug)
    print(f"Found {len(bboxes_contours)} potential cards using contour detection.")


    card_album_counts = []
    for i, (x, y, w, h) in enumerate(bboxes_contours):
        print(f"\n--- Processing Card {i+1} (Region: x={x}, y={y}, w={w}, h={h}) ---")
        card_img = pil_img.crop((x, y, x + w, y + h))
               
        extracted_info, count = extract_info_with_gemini(genai_client, card_img)
        
        print(f"Gemini LLM extracted information for Card {i+1}:\n{extracted_info}\n{'-'*80}")
        
        if debug:
            display_card_and_gemini_text(i + 1, card_img, extracted_info)
        
        
        print(f"Parsed album count for Card {i+1}: {count}")
        card_album_counts.append((i, (x, y, w, h), count))

    selected_indices = []
    for i, (x, y, w, h) in enumerate(bboxes_contours):
        card_img = pil_img.crop((x, y, x + w, y + h))
        if card_is_selected(card_img):
            selected_indices.append(i)
    
    # Sort by album count descending
    card_album_counts.sort(key=lambda e: e[2], reverse=True)  
    
    highest_count = card_album_counts[0][2]
    
    # Get all indices with the highest count
    top_cards = [e for e in card_album_counts if e[2] == highest_count]
    
    # Find if any top card is already selected
    best_card_index = None
    best_bbox = None
    best_count = highest_count
    
    for idx, bbox, count in top_cards:
        if idx in selected_indices:
            best_card_index, best_bbox = idx, bbox
            print(f"Keeping already selected card {best_card_index + 1} as best among ties.")
            break
    
    # If none of the top cards is already selected, pick the first top card
    if best_card_index is None:
        best_card_index, best_bbox, best_count = top_cards[0]
        print(f"No top card selected yet; selecting card {best_card_index + 1}.")
    
    print(f"\nFinal best card: {best_card_index + 1}. Selected indices before update: {selected_indices}")
    
    # Deselect any other selected cards except the best one
    for i in selected_indices:
        if i != best_card_index:
            x, y, w, h = bboxes_contours[i]
            click_x = SCREEN_LEFT + x + w // 2
            click_y = SCREEN_TOP + y + h // 2
            print(f"Deselecting card {i+1} at ({click_x},{click_y})")
            pyautogui.moveTo(click_x, click_y)
            pyautogui.click()
    
    # Select best if not already selected
    if best_card_index not in selected_indices:
        x, y, w, h = best_bbox
        click_x = SCREEN_LEFT + x + w // 2
        click_y = SCREEN_TOP + y + h // 2
        print(f"Selecting card {best_card_index + 1} at ({click_x},{click_y})")
        pyautogui.moveTo(click_x, click_y)
        pyautogui.click()
    else:
        print("Best card already selected; no need to click.")

