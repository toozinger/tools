from PIL import ImageGrab, Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv # Import to load .env file
from google import genai       # Import the Google Generative AI SDK

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

def take_screenshot(left, top, width, height):
    """Takes a screenshot of a specified region."""
    bbox = (left, top, left + width, top + height)
    img = ImageGrab.grab(bbox)
    return img

def find_card_bounding_boxes_canny(img, debug=False):
    """
    Finds potential card bounding boxes using Canny edge detection and contour analysis.
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

    if debug:
        vis = img_np.copy()
        for i_rect, (x, y, w, h) in enumerate(rectangles):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0,255,0), 3)
            cv2.putText(vis, f'{i_rect+1}', (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        show_image_matplotlib(vis, title="Detected Card Bounding Boxes")

    return sorted(rectangles, key=lambda r: (r[1], r[0])) # Sort by y then x for consistent ordering

# --- New Gemini LLM Integration Function ---

def extract_info_with_gemini(gemini_client, card_pil_img: Image.Image):
    """
    Sends a PIL Image of a card to Google Gemini 2.5 Flash LLM for information extraction.
    """
    
    # crop down:
    w, h = card_pil_img.size
    card = card_pil_img.crop((0, h * 0.7, w, h)).convert("L")

    
    # Define the prompt for the LLM. Be specific about what you want it to extract.
    prompt_text = (
        "Extract all of the words from the given image. Return only this text."
    )
    
    # The `contents` argument accepts a list of text strings and PIL Image objects.
    contents = [prompt_text, card_pil_img]

    print("Sending image to Gemini LLM for analysis...")
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite", # Use the specified model
        contents=contents)

    return response.text

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
    
    # Wrap text properly to fit into the box.
    # We might need to handle newline characters and general wrapping for long LLM outputs.
    # A simple way for display is to just put it as text.
    # For very long outputs, you might need more sophisticated text wrapping or a scrollable view.
    wrapped_text = "\n".join(gemini_text.splitlines())
    ax_text.text(0, 1, wrapped_text, fontsize=10, va='top', ha='left', wrap=True,
                 bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv() 

    # Initialize Google Gemini client
    genai_client = None
    google_api_key = os.getenv("GOOGLE_API_KEY")
    genai_client = genai.Client(api_key=google_api_key) 
    
    # Try loading a screenshot from file, otherwise take a live one
    try:
        pil_img = Image.open("screenshot.PNG") 
        print("Loaded image from 'screenshot.PNG'.")
    except FileNotFoundError:
        print("File 'screenshot.PNG' not found. Taking a live screenshot...")
        # Adjust these coordinates (left, top, width, height) to your specific screen area
        pil_img = take_screenshot(450, 250, 1750, 1000) 

    print(f"Screenshot size: {pil_img.size}")

    # Find bounding boxes of cards using your Canny detection method
    bboxes_contours = find_card_bounding_boxes_canny(pil_img, debug=True)
    print(f"Found {len(bboxes_contours)} potential cards using contour detection.")

    if not bboxes_contours:
        print("No cards found. Consider adjusting screenshot region or contour detection parameters (e.g., Canny thresholds, min_area, aspect ratio, min_width/height).")

    # Process each detected card
    for i, (x, y, w, h) in enumerate(bboxes_contours):
        print(f"\n--- Processing Card {i+1} (Region: x={x}, y={y}, w={w}, h={h}) ---")
        card_img = pil_img.crop((x, y, x + w, y + h))
        
        # Optional: show the cropped card image before sending to LLM
        # show_image_matplotlib(np.array(card_img), title=f"Cropped Card {i+1} for LLM")
        
        # Use the Gemini LLM to extract information from the card image
        extracted_info = extract_info_with_gemini(genai_client, card_img)
        
        print(f"Gemini LLM extracted information for Card {i+1}:\n{extracted_info}\n{'-'*80}")
        
        # Display the card image and the extracted text side-by-side
        display_card_and_gemini_text(i + 1, card_img, extracted_info)

