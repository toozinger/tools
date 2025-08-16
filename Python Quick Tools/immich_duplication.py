from PIL import ImageGrab, Image
import numpy as np
import cv2
import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np
from PIL import Image

def take_screenshot(left, top, width, height):
    """Takes a screenshot of a specific area of the screen."""
    bbox = (left, top, left + width, top + height)
    img = ImageGrab.grab(bbox)
    return img



def find_card_bounding_boxes_canny(img, debug=False):
    """
    Finds card bounding boxes using only Canny edge detection and contour polygon approximation.
    Works well for cards with rounded corners and clear borders.
    """
    img_np = np.array(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Use Canny edge detection (tune thresholds as needed)
    edges = cv2.Canny(gray, 130, 140, 1, L2gradient  = False)
    # if debug:
    #     Image.fromarray(edges).show(title="Canny Edges")
        
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if debug:
        Image.fromarray(edges).show(title = "edges_modified")


    # Find external contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualize all contours
    contour_vis = img_np.copy()
    cv2.drawContours(contour_vis, contours, -1, (255,0,0), 2)
    Image.fromarray(contour_vis).show(title="All Contours")

    rectangles = []
    min_area = 200  # Minimum reasonable area, adjust as needed
    img_area = img_np.shape[0] * img_np.shape[1]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)  # Adjust approximation as needed

        # Check for 4-point polygons (rectangle or rounded-rectangle)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect = w / float(h)

            if (min_area < area < 0.8 * img_area and
                0.45 < aspect < 0.95 and   # Typical card aspect
                w > 100 and h > 150):
                rectangles.append((x, y, w, h))

    # Visualization
    if debug:
        vis = img_np.copy()
        for i, (x, y, w, h) in enumerate(rectangles):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0,255,0), 3)
            cv2.putText(vis, f'{i+1}', (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        Image.fromarray(vis).show(title="Canny Contour Results")
    # Sort by y then x for consistency
    return sorted(rectangles, key=lambda r: (r[1], r[0]))

if __name__ == "__main__":
    # Load image
    try:
        pil_img = Image.open("screenshot.PNG") 
        print("Loaded image from file.")
    except FileNotFoundError:
        print("Taking a live screenshot...")
        pil_img = take_screenshot(450, 250, 1750, 1000)

    print(f"Screenshot size: {pil_img.size}")

    bboxes_contours = find_card_bounding_boxes_canny(pil_img, debug=True)
    print(f"Found {len(bboxes_contours)} cards using contour detection.")
    for i, (x, y, w, h) in enumerate(bboxes_contours):
        print(f"Card {i+1}: x={x}, y={y}, w={w}, h={h}")

    