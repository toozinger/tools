from PIL import ImageGrab, Image
import numpy as np
import cv2

def take_screenshot(left, top, width, height):
    """Takes a screenshot of a specific area of the screen."""
    bbox = (left, top, left + width, top + height)
    img = ImageGrab.grab(bbox)
    return img

def find_card_bounding_boxes_hough(img, debug=False):
    """
    Finds card-like bounding boxes using Hough line detection to find rectangles.
    This approach is better for clean, rectangular card layouts.
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    if debug:
        Image.fromarray(gray).show(title="1. Grayscale Image")

    # Get clean edges
    edges = cv2.Canny(gray, 50, 150)
    
    if debug:
        Image.fromarray(edges).show(title="2. Canny Edges")

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=10)
    
    if lines is None:
        print("No lines detected!")
        return []
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 - x1 == 0:  # Vertical line
            angle = 90
        else:
            angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        
        if angle > 85:  # Nearly vertical (85-95 degrees)
            vertical_lines.append((x1, y1, x2, y2))
        elif angle < 5:  # Nearly horizontal (0-5 degrees)
            horizontal_lines.append((x1, y1, x2, y2))
    
    if debug:
        # Visualize detected lines
        line_vis = img_np.copy()
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(line_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        Image.fromarray(line_vis).show(title="3. Detected Lines (Green=Horizontal, Blue=Vertical)")
    
    # Group similar lines together
    def group_lines(lines, is_vertical=False):
        if not lines:
            return []
        
        grouped = []
        tolerance = 20  # pixels
        
        for line in lines:
            x1, y1, x2, y2 = line
            # For vertical lines, group by x-coordinate
            # For horizontal lines, group by y-coordinate
            coord = x1 if is_vertical else y1
            
            # Find existing group or create new one
            added = False
            for group in grouped:
                if abs(group['coord'] - coord) < tolerance:
                    group['lines'].append(line)
                    group['coord'] = np.mean([group['coord'], coord])  # Update average
                    added = True
                    break
            
            if not added:
                grouped.append({'coord': coord, 'lines': [line]})
        
        return grouped
    
    h_groups = group_lines(horizontal_lines, is_vertical=False)
    v_groups = group_lines(vertical_lines, is_vertical=True)
    
    # Sort groups by coordinate
    h_groups.sort(key=lambda g: g['coord'])
    v_groups.sort(key=lambda g: g['coord'])
    
    if debug:
        print(f"Found {len(h_groups)} horizontal line groups and {len(v_groups)} vertical line groups")
    
    # Find rectangles by intersecting line groups
    rectangles = []
    
    for i in range(len(h_groups) - 1):
        for j in range(len(v_groups) - 1):
            # Get coordinates of potential rectangle
            top = int(h_groups[i]['coord'])
            bottom = int(h_groups[i + 1]['coord'])
            left = int(v_groups[j]['coord'])
            right = int(v_groups[j + 1]['coord'])
            
            # Calculate dimensions
            width = right - left
            height = bottom - top
            
            # Filter for card-like rectangles
            if width > 0 and height > 0:
                aspect = width / height
                area = width * height
                
                if (0.5 < aspect < 0.9 and  # Card-like aspect ratio
                    area > 15000 and        # Minimum area
                    width > 150 and height > 200):  # Minimum dimensions
                    
                    rectangles.append((left, top, width, height))
    
    # Remove duplicates and overlapping rectangles
    def remove_overlaps(rects):
        if not rects:
            return []
        
        # Sort by area (largest first)
        rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
        filtered = []
        
        for rect in rects:
            x1, y1, w1, h1 = rect
            
            # Check if this rectangle significantly overlaps with any accepted rectangle
            overlaps = False
            for accepted in filtered:
                x2, y2, w2, h2 = accepted
                
                # Calculate intersection
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                
                if ix1 < ix2 and iy1 < iy2:  # There is intersection
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    min_area = min(w1 * h1, w2 * h2)
                    
                    if intersection_area / min_area > 0.5:  # 50% overlap threshold
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(rect)
        
        return filtered
    
    rectangles = remove_overlaps(rectangles)
    
    # Sort final rectangles top-to-bottom, left-to-right
    rectangles = sorted(rectangles, key=lambda r: (r[1] // 100, r[0]))
    
    if debug and rectangles:
        # Visualize final rectangles
        final_vis = img_np.copy()
        for i, (x, y, w, h) in enumerate(rectangles):
            cv2.rectangle(final_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(final_vis, f'{i+1}', (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        Image.fromarray(final_vis).show(title="4. Final Detected Rectangles")
    
    return rectangles

# Alternative approach using template matching for grid layouts
def find_card_bounding_boxes_grid(img, debug=False):
    """
    Alternative approach: Assume cards are in a regular grid and detect the grid structure.
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Get edges
    edges = cv2.Canny(gray, 50, 150)
    
    if debug:
        Image.fromarray(edges).show(title="Edges for Grid Detection")
    
    # Project edges onto axes to find grid lines
    h_projection = np.sum(edges, axis=1)  # Sum along horizontal axis
    v_projection = np.sum(edges, axis=0)  # Sum along vertical axis
    
    if debug:
        # Plot projections
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(h_projection)
        ax1.set_title('Horizontal Projection')
        ax1.set_xlabel('Y coordinate')
        ax1.set_ylabel('Edge intensity sum')
        
        ax2.plot(v_projection)
        ax2.set_title('Vertical Projection')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Edge intensity sum')
        plt.show()
    
    # Find peaks in projections (these correspond to grid lines)
    from scipy.signal import find_peaks
    
    # Find horizontal grid lines (peaks in horizontal projection)
    h_peaks, _ = find_peaks(h_projection, height=np.max(h_projection) * 0.3, distance=50)
    
    # Find vertical grid lines (peaks in vertical projection)
    v_peaks, _ = find_peaks(v_projection, height=np.max(v_projection) * 0.3, distance=50)
    
    if debug:
        print(f"Found {len(h_peaks)} horizontal grid lines and {len(v_peaks)} vertical grid lines")
        print(f"H-lines at: {h_peaks}")
        print(f"V-lines at: {v_peaks}")
    
    # Create rectangles from grid intersections
    rectangles = []
    
    for i in range(len(h_peaks) - 1):
        for j in range(len(v_peaks) - 1):
            top = h_peaks[i]
            bottom = h_peaks[i + 1]
            left = v_peaks[j]
            right = v_peaks[j + 1]
            
            width = right - left
            height = bottom - top
            
            if width > 100 and height > 150:  # Minimum card size
                # Add some padding to avoid including borders
                padding = 10
                rectangles.append((left + padding, top + padding, 
                                width - 2*padding, height - 2*padding))
    
    if debug and rectangles:
        final_vis = img_np.copy()
        for i, (x, y, w, h) in enumerate(rectangles):
            cv2.rectangle(final_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(final_vis, f'{i+1}', (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        Image.fromarray(final_vis).show(title="Grid-Based Detection Results")
    
    return rectangles

if __name__ == "__main__":
    # Load image
    try:
        pil_img = Image.open("screenshot_for_detection.png") 
        print("Loaded image from file.")
    except FileNotFoundError:
        print("Taking a live screenshot...")
        pil_img = take_screenshot(450, 250, 1750, 1000)

    print(f"Screenshot size: {pil_img.size}")

    # Try both approaches
    print("\n=== Hough Line Approach ===")
    bboxes_hough = find_card_bounding_boxes_hough(pil_img, debug=True)
    print(f"Detected {len(bboxes_hough)} cards with Hough lines.")
    
    print("\n=== Grid Detection Approach ===")
    bboxes_grid = find_card_bounding_boxes_grid(pil_img, debug=True)
    print(f"Detected {len(bboxes_grid)} cards with grid detection.")
    
    # Print results
    for i, (x, y, w, h) in enumerate(bboxes_hough):
        print(f"Hough Card {i+1}: Region (x={x}, y={y}, w={w}, h={h})")
    
    for i, (x, y, w, h) in enumerate(bboxes_grid):
        print(f"Grid Card {i+1}: Region (x={x}, y={y}, w={w}, h={h})")
