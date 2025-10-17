import os
import cv2
import numpy as np
import shutil

# --- CONFIGURATION ---
RAW_DIR = '/content/plant-disease-predictor/ai-service/data/raw'
PROCESSED_DIR = '/content/plant-disease-predictor/ai-service/data/processed'
# This threshold is now for our more advanced score, so we can start with a lower value.
SEVERE_THRESHOLD = 0.12

# Weights for our "Smart Score". This tells the algorithm what's most important.
WEIGHTS = {
    'total_area': 0.5,
    'lesion_count': 0.2,
    'max_lesion_area': 0.3
}

def calculate_smart_severity(image_path):
    """
    Calculates a nuanced severity score based on lesion analysis.
    Returns a score between 0 and 1.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0.0
    
    image_area = image.shape[0] * image.shape[1]

    # 1. More Robust Spot Detection
    # Convert to HSV color space, which is better for color-based filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define a range for brown/black colors (typical for disease spots)
    lower_bound = np.array([10, 50, 20])
    upper_bound = np.array([30, 255, 200]) # Wider range for browns
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Add another mask for very dark spots
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    
    # Combine the masks
    combined_mask = cv2.bitwise_or(mask, dark_mask)

    # Clean the mask to remove noise
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # 2. Find Individual Lesions (Contours)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise) and very large ones (background)
    min_area = 50
    max_area = image_area * 0.5
    lesions = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    if not lesions:
        return 0.0

    # 3. Calculate Advanced Metrics
    total_lesion_area = sum(cv2.contourArea(c) for c in lesions)
    lesion_count = len(lesions)
    max_lesion_area = max(cv2.contourArea(c) for c in lesions)
    
    # 4. Normalize metrics and calculate final score
    norm_total_area = total_lesion_area / image_area
    # Normalize count by a plausible max (e.g., 50 spots)
    norm_lesion_count = min(lesion_count / 50.0, 1.0) 
    norm_max_area = max_lesion_area / image_area

    severity_score = (WEIGHTS['total_area'] * norm_total_area +
                      WEIGHTS['lesion_count'] * norm_lesion_count +
                      WEIGHTS['max_lesion_area'] * norm_max_area)
    
    return severity_score


def main():
    """Main function to process all folders."""
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR) # Start fresh

    CLASSES = {
        'Potato___Early_blight': ('Early_Blight_Mild', 'Early_Blight_Severe'),
        'Potato___Late_blight': ('Late_Blight_Mild', 'Late_Blight_Severe'),
        'Potato___healthy': ('Healthy', None)
    }

    print("Starting smart image sorting...")
    for raw_folder, (mild, severe) in CLASSES.items():
        source_path = os.path.join(RAW_DIR, raw_folder)
        if not os.path.isdir(source_path): continue

        # Create destination folders
        if mild: os.makedirs(os.path.join(PROCESSED_DIR, mild), exist_ok=True)
        if severe: os.makedirs(os.path.join(PROCESSED_DIR, severe), exist_ok=True)

        for filename in os.listdir(source_path):
            src_file = os.path.join(source_path, filename)
            dest_folder = mild
            if severe: # This handles the disease classes
                score = calculate_smart_severity(src_file)
                if score >= SEVERE_THRESHOLD:
                    dest_folder = severe
            
            shutil.copy(src_file, os.path.join(PROCESSED_DIR, dest_folder, filename))
            
    print("\n✅ Smart sorting complete. Final counts:")
    for folder in sorted(os.listdir(PROCESSED_DIR)):
        count = len(os.listdir(os.path.join(PROCESSED_DIR, folder)))
        print(f"- {folder}: {count} images")

if __name__ == "__main__":
    main()
