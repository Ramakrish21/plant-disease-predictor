import os
import cv2
import numpy as np
import shutil

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

THRESHOLD = 0.10  # 10% of image → considered severe

def calculate_disease_severity(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 90])
    mask = cv2.inRange(hsv, lower, upper)
    severity = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return severity

def process_folder(folder_name):
    folder_path = os.path.join(RAW_DIR, folder_name)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        severity = calculate_disease_severity(img_path)
        if "Healthy" in folder_name:
            target_folder = "Healthy"
        elif "Early" in folder_name:
            target_folder = "Early_Blight_Severe" if severity >= THRESHOLD else "Early_Blight_Mild"
        elif "Late" in folder_name:
            target_folder = "Late_Blight_Severe" if severity >= THRESHOLD else "Late_Blight_Mild"
        else:
            continue

        dest = os.path.join(PROCESSED_DIR, target_folder, img_file)
        shutil.copy(img_path, dest)
        print(f"{img_file} ➜ {target_folder} ({severity:.2%})")

def main():
    folders = ["Potato___Healthy", "Potato___Early_blight", "Potato___Late_blight"]
    for folder in folders:
        process_folder(folder)

if __name__ == "__main__":
    main()






# workflow process to transform from raw folders images to processed folder images to be consider following steps

# 📁 data/raw/
# │
# ├── Potato___Healthy/
# ├── Potato___Early_blight/
# └── Potato___Late_blight/
#       │
#       ▼
# 🎯 Python Script: label_severity.py
#    └── For each image:
#        ├── Reads image with OpenCV
#        ├── Converts to HSV color space
#        ├── Detects black/dark/brown areas (possible disease)
#        ├── Calculates % of image that’s diseased
#        │
#        ├── Healthy → Processed/Healthy/
#        ├── Early blight:
#        │     └── <10% → Early_Blight_Mild/
#        │     └── ≥10% → Early_Blight_Severe/
#        └── Late blight:
#              └── <10% → Late_Blight_Mild/
#              └── ≥10% → Late_Blight_Severe/



          #  Summary Table
        # ------------------------
    
# Folder Name   	     % Infected  	    Goes To Folder
# Potato___Healthy  	      0%  	    processed/Healthy/
# Potato___Early_blight 	< 10%	    processed/Early_Blight_Mild/
# Potato___Early_blight	    ≥ 10%	    processed/Early_Blight_Severe/
# Potato___Late_blight	    < 10%	    processed/Late_Blight_Mild/
# Potato___Late_blight	    ≥ 10%	    processed/Late_Blight_Severe/

