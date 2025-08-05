import pandas as pd
import numpy as np
import os
import random
from datetime import date, timedelta

# --- Configuration ---
PROCESSED_DIR = 'ai-model/data/processed'
OUTPUT_CSV = 'ai-model/data/weather_data.csv'
START_DATE = date(2023, 6, 1)
DAYS = 30
TEMP_RANGE = (15, 28)
HUMIDITY_RANGE = (60, 95)

def generate_weather_data():
    all_images = []
    for class_folder in os.listdir(PROCESSED_DIR):
        class_path = os.path.join(PROCESSED_DIR, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                all_images.append(filename)

    weather_records = []
    for filename in all_images:
        random_day = random.randint(0, DAYS - 1)
        image_date = START_DATE + timedelta(days=random_day)
        temp = round(random.uniform(*TEMP_RANGE), 1)
        humidity = round(random.uniform(*HUMIDITY_RANGE), 1)
        weather_records.append({
            'image_filename': filename,
            'date': image_date.strftime("%Y-%m-%d"),
            'temperature_celsius': temp,
            'humidity_percent': humidity
        })

    df = pd.DataFrame(weather_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully created weather data with {len(df)} rows at: {OUTPUT_CSV}")
    print("Sample data:")
    print(df.head())

if __name__ == '__main__':
    generate_weather_data()