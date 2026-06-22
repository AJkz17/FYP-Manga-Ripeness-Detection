import requests
import json
import os

def predict_ripeness(image_path):
    print(f"--- Processing {image_path} ---")

    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": "cb31129c609a6b125520e6e911317a680d7732d663"}
    data = {
        "model": "https://hub.ultralytics.com/models/AeO53n3gJlbaErbTJWKy",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.45
    }
    all_detections = []

    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files={"file": f}
            )

        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return []

        data = response.json()

        if 'images' in data and len(data['images']) > 0:
            first_image = data['images'][0]

            if 'results' in first_image and len(first_image['results']) > 0:
                for result in first_image['results']:
                    prediction_class = result.get('name', 'Unknown')
                    confidence = result.get('confidence', 0.0)
                    box = result.get('box')
                    all_detections.append({
                        'class': str(prediction_class).capitalize(),
                        'confidence': round(confidence * 100, 2),
                        'box': box
                    })

        return all_detections

    except Exception as e:
        print(f"Critical Error: {e}")
        return []