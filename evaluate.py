from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np
model = YOLO("runs/detect/train-4/weights/best.pt")
def predict(data):   
    confidences = []
    texts=[]
    img = np.array(data)
    # ensure BGR format (important for OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model.predict(source=img,conf=0.25,save=True)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
         # crop license plate
        plate = img[y1:y2, x1:x2]
        # preprocessing (IMPORTANT for accuracy)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # OCR
        text = pytesseract.image_to_string(gray, config='--psm 7')

        texts.append(text.strip())
        confidence = float(box.conf[0])*100
        confidences.append(round(confidence,2))
    print(confidences)
    print(texts)
    return results[0],confidences,texts


