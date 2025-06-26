import cv2
from ultralytics import YOLO
import threading
import time
import pygame
import os
import requests
from twilio.rest import Client
import telebot
import json
from datetime import datetime

# === Telegram Bot Setup ===
telegram_token = "8199032431:AAEGavRUynhyz3d40XhMNfvkwHBOccdOp7s"
chat_id = "6625535153"
bot = telebot.TeleBot(telegram_token)

# === Twilio Setup ===
twilio_account_sid = 'ACa9e3e8a50f52924ae95da1ba02021f9e'
twilio_auth_token = 'abbc39e74cdfc8c7da11d599a3e995e6'
twilio_phone_number = '+16206590991'
to_phone_number = '+918838154112'

client = Client(twilio_account_sid, twilio_auth_token)
pygame.mixer.init()
ALERT_SOUND = "backend/alert1.wav"
DETECTION_DIR = "backend/detection"
os.makedirs(DETECTION_DIR, exist_ok=True)

# === LOAD YOLOv8 MODELS ===
model_human = YOLO("backend/best1.pt")
model_elephant = YOLO("backend/elephant.pt")
model_pig = YOLO("backend/pig.pt")
model_rat = YOLO("backend/rat.pt")
model_other_animals = YOLO("backend/best.pt")

# === FLAGS & THREADING ===
is_alerting = False
stop_thread = False
last_detection_time = None
sound_thread = None
detected_labels = set()  # To keep track of already detected labels

# === CAMERA ===
cap = cv2.VideoCapture(0)

# === SOUND ALERT FUNCTION ===
def play_alert():
    pygame.mixer.music.load(ALERT_SOUND)
    pygame.mixer.music.play(-1)
    while not stop_thread:
        time.sleep(1)
    pygame.mixer.music.stop()

# === TELEGRAM ALERT FUNCTION ===
def send_telegram_alert(label, image_path):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = f"⚠ Alert: {label} detected!\nTime: {timestamp}"
        with open(image_path, "rb") as img:
            requests.post(
                f"https://api.telegram.org/bot{telegram_token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": img}
            )
        print("[✓] Telegram alert sent.")
    except Exception as e:
        print("[x] Telegram error:", e)

# === Django Upload Function ===
def upload_to_django(img_path, label):
    url = "http://192.168.218.171:8000/api/upload/"  # Django API endpoint for uploading detection
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Open image file
        with open(img_path, "rb") as img_file:
            files = {"image": img_file}  # Image file to be uploaded
            data = {
                "label": label,  # Animal label
                "timestamp": timestamp  # Timestamp for when the image was captured
            }

            # Send POST request to Django API
            response = requests.post(url, files=files, data=data)

            if response.status_code == 201:
                print(f"[✓] Detection uploaded to Django: {label} at {timestamp}")
            else:
                print(f"[x] Upload failed: {response.text}")
    except Exception as e:
        print(f"[x] Error uploading detection to Django: {e}")

# === TRIGGER ALERT FUNCTION ===
def trigger_alert(frame, label):
    global is_alerting, sound_thread, last_detection_time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}_{timestamp}.jpg"
    img_path = os.path.join(DETECTION_DIR, filename)
    cv2.imwrite(img_path, frame)

    send_telegram_alert(label, img_path)

    try:
        client.calls.create(
            twiml='<Response><Say>Alert! Wild animal detected. Check your camera feed immediately.</Say></Response>',
            to=to_phone_number,
            from_=twilio_phone_number
        )
        print("[✓] Twilio call triggered.")
    except Exception as e:
        print("[x] Twilio error:", e)

    sound_thread = threading.Thread(target=play_alert)
    sound_thread.start()
    is_alerting = True
    last_detection_time = datetime.now()

    # Upload to Django
    upload_to_django(img_path, label)

# === MAIN LOOP ===
print("[INFO] Detection system started.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Camera error.")
            break

        frame = cv2.resize(frame, (640, 480))
        display = frame.copy()

        # === 1. HUMAN CHECK ===
        result = model_human(frame, verbose=False)
        if result[0].boxes and len(result[0].boxes.conf) > 0:
            print("[INFO] Human detected — No alert.")
            cv2.imshow("Wild Animal Detection", result[0].plot())
            continue

        # === 2. ELEPHANT ===
        result = model_elephant(frame, verbose=False)
        if result[0].boxes and "Elephant" not in detected_labels:
            print("[ALERT] Elephant Detected")
            trigger_alert(frame, "Elephant")
            detected_labels.add("Elephant")
            continue

        # === 3. PIG ===
        result = model_pig(frame, verbose=False)
        if result[0].boxes and "Pig" not in detected_labels:
            print("[ALERT] Pig Detected")
            trigger_alert(frame, "Pig")
            detected_labels.add("Pig")
            continue

        # === 4. RAT ===
        result = model_rat(frame, verbose=False)
        if result[0].boxes and "Rat" not in detected_labels:
            print("[ALERT] Rat Detected")
            trigger_alert(frame, "Rat")
            detected_labels.add("Rat")
            continue

        # === 5. MOUSE (Handle Mouse detection) ===
        result = model_other_animals(frame, verbose=False)
        if result[0].boxes:
            cls_id = int(result[0].boxes.cls[0])
            name = result[0].names[cls_id].lower()
            if name == "mouse" and "Mouse" not in detected_labels:
                print("[ALERT] Mouse Detected")
                trigger_alert(frame, "Mouse")
                detected_labels.add("Mouse")
                continue
            elif name == "dog" and "Dog" not in detected_labels:
                print("[ALERT] Dog Detected")
                trigger_alert(frame, "Dog")
                detected_labels.add("Dog")
                continue
            else:
                print(f"[INFO] Detected {name}, not in threat list.")

        # === RESET ALERT ===
        if is_alerting and last_detection_time:
            if (datetime.now() - last_detection_time).seconds > 60:
                print("[INFO] Resetting alert system.")
                is_alerting = False
                stop_thread = True
                if sound_thread:
                    sound_thread.join()
                    sound_thread = None
                stop_thread = False
                detected_labels.clear()  # Reset detected labels after reset

        cv2.imshow("Wild Animal Detection", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted manually.")
finally:
    stop_thread = True
    if sound_thread:
        sound_thread.join()
    cap.release()
    cv2.destroyAllWindows()
