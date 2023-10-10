import speech_recognition as sr
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import math


class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "telephone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

object_dimensions = {                         
    "bird" : "0.10",
    "cat" : "0.45",
    "backpack" : "0.55",
    "umbrella" : "0.50",
    "bottle" : "0.20",
    "wine glass" : "0.25",
    "cup" : "0.15",
    "fork" : "0.15",
    "knife" : "0.25",
    "spoon" : "0.15",
    "banana" : "0.20",
    "apple" : "0.07",
    "sandwich" : "0.20",
    "orange" : "0.08",
    "chair" : "0.50",
    "laptop" : "0.40",
    "mouse" : "0.10",
    "remote" : "0.20",
    "keyboard" : "0.30",
    "phone" : "0.15",
    "book" : "0.18",
    "toothbrush" : "0.16"
}
def voice_notification(obj_name, direction, distance):
    engine = pyttsx3.init()
    text = "{} is at {}. It is {:.2f} meters away.".format(obj_name, direction, distance)
    engine.say(text)
    engine.runAndWait()

def get_last_word(sentence):
    words = sentence.split()
    return words[-1]

def voice_command():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Waiting for voice command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    target_object = ""  
    real_width = 0.15  

    try:
        command = recognizer.recognize_google(audio, language="en-US")
        print("Recognized command:", command)
        last_word = get_last_word(command.lower())  
        if last_word:
            print("Last word:", last_word)

        target_object = last_word.lower()
        
        if target_object in object_dimensions:
            real_width = float(object_dimensions[target_object])
            print(real_width)
        else:
            print(f"No length information found for {target_object}, using the default value of 0.15.")
    except sr.UnknownValueError:
        print("Voice cannot be understood.")
    except sr.RequestError as e:
        print("Voice recognition error; {0}".format(e))

    return target_object, real_width

def main():
    # Load the YOLO model
    model = YOLO("yolov8n.pt")
    
    # Get video frame dimensions for calculating 
    cap = cv2.VideoCapture(0)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    center_x = int(frame_width // 2)
    center_y = int(frame_height // 2)
    radius = min(center_x, center_y) - 30  # Radius of the circle where clock hands are drawn
    
    #The target object the user wants to search for via voice command and its real-world average size
    target_object, real_width = voice_command()

    while True:
        success, img = cap.read()
        
        # Predict objects using the YOLO model
        results = model.predict(img, stream=True)
        
        # Draw clock
        for i in range(1, 13):
            angle = math.radians(360 / 12 * i - 90)
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            if i % 3 == 0:
                thickness = 3
                length = 20
            else:
                thickness = 1
                length = 10

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), (x - 10, y + 10), font, 0.5, (0, 255, 0), thickness)
        
        # detect and process objects recognized by model
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  

                cls = int(box.cls)

                if class_names[cls].lower() == target_object:
                    camera_width = x2 - x1
                    distance = (real_width * frame_width) / camera_width
                    #voice_notification(target_object)

                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2

                    camera_middle_x = frame_width // 2
                    camera_middle_y = frame_height // 2

                    vector_x = obj_center_x - camera_middle_x
                    vector_y = obj_center_y - camera_middle_y

                    angle_deg = math.degrees(math.atan2(vector_y, vector_x))
                    #direction = ''
                    if angle_deg < 0:
                        angle_deg += 360

                    if 0 <= angle_deg < 30:
                        direction = "3 o'clock"
                    elif 30 <= angle_deg < 60:
                        direction = "4 o'clock"
                    elif 60 <= angle_deg < 90:
                        direction = "5 o'clock"
                    elif 90 <= angle_deg < 120:
                        direction = "6 o'clock"
                    elif 120 <= angle_deg < 150:
                        direction = "7 o'clock"
                    elif 150 <= angle_deg < 180:
                        direction = "8 o'clock"
                    elif 180 <= angle_deg < 210:
                        direction = "9 o'clock"
                    elif 210 <= angle_deg < 240:
                        direction = "10 o'clock"
                    elif 240 <= angle_deg < 270:
                        direction = "11 o'clock"
                    elif 270 <= angle_deg < 300:
                        direction = "12 o'clock"
                    elif 300 <= angle_deg < 330:
                        direction = "1 o'clock"
                    elif 330 <= angle_deg < 360:
                        direction = "2 o'clock"
                    else:
                        direction = "Unknown Clock Position"

                    cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, "Distance: {:.2f} meters".format(distance), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    if boxes is not None:
                        
                        voice_notification(target_object, direction, distance)

        
        cv2.imshow("Webcam", img)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
