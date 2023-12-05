import cv2
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array

# Load the pre-trained age detection model
model = load_model("best_model.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict age from an image array
def predict_age(image_array):
    image = preprocess_input(image_array)
    input_arr = np.array([image])
    pred = np.argmax(model.predict(input_arr))

    if pred == 0:
        return "age = 20-38"
    elif pred == 1:
        return "age = 0-3"
    elif pred == 2:
        return "age = 4-12"
    elif pred == 3:
        return "age = 39-90"
    else:
        return "age = 13-19"

# OpenCV setup for webcam access
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()  # Read frames from the webcam

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Crop the face region
        resized_img = cv2.resize(face_img, (256, 256))  # Resize the face image
        img_array = img_to_array(resized_img)  # Convert to array
        age = predict_age(img_array)  # Predict age

        # Draw rectangle around the face and add text with predicted age
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, age, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Age Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
