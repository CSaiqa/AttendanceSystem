import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a folder to store known face images if it doesn't exist
if not os.path.exists('Known_faces'):
    os.makedirs('Known_faces')

# Initialize face data and labels (used for training the KNN model)
face_data = []
labels = []
label_dict = {}  # Dictionary to map label numbers to names
label_counter = 0

# Load known face images and assign labels
def load_Known_faces():
    global label_counter
    for filename in os.listdir('Known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image
            img = cv2.imread(os.path.join('Known_faces', filename))
            if img is None:
                print(f"Error loading image: {filename}")
                continue  # Skip this image if it's not loaded correctly
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face in the image
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"No face detected in: {filename}")
                continue  # Skip this image if no face is detected

            for (x, y, w, h) in faces:
                # Extract the face ROI and resize it to a fixed size (for uniformity)
                face_roi = gray_img[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))  # Resize to 100x100
                
                # Flatten the face data
                face_flattened = face_resized.flatten()

                # Append the face data and corresponding label
                face_data.append(face_flattened)

                # Extract the person's name from the file name and assign a label
                name = os.path.splitext(filename)[0]
                if name not in label_dict:
                    label_dict[name] = label_counter
                    label_counter += 1

                labels.append(label_dict[name])

# Train KNN model
def train_knn():
    # Ensure face_data and labels are not empty
    if len(face_data) == 0 or len(labels) == 0:
        print("No face data or labels available for training.")
        return None

    # Convert data to numpy arrays
    X = np.array(face_data)
    y = np.array(labels)

    # Initialize KNN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

# Load known faces and train KNN
load_Known_faces()
knn_classifier = train_knn()

if knn_classifier is None:
    print("KNN training failed due to insufficient data.")
else:
    # Start real-time face detection and recognition
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture video frame.")
            break

        # Convert the frame to grayscale (HaarCascade requires grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the face ROI, resize it to match training data size, and flatten
            face_roi = gray_frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            face_flattened = face_resized.flatten().reshape(1, -1)

            # Use KNN to predict the label of the detected face
            prediction = knn_classifier.predict(face_flattened)

            # Get the name corresponding to the predicted label
            name = list(label_dict.keys())[list(label_dict.values()).index(prediction[0])]

            # Display the predicted name on the frame
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the video feed
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()




