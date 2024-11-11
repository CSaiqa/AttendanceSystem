import cv2
import os

# Path to Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess the image (resize while keeping color channels)
def preprocess_image(image, target_size=(100, 100)):
    resized_image = cv2.resize(image, target_size)  # Resize to target size (100x100)
    return resized_image

# Capture images from the webcam and save to a dataset folder (in color)
def capture_images(person_name, num_images=50):
    # Open the webcam
    cam = cv2.VideoCapture(0)  # Try changing to 1 or 2 if this doesn't work
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Create the dataset folder for the person if it doesn't exist
    os.makedirs(f'dataset/{person_name}', exist_ok=True)
    count = 0

    # Loop to capture the required number of images
    while count < num_images:
        ret, frame = cam.read()  # Capture frame
        if not ret:
            print("Error: Failed to capture an image from the webcam.")
            break

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        # Display the frame even if no face is detected (so you can see the camera feed)
        cv2.imshow('Capturing Faces', frame)

        # If no face is detected, continue showing the feed
        if len(faces) == 0:
            print("No face detected. Make sure your face is visible in the camera.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Loop through all detected faces
        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y + h, x:x + w]  # Capture face region
            face_resized = preprocess_image(face)  # Resize and keep color

            # Save the face image to the folder in color
            save_path = f'dataset/{person_name}/{count}.jpg'
            cv2.imwrite(save_path, face_resized)
            print(f"Captured {count}/{num_images} images for {person_name} (saved at {save_path})")

            # Display the image with a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing {count}/{num_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the current frame with face detection and save feedback
        cv2.imshow('Capturing Faces', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()  # Release the camera
    cv2.destroyAllWindows()  # Close OpenCV windows

    if count == num_images:
        print(f"Successfully captured {num_images} images for {person_name}.")
    else:
        print(f"Stopped after capturing {count} images for {person_name}.")

if __name__ == "__main__":
    # Take the person's name as input and start capturing
    person_name = input("Enter the person's name: ")
    recognized_student_id=input("Enter ID:")
    capture_images(person_name, num_images=5)

