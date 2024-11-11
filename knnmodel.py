import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Preprocess images (resize and flatten)
def preprocess_image(image, target_size=(100, 100)):
    return cv2.resize(image, target_size).flatten()

# Train the KNN model and save it to a file
def train_knn_model():
    X, y = [], []
    people = os.listdir('dataset')

    for person in people:
        person_folder = f'dataset/{person}'
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                processed_image = preprocess_image(image)
                X.append(processed_image)
                y.append(person)
    
    X, y = np.array(X), np.array(y)
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X, y)

    # Save the model to a file
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    print("KNN model trained and saved as 'knn_model.pkl'.")

if __name__ == "__main__":
    train_knn_model()


