import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle

# Path to the dataset folder
DATASET_PATH = r"D:\MCA\Final\Project\myenv\project\dataset"  # Corrected path
CATEGORIES = ["cancer", "non_cancer"]
IMAGE_SIZE = (50, 50)  # Resizing the images to 50x50 pixels

# Function to load and preprocess the data
def load_data():
    data = []
    labels = []
    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category)
        label = CATEGORIES.index(category)
        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img = img.resize(IMAGE_SIZE)  # Resize image to 50x50
                data.append(np.array(img).flatten())  # Flatten the image
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
    return np.array(data), np.array(labels)

# Load the data
data, labels = load_data()

# Split into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("cancer_detection_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved as 'cancer_detection_model.pkl'")
