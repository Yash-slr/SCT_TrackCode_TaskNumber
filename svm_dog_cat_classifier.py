import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
def load_data(cat_folder, dog_folder, image_size):
    X, y = [], []
    print("Loading cat images...")
    for filename in tqdm(os.listdir(cat_folder)):
        try:
            img = cv2.imread(os.path.join(cat_folder, filename), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (image_size, image_size))
            X.append(img.flatten())
            y.append(0)
        except:
            continue
    print("Loading dog images...")
    for filename in tqdm(os.listdir(dog_folder)):
        try:
            img = cv2.imread(os.path.join(dog_folder, filename), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (image_size, image_size))
            X.append(img.flatten())
            y.append(1)
        except:
            continue
    return np.array(X), np.array(y)
def show_predictions(X_test, y_pred, image_size):
    import matplotlib.pyplot as plt
    for idx in np.random.choice(len(X_test), 5, replace=False):
        img = X_test[idx].reshape(image_size, image_size, 3)
        label = "Dog" if y_pred[idx] == 1 else "Cat"
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted: {label}")
        plt.axis('off')
        plt.show()
if __name__ == "__main__":
    IMAGE_SIZE = 64
    cat_folder = "Cat"
    dog_folder = "Dog"
    X, y = load_data(cat_folder, dog_folder, IMAGE_SIZE)
    print(f"Total images loaded: {len(X)}")
    if len(X) == 0:
        print(" No images were loaded. Check the paths.")
        exit()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training SVM classifier...")
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    show_predictions(X_test, y_pred, IMAGE_SIZE)

