import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
# Function to load train images from a folder and assign labels
def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                            images.append(img)
                            labels.append(subfolder)
                            print('Labels \n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return images, labels

# Function to load test images from a folder and assign labels
def load_test_images_from_folder(folder, target_shape=None):
    test_images = []
    test_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                            test_images.append(img)
                            test_labels.append(subfolder)
                            print('Labels \n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return test_images, test_labels

def load_val_images_from_folder(folder, target_shape=None):
    val_images = []
    val_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                            val_images.append(img)
                            val_labels.append(subfolder)
                            print('Labels \n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return val_images, val_labels


# Folder paths
data_folder = './train1'  # Folder with training data
test_folder = './test'  # Folder with test data
val_folder = './val'  # Folder with validation data

# Load images and labels from the 'dataset' folder and resize them to (200, 200)
images, labels = load_train_images_from_folder(data_folder, target_shape=(250, 250))

# Load validation images and labels from the 'val' folder
val_images, val_labels = load_val_images_from_folder(val_folder, target_shape=(250, 250))

# Combine training and validation data
images += val_images
labels += val_labels


# Convert labels to binary (0 or 1)
labels_binary = [1 if label == 'NORMAL' else 0 for label in labels]

# Reshape the images and convert them to grayscale
image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]

# Convert the list of 1D arrays to a 2D numpy array
image_data = np.array(image_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)


# Load TEST images from the 'test' folder
test_images, test_labels = load_test_images_from_folder(test_folder, target_shape=(250, 250))

# Convert labels to binary (0 or 1), reshape and convert to grayscale and scale dthe TEST data
test_labels_binary = [1 if label == 'NORMAL' else 0 for label in test_labels]
test_image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in test_images]
test_image_data = np.array(test_image_data)
scaled_test_data = scaler.transform(test_image_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(scaled_data, labels_binary, test_size=0.2, random_state=42)

# Train a Random Forest model
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Predictions on the validation set
validation_predictions = random_forest_model.predict(X_val)


### TESTING ###

### RANDOM FOREST###

# Evaluate performance on the validation set
accuracy_val = accuracy_score(y_val, validation_predictions)
precision_val = precision_score(y_val, validation_predictions)
recall_val = recall_score(y_val, validation_predictions, zero_division=1)
f1_val = f1_score(y_val, validation_predictions)
confusion_matrix_val = confusion_matrix(y_val, validation_predictions)

print("Performance on Validation Set:")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_val)

# Predictions on the test set
test_predictions = random_forest_model.predict(scaled_test_data)
train_predictions = random_forest_model.predict(scaled_data)
#Afisez vectorul de predictie
print("test_prediction: ")
print(test_predictions)

# Plotarea imaginilor din test cu labelul asociat (din 5 in 5 pt ca sunt multe(187))
for i, test_image in enumerate(test_images[::5]):
    
    if test_predictions[i] == 1:
        print(f"Test Image {i + 1} - ==NORMAL==")
    else:
        print(f"Test Image {i + 1} - ==PNEUMONIA==")

    # Visualize the test image with its predicted value
    # plt.figure()
    # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    # if test_predictions[i] == 1:
    #     plt.title(f"Test Image {i + 1} - ==NORMAL==")
    # else:
    #     plt.title(f"Test Image {i + 1} - ==PNEUMONIA==")
    # plt.axis('off')
    # plt.show()

# Function to calculate accuracy and confusion matrix
def calculate_accuracy_and_confusion_matrix(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    return accuracy, confusion_mat

# Function to plot confusion matrix
def plot_confusion_matrix(conf_mat, classes, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# True labels for training and test sets
train_true_labels = [1 if label == 'NORMAL' else 0 for label in labels]
test_true_labels = [1 if label == 'NORMAL' else 0 for label in test_labels]

# Calculate accuracy and confusion matrix for training set
accuracy_train, confusion_matrix_train = calculate_accuracy_and_confusion_matrix(train_true_labels, train_predictions)
print("Accuracy on Training Set:", accuracy_train)
print("Confusion Matrix on Training Set:\n", confusion_matrix_train)

# Calculate accuracy and confusion matrix for test set
accuracy_test, confusion_matrix_test = calculate_accuracy_and_confusion_matrix(test_true_labels, test_predictions)
print("\nAccuracy on Test Set:", accuracy_test)
print("Confusion Matrix on Test Set:\n", confusion_matrix_test)

# Plot confusion matrix for training set
plot_confusion_matrix(confusion_matrix_train, classes=["PNEUMONIA", "NORMAL"], title="Confusion Matrix on Training Set")

# Plot confusion matrix for test set
plot_confusion_matrix(confusion_matrix_test, classes=["PNEUMONIA", "NORMAL"], title="Confusion Matrix on Test Set")

# Plot accuracy on training and test sets
plt.figure(figsize=(10, 4))
plt.bar(["Training Set", "Test Set"], [accuracy_train, accuracy_test], color=['blue', 'orange'])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
