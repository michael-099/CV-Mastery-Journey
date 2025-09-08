import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


data = load_digits()
X = data.images
y = data.target

# Flatten the 8x8 images into 64 feature vectors
n_samples = len(X)
X = X.reshape((n_samples, -1))
print("np_version",np.__version__)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

    # Evaluation
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))


#Support Vector Machine (SVM)
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

    #Evaluation
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  Integration with OpenCV (Optional)
img = cv2.imread('images.jpeg', 0)
img = cv2.resize(img, (8, 8))
img = 16 - (img // 16) # Normalize to 0–16 range
img_flat = img.flatten().reshape(1, -1)
predicted_digit = knn.predict(img_flat)
print("Predicted Digit:", predicted_digit[0])