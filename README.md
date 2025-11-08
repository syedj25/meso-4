# Syed Javed CYFI 445 Forensic Data Analysis - Final Project (meso-4)

## Original Source Code
- Original Source Code: https://github.com/kiteco/python-youtube-code/tree/master/Deepfake-detection
- Original Source Code YouTube video: https://www.youtube.com/watch?v=kYeLBZMTLjk

## Abstract
Deepfake technology poses significant challenges
for digital forensics, as it enables the creation of highly realistic
fake media that can confuse investigations and compromise the
reliability of evidence. This study examines the application of
Artificial Intelligence (AI) and Machine Learning (ML) methods
to detect deepfake images in forensic settings. We used the Meso-4
Convolutional Neural Network (CNN), which is a compact design
designed for identifying face alterations, and tested it on a set of
real and edited images gathered from real videos and Deepfake
videos. The process included preparing the data, training the
model, and checking how well it performs to measure accuracy
and how well it holds up. The results show that the Meso-4 model
works well in telling the difference between real and fake images,
showing its value as a useful tool for forensic experts. These
results emphasize the need to include AI-based tools in forensic
processes to better confirm evidence and reduce the risks from
fake media. Future research will aim to improve the model’s
ability to detect high-quality images and make it more resistant
to attempts to fool it.

## Research Paper
https://www.overleaf.com/read/rmdcsvsxbxyr#6892e3

## 🛠️ Tools for Installation
Make sure you have the following installed before running the project: 

- Python 3.10.11
- numpy
- matplotlib
- tensorflow
- scikit-learn
- PyCharm IDE (prefered to run smoothly)

## 🚀 To Run the code

- PyCharm IDE is prefered to run this code
- Download whole file as Zip file (data, weights, meso-4.ipynb) and unzip
- Open unzip folder in PyCharm IDE and run "meso-4.ipynb"

## 🔄 Updates & Fixes

This project has been updated to align with newer versions of Keras/TensorFlow:

- **Optimizer Argument**
- In newer versions of Keras/TensorFlow, the Adam optimizer no longer accepts the argument lr. Instead, it expects learning_rate.
  ```python
  # Old
  optimizer = Adam(lr = learning_rate)

  # New
  optimizer = Adam(learning_rate=learning_rate)

- **Model Weights Loading**
  ```python
  # Old
  meso.load('./weights/Meso4_DF')

  # New
  meso.load('./weights/Meso4_DF.h5')

- **Generator Iteration**
- In modern Keras/TensorFlow, the DirectoryIterator (from ImageDataGenerator.flow_from_directory) doesn’t have a .next() method. Use Python’s built‑in next() function instead.
  ```python
  # Old
  X, y = generator.next()

  # New
  X, y = next(generator)

- **LeakyReLU Argument**
- The alpha argument in LeakyReLU has been deprecated in favor of negative_slope.
  ```python
  # Old
  y = LeakyReLU(alpha=0.1)(y)

  # New
  y = LeakyReLU(negative_slope=0.1)(y)

## 📌 Modified and Added

This block evaluates the Meso4 model by predicting labels for each image, categorizing correct and incorrect classifications, calculating overall accuracy, and visualizing how accuracy progresses as more predictions are made.

```python
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
 
# Lists for classification results
correct_real, correct_real_pred = [], []
correct_deepfake, correct_deepfake_pred = [], []
misclassified_real, misclassified_real_pred = [], []
misclassified_deepfake, misclassified_deepfake_pred = [], []

# Lists for metrics
true_labels, predicted_labels = [], []

# Iterate once through the generator
for i in range(len(generator.labels)):
    # Load next picture and predict
    X, y = next(generator)
    pred = meso.predict(X)[0][0]
    rounded_pred = int(round(pred))

    # Store true and predicted labels
    true_labels.append(int(y[0]))
    predicted_labels.append(rounded_pred)

    # Sort into categories
    if rounded_pred == y[0] and y[0] == 1:
        correct_real.append(X)
        correct_real_pred.append(pred)
    elif rounded_pred == y[0] and y[0] == 0:
        correct_deepfake.append(X)
        correct_deepfake_pred.append(pred)
    elif y[0] == 1:
        misclassified_real.append(X)
        misclassified_real_pred.append(pred)
    else:
        misclassified_deepfake.append(X)
        misclassified_deepfake_pred.append(pred)

    # Status update
    if i % 1000 == 0:
        print(i, ' predictions completed.')
    if i == len(generator.labels) - 1:
        print("All", len(generator.labels), "predictions completed")

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"\n--- Accuracy Metrics ---")
print(f"Accuracy: {accuracy:.2f}")

# Accuracy progression graph
accuracies = []
current_true, current_pred = [], []

for i in range(len(true_labels)):
    current_true.append(true_labels[i])
    current_pred.append(predicted_labels[i])
    acc = accuracy_score(current_true, current_pred)
    accuracies.append(acc)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(accuracies)+1), accuracies, marker='o', linestyle='-', color='darkorange')
plt.title('Meso4 Model Accuracy Progression')
plt.xlabel('Number of Predictions')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_progression_graph.png')
plt.show()
