# Syed Javed CYFI 445 Forensic Data Analysis - Final Project (meso-4)

Source Code: https://github.com/kiteco/python-youtube-code/tree/master/Deepfake-detection

Source Code YouTube video: https://www.youtube.com/watch?v=kYeLBZMTLjk

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
