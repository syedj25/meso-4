# Syed Javed CYFI 445 Forensic Data Analysis - Final Project (meso-4)

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

 ## Original Source Code

- Original Source Code: https://github.com/kiteco/python-youtube-code/tree/master/Deepfake-detection

- Original Source Code YouTube video: https://www.youtube.com/watch?v=kYeLBZMTLjk
