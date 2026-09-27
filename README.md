# AI-ML-RECRUITMENT-2026-LAKSHYA-CHOUDHARY-
# Task 2: Neural Network – MNIST Handwritten Digit Classification

## 1. Candidate Details

| Field          | Details                                      |
| -------------- | -------------------------------------------- |
| Name           | Lakshya Choudhary                            |
| Degree         | B.Tech Computer Science and Engineering      |
| Specialization | Cloud Computing                              |
| Institution    | SRM Institute of Science and Technology, KTR |
| Recruitment    | AI/ML Recruitment 2026                       |

## 2. Task Completed

**Task 2 – Neural Network: MNIST Handwritten Digit Classification**

Status: Completed

## 3. Problem Statement

The objective of this task was to develop and train an Artificial Neural Network (ANN) capable of recognizing handwritten digits from images.

The model uses the MNIST dataset, which consists of grayscale images of handwritten digits ranging from 0 to 9.

The goal was to build a neural network that learns patterns from pixel values and accurately classifies unseen handwritten digit images into their respective categories.

## 4. Approach

1. Loaded the MNIST dataset using TensorFlow/Keras.
2. Explored the dataset dimensions, image shapes, pixel values, and labels.
3. Normalized pixel intensities from the range 0–255 to 0–1.
4. Flattened each 28 × 28 image into a 784-dimensional input vector.
5. Designed a feedforward Artificial Neural Network.
6. Configured the model using appropriate activation functions, optimizer, and loss function.
7. Trained the neural network using the training dataset.
8. Generated predictions on the test dataset.
9. Evaluated model performance using accuracy, precision, recall, and F1-score.
10. Saved the trained model in `.keras` format for future inference.

## 5. Technologies Used

| Technology       | Purpose                                       |
| ---------------- | --------------------------------------------- |
| Python           | Programming language                          |
| TensorFlow       | Deep learning framework                       |
| Keras            | Neural network development and training       |
| NumPy            | Numerical computations and array manipulation |
| Matplotlib       | Image visualization and plotting              |
| Scikit-learn     | Classification evaluation metrics             |
| Jupyter Notebook | Interactive development and experimentation   |
| Git & GitHub     | Version control and project hosting           |

## 6. Results

The trained Artificial Neural Network successfully performs handwritten digit classification across 10 classes (0–9).

### Evaluation Metrics

| Metric        | Result                 |
| ------------- | ---------------------- |
| Test Accuracy | Add your actual result |
| Precision     | Add your actual result |
| Recall        | Add your actual result |
| F1-score      | Add your actual result |

### Model Artifact

The trained model is saved as:

`mnist_model.keras`

This file contains the trained neural network architecture and learned weights, allowing the model to be loaded for predictions without retraining.

## 7. Key Learnings

1. **Neural Network Fundamentals:** Understood neurons, weights, biases, activation functions, forward propagation, and how neural networks learn patterns from data.

2. **Gradient Descent and Optimization:** Learned how loss functions measure prediction errors and how optimization algorithms update weights to minimize the loss.

3. **Image Preprocessing:** Understood the importance of normalizing pixel values and flattening image matrices before feeding them into a fully connected neural network.

4. **Model Evaluation:** Learned how accuracy, precision, recall, and F1-score provide different perspectives on classification performance.

5. **Model Persistence:** Gained practical experience saving and loading trained neural networks using the Keras `.keras` format.

## 8. Challenges and Solutions

### Challenge 1: Preparing Image Data for the Neural Network

**Problem:** MNIST images are represented as 28 × 28 pixel matrices, whereas a fully connected neural network requires a one-dimensional input vector.

**Solution:** Flattened each image into a 784-dimensional vector and normalized pixel values from 0–255 to 0–1.

### Challenge 2: Understanding Model Performance

**Problem:** Accuracy alone does not provide a complete understanding of classification performance across individual digit classes.

**Solution:** Evaluated the model using precision, recall, and F1-score alongside accuracy to obtain a more comprehensive assessment.

### Challenge 3: Saving the Trained Model

**Problem:** Retraining the neural network every time predictions were required would be inefficient.

**Solution:** Saved the trained model as `mnist_model.keras`, allowing it to be loaded and reused for future predictions.

---

## Project Structure

```text
Task_2/
│
├── MNIST.ipynb
├── mnist_model.keras
└── README.md
```

## Loading the Trained Model

```python
from tensorflow.keras.models import load_model

model = load_model("mnist_model.keras")
```

The loaded model can be used to classify appropriately preprocessed handwritten digit images.

---

**Task Status: Completed**

**Submitted by:** Lakshya Choudhary
**AI/ML Recruitment Assignment – 2026**
