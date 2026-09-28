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

---

## 2. Task Completed

**Task 2 – Neural Network: MNIST Handwritten Digit Classification**

**Status: Completed**

The task involved developing and training an Artificial Neural Network (ANN) to classify handwritten digits using the MNIST dataset. An interactive custom input interface was also developed to test the trained model using user-drawn digits.

---

## 3. Problem Statement

The objective of this task was to develop and train an Artificial Neural Network (ANN) capable of recognizing handwritten digits from images.

The model uses the MNIST dataset, which consists of grayscale images of handwritten digits ranging from 0 to 9.

The goal was to build a neural network that learns patterns from pixel values and accurately classifies unseen handwritten digit images into their respective categories.

Additionally, an interactive graphical interface was developed to allow users to draw their own digits and observe the predictions made by the trained model.

---

## 4. Approach

### Dataset Preparation and Preprocessing

1. Loaded the MNIST dataset using TensorFlow/Keras.
2. Explored dataset dimensions, image shapes, pixel values, and labels.
3. Normalized pixel intensities from the range 0–255 to 0–1.
4. Flattened each 28 × 28 image into a 784-dimensional input vector suitable for a fully connected neural network.

### Neural Network Development

5. Designed a feedforward Artificial Neural Network.
6. Configured the model using appropriate activation functions, optimizer, and loss function.
7. Trained the neural network using the training dataset.
8. Generated predictions on the test dataset.
9. Evaluated model performance using accuracy, precision, recall, and F1-score.
10. Saved the trained model in `.keras` format for future inference.

### Interactive Custom Input Interface

11. Developed an interactive graphical interface using Python and Tkinter.
12. Enabled users to draw handwritten digits (0–9) directly on a canvas using their mouse.
13. Implemented preprocessing to resize, normalize, and flatten the user-drawn image to match the MNIST input format.
14. Integrated the saved neural network to predict the digit drawn by the user.
15. Displayed the predicted digit along with its confidence score.
16. Added Predict and Clear buttons to make the application interactive and user-friendly.

---

## 5. Technologies Used

| Technology       | Purpose                                       |
| ---------------- | --------------------------------------------- |
| Python           | Programming language                          |
| TensorFlow       | Deep learning framework                       |
| Keras            | Neural network development and training       |
| NumPy            | Numerical computations and array manipulation |
| Matplotlib       | Image visualization and plotting              |
| Scikit-learn     | Classification evaluation metrics             |
| Pillow (PIL)     | Image processing and preprocessing            |
| Tkinter          | Interactive graphical user interface          |
| Jupyter Notebook | Interactive development and experimentation   |
| Git & GitHub     | Version control and project hosting           |

---

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

### Interactive Custom Input Interface

An interactive handwritten digit recognition application was developed to test the practical working of the trained model.

The application allows users to draw digits using their mouse and obtain predictions from the trained neural network.

**Features:**

* Interactive drawing canvas for handwritten digit input.
* Predict button to classify the drawn digit.
* Clear button to reset the drawing canvas.
* Image preprocessing to match the MNIST input format.
* Displays the predicted digit and confidence score.
* Uses the saved `mnist_model.keras` model without requiring retraining.

The interface demonstrates how a trained neural network can be integrated into a graphical application to perform inference on custom user input.

---

## 7. Key Learnings

1. **Neural Network Fundamentals:** Understood neurons, weights, biases, activation functions, forward propagation, and how neural networks learn patterns from data.

2. **Gradient Descent and Optimization:** Learned how loss functions measure prediction errors and how optimization algorithms update weights to minimize the loss.

3. **Image Preprocessing:** Understood the importance of normalizing pixel values and flattening image matrices before feeding them into a fully connected neural network.

4. **Model Evaluation:** Learned how accuracy, precision, recall, and F1-score provide different perspectives on classification performance.

5. **Model Persistence:** Gained practical experience saving and loading trained neural networks using the Keras `.keras` format.

6. **Interactive Model Deployment:** Learned how to integrate a trained neural network into a graphical user interface, accept custom user input, preprocess images, and perform predictions using the saved model.

7. **End-to-End ML Workflow:** Gained experience connecting dataset preparation, model training, evaluation, model saving, and interactive inference into a complete machine learning application.

---

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

### Challenge 4: Processing Custom User-Drawn Digits

**Problem:** User-drawn digits on the interface have different dimensions and positioning compared to the original MNIST images.

**Solution:** Implemented preprocessing to crop the drawn digit, resize it while preserving its aspect ratio, center it within a 28 × 28 image, normalize pixel values, and flatten it into the required input format.

### Challenge 5: Integrating the Model with a Graphical Interface

**Problem:** The trained neural network needed to accept custom user input and display predictions interactively.

**Solution:** Developed a Tkinter-based interface that captures mouse movements, processes the drawing, passes it to the saved model, and displays the predicted digit with its confidence score.

---

## Project Structure

```text
AI-ML-RECRUITMENT-2026-LAKSHYA-CHOUDHARY-/
│
├── MNIST.ipynb
├── mnist_model.keras
├── custom_input_interface.py
├── README.md
└── .gitignore
```

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/lakshCh123/AI-ML-RECRUITMENT-2026-LAKSHYA-CHOUDHARY-.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-ML-RECRUITMENT-2026-LAKSHYA-CHOUDHARY-
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn pillow
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `MNIST.ipynb` to explore the dataset, train the neural network, and evaluate its performance.

### 5. Run the Interactive Digit Recognition Interface

Ensure that `mnist_model.keras` and `custom_input_interface.py` are present in the same directory.

Execute:

```bash
python custom_input_interface.py
```

A graphical window will open where users can draw handwritten digits and test the trained model.

---

## Loading the Trained Model

The saved model can also be loaded independently using TensorFlow/Keras:

```python
from tensorflow.keras.models import load_model

model = load_model("mnist_model.keras")
```

The loaded model can then be used to classify appropriately preprocessed handwritten digit images without retraining.

---

## Conclusion

This task provided practical experience in developing an end-to-end machine learning solution, from dataset preprocessing and neural network training to model evaluation and interactive inference.

The addition of the custom input interface demonstrates the practical application of the trained model by allowing users to draw handwritten digits and observe the predictions generated by the neural network.

---

**Task Status: Completed**

**Submitted by:** Lakshya Choudhary
**AI/ML Recruitment Assignment – 2026**


