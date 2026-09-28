
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Load your trained MNIST model
model = load_model("mnist_model.keras")

# Canvas settings
CANVAS_SIZE = 280
BRUSH_SIZE = 20

# Create application window
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("420x550")
root.resizable(False, False)

# Drawing canvas
canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross"
)
canvas.pack(pady=20)

# PIL image stores the actual drawing
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)

# Result display
result_label = tk.Label(
    root,
    text="Draw a digit (0-9)",
    font=("Arial", 20, "bold")
)
result_label.pack(pady=10)

confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)
confidence_label.pack(pady=5)


# Draw using mouse
def start_draw(event):
    draw_digit(event)


def draw_digit(event):
    x, y = event.x, event.y
    r = BRUSH_SIZE // 2

    # Draw on visible canvas
    canvas.create_oval(
        x-r, y-r, x+r, y+r,
        fill="white",
        outline="white"
    )

    # Draw on PIL image
    draw.ellipse(
        [x-r, y-r, x+r, y+r],
        fill=255
    )


# Preprocess drawing for MNIST
def preprocess():
    img = image.copy()

    # Find bounding box of digit
    bbox = img.getbbox()

    if bbox is None:
        return None

    img = img.crop(bbox)

    # Resize digit while preserving aspect ratio
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place digit at center of 28x28 image
    new_img = Image.new("L", (28, 28), 0)

    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    new_img.paste(img, (x, y))

    # Normalize pixel values
    img_array = np.array(new_img).astype("float32") / 255.0

    # Flatten for ANN input
    img_array = img_array.reshape(1, 784)

    return img_array


# Predict digit
def predict_digit():
    processed = preprocess()

    if processed is None:
        result_label.config(text="Please draw a digit!")
        confidence_label.config(text="")
        return

    predictions = model.predict(processed, verbose=0)[0]

    predicted_digit = np.argmax(predictions)
    confidence = predictions[predicted_digit] * 100

    result_label.config(
        text=f"Predicted Digit: {predicted_digit}"
    )

    confidence_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )


# Clear canvas
def clear_canvas():
    canvas.delete("all")

    global image, draw
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(image)

    result_label.config(text="Draw a digit (0-9)")
    confidence_label.config(text="")


# Mouse events
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 14),
    width=12,
    command=predict_digit
)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 14),
    width=12,
    command=clear_canvas
)
clear_button.grid(row=0, column=1, padx=10)

root.mainloop()