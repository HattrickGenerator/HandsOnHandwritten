import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Constants for the drawing area
GRID_SIZE = 28
CANVAS_SIZE = 280
SCALE = CANVAS_SIZE // GRID_SIZE

class Drawer:
    def __init__(self, root, estimator=None):
        self.root = root
        self.root.title("Try the estimator yourself!")
        self.estimator = estimator

        self.img = Image.new("L", (GRID_SIZE, GRID_SIZE), 255)
        self.draw = ImageDraw.Draw(self.img)

        self.canvas = tk.Canvas(self.root, height=CANVAS_SIZE, bg='white', width=CANVAS_SIZE)
        self.canvas.pack()

        self.old_x = None
        self.old_y = None
        self.save_button = tk.Button(self.root, text="Estimate Number", command=self.save_and_estimate)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.save_button.pack()

    def on_mouse_click(self, event):
        self.old_x, self.old_y = event.x, event.y

    def paint(self, event):
        if self.old_x is not None and self.old_y is not None:
            # Draw on the canvas
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y, fill='black',
                width=SCALE, capstyle=tk.ROUND, smooth=tk.TRUE)

            # Draw on the small image
            scaled_event_x = event.x // SCALE
            scaled_event_y = event.y // SCALE
            scaled_old_x = self.old_x // SCALE
            scaled_old_y = self.old_y // SCALE
            self.draw.line(
                (scaled_old_x, scaled_old_y, scaled_event_x, scaled_event_y),
                fill=0, width=1
            )

        self.old_x, self.old_y = event.x, event.y

    def reset(self):
        self.old_x, self.old_y = None, None
        self.canvas.delete("all")
        self.img = Image.new("L", (GRID_SIZE, GRID_SIZE), 255)
        self.draw = ImageDraw.Draw(self.img)

    def get_scaled_pixel_vector(self):
        # Convert the PIL Image to a NumPy array
        array = 1 - np.array(self.img, dtype='float32') / 255.0
        # Ensure the array is in the correct shape (in case the dimensions need adjustment)
        img_array = np.reshape(array, (1, GRID_SIZE * GRID_SIZE))
        return img_array

    def save_and_estimate(self):
        filename = "digit.png"
        self.img.save(filename)
        print(f"Image saved as {filename}")

        # Estimate
        if self.estimator:
            vec = self.get_scaled_pixel_vector()
            estimated_vec = self.estimator.predict(vec)
            print(f'Estimator predicts that the number drawn is {np.argmax(estimated_vec)}')

        self.reset()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()

    model = tf.keras.models.load_model('models/mnist')
    app = Drawer(root, model)
    app.run()
