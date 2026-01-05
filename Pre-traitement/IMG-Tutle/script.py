from turtle import *
import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab
from PIL import EpsImagePlugin
import numpy as np

def create_segmentation_from_turtle(output_path="segmentation_mask.png"):
    # Get the turtle canvas
    canvas = getcanvas()

    # Save the canvas as PostScript
    ps_path = "temp_canvas.eps"
    canvas.postscript(file=ps_path)

    # Open the PostScript file and convert to PNG
    img = Image.open(ps_path)
    img = img.convert('RGB')

    # Create segmentation mask by replacing white/light colors with black
    pixels = np.array(img)

    # Replace white and light gray background with black
    # Keep colors that are distinctly blue or green
    mask = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] > 200) & (pixels[:, :, 2] > 200)  # White areas

    pixels[mask] = [0, 0, 0]  # Replace with black

    # Create final image
    seg_image = Image.fromarray(pixels)
    seg_image.save(output_path)
    print(f"Segmentation mask saved to: {output_path}")

    # Clean up temporary file
    import os
    if os.path.exists(ps_path):
        os.remove(ps_path)

    return seg_image

def draw_shapes():
    # Set up the screen
    setup(800, 800)
    bgcolor("lightgray")
    speed(0)

    # Draw white square area (canvas)
    penup()
    goto(-200, 200)
    pendown()
    color("black")
    fillcolor("white")
    begin_fill()
    for _ in range(4):
        forward(400)
        right(90)
    end_fill()

    # Draw blue rectangle in top right
    penup()
    goto(50, 150)
    pendown()
    color("blue")
    fillcolor("blue")
    begin_fill()
    for _ in range(2):
        forward(100)
        right(90)
        forward(50)
        right(90)
    end_fill()

    # Draw green circle in bottom left
    penup()
    goto(-100, -100)
    pendown()
    color("green")
    fillcolor("green")
    begin_fill()
    circle(50)
    end_fill()
    hideturtle()

# Draw the shapes
draw_shapes()

# Create segmentation mask from the turtle drawing
create_segmentation_from_turtle(output_path="segmentation_mask.png")

done()
