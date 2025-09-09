import os
from flask import Flask, request, render_template, send_file
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_image(src: str, output_path: str):
    # Step 1: Linear enhancement
    im = Image.open(src).convert("RGB")
    dt = np.array(im.getdata(), dtype=np.uint8).reshape((im.height, im.width, 3))
    for i, (mn, mx) in enumerate(im.getextrema()):
        dt[:, :, i] = (dt[:, :, i] - mn) / (mx - mn) * 255
    enhanced_image = Image.fromarray(dt, mode="RGB")
    enhanced_image.save(output_path)

    # Step 2: NDVI Calculation
    w, h = enhanced_image.width, enhanced_image.height
    b0 = np.array(enhanced_image.getdata(0), dtype=np.float32).reshape((h, w))
    b1 = np.array(enhanced_image.getdata(1), dtype=np.float32).reshape((h, w))
    ndvi = np.divide(b0 - b1, b0 + b1, where=(b0 + b1) != 0)
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    dt = np.vectorize(lambda _: 255 if 0.2 <= _ <= 0.9 else 0, otypes=[np.uint8])(ndvi)
    imx = Image.fromarray(dt, mode='L')
    imx.save(output_path.replace(".jpeg", "-ndvi.jpeg"))

    # Step 3: Kernel Application
    newkernel = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1]
    ])
    for i in range(0, h - 7):
        for j in range(0, w - 7):
            b = dt[i:i + 7, j:j + 7]
            r = sum(b[k, m] * newkernel[k, m] for k in range(7) for m in range(7)) / 49
            dt[i + 3, j + 3] = r if 30 <= r <= 50 else 0
    imx = Image.fromarray(dt, mode='L')
    imx.save(output_path.replace(".jpeg", "-content5.jpeg"))

    # Step 4: Resizing and Block-wise Processing
    newheight = (h // 50) * 50
    newwidth = (w // 50) * 50
    newsize = (newwidth, newheight)
    im1 = imx.resize(newsize)
    dt1 = np.array(im1)
    dt = dt1.reshape(newheight, newwidth)

    b1 = []
    for i in range(0, newheight, 50):
        for j in range(0, newwidth, 50):
            b = dt[i:i + 50, j:j + 50]
            c = np.sum(b >= 40)
            b1.append([i, i + 50, j, j + 50, c])

    tree = [x for x in b1 if x[4] >= 50]
    nontree = [x for x in b1 if x[4] < 50]

    print("Tree list:", tree)
    print("Non-tree list:", nontree)

    # Step 5: Visualization
    im = Image.open(output_path.replace(".jpeg", "-content5.jpeg"))
    dt = np.array(im, dtype=np.uint8)
    imx = im.copy()
    draw = ImageDraw.Draw(imx)
    block_size = 50
    for i in range(0, dt.shape[0] - block_size + 1, block_size):
        for j in range(0, dt.shape[1] - block_size + 1, block_size):
            block = dt[i:i + block_size, j:j + block_size]
            R = np.sum(block)
            left, top, right, bottom = j, i, j + block_size, i + block_size
            if R > 10:
                tree_label, rect_color, label_color = 'tree', 'green', 'white'
            else:
                tree_label, rect_color, label_color = 'NT', 'black', 'red'
            draw.rectangle([left, top, right, bottom], outline=rect_color, width=2)
            label_x, label_y = left + (block_size // 2), top + (block_size // 2)
            draw.text((label_x, label_y), tree_label, fill=label_color)

    labeled_image_path = output_path.replace(".jpeg", "-labeled.jpeg")
    imx.save(labeled_image_path)
    return labeled_image_path

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            output_path = os.path.join(OUTPUT_FOLDER, file.filename)
            labeled_image_path = process_image(file_path, output_path)

            return send_file(labeled_image_path, as_attachment=True)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
