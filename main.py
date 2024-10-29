from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from picamzero import Camera
import time
import svgwrite
import ezdxf
from xml.dom import minidom
app = Flask(__name__)
IMAGE_FOLDER = "./static/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

captured_images = []  # List to store filenames of captured images

def capture_image():
        camera = Camera()
        camera.start_preview()
        time.sleep(2)  # Allow camera to adjust settings
        filename = f"{IMAGE_FOLDER}/image_{len(captured_images)}.png"
        camera.take_photo(filename)
        return filename

def process_images():
    if not captured_images:
        raise ValueError("No images to process.")

    combined_image = None

    # Combine all captured images into one PNG
    for image_file in captured_images:
        img = cv2.imread(image_file)
        if combined_image is None:
            combined_image = img
        else:
            combined_image = np.vstack((combined_image, img))

    # Save the combined image
    combined_image_path = f"{IMAGE_FOLDER}/combined_image.png"
    cv2.imwrite(combined_image_path, combined_image)

    # Extract contours
    gray = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an SVG file with contours and a rectangle
    svg_file_path = f"{IMAGE_FOLDER}/output.svg"
    create_svg_with_contours(svg_file_path, contours)

    return svg_file_path

def create_svg_with_contours(svg_file_path, contours, rect_size=(50, 50)):
    dwg = svgwrite.Drawing(svg_file_path, profile='tiny', size=(640, 480))

    # Create a rectangle at the top-left corner
    dwg.add(dwg.rect(insert=(0, 0), size=rect_size, fill='none', stroke='black'))

    # Add contours to SVG
    for contour in contours:
        points = [(pt[0][0], pt[0][1]) for pt in contour]
        dwg.add(dwg.polygon(points, fill='none', stroke='black'))

    dwg.save()

def convert_svg_to_dxf(svg_file_path, dxf_file_path):
    # Load SVG file and parse it
    svg_doc = minidom.parse(svg_file_path)
    paths = svg_doc.getElementsByTagName("polygon")  # Only works if contours are saved as <polygon>

    # Create a new DXF document
    dwg = ezdxf.new(dxfversion='R2010')
    msp = dwg.modelspace()

    # Parse each polygon and add as polyline in DXF
    for path in paths:
        points_str = path.getAttribute("points")
        points = []
        
        # Parse the points in "x1,y1 x2,y2 ... xn,yn" format
        for point_str in points_str.split():
            x, y = map(float, point_str.split(","))
            points.append((x, y))
        
        # Add polyline to DXF
        if points:
            msp.add_lwpolyline(points, is_closed=True)

    # Save the DXF document
    dwg.saveas(dxf_file_path)
    svg_doc.unlink()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    try:
        filename = capture_image()
        captured_images.append(filename)
        return jsonify({"image_url": f"/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process", methods=["POST"])
def process():
    try:
        svg_path = process_images()
        return jsonify({"svg_url": f"/{svg_path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/convert_to_dxf", methods=["POST"])
def convert_to_dxf():
    svg_file_path = f"{IMAGE_FOLDER}/output.svg"  # Ensure this matches your SVG output path
    dxf_file_path = f"{IMAGE_FOLDER}/output.dxf"
    
    try:
        convert_svg_to_dxf(svg_file_path, dxf_file_path)
        return send_file(dxf_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
