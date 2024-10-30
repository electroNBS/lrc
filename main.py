from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from picamzero import Camera
import svgwrite
import ezdxf
from xml.dom import minidom
app = Flask(__name__)
IMAGE_FOLDER = "./static/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

count = 1

def capture_image(n):
     os.system("rpicam-still -e png -o image"+str(n)+".png")
     count+=1
def capture_image():
    return 

def process_images():
    n = int(np.ceil(np.sqrt(count)))
    # Combine all captured images into one PNG
    # Save the combined image
    os.system(f"montage {IMAGE_FOLDER}/*.png -tile "+str(n)+"x"+str(n)+" image.png")

   # Load the combined montage image
    combined_image_path = "image.png"
    img = cv2.imread(combined_image_path)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

    # Find initial contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process tolerance using dilation and erosion
    tolerance = 5  # Example tolerance value for contour thickness
    kernel = np.ones((tolerance, tolerance), np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

    # Re-find contours on the tolerance-processed image
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply offset to each contour
    offset = 10  # Example offset value (in pixels)
    offset_contours = []
    for contour in contours:
    # Offset each point in the contour
        offset_contour = np.array([[[point[0][0] + offset, point[0][1] + offset]] for point in contour])
        offset_contours.append(offset_contour)

# Draw offset contours on a blank image for preview
    contour_img = np.zeros_like(gray_img)
    cv2.drawContours(contour_img, offset_contours, -1, (255), thickness=cv2.FILLED)

# Save or display the final image with contours
    cv2.imwrite("contours_with_offset.png", contour_img)

# Show the image with contours for preview (optional)
    cv2.imshow("Contours with Offset", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        #capture_image(count)
        capture_image()
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
