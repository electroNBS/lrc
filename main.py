from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import cv2
import numpy as np
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# from picamzero import Camera
import svgwrite
import ezdxf
from xml.dom import minidom


app = Flask(__name__)
IMAGE_FOLDER = "./static/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# UNCOMMENT BEFORE USING WITH RASPI AND COMMENT THE NEXT LINE
# count = 1
count = 2


# UNCOMMENT BEFORE USING WITH RASPI AND COMMENT THE NEXT LINE
# def capture_image(n):
#      os.system("rpicam-still -e png -o image"+str(n)+".png")
#      count+=1
def capture_image():
    return 1


def process_images():
    # Get offset and tolerance from the request
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Expected JSON data"}), 400
    offset = int(data.get("offset", 10))  # Default to 10 if not provided
    tolerance = int(data.get("tolerance", 5))  # Default to 5 if not provided

    # Retrieve and sort the image files
    image_files = sorted(
        [
            f
            for f in os.listdir(IMAGE_FOLDER)
            if f.startswith("image") and f.endswith(".png")
        ]
    )
    count = len(image_files)
    n = int(np.ceil(np.sqrt(count)))  # Number of images per row/column in the grid

    # Determine canvas size based on maximum image dimensions
    max_width = max(cv2.imread(f"{IMAGE_FOLDER}/{f}").shape[1] for f in image_files)
    max_height = max(cv2.imread(f"{IMAGE_FOLDER}/{f}").shape[0] for f in image_files)

    # Create a white canvas to hold all images
    canvas_width = max_width * n
    canvas_height = max_height * n
    canvas = (
        np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    )  # Initialize with white background

    # Process each image and place it on the canvas
    for idx, image_file in enumerate(image_files):
        img = cv2.imread(f"{IMAGE_FOLDER}/{image_file}")

        # Convert to grayscale and threshold to binary
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(
            gray_img, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # Process tolerance using dilation and erosion
        kernel = np.ones((tolerance, tolerance), np.uint8)
        dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
        eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

        # Apply Gaussian blur to make edges smoother and more connected
        blurred_img = cv2.GaussianBlur(eroded_img, (1, 1), 0)

        # Find contours after tolerance adjustments
        edges = cv2.Canny(blurred_img, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        simplified_contours = []
        for contour in contours:
            epsilon = 0.00000000000000000000001 * cv2.arcLength(
                contour, True
            )  # Tuning this factor can control smoothness
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(simplified_contour)

        # Create a white background for the contour image
        contour_img = np.ones_like(gray_img) * 255  # White background
        # Draw contours in black
        cv2.drawContours(contour_img, simplified_contours, -1, (0, 0, 0), 1)

        # Calculate position on canvas
        row, col = divmod(idx, n)
        y_start, x_start = row * max_height, col * max_width

        # Place the processed image on the canvas
        canvas[
            y_start : y_start + contour_img.shape[0],
            x_start : x_start + contour_img.shape[1],
        ] = contour_img

    # Save the combined image
    combined_image_path = f"{IMAGE_FOLDER}/combined_image.png"
    cv2.imwrite(combined_image_path, canvas)

    # Extract contours from the combined image if needed
    _, combined_binary_img = cv2.threshold(
        canvas, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    final_edges = cv2.Canny(combined_binary_img, 50, 150)

    final_contours, _ = cv2.findContours(
        final_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create SVG with final contours and offset rectangle
    svg_file_path = f"{IMAGE_FOLDER}/output.svg"
    # create_svg_with_contours(
    #     svg_file_path, final_contours, offset, canvas_width, canvas_height
    # )
    png_to_svg(combined_image_path, svg_file_path)

    return svg_file_path, combined_image_path


def png_to_svg(png_path, svg_path):
    # Path to your ChromeDriver executable
    chrome_driver_path = "./chromedriver.exe"

    # Path to the input PNG file and output SVG file
    input_file_path = os.path.abspath(png_path)
    output_file_path = os.path.abspath(svg_path)

    # Set up Chrome options for file download
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": os.path.dirname(output_file_path),
        "download.prompt_for_download": False,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Initialize WebDriver
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 20)

    try:
        # Step 1: Go to the PNG-to-SVG conversion page
        driver.get("https://image.online-convert.com/convert/png-to-svg")

        # Step 2: Find and interact with the file upload button
        file_upload = wait.until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "input[type='file']",
                )
            )
        )
        file_upload.send_keys(input_file_path)  # Automatically uploads the file

        # Step 3: Start the conversion process
        convert_button = wait.until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "button[class='btn btn-lg submit-btn white-space-nowrap mb-0 d-flex align-items-center gap-10']",
                )
            )
        )
        convert_button.click()

        # Step 4: Wait for the download link to appear
        download_link = wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="content"]/div[3]/div[1]/div[3]/div/div[3]/div[2]/a',
                )
            )
        )
        download_url = download_link.get_attribute("href")
        # print(download_url)
        # Step 5: Download the file using Selenium
        driver.get(download_url)
        time.sleep(10)  # Give it time to download

        # Rename the downloaded file to the specified output path
        download_dir = prefs["download.default_directory"]
        for file_name in os.listdir(download_dir):
            if file_name.endswith(".svg"):
                os.rename(os.path.join(download_dir, file_name), output_file_path)
                print(f"SVG file downloaded as {output_file_path}")
                break

    finally:
        driver.quit()


def convert_svg_to_dxf(svg_file_path, dxf_file_path):
    # Load SVG file and parse it
    svg_doc = minidom.parse(svg_file_path)
    paths = svg_doc.getElementsByTagName("path")  # Look for <path> elements

    # Create a new DXF document
    dwg = ezdxf.new(dxfversion="R2010")
    msp = dwg.modelspace()

    # Parse each path and add as polyline in DXF
    for path in paths:
        d = path.getAttribute("d")
        if not d:
            print("No 'd' attribute found for path.")
            continue

        print(f"Processing path with d attribute: {d}")  # Debugging line
        points = []

        # Regular expression to extract commands and coordinates
        commands = re.findall(r"([MmLlZz])([^MmLlZz]*)", d)

        current_position = None

        for command, coord_str in commands:
            coords = coord_str.strip().split()
            coords = [
                coord.replace(",", ".") for coord in coords if coord
            ]  # Replace commas with dots

            # Convert coordinates to float
            try:
                coords = [float(coord) for coord in coords]
            except ValueError as e:
                print(f"Error converting coordinates: {coords}. Error: {e}")
                continue  # Skip to the next command if there is an error

            if command in ["M", "m"]:  # Move to
                if len(coords) >= 2:
                    current_position = (coords[0], coords[1])
                    if command == "m":  # Relative move
                        current_position = (
                            current_position[0] + points[-2][0],
                            current_position[1] + points[-2][1],
                        )
            elif command in ["L", "l"]:  # Line to
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point = (coords[i], coords[i + 1])
                        if command == "l":  # Relative line
                            point = (
                                point[0] + current_position[0],
                                point[1] + current_position[1],
                            )
                        points.append(point)
                        current_position = point
            elif command in ["Z", "z"]:  # Close path
                if points and current_position:
                    points.append(points[0])  # Close the loop

        # Add polyline to DXF if points are found
        if points:
            msp.add_lwpolyline(points, is_closed=True)
            print(f"Added polyline with points: {points}")  # Debugging line
        else:
            print("No points found for this path.")

    # Save the DXF document
    dwg.saveas(dxf_file_path)
    print(f"DXF saved at: {dxf_file_path}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    try:
        # UNCOMMENT BEFORE USING WITH RASPI AND COMMENT THE NEXT LINE
        # capture_image(count)
        capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process_image", methods=["POST"])
def process():
    try:
        svg_file_path, processed_image_path = process_images()
        return jsonify({"svg_url": svg_file_path, "image_url": processed_image_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/convert_to_dxf", methods=["POST"])
def convert_to_dxf():
    svg_file_path = (
        f"{IMAGE_FOLDER}/SVGnest-output.svg"  # Ensure this matches your SVG output path
    )
    dxf_file_path = f"{IMAGE_FOLDER}/output.dxf"

    try:
        convert_svg_to_dxf(svg_file_path, dxf_file_path)
        return send_file(dxf_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/index2")
def index2():
    # Render the SVGnest index page as a template
    return render_template("svgnest_index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
