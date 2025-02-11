import pyrealsense2 as rs
import requests
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import cv2
import os
import base64

REALSENSE = 0


# Define the server URL
url = "http://0.0.0.0:8000/predict"  # Note the '/predict' endpoint
# url = "http://127.0.0.1:8000/predict"  # Note the '/predict' endpoint
# url = "http://10.138.177.122:9000/predict"  # Note the '/predict' endpoint

prompt = "detect a person"


if REALSENSE:
    # RealSense camera setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
else:
    # CV2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

try:
    # pipeline.start(config)
    print("Streaming started. Press 'q' to quit.")
    save_dir = "image_files"
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    color_image = None
    depth_image = None

    while True:
        if REALSENSE:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Error: Could not read frame.")
                continue

            # Convert RealSense frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Normalize depth image for display
            depth_image_display = cv2.convertScaleAbs(depth_image, alpha=0.03)

            # Send the POST requestDisplay images
            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_image_display)

            # Convert the color frame to PNG format
            _, buffer = cv2.imencode('.png', color_image)
        else:
            # CV2
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            # cv2.imshow("Webcam Feed", frame)

            # CV2
            _, buffer = cv2.imencode('.png', frame)


        count += 1

        # Prepare the payload
        files = {
            "text_prompt": (None, prompt),
            "image": ("image.png", buffer.tobytes(), "image/png")
        }

        # print("Send the POST request")
        # Send the POST request
        response = requests.post(url, files=files)

        if response.status_code == 200:
            response_json = response.json()
            print(f"Processed output received for frame {count}")

            # Decode the base64-encoded image
            img_data = base64.b64decode(response_json["segmentation_image"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            segmentation_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Display the processed output image
            cv2.imshow("Segmentation Image", segmentation_image)

            # Decode the base64-encoded image
            img_data = base64.b64decode(response_json["bounding_box_image"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            bounding_box_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Display the processed output image
            cv2.imshow("Bounding Box Image", bounding_box_image)

            # Decode the base64-encoded image
            img_data = base64.b64decode(response_json["mask_image"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            # img_array = (img_array * 255).astype(np.uint8)
            mask_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # mask_image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            # print("mask_image: ", mask_image)
            # Display the processed output image
            cv2.imshow("Mask Image", mask_image)




        else:
            print(f"Error: {response.status_code}")
            print(response.text)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count == 100000000:
            print("Break the loop")
            break

finally:
    if REALSENSE:
        pipeline.stop()

    # Save the last captured color and depth images
    segmentation_image_path = os.path.join(save_dir, "segmentation_image.png")
    bounding_box_image_path = os.path.join(save_dir, "bounding_box_image.png")
    mask_image_path = os.path.join(save_dir, "mask_image.png")

    cv2.imwrite(segmentation_image_path, segmentation_image)
    cv2.imwrite(bounding_box_image_path, bounding_box_image)
    cv2.imwrite(mask_image_path, mask_image)

    print(f"Segmentation image saved as {segmentation_image_path}")
    print(f"Bounding box image saved as {bounding_box_image_path}")
    print(f"Mask image saved as {mask_image_path}")

    if not REALSENSE:
        # CV2
        cap.release()

    cv2.destroyAllWindows()
