import cv2
import pyzed.sl as sl
import numpy as np
import os
import time

image_dir = "/shared"
flag_file = "/shared/capture_complete.flag"

def zed_image_capture_auto(init_params):
    # Create a ZED camera object
    zed = sl.Camera()

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit()

    runtime_params = sl.RuntimeParameters()
    mat = sl.Mat() 
    
    err = zed.grab(runtime_params) 
    if err == sl.ERROR_CODE.SUCCESS: # Check that a new image is successfully acquired
        zed.retrieve_image(mat, sl.VIEW.LEFT) # Retrieve left image
        cvImage = mat.get_data() # Convert sl.Mat to cv2.Mat
        
        # Get current date and time
        current_time = time.strftime("%Y%m%d_%H%M%S")
        
        # Define the directory to save the image
        image_dir = "/shared/captured_images"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # Define the file name with date and time
        filename = os.path.join(image_dir, f"{current_time}.png")
        
        # Save the image
        cv2.imwrite(filename, cvImage)
        # Erstelle die Flag-Datei
        with open(flag_file, 'w') as f:
            f.write(filename)
        print(f'Image captured and saved as {filename}')

    else:
        print("Error during capture : ", err)


    # Close the camera 
    zed.close()


if __name__ == "__main__":
    # Load init_params from configuration file
    init_params = sl.InitParameters()
    init_params.load("init_params.conf")
    zed_image_capture_auto(init_params)
