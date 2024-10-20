
"""
This class is used only to retrieve images from the camera.
Retrieved images are saved to a temporary directory and the path to the image is returned.
"""

import FuncHub
import cv2
import tempfile
import os
import sys
import time
import logger
class See:
    def __init__(self, config = None):
        self.logger = logger.Logger()
        self.config = FuncHub.open_yaml(config,'See') if config else None
        self.cam_port = self.config['cam_port'] if (self.config and 'cam_port' in self.config) else 0
        self.cam = cv2.VideoCapture(self.cam_port) 
        self.cam.read()
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logger.log('See initialized a fresh instance')
        self.logger.log(f'camera port: {self.cam_port}')
    
    def see(self):
        result, image = self.cam.read() 
        if result: 
            temp_image_path = os.path.join(self.temp_dir.name, f"echosee_temp_{time.time()}.png")
            cv2.imwrite(temp_image_path, image)
            self.logger.log(f"Image saved to {temp_image_path}")
        else: 
            self.logger.log("Failed to take picture in echosee See class. Check camera connection, permissions, and camera port.")
            raise Exception("Failed to take picture in echosee See class. Check camera connection, permissions, and camera port.")
        return temp_image_path
    

if __name__ == "__main__":

    see = See('dev/code/config/echosee.yaml')
    # 1 sec sleep to allow camera to warmup
    time.sleep(1)
    cv2.imshow('image', cv2.imread(see.see()))
    cv2.waitKey(0)