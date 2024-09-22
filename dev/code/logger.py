import os
import logging

class Logger:
    def __init__(self):
        # Create the log folder if it doesn't exist
        if not os.path.exists('dev/log'):
            os.makedirs('dev/log')

        # Set up the logger
        logging.basicConfig(filename='dev/log/app.log', level=logging.DEBUG)

    def log(self, message):
        logging.info(message)