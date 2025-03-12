import logging
import os
from utils.utils import open_yaml
class Logger:
    def __init__(self, filename: str):
        self.config = open_yaml('config.yml', 'logger')
        self.logs_dir = self.config['LOGS']
        os.makedirs(self.logs_dir, exist_ok=True)
        self.filename = os.path.join(self.logs_dir, f'{filename}.log')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler = logging.FileHandler(self.filename)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def trial():
    logger = Logger('test')
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
if __name__ == '__main__':
    trial()