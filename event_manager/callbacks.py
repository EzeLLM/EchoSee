from utils.utils import open_yaml
import CONSTANTS
from playsound import playsound
config = open_yaml(CONSTANTS.CONFIG_PATH, 'callbacks')
def alarm():
    playsound(config['alarm']['sound'])

if __name__ == '__main__':
    alarm()