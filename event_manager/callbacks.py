from utils.utils import open_yaml
import CONSTANTS
from playsound import playsound
config = open_yaml(CONSTANTS.CONFIG_PATH, 'callbacks')
from threading import Event, Thread
########### ALARM ###########
class Alarm:
    def __init__(self):
        self._stop_event = Event()
        self._alarm_thread = None
    def alarm(self):
        self._stop_event.clear()
        def _play_loop():
            while not self._stop_event.is_set():
                playsound(config['alarm']['sound'])
                if self._stop_event.is_set():
                    break
        self._alarm_thread = Thread(target=_play_loop, daemon=True)
        self._alarm_thread.start()
    def stop_alarm(self):
        self._stop_event.set()
        self._alarm_thread.join()
        self._alarm_thread = None
alarm = Alarm()
########### ALARM ###########
if __name__ == '__main__':
    alrm = Alarm()
    alrm.alarm()
    import time
    time.sleep(5)
    alrm.stop_alarm()