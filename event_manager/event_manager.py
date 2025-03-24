import time
import threading
from datetime import datetime, timedelta
import heapq
from typing import Callable, Any, Tuple
import logging

class EventManager:
    def __init__(self):
        self.events = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger("EventManager")
        self.event_counter = 0  # For unique event IDs
        
    def start(self) -> None:
        """Start the event manager thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info("Event manager started")
    
    def stop(self) -> None:
        """Stop the event manager thread gracefully"""
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()
            self.logger.info("Event manager stopped")
    
    def add_event(
        self,
        event_time: datetime,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> int:
        """
        Add a new event to be triggered at event_time
        Returns event ID that can be used to cancel the event
        """
        # check if event already exists
        for event_t,_,call_b,args_,kwargs_ in self.events:
            if event_t == event_time and call_b == callback and args_ == args and kwargs_ == kwargs:
                return -1
        with self.lock:
            self.event_counter += 1
            event_id = self.event_counter
            heapq.heappush(
                self.events, 
                (event_time, event_id, callback, args, kwargs)
            )
            self.logger.debug(f"Added event {event_id} for {event_time}")
            return event_id
    
    def cancel_event(self, event_id: int) -> bool:
        """Cancel an event by its ID"""
        with self.lock:
            for i, (_, e_id, *_) in enumerate(self.events):
                if e_id == event_id:
                    self.events.pop(i)
                    heapq.heapify(self.events)  # Re-heapify after removal
                    self.logger.debug(f"Cancelled event {event_id}")
                    return True
            return False
    
    def add_recurring_event(
        self,
        interval: timedelta,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> int:
        """
        Add a recurring event that runs at regular intervals
        Returns event ID that can be used to cancel the event
        """
        def recurring_wrapper():
            callback(*args, **kwargs)
            # Reschedule the event
            new_time = datetime.now() + interval
            return self.add_event(new_time, recurring_wrapper)
        
        initial_time = datetime.now() + interval
        return self.add_event(initial_time, recurring_wrapper)
    
    def _run(self) -> None:
        """Main event loop running in a separate thread"""
        self.logger.info("Event manager running")
        
        while self.running:
            now = datetime.now()
            events_to_run = []
            
            with self.lock:
                # Get all events that are due
                while self.events and self.events[0][0] <= now:
                    events_to_run.append(heapq.heappop(self.events))
            
            # Execute callbacks outside the lock
            for event in events_to_run:
                event_time, event_id, callback, args, kwargs = event
                try:
                    self.logger.debug(f"Executing event {event_id}")
                    callback(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error executing event {event_id}: {e}")
            
            # Sleep for a short time to prevent busy waiting
            time.sleep(0.1)

def play_alarm(message):
    print(f"ALARM: {message}")
    # Here you would add code to play sound using winsound, playsound, etc.

# Create and start the event manager
manager = EventManager()
manager.start()

# Add some events
now = datetime.now()
manager.add_event(now + timedelta(seconds=5), play_alarm, "Wake up!")
manager.add_event(now + timedelta(seconds=10), play_alarm, "Time for lunch!")

# Keep the program running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    manager.stop()