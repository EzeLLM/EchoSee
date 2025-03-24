from langchain_core.tools import tool
from utils.utils import event_manager_instance as em
from event_manager.callbacks import *
from datetime import datetime, timedelta 
event_ids = []

@tool
def set_alarm_at_specific_time(time: str) -> bool:
    """
    Sets an alarm to trigger at an absolute date/time in the future.

    Parameters:
        time (str): Exact trigger time in strict format:
            "YYYY:MM:DD:HH:MM:SS" where:
            - YYYY: 4-digit year (e.g., 2024)
            - MM: Month (01-12, zero-padded)
            - DD: Day (01-31, zero-padded)
            - HH: Hour (00-23, 24-hour format)
            - MM: Minute (00-59)
            - SS: Second (00-59)
            Example: "2024:03:15:14:30:00" = March 15, 2024 at 2:30 PM

    Returns:
        bool: Alarm status:
        - True: Alarm successfully scheduled
        - False: Invalid time format, past time, duplicate alarm, 
                 or invalid date components

    Example:
        >>> set_alarm_at_specific_time("2024:12:25:08:00:00")
        True  # Sets alarm for Christmas morning 2024 at 8 AM

    Notes:
        - All components must be zero-padded (2 digits except year)
        - Time is interpreted in the system's local timezone
        - Automatically rejects past times
        - Performs full date validation (leap years, month lengths, etc)
        - Duplicate check compares exact datetime values
    """
    try:
        event_time = datetime.strptime(time, "%Y:%m:%d:%H:%M:%S")
        if event_time <= datetime.now():
            return False
    except ValueError:
        return False

    state = em.add_event(event_time=event_time, callback=alarm)
    if state != -1:
        event_ids.append(state)
        return True
    return False  # Explicit boolean return for failures


@tool
def set_alarm_with_time_delta(delta: str) -> bool:
    """
    Sets an alarm to trigger after a specified duration from the current time.

    Parameters:
        delta (str): Time duration until alarm trigger. Strict format:
            "DD:HH:MM:SS" where:
            - DD: Days (00-99)
            - HH: Hours (00-23)
            - MM: Minutes (00-59)
            - SS: Seconds (00-59)
            Example: "02:12:30:45" = 2 days 12h 30m 45s

    Returns:
        bool: Alarm status:
        - True: Alarm successfully set
        - False: Alarm already exists or invalid input format
    """
        # Split duration into components
    days, hours, minutes, seconds = map(int, delta.split(':'))
    
    # Create timedelta object
    delta_td = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )
    
    event_time = datetime.now() + delta_td
    state = em.add_event(event_time=event_time, callback=alarm)
    
    if state != -1:
        event_ids.append(state)
        return True
    return False





tools = [set_alarm_at_specific_time]


if __name__ == "__main__":
    # invoke tool
    print(set_alarm_with_time_delta.invoke({"delta": "00:00:00:05"}))
    import time
    time.sleep(10)