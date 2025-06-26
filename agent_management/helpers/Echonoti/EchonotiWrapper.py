"""
This is just the module that makes the http requests to the echonoti server. no cases are handled here.
"""

import requests
from CONSTANTS import ECHONOTI_API_URL

def send_notification(headline: str, summary: str, content: str, notification_type: str):

    data = {
        "headline": headline,
        "summary": summary,
        "content": content,
        "type": notification_type
    }
    response = requests.post(ECHONOTI_API_URL, json=data)
    return response.json()
