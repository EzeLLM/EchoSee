"""Echonoti notification wrapper for sending notifications."""

import os
import requests
from datetime import datetime
from typing import Dict, Any


ECHONOTI_API_URL = os.environ.get("ECHONOTI_API_URL", "http://localhost:3000/api/notifications")


def send_notification(
    headline: str, 
    summary: str, 
    content: str, 
    notification_type: str
) -> Dict[str, Any]:
    """Send a notification through the Echonoti service.
    
    Args:
        headline: Brief title for the notification
        summary: Short description for notification preview
        content: Detailed notification content (supports markdown)
        notification_type: Category/type of notification
        
    Returns:
        Dict with notification status and details
    """
    try:
        payload = {
            "headline": headline,
            "summary": summary,
            "content": content,
            "type": notification_type,
            "createdAt": datetime.utcnow().isoformat()
        }
        
        response = requests.post(ECHONOTI_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}