import agent_management.helpers.Echonoti.EchonotiWrapper as EchonotiWrapper
from langchain.agents import Tool
from utils.utils import llm
from utils.cot import *
from typing import Literal, Dict, Any
import re
import json
import logging
import requests
from langchain_core.tools import tool

@tool
def send_notification(headline: str, summary: str, content: str, notification_type: str) -> bool:
    """
    Send a notification to the user's main device using the Echonoti service.
    
    This tool delivers notifications directly to the user's device (phone, computer, etc.) 
    with structured information including headline, summary, and detailed content.
    
    PARAMETERS:
    
    headline (str, required): 
        - Brief, descriptive title for the notification (max 100 characters)
        - Should clearly indicate what the notification is about
        - MUST BE PLAIN TEXT (no markdown formatting)
        - Examples: "Election Results", "Hints regarding LeetCode 141", "Code regarding LeetCode 141"
        - Keep concise but descriptive
    
    summary (str, required):
        - Short description that appears in the device notification preview (max 500 characters)
        - Usually a couple of words or brief sentence describing the notification content
        - MUST BE PLAIN TEXT (no markdown formatting)
        - Examples: "Connection failed", "Solution for a DP problem with memoization", "Solution for a DP problem with tabulation"
        - This is what the user sees first on their device
    
    content (str, required):
        - The detailed message content sent to the user's device (no character limit)
        - Contains the actual information the user requested to be sent to them
        - SUPPORTS MARKDOWN FORMATTING for rich text display on the device
        - Can include technical details, instructions, code blocks, or full context
        - Examples: "Database connection failed after 3 retry attempts. Check server status.", 
                    "Your scheduled backup completed at 3:15 AM. All 1,247 files backed up successfully.",
                    "```python\ndef solution(nums):\n    return max(nums)\n```"
    
    notification_type (str, required):
        - Category/classification of the notification (max 50 characters)
        - Helps the user understand the nature and urgency of the message
        - Common types: "Code", "Leetcode", "Elections", "News", "Plan"
        - Domain-specific types: "code", "weather", "news", "system", "security", "backup"
        - Choose the most appropriate category for proper notification handling on the device
    
    RETURNS:
        str: Success message with timestamp if notification sent successfully,
             or error message with specific failure reason if sending fails
    
    USAGE GUIDELINES:
    - Always provide meaningful, user-friendly content that makes sense on a mobile/desktop device
    - Use appropriate notification types to help users prioritize and organize their notifications  
    - Keep headline and summary concise since they appear in notification previews
    - Include relevant context in content so the user understands the notification without needing additional information
    - Try to use markdown formatting for the content to make it more readable on the device, avoid long paragraphs.
    """
    

    if not headline:
        return "Headline is required but not provided, please provide a headline."
    if not summary:
        return "Summary is required but not provided, please provide a summary."
    if not content:
        return "Content is required but not provided, please provide a content."
    if not notification_type:
        return "Type is required but not provided, please provide a type."
    
    # this is a choice that made arbitrarily, it can be changed later. this is not applied on content.
    max_length = 500

    if len(headline) > max_length*0.5:
        return "Headline is too long, please provide a headline that is less than 100 characters."
    if len(summary) > max_length:
        return "Summary is too long, please provide a summary that is less than 100 characters."
    if len(notification_type) > max_length*0.1:
        return "Type is too long, please provide a type that is less than 100 characters."
    


    state = EchonotiWrapper.send_notification(headline=headline, summary=summary, content=content, notification_type=notification_type)

    """missing fields are already handled above. still, for the sake of debugging we will recheck all of possible errors by 
    returning the error message if any.
    the success message in"""
    if "error" in state.keys():
        return f"Error while sending notification to the device with echonoti: {state['error']}"
    else:
        return f"Successfully sent notification to the device with echonoti. Time of notification: {state['createdAt']}."

if __name__ == "__main__":
    # Test the notification tool
    result = send_notification.invoke({"headline": "Test Notification", "summary": "Testing the Echonoti system", "content": "This is a **test notification** with markdown support!", "notification_type": "Code"})
    print("Test result:", result)