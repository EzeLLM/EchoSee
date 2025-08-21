"""Alarm MCP server providing alarm management functionality."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
import pygame
import os

from mcp.server.fastmcp import Context, FastMCP
from mcp_servers.utils import write_audit

app = FastMCP("alarm")

# In-memory alarm storage
active_alarms: Dict[str, dict] = {}
alarm_sound_file = os.path.join(os.path.dirname(__file__), "../../assets/alarm1.wav")


class AlarmManager:
    """Manages alarm playback and state."""
    
    def __init__(self):
        pygame.mixer.init()
        self.current_alarm_task: Optional[asyncio.Task] = None
        self.is_playing = False
    
    async def play_alarm(self, alarm_id: str):
        """Play alarm sound continuously until stopped."""
        if self.is_playing:
            return
        
        self.is_playing = True
        try:
            sound = pygame.mixer.Sound(alarm_sound_file)
            while self.is_playing and alarm_id in active_alarms:
                sound.play()
                await asyncio.sleep(sound.get_length())
        except Exception as e:
            print(f"Error playing alarm: {e}")
        finally:
            self.is_playing = False
    
    def stop_alarm(self):
        """Stop currently playing alarm."""
        self.is_playing = False
        pygame.mixer.stop()


alarm_manager = AlarmManager()


async def trigger_alarm(alarm_id: str):
    """Callback function to trigger when alarm time is reached."""
    if alarm_id in active_alarms:
        active_alarms[alarm_id]["status"] = "ringing"
        await alarm_manager.play_alarm(alarm_id)


@app.custom_route("/healthz", methods=["GET"])
async def healthz():
    return {"ok": True}


@app.tool()
async def set_alarm_at_time(time: str, label: Optional[str] = None, ctx: Context | None = None):
    """Set an alarm to trigger at a specific date/time.
    
    Args:
        time: Exact trigger time in format "YYYY:MM:DD:HH:MM:SS"
              Example: "2024:03:15:14:30:00" = March 15, 2024 at 2:30 PM
        label: Optional label for the alarm
        ctx: MCP context
        
    Returns:
        Alarm ID if successful, error if time is invalid or in the past
    """
    rid = ctx.request_id if ctx else None
    try:
        alarm_time = datetime.strptime(time, "%Y:%m:%d:%H:%M:%S")
        if alarm_time <= datetime.now():
            write_audit({"request_id": rid, "tool": "set_alarm_at_time", "ok": False, "error": "past_time"})
            return {
                "ok": False,
                "error": {"code": "invalid_time", "message": "Cannot set alarm for past time"},
                "meta": {}
            }
        
        alarm_id = str(uuid.uuid4())
        active_alarms[alarm_id] = {
            "id": alarm_id,
            "time": alarm_time.isoformat(),
            "label": label,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        # Schedule the alarm
        delay = (alarm_time - datetime.now()).total_seconds()
        asyncio.create_task(asyncio.sleep(delay)).add_done_callback(
            lambda _: asyncio.create_task(trigger_alarm(alarm_id))
        )
        
        write_audit({"request_id": rid, "tool": "set_alarm_at_time", "ok": True, "alarm_id": alarm_id})
        return {"ok": True, "data": {"alarm_id": alarm_id, "scheduled_time": alarm_time.isoformat()}, "meta": {}}
        
    except ValueError as e:
        write_audit({"request_id": rid, "tool": "set_alarm_at_time", "ok": False, "error": str(e)})
        return {
            "ok": False,
            "error": {"code": "invalid_format", "message": f"Invalid time format: {str(e)}"},
            "meta": {}
        }


@app.tool()
async def set_alarm_with_delta(delta: str, label: Optional[str] = None, ctx: Context | None = None):
    """Set an alarm to trigger after a specified duration from now.
    
    Args:
        delta: Time duration in format "DD:HH:MM:SS"
               Example: "02:12:30:45" = 2 days, 12 hours, 30 minutes, 45 seconds
        label: Optional label for the alarm
        ctx: MCP context
        
    Returns:
        Alarm ID if successful, error if format is invalid
    """
    rid = ctx.request_id if ctx else None
    try:
        days, hours, minutes, seconds = map(int, delta.split(':'))
        delta_td = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        alarm_time = datetime.now() + delta_td
        
        alarm_id = str(uuid.uuid4())
        active_alarms[alarm_id] = {
            "id": alarm_id,
            "time": alarm_time.isoformat(),
            "label": label,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        # Schedule the alarm
        asyncio.create_task(asyncio.sleep(delta_td.total_seconds())).add_done_callback(
            lambda _: asyncio.create_task(trigger_alarm(alarm_id))
        )
        
        write_audit({"request_id": rid, "tool": "set_alarm_with_delta", "ok": True, "alarm_id": alarm_id})
        return {"ok": True, "data": {"alarm_id": alarm_id, "scheduled_time": alarm_time.isoformat()}, "meta": {}}
        
    except (ValueError, AttributeError) as e:
        write_audit({"request_id": rid, "tool": "set_alarm_with_delta", "ok": False, "error": str(e)})
        return {
            "ok": False,
            "error": {"code": "invalid_format", "message": f"Invalid delta format: {str(e)}"},
            "meta": {}
        }


@app.tool()
async def stop_alarm(alarm_id: Optional[str] = None, ctx: Context | None = None):
    """Stop a currently ringing alarm or cancel a scheduled alarm.
    
    Args:
        alarm_id: Specific alarm ID to stop. If not provided, stops any currently ringing alarm.
        ctx: MCP context
        
    Returns:
        Success status
    """
    rid = ctx.request_id if ctx else None
    
    if alarm_id:
        if alarm_id in active_alarms:
            del active_alarms[alarm_id]
            alarm_manager.stop_alarm()
            write_audit({"request_id": rid, "tool": "stop_alarm", "ok": True, "alarm_id": alarm_id})
            return {"ok": True, "data": {"message": f"Alarm {alarm_id} stopped"}, "meta": {}}
        else:
            write_audit({"request_id": rid, "tool": "stop_alarm", "ok": False, "error": "alarm_not_found"})
            return {
                "ok": False,
                "error": {"code": "not_found", "message": f"Alarm {alarm_id} not found"},
                "meta": {}
            }
    else:
        # Stop any currently ringing alarm
        alarm_manager.stop_alarm()
        ringing_alarms = [aid for aid, alarm in active_alarms.items() if alarm["status"] == "ringing"]
        for aid in ringing_alarms:
            active_alarms[aid]["status"] = "stopped"
        
        write_audit({"request_id": rid, "tool": "stop_alarm", "ok": True})
        return {"ok": True, "data": {"message": "All ringing alarms stopped"}, "meta": {}}


@app.tool()
async def list_alarms(ctx: Context | None = None):
    """List all active alarms with their status.
    
    Returns:
        List of active alarms with their details
    """
    rid = ctx.request_id if ctx else None
    alarms = list(active_alarms.values())
    write_audit({"request_id": rid, "tool": "list_alarms", "ok": True})
    return {"ok": True, "data": {"alarms": alarms}, "meta": {"count": len(alarms)}}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7015)
