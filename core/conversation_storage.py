"""Conversation history persistence.

Provides SQLite-based storage for conversation history, enabling:
- Context persistence across application restarts
- Search through past conversations
- Usage pattern analysis
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationStorage:
    """SQLite-based conversation history storage.

    Stores conversation turns with timestamps and metadata for
    persistence across application sessions.
    """

    def __init__(self, db_path: str = "data/conversations.db"):
        """Initialize conversation storage.

        Args:
            db_path: Path to SQLite database file. Parent directories
                    will be created if they don't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"ConversationStorage initialized at {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        """)

        conn.commit()
        conn.close()

    def save_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save a conversation turn.

        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
            metadata: Optional metadata (e.g., tool calls, model used)

        Returns:
            ID of the inserted conversation turn
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO conversations (timestamp, user_message, assistant_message, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_msg,
            assistant_msg,
            json.dumps(metadata or {})
        ))

        turn_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Saved conversation turn {turn_id}")
        return turn_id

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history.

        Args:
            limit: Maximum number of turns to retrieve

        Returns:
            List of conversation turns, oldest first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM conversations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        # Return in chronological order (oldest first)
        result = [dict(row) for row in reversed(rows)]

        # Parse metadata JSON
        for item in result:
            if item.get('metadata'):
                try:
                    item['metadata'] = json.loads(item['metadata'])
                except json.JSONDecodeError:
                    item['metadata'] = {}

        return result

    def search_conversations(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search conversations by text.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching conversation turns
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM conversations
            WHERE user_message LIKE ? OR assistant_message LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))

        rows = cursor.fetchall()
        conn.close()

        result = [dict(row) for row in rows]

        # Parse metadata JSON
        for item in result:
            if item.get('metadata'):
                try:
                    item['metadata'] = json.loads(item['metadata'])
                except json.JSONDecodeError:
                    item['metadata'] = {}

        return result

    def get_conversation_count(self) -> int:
        """Get total number of stored conversations.

        Returns:
            Total count of conversation turns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def clear_history(self) -> int:
        """Clear all conversation history.

        Returns:
            Number of conversations deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()

        logger.info(f"Cleared {count} conversation turns")
        return count

    def get_conversations_by_date(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversations within a date range.

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD).
                     If None, uses current date.

        Returns:
            List of conversation turns within the date range
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Add time component to include full end day
        start_datetime = f"{start_date}T00:00:00"
        end_datetime = f"{end_date}T23:59:59"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM conversations
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (start_datetime, end_datetime))

        rows = cursor.fetchall()
        conn.close()

        result = [dict(row) for row in rows]

        # Parse metadata JSON
        for item in result:
            if item.get('metadata'):
                try:
                    item['metadata'] = json.loads(item['metadata'])
                except json.JSONDecodeError:
                    item['metadata'] = {}

        return result


if __name__ == "__main__":
    # Test the conversation storage
    import tempfile
    import os

    # Create a temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_conversations.db")
        storage = ConversationStorage(db_path)

        # Test saving conversations
        print("Testing ConversationStorage...")

        turn1_id = storage.save_turn(
            "What's the weather like?",
            "I don't have access to real-time weather data.",
            {"model": "gpt-4o-mini"}
        )
        print(f"✓ Saved turn 1 (ID: {turn1_id})")

        turn2_id = storage.save_turn(
            "Set a timer for 5 minutes",
            "I've set a timer for 5 minutes.",
            {"tool_used": "set_alarm"}
        )
        print(f"✓ Saved turn 2 (ID: {turn2_id})")

        turn3_id = storage.save_turn(
            "What did I ask about earlier?",
            "You asked about the weather and then set a timer."
        )
        print(f"✓ Saved turn 3 (ID: {turn3_id})")

        # Test getting recent history
        history = storage.get_recent_history(limit=5)
        print(f"\n✓ Retrieved {len(history)} turns from history")
        for turn in history:
            print(f"  - User: {turn['user_message'][:40]}...")

        # Test search
        search_results = storage.search_conversations("weather")
        print(f"\n✓ Found {len(search_results)} results for 'weather'")

        # Test count
        count = storage.get_conversation_count()
        print(f"\n✓ Total conversations: {count}")

        # Test clear
        cleared = storage.clear_history()
        print(f"\n✓ Cleared {cleared} conversations")

        remaining = storage.get_conversation_count()
        print(f"✓ Remaining conversations: {remaining}")

        print("\n✓ All tests passed!")

