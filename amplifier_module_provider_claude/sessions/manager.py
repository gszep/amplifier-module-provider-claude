"""Session management for Claude provider.

Follows the amplifier-claude branch pattern for disk-persisted sessions.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import SessionMetadata, SessionState


class SessionManager:
    """Manager for creating, loading, and persisting sessions.

    Handles session lifecycle including:
    - Creating new sessions with unique IDs
    - Loading existing sessions for re-entrancy
    - Saving session state to disk
    - Cleaning up old sessions
    - Mapping Amplifier session IDs to Claude CLI session IDs
    """

    def __init__(self, session_dir: Path | None = None):
        """Initialize session manager.

        Args:
            session_dir: Directory to store sessions.
                        Defaults to ~/.amplifier-claude/sessions
        """
        self.session_dir = session_dir or (
            Path.home() / ".amplifier-claude" / "sessions"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self, name: str = "unnamed", tags: list[str] | None = None
    ) -> SessionState:
        """Create a new session.

        Args:
            name: Human-readable session name
            tags: Optional tags for categorization

        Returns:
            New SessionState instance
        """
        metadata = SessionMetadata(name=name, tags=tags or [])
        return SessionState(metadata=metadata)

    def load_session(self, session_id: str) -> SessionState | None:
        """Load an existing session.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        session_file = self.session_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file) as f:
            data = json.load(f)

        # Convert datetime strings back to datetime objects
        if "metadata" in data:
            if "created_at" in data["metadata"]:
                data["metadata"]["created_at"] = datetime.fromisoformat(
                    data["metadata"]["created_at"]
                )
            if "updated_at" in data["metadata"]:
                data["metadata"]["updated_at"] = datetime.fromisoformat(
                    data["metadata"]["updated_at"]
                )

        return SessionState(**data)

    def save_session(self, session: SessionState) -> Path:
        """Save session to disk.

        Args:
            session: Session to save

        Returns:
            Path to saved session file
        """
        session_file = self.session_dir / f"{session.metadata.session_id}.json"

        # Convert to JSON-serializable format
        data = session.model_dump()

        # Convert datetime objects to ISO format strings
        if "metadata" in data:
            if "created_at" in data["metadata"]:
                data["metadata"]["created_at"] = data["metadata"][
                    "created_at"
                ].isoformat()
            if "updated_at" in data["metadata"]:
                data["metadata"]["updated_at"] = data["metadata"][
                    "updated_at"
                ].isoformat()

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return session_file

    def list_sessions(self, days_back: int = 7) -> list[SessionMetadata]:
        """List recent sessions.

        Args:
            days_back: How many days back to look

        Returns:
            List of session metadata
        """
        sessions = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        for session_file in self.session_dir.glob("*.json"):
            # Check file modification time
            mtime = datetime.fromtimestamp(
                session_file.stat().st_mtime, tz=timezone.utc
            )
            if mtime < cutoff:
                continue

            try:
                session = self.load_session(session_file.stem)
                if session:
                    sessions.append(session.metadata)
            except Exception:
                # Skip corrupted sessions
                continue

        # Sort by updated time, newest first
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return sessions

    def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """Remove sessions older than specified days.

        Args:
            days_to_keep: Keep sessions newer than this many days

        Returns:
            Number of sessions removed
        """
        cutoff = time.time() - (days_to_keep * 86400)
        removed = 0

        for session_file in self.session_dir.glob("*.json"):
            if session_file.stat().st_mtime < cutoff:
                session_file.unlink()
                removed += 1

        return removed

    def get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to session file
        """
        return self.session_dir / f"{session_id}.json"

    def get_or_create_session(
        self,
        session_id: str | None = None,
        name: str = "unnamed",
        tags: list[str] | None = None,
    ) -> SessionState:
        """Get existing session or create a new one.

        Args:
            session_id: Optional session ID to load
            name: Name for new session if creating
            tags: Tags for new session if creating

        Returns:
            Existing or new SessionState
        """
        if session_id:
            session = self.load_session(session_id)
            if session:
                return session

        return self.create_session(name=name, tags=tags)

    def find_by_claude_session_id(self, claude_session_id: str) -> SessionState | None:
        """Find a session by its Claude CLI session ID.

        This allows resuming sessions based on the Claude CLI session ID
        for optimal cache utilization.

        Args:
            claude_session_id: The Claude CLI session ID

        Returns:
            SessionState if found, None otherwise
        """
        for session_file in self.session_dir.glob("*.json"):
            try:
                session = self.load_session(session_file.stem)
                if session and session.metadata.claude_session_id == claude_session_id:
                    return session
            except Exception:
                continue
        return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get aggregate cache statistics across all sessions.

        Returns:
            Dictionary with cache statistics
        """
        total_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        session_count = 0

        for session_file in self.session_dir.glob("*.json"):
            try:
                session = self.load_session(session_file.stem)
                if session:
                    total_tokens += session.metadata.total_tokens
                    cache_read_tokens += session.metadata.cache_read_tokens
                    cache_creation_tokens += session.metadata.cache_creation_tokens
                    session_count += 1
            except Exception:
                continue

        efficiency = cache_read_tokens / total_tokens if total_tokens > 0 else 0.0

        return {
            "session_count": session_count,
            "total_tokens": total_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_efficiency": efficiency,
        }
