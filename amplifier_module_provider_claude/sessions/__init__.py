"""Session management for Claude provider.

Provides disk-persisted sessions for efficient prompt caching.
"""

from .manager import SessionManager
from .models import SessionMetadata, SessionState

__all__ = ["SessionManager", "SessionMetadata", "SessionState"]
