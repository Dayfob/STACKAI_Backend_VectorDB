"""Read-Write Lock implementation for thread-safe operations."""

import threading
from typing import Any

__all__ = ["RWLock", "ReadLock", "WriteLock"]


class RWLock:
    """Read-Write Lock for thread-safe concurrent access.

    Allows multiple readers or a single writer at a time.
    Writers have priority to prevent writer starvation.
    """

    def __init__(self) -> None:
        """Initialize the RWLock."""
        self._readers = 0  # Number of active readers
        self._writers = 0  # Number of active writers (0 or 1)
        self._writers_waiting = 0  # Number of writers waiting (for priority)
        self._lock = threading.Lock()  # Protects the above counters
        self._readers_ok = threading.Condition(self._lock)  # Signals readers can proceed
        self._writers_ok = threading.Condition(self._lock)  # Signals writers can proceed

    def acquire_read(self) -> None:
        """Acquire a read lock.

        Multiple readers can hold the lock simultaneously.
        Blocks if a writer holds the lock or writers are waiting (writer priority).
        """
        self._readers_ok.acquire()
        try:
            # Wait while there are active writers or waiting writers (writer priority)
            while self._writers > 0 or self._writers_waiting > 0:
                self._readers_ok.wait()
            self._readers += 1
        finally:
            self._readers_ok.release()

    def release_read(self) -> None:
        """Release a read lock.

        Notifies waiting writers if this was the last reader.
        """
        self._readers_ok.acquire()
        try:
            self._readers -= 1
            # If no more readers, wake up a waiting writer
            if self._readers == 0:
                self._writers_ok.notify()
        finally:
            self._readers_ok.release()

    def acquire_write(self) -> None:
        """Acquire a write lock.

        Only one writer can hold the lock at a time.
        Blocks if any readers or writers hold the lock.
        Has priority over readers to prevent writer starvation.
        """
        self._writers_ok.acquire()
        try:
            # Indicate that a writer is waiting (for priority)
            self._writers_waiting += 1
            # Wait while there are active readers or active writers
            while self._readers > 0 or self._writers > 0:
                self._writers_ok.wait()
            # We got the lock, no longer waiting
            self._writers_waiting -= 1
            self._writers += 1
        finally:
            self._writers_ok.release()

    def release_write(self) -> None:
        """Release a write lock.

        Prioritizes waiting writers, then notifies all waiting readers.
        """
        self._writers_ok.acquire()
        try:
            self._writers -= 1
            # Writer priority: wake up waiting writers first
            if self._writers_waiting > 0:
                self._writers_ok.notify()
            else:
                # No waiting writers, wake up all waiting readers
                self._readers_ok.notify_all()
        finally:
            self._writers_ok.release()

    def reader(self) -> "ReadLock":
        """Get a context manager for read lock.

        Usage:
            with rwlock.reader():
                # perform read operations
                pass
        """
        return ReadLock(self)

    def writer(self) -> "WriteLock":
        """Get a context manager for write lock.

        Usage:
            with rwlock.writer():
                # perform write operations
                pass
        """
        return WriteLock(self)


class ReadLock:
    """Context manager for read lock."""

    def __init__(self, rwlock: RWLock) -> None:
        """Initialize with the parent RWLock."""
        self._rwlock = rwlock

    def __enter__(self) -> "ReadLock":
        """Acquire read lock on entry."""
        self._rwlock.acquire_read()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release read lock on exit."""
        self._rwlock.release_read()


class WriteLock:
    """Context manager for write lock."""

    def __init__(self, rwlock: RWLock) -> None:
        """Initialize with the parent RWLock."""
        self._rwlock = rwlock

    def __enter__(self) -> "WriteLock":
        """Acquire write lock on entry."""
        self._rwlock.acquire_write()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release write lock on exit."""
        self._rwlock.release_write()
