"""Tests for RWLock implementation."""

import threading
import time
from typing import List

import pytest

from src.infrastructure.concurrency.rwlock import RWLock


class TestRWLock:
    """Test cases for RWLock."""

    def test_single_reader(self) -> None:
        """Test that a single reader can acquire the lock."""
        lock = RWLock()
        lock.acquire_read()
        assert lock._readers == 1
        lock.release_read()
        assert lock._readers == 0

    def test_multiple_readers(self) -> None:
        """Test that multiple readers can hold the lock simultaneously."""
        lock = RWLock()
        num_readers = 5
        results: List[int] = []

        def reader(reader_id: int) -> None:
            lock.acquire_read()
            results.append(reader_id)
            time.sleep(0.1)  # Hold the lock briefly
            lock.release_read()

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(num_readers)]
        for t in threads:
            t.start()

        # Wait a bit for threads to acquire locks
        time.sleep(0.05)

        # All readers should be able to acquire the lock
        assert lock._readers == num_readers

        for t in threads:
            t.join()

        assert lock._readers == 0
        assert len(results) == num_readers

    def test_single_writer(self) -> None:
        """Test that a single writer can acquire the lock."""
        lock = RWLock()
        lock.acquire_write()
        assert lock._writers == 1
        lock.release_write()
        assert lock._writers == 0

    def test_writer_blocks_readers(self) -> None:
        """Test that a writer blocks readers."""
        lock = RWLock()
        results: List[str] = []

        def writer() -> None:
            lock.acquire_write()
            results.append("writer_start")
            time.sleep(0.2)
            results.append("writer_end")
            lock.release_write()

        def reader() -> None:
            time.sleep(0.05)  # Let writer acquire first
            lock.acquire_read()
            results.append("reader")
            lock.release_read()

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Reader should wait for writer to finish
        assert results == ["writer_start", "writer_end", "reader"]

    def test_readers_block_writer(self) -> None:
        """Test that readers block writers."""
        lock = RWLock()
        results: List[str] = []

        def reader() -> None:
            lock.acquire_read()
            results.append("reader_start")
            time.sleep(0.2)
            results.append("reader_end")
            lock.release_read()

        def writer() -> None:
            time.sleep(0.05)  # Let reader acquire first
            lock.acquire_write()
            results.append("writer")
            lock.release_write()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()

        reader_thread.join()
        writer_thread.join()

        # Writer should wait for reader to finish
        assert results == ["reader_start", "reader_end", "writer"]

    def test_writer_priority(self) -> None:
        """Test that writers have priority over readers (no writer starvation)."""
        lock = RWLock()
        results: List[str] = []

        def reader(reader_id: int, delay: float = 0.0) -> None:
            if delay > 0:
                time.sleep(delay)
            lock.acquire_read()
            results.append(f"reader_{reader_id}")
            time.sleep(0.05)
            lock.release_read()

        def writer(writer_id: int, delay: float = 0.0) -> None:
            if delay > 0:
                time.sleep(delay)
            lock.acquire_write()
            results.append(f"writer_{writer_id}")
            time.sleep(0.05)
            lock.release_write()

        # Start first reader (acquires immediately)
        r1 = threading.Thread(target=reader, args=(1, 0.0))
        r1.start()
        time.sleep(0.01)  # Ensure r1 acquires the lock

        # Start writer (will wait for r1, but increments _writers_waiting immediately)
        w1 = threading.Thread(target=writer, args=(1, 0.0))
        w1.start()
        time.sleep(0.01)  # Ensure w1 starts waiting with _writers_waiting=1

        # Start second reader (should wait for writer due to writer priority)
        r2 = threading.Thread(target=reader, args=(2, 0.0))
        r2.start()

        r1.join()
        w1.join()
        r2.join()

        # Writer should go before reader_2 due to priority
        assert results == ["reader_1", "writer_1", "reader_2"]

    def test_context_manager_reader(self) -> None:
        """Test reader context manager."""
        lock = RWLock()

        with lock.reader():
            assert lock._readers == 1

        assert lock._readers == 0

    def test_context_manager_writer(self) -> None:
        """Test writer context manager."""
        lock = RWLock()

        with lock.writer():
            assert lock._writers == 1

        assert lock._writers == 0

    def test_context_manager_exception_handling_reader(self) -> None:
        """Test that reader lock is released even when exception occurs."""
        lock = RWLock()

        try:
            with lock.reader():
                assert lock._readers == 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released despite exception
        assert lock._readers == 0

    def test_context_manager_exception_handling_writer(self) -> None:
        """Test that writer lock is released even when exception occurs."""
        lock = RWLock()

        try:
            with lock.writer():
                assert lock._writers == 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released despite exception
        assert lock._writers == 0

    def test_concurrent_readers_and_writers(self) -> None:
        """Test concurrent access with multiple readers and writers."""
        lock = RWLock()
        shared_data = {"value": 0}
        read_values: List[int] = []

        def writer(increment: int) -> None:
            with lock.writer():
                current = shared_data["value"]
                time.sleep(0.01)  # Simulate work
                shared_data["value"] = current + increment

        def reader() -> None:
            with lock.reader():
                read_values.append(shared_data["value"])
                time.sleep(0.01)  # Simulate work

        # Create multiple readers and writers
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(1,)))
            threads.append(threading.Thread(target=reader))

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Final value should be 5 (5 writers each adding 1)
        assert shared_data["value"] == 5
        # All reads should be valid values (0-5)
        assert all(0 <= v <= 5 for v in read_values)
