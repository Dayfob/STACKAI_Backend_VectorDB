"""Unit tests for storage implementations."""

import json
import pickle
import tempfile
from pathlib import Path

import pytest

from src.infrastructure.persistence.disk_storage import DiskStorage
from src.infrastructure.persistence.memory_storage import InMemoryStorage


class TestInMemoryStorage:
    """Test cases for InMemoryStorage."""

    def test_save_and_load(self):
        """Test saving and loading data."""
        storage = InMemoryStorage()
        test_data = {"key": "value", "number": 42}

        storage.save("test_key", test_data)
        loaded_data = storage.load("test_key")

        assert loaded_data == test_data

    def test_load_nonexistent_key(self):
        """Test loading a non-existent key returns None."""
        storage = InMemoryStorage()
        result = storage.load("nonexistent")

        assert result is None

    def test_delete(self):
        """Test deleting data."""
        storage = InMemoryStorage()
        storage.save("test_key", "test_value")

        assert storage.exists("test_key")

        storage.delete("test_key")

        assert not storage.exists("test_key")
        assert storage.load("test_key") is None

    def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key doesn't raise error."""
        storage = InMemoryStorage()
        storage.delete("nonexistent")  # Should not raise

    def test_exists(self):
        """Test checking if key exists."""
        storage = InMemoryStorage()

        assert not storage.exists("test_key")

        storage.save("test_key", "value")

        assert storage.exists("test_key")

    def test_list_keys(self):
        """Test listing all keys."""
        storage = InMemoryStorage()
        storage.save("key1", "value1")
        storage.save("key2", "value2")
        storage.save("key3", "value3")

        keys = storage.list_keys()

        assert len(keys) == 3
        assert set(keys) == {"key1", "key2", "key3"}

    def test_list_keys_empty(self):
        """Test listing keys when storage is empty."""
        storage = InMemoryStorage()
        keys = storage.list_keys()

        assert keys == []

    def test_save_overwrites_existing_key(self):
        """Test that saving with existing key overwrites the value."""
        storage = InMemoryStorage()
        storage.save("key", "old_value")
        storage.save("key", "new_value")

        assert storage.load("key") == "new_value"


class TestDiskStorageJSON:
    """Test cases for DiskStorage with JSON format."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading data in JSON format."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        storage.save("test_key", test_data)
        loaded_data = storage.load("test_key")

        assert loaded_data == test_data

    def test_json_file_is_human_readable(self, tmp_path):
        """Test that JSON files are human-readable."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")
        test_data = {"message": "Hello, World!"}

        storage.save("test_key", test_data)

        # Read the file directly
        file_path = tmp_path / "test_key.json"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Hello, World!" in content

    def test_load_nonexistent_key_json(self, tmp_path):
        """Test loading a non-existent key returns None."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")
        result = storage.load("nonexistent")

        assert result is None

    def test_delete_json(self, tmp_path):
        """Test deleting data in JSON format."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")
        storage.save("test_key", "test_value")

        file_path = tmp_path / "test_key.json"
        assert file_path.exists()

        storage.delete("test_key")

        assert not file_path.exists()
        assert not storage.exists("test_key")

    def test_exists_json(self, tmp_path):
        """Test checking if key exists in JSON format."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")

        assert not storage.exists("test_key")

        storage.save("test_key", "value")

        assert storage.exists("test_key")

    def test_list_keys_json(self, tmp_path):
        """Test listing all keys in JSON format."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")
        storage.save("key1", "value1")
        storage.save("key2", "value2")
        storage.save("key3", "value3")

        keys = storage.list_keys()

        assert len(keys) == 3
        assert set(keys) == {"key1", "key2", "key3"}


class TestDiskStoragePickle:
    """Test cases for DiskStorage with Pickle format."""

    def test_save_and_load_pickle(self, tmp_path):
        """Test saving and loading data in Pickle format."""
        storage = DiskStorage(base_path=str(tmp_path), format="pickle")
        test_data = {"key": "value", "number": 42, "complex": (1, 2, {3: 4})}

        storage.save("test_key", test_data)
        loaded_data = storage.load("test_key")

        assert loaded_data == test_data

    def test_pickle_supports_complex_objects(self, tmp_path):
        """Test that Pickle can handle complex Python objects."""
        storage = DiskStorage(base_path=str(tmp_path), format="pickle")

        # Use built-in types and nested structures instead of custom classes
        # (local classes can't be pickled)
        test_obj = {
            "nested": {"list": [1, 2, 3], "tuple": (4, 5, 6)},
            "set": {7, 8, 9},
            "complex_number": complex(1, 2),
        }
        storage.save("test_key", test_obj)
        loaded_obj = storage.load("test_key")

        assert loaded_obj == test_obj
        assert loaded_obj["nested"]["list"] == [1, 2, 3]
        assert loaded_obj["complex_number"] == complex(1, 2)

    def test_load_nonexistent_key_pickle(self, tmp_path):
        """Test loading a non-existent key returns None."""
        storage = DiskStorage(base_path=str(tmp_path), format="pickle")
        result = storage.load("nonexistent")

        assert result is None

    def test_delete_pickle(self, tmp_path):
        """Test deleting data in Pickle format."""
        storage = DiskStorage(base_path=str(tmp_path), format="pickle")
        storage.save("test_key", "test_value")

        file_path = tmp_path / "test_key.pkl"
        assert file_path.exists()

        storage.delete("test_key")

        assert not file_path.exists()
        assert not storage.exists("test_key")

    def test_exists_pickle(self, tmp_path):
        """Test checking if key exists in Pickle format."""
        storage = DiskStorage(base_path=str(tmp_path), format="pickle")

        assert not storage.exists("test_key")

        storage.save("test_key", "value")

        assert storage.exists("test_key")

    def test_list_keys_pickle(self, tmp_path):
        """Test listing all keys in Pickle format."""
        storage = DiskStorage(base_path=str(tmp_path), format="pickle")
        storage.save("key1", "value1")
        storage.save("key2", "value2")
        storage.save("key3", "value3")

        keys = storage.list_keys()

        assert len(keys) == 3
        assert set(keys) == {"key1", "key2", "key3"}


class TestDiskStorageEdgeCases:
    """Test edge cases for DiskStorage."""

    def test_base_path_created_automatically(self, tmp_path):
        """Test that base path is created if it doesn't exist."""
        new_path = tmp_path / "new_directory" / "nested"
        storage = DiskStorage(base_path=str(new_path), format="json")

        assert new_path.exists()

        storage.save("test", "value")
        assert storage.load("test") == "value"

    def test_json_format_rejects_non_serializable(self, tmp_path):
        """Test that JSON format raises error for non-serializable data."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")

        # Lambda functions are not JSON serializable
        with pytest.raises(TypeError):
            storage.save("test", lambda x: x)

    def test_list_keys_filters_by_format(self, tmp_path):
        """Test that list_keys only returns keys for current format."""
        json_storage = DiskStorage(base_path=str(tmp_path), format="json")
        pickle_storage = DiskStorage(base_path=str(tmp_path), format="pickle")

        json_storage.save("json_key", "value")
        pickle_storage.save("pickle_key", "value")

        json_keys = json_storage.list_keys()
        pickle_keys = pickle_storage.list_keys()

        assert "json_key" in json_keys
        assert "pickle_key" not in json_keys

        assert "pickle_key" in pickle_keys
        assert "json_key" not in pickle_keys

    def test_delete_nonexistent_key(self, tmp_path):
        """Test deleting a non-existent key doesn't raise error."""
        storage = DiskStorage(base_path=str(tmp_path), format="json")
        storage.delete("nonexistent")  # Should not raise

    def test_list_keys_on_nonexistent_directory(self, tmp_path):
        """Test list_keys returns empty list for non-existent directory."""
        non_existent = tmp_path / "does_not_exist"
        storage = DiskStorage(base_path=str(non_existent), format="json")

        # Delete the directory that was auto-created
        non_existent.rmdir()

        keys = storage.list_keys()
        assert keys == []
