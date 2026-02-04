"""Tests for the load module."""

import pytest
import pandas as pd
from pathlib import Path

from src.load.loader import CSVLoader, JSONLoader, ParquetLoader, DataLoader


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output = tmp_path / "out.csv"
        result_path = CSVLoader(output).load(df)
        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["a", "b"]

    def test_load_csv_creates_directory(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        output = tmp_path / "subdir" / "out.csv"
        CSVLoader(output).load(df)
        assert output.exists()


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_load_json(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output = tmp_path / "out.json"
        result_path = JSONLoader(output).load(df)
        assert result_path.exists()
        loaded = pd.read_json(result_path)
        assert len(loaded) == 2


class TestDataLoader:
    """Tests for the DataLoader factory."""

    def test_to_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = DataLoader.to_csv(df, tmp_path / "test.csv")
        assert path.exists()

    def test_to_json(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = DataLoader.to_json(df, tmp_path / "test.json")
        assert path.exists()

    def test_auto_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = DataLoader.auto(df, tmp_path / "test.csv")
        assert path.exists()

    def test_auto_json(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = DataLoader.auto(df, tmp_path / "test.json")
        assert path.exists()

    def test_auto_unsupported_format(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported file format"):
            DataLoader.auto(df, tmp_path / "test.xyz")


class TestJSONLoaderExport:
    """Test that JSONLoader is properly exported."""

    def test_json_loader_importable_from_package(self):
        from src.load import JSONLoader as LoaderFromInit
        assert LoaderFromInit is JSONLoader
