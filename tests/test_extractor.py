"""Tests for the extract module."""

import pytest
import pandas as pd
from pathlib import Path

from src.extract.extractor import (
    DictExtractor,
    CSVExtractor,
    DataExtractor,
)


class TestDictExtractor:
    """Tests for DictExtractor."""

    def test_extract_basic(self):
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        extractor = DictExtractor(data)
        df = extractor.extract()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_extract_single_column(self):
        data = {"col": [10, 20]}
        df = DictExtractor(data).extract()
        assert len(df) == 2
        assert df["col"].tolist() == [10, 20]

    def test_extract_empty(self):
        data = {"a": [], "b": []}
        df = DictExtractor(data).extract()
        assert len(df) == 0
        assert list(df.columns) == ["a", "b"]


class TestCSVExtractor:
    """Tests for CSVExtractor."""

    def test_extract_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,value\nalice,1\nbob,2\n")
        extractor = CSVExtractor(csv_file)
        df = extractor.extract()
        assert len(df) == 2
        assert list(df.columns) == ["name", "value"]

    def test_extract_csv_with_kwargs(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name|value\nalice|1\nbob|2\n")
        df = CSVExtractor(csv_file, sep="|").extract()
        assert len(df) == 2

    def test_extract_csv_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CSVExtractor("/nonexistent/file.csv").extract()


class TestDataExtractor:
    """Tests for the DataExtractor factory."""

    def test_from_dict(self):
        data = {"x": [1, 2], "y": [3, 4]}
        df = DataExtractor.from_dict(data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_from_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        df = DataExtractor.from_csv(csv_file)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]
