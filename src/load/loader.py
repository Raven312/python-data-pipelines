"""Data loading module - Write data to various destinations."""

import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, df: pd.DataFrame) -> Path:
        """Load DataFrame to destination."""
        pass


class CSVLoader(BaseLoader):
    """Load data to CSV files."""

    def __init__(self, file_path: Union[str, Path], **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs

    def load(self, df: pd.DataFrame) -> Path:
        """Write DataFrame to CSV file."""
        logger.info(f"Loading data to CSV: {self.file_path}")

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.file_path, index=False, **self.kwargs)
        logger.info(f"Successfully loaded {len(df)} rows to {self.file_path}")

        return self.file_path


class ParquetLoader(BaseLoader):
    """Load data to Parquet files."""

    def __init__(self, file_path: Union[str, Path], **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs

    def load(self, df: pd.DataFrame) -> Path:
        """Write DataFrame to Parquet file."""
        logger.info(f"Loading data to Parquet: {self.file_path}")

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(self.file_path, index=False, **self.kwargs)
        logger.info(f"Successfully loaded {len(df)} rows to {self.file_path}")

        return self.file_path


class JSONLoader(BaseLoader):
    """Load data to JSON files."""

    def __init__(self, file_path: Union[str, Path], **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs

    def load(self, df: pd.DataFrame) -> Path:
        """Write DataFrame to JSON file."""
        logger.info(f"Loading data to JSON: {self.file_path}")

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_json(self.file_path, orient="records", **self.kwargs)
        logger.info(f"Successfully loaded {len(df)} rows to {self.file_path}")

        return self.file_path


class DataLoader:
    """
    Unified data loader that handles multiple destinations.

    Factory pattern for creating appropriate loaders.
    """

    @staticmethod
    def to_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
        """Load to CSV file."""
        return CSVLoader(file_path, **kwargs).load(df)

    @staticmethod
    def to_parquet(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
        """Load to Parquet file."""
        return ParquetLoader(file_path, **kwargs).load(df)

    @staticmethod
    def to_json(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
        """Load to JSON file."""
        return JSONLoader(file_path, **kwargs).load(df)

    @staticmethod
    def auto(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
        """
        Automatically detect format from file extension and load.

        Supports: .csv, .parquet, .json
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return DataLoader.to_csv(df, file_path, **kwargs)
        elif suffix == ".parquet":
            return DataLoader.to_parquet(df, file_path, **kwargs)
        elif suffix == ".json":
            return DataLoader.to_json(df, file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
