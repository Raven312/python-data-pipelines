"""Data extraction module - Load data from various sources."""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Abstract base class for data extractors."""

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data and return as Pandas DataFrame."""
        pass


class DictExtractor(BaseExtractor):
    """Extract data from Python dictionaries."""

    def __init__(self, data: Dict[str, list]):
        self.data = data

    def extract(self) -> pd.DataFrame:
        logger.info("Extracting data from dictionary")
        df = pd.DataFrame(self.data)
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
        return df


class CSVExtractor(BaseExtractor):
    """Extract data from CSV files."""

    def __init__(self, file_path: Union[str, Path], **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs

    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from CSV: {self.file_path}")
        df = pd.read_csv(self.file_path, **self.kwargs)
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
        return df


class ParquetExtractor(BaseExtractor):
    """Extract data from Parquet files."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from Parquet: {self.file_path}")
        df = pd.read_parquet(self.file_path)
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
        return df


class APIExtractor(BaseExtractor):
    """Extract data from REST APIs."""

    def __init__(self, url: str, params: Optional[Dict] = None):
        self.url = url
        self.params = params or {}

    def extract(self) -> pd.DataFrame:
        import urllib.request
        import json

        logger.info(f"Extracting data from API: {self.url}")

        try:
            with urllib.request.urlopen(self.url) as response:
                data = json.loads(response.read().decode())

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected data format from API: {type(data)}")

            logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Failed to extract from API: {str(e)}")
            raise


class DataExtractor:
    """
    Unified data extractor that handles multiple sources.

    Factory pattern for creating appropriate extractors.
    """

    @staticmethod
    def from_dict(data: Dict[str, list]) -> pd.DataFrame:
        """Extract from dictionary."""
        return DictExtractor(data).extract()

    @staticmethod
    def from_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Extract from CSV file."""
        return CSVExtractor(file_path, **kwargs).extract()

    @staticmethod
    def from_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
        """Extract from Parquet file."""
        return ParquetExtractor(file_path).extract()

    @staticmethod
    def from_api(url: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Extract from REST API."""
        return APIExtractor(url, params).extract()


# Convenience functions
def extract_from_dict(data: Dict[str, list]) -> pd.DataFrame:
    """Extract data from a dictionary."""
    return DataExtractor.from_dict(data)


def extract_from_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Extract data from a CSV file."""
    return DataExtractor.from_csv(file_path, **kwargs)


def extract_from_api(url: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Extract data from an API endpoint."""
    return DataExtractor.from_api(url, params)
