"""Data transformation module."""

import pandas as pd
import logging
from typing import Dict, List, Callable, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame."""
        pass


class DataTransformer:
    """
    Flexible data transformer using method chaining.

    Supports common transformation operations on Pandas DataFrames.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._operations: List[str] = []

    def select(self, columns: List[str]) -> "DataTransformer":
        """Select specific columns."""
        self.df = self.df[columns]
        self._operations.append(f"select({columns})")
        return self

    def rename(self, mapping: Dict[str, str]) -> "DataTransformer":
        """Rename columns."""
        self.df = self.df.rename(columns=mapping)
        self._operations.append(f"rename({list(mapping.keys())})")
        return self

    def filter(self, condition: pd.Series) -> "DataTransformer":
        """Filter rows based on condition."""
        before = len(self.df)
        self.df = self.df[condition]
        after = len(self.df)
        self._operations.append(f"filter(removed {before - after} rows)")
        return self

    def join(
        self,
        other: pd.DataFrame,
        left_on: str,
        right_on: str,
        how: str = "inner"
    ) -> "DataTransformer":
        """Join with another DataFrame."""
        before = len(self.df)
        self.df = self.df.merge(other, left_on=left_on, right_on=right_on, how=how)
        after = len(self.df)
        self._operations.append(f"join({how}, {before} -> {after} rows)")
        return self

    def with_column(self, column: str, values) -> "DataTransformer":
        """Add or modify a column."""
        self.df[column] = values
        self._operations.append(f"with_column({column})")
        return self

    def uppercase(self, column: str, alias: Optional[str] = None) -> "DataTransformer":
        """Convert column to uppercase."""
        alias = alias or column
        self.df[alias] = self.df[column].str.upper()
        self._operations.append(f"uppercase({column})")
        return self

    def lowercase(self, column: str, alias: Optional[str] = None) -> "DataTransformer":
        """Convert column to lowercase."""
        alias = alias or column
        self.df[alias] = self.df[column].str.lower()
        self._operations.append(f"lowercase({column})")
        return self

    def drop_nulls(self, columns: Optional[List[str]] = None) -> "DataTransformer":
        """Drop rows with null values."""
        before = len(self.df)
        if columns:
            self.df = self.df.dropna(subset=columns)
        else:
            self.df = self.df.dropna()
        after = len(self.df)
        self._operations.append(f"drop_nulls(removed {before - after} rows)")
        return self

    def drop_duplicates(self, columns: Optional[List[str]] = None) -> "DataTransformer":
        """Drop duplicate rows."""
        before = len(self.df)
        if columns:
            self.df = self.df.drop_duplicates(subset=columns)
        else:
            self.df = self.df.drop_duplicates()
        after = len(self.df)
        self._operations.append(f"drop_duplicates(removed {before - after} rows)")
        return self

    def apply(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> "DataTransformer":
        """Apply a custom transformation function."""
        self.df = func(self.df)
        self._operations.append("apply(custom)")
        return self

    def get_result(self) -> pd.DataFrame:
        """Get the transformed DataFrame."""
        logger.info(f"Applied {len(self._operations)} transformations")
        for op in self._operations:
            logger.debug(f"  - {op}")
        return self.df

    def get_operations(self) -> List[str]:
        """Get list of applied operations."""
        return self._operations.copy()


class SupplierTransformer:
    """
    Specialized transformer for supplier data.

    Implements the specific transformation logic:
    1. Join supplier to nation
    2. Uppercase supplier name
    3. Select and rename output columns

    Note: This does not inherit from BaseTransformer because it requires
    two DataFrames (supplier + nation) rather than the single-DataFrame
    interface defined by BaseTransformer.transform().
    """

    def __init__(
        self,
        supplier_key_col: str = "s_nationkey",
        nation_key_col: str = "n_nationkey",
        supplier_name_col: str = "s_name",
        supplier_phone_col: str = "s_phone",
        nation_name_col: str = "n_name"
    ):
        self.supplier_key_col = supplier_key_col
        self.nation_key_col = nation_key_col
        self.supplier_name_col = supplier_name_col
        self.supplier_phone_col = supplier_phone_col
        self.nation_name_col = nation_name_col

    def transform(
        self,
        supplier_df: pd.DataFrame,
        nation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform supplier and nation data.

        Args:
            supplier_df: Supplier DataFrame
            nation_df: Nation DataFrame

        Returns:
            Transformed DataFrame with joined and cleaned data
        """
        logger.info("Starting supplier data transformation")

        # Step 1: Join supplier to nation
        joined_df = supplier_df.merge(
            nation_df,
            left_on=self.supplier_key_col,
            right_on=self.nation_key_col,
            how="inner"
        )
        logger.info(f"Joined DataFrames: {len(joined_df)} rows")

        # Step 2 & 3: Uppercase and select columns
        transformed_df = pd.DataFrame({
            "supplier_name": joined_df[self.supplier_name_col].str.upper(),
            "supplier_phone_number": joined_df[self.supplier_phone_col],
            "supplier_nation": joined_df[self.nation_name_col]
        })

        logger.info(f"Transformation complete: {len(transformed_df)} rows")
        return transformed_df
