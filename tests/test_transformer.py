"""Tests for the transform module."""

import pytest
import pandas as pd

from src.transform.transformer import DataTransformer, SupplierTransformer


class TestDataTransformer:
    """Tests for DataTransformer method chaining."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
            "city": ["NYC", "LA", "NYC"],
        })

    def test_select(self, sample_df):
        result = DataTransformer(sample_df).select(["name", "age"]).get_result()
        assert list(result.columns) == ["name", "age"]

    def test_rename(self, sample_df):
        result = DataTransformer(sample_df).rename({"name": "full_name"}).get_result()
        assert "full_name" in result.columns
        assert "name" not in result.columns

    def test_filter(self, sample_df):
        result = DataTransformer(sample_df).filter(sample_df["age"] > 28).get_result()
        assert len(result) == 2

    def test_join(self, sample_df):
        other = pd.DataFrame({"city": ["NYC", "LA"], "state": ["NY", "CA"]})
        result = (
            DataTransformer(sample_df)
            .join(other, left_on="city", right_on="city", how="inner")
            .get_result()
        )
        assert "state" in result.columns
        assert len(result) == 3

    def test_uppercase(self, sample_df):
        result = DataTransformer(sample_df).uppercase("name").get_result()
        assert result["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]

    def test_uppercase_with_alias(self, sample_df):
        result = DataTransformer(sample_df).uppercase("name", alias="NAME_UPPER").get_result()
        assert "NAME_UPPER" in result.columns
        assert result["NAME_UPPER"].tolist() == ["ALICE", "BOB", "CHARLIE"]

    def test_lowercase(self, sample_df):
        result = DataTransformer(sample_df).lowercase("city").get_result()
        assert result["city"].tolist() == ["nyc", "la", "nyc"]

    def test_drop_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = DataTransformer(df).drop_nulls().get_result()
        assert len(result) == 1

    def test_drop_nulls_subset(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = DataTransformer(df).drop_nulls(columns=["a"]).get_result()
        assert len(result) == 2

    def test_drop_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = DataTransformer(df).drop_duplicates().get_result()
        assert len(result) == 2

    def test_drop_duplicates_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
        result = DataTransformer(df).drop_duplicates(columns=["a"]).get_result()
        assert len(result) == 2

    def test_with_column(self, sample_df):
        result = DataTransformer(sample_df).with_column("score", 100).get_result()
        assert "score" in result.columns
        assert all(result["score"] == 100)

    def test_apply(self, sample_df):
        result = (
            DataTransformer(sample_df)
            .apply(lambda df: df.assign(age_doubled=df["age"] * 2))
            .get_result()
        )
        assert "age_doubled" in result.columns
        assert result["age_doubled"].tolist() == [60, 50, 70]

    def test_chaining_multiple_operations(self, sample_df):
        result = (
            DataTransformer(sample_df)
            .select(["name", "age"])
            .rename({"name": "full_name"})
            .uppercase("full_name")
            .get_result()
        )
        assert list(result.columns) == ["full_name", "age"]
        assert result["full_name"].tolist() == ["ALICE", "BOB", "CHARLIE"]

    def test_get_operations(self, sample_df):
        transformer = DataTransformer(sample_df).select(["name"]).uppercase("name")
        ops = transformer.get_operations()
        assert len(ops) == 2
        assert "select" in ops[0]
        assert "uppercase" in ops[1]

    def test_does_not_mutate_original(self, sample_df):
        original_names = sample_df["name"].tolist()
        DataTransformer(sample_df).uppercase("name").get_result()
        assert sample_df["name"].tolist() == original_names


class TestSupplierTransformer:
    """Tests for SupplierTransformer."""

    @pytest.fixture
    def supplier_df(self):
        return pd.DataFrame({
            "s_suppkey": [1, 2],
            "s_name": ["Supplier#001", "Supplier#002"],
            "s_nationkey": [0, 1],
            "s_phone": ["10-111-1111", "20-222-2222"],
        })

    @pytest.fixture
    def nation_df(self):
        return pd.DataFrame({
            "n_nationkey": [0, 1],
            "n_name": ["ALGERIA", "BRAZIL"],
            "n_regionkey": [0, 1],
        })

    def test_transform(self, supplier_df, nation_df):
        transformer = SupplierTransformer()
        result = transformer.transform(supplier_df, nation_df)
        assert list(result.columns) == [
            "supplier_name", "supplier_phone_number", "supplier_nation"
        ]
        assert len(result) == 2

    def test_transform_uppercases_names(self, supplier_df, nation_df):
        result = SupplierTransformer().transform(supplier_df, nation_df)
        assert result["supplier_name"].tolist() == ["SUPPLIER#001", "SUPPLIER#002"]

    def test_transform_joins_nation(self, supplier_df, nation_df):
        result = SupplierTransformer().transform(supplier_df, nation_df)
        assert result["supplier_nation"].tolist() == ["ALGERIA", "BRAZIL"]

    def test_transform_inner_join_drops_unmatched(self):
        supplier_df = pd.DataFrame({
            "s_suppkey": [1, 2],
            "s_name": ["Supplier#001", "Supplier#002"],
            "s_nationkey": [0, 99],  # 99 doesn't exist in nations
            "s_phone": ["10-111-1111", "20-222-2222"],
        })
        nation_df = pd.DataFrame({
            "n_nationkey": [0],
            "n_name": ["ALGERIA"],
            "n_regionkey": [0],
        })
        result = SupplierTransformer().transform(supplier_df, nation_df)
        assert len(result) == 1

    def test_is_not_base_transformer(self):
        """SupplierTransformer should NOT inherit from BaseTransformer."""
        from src.transform.transformer import BaseTransformer
        assert not issubclass(SupplierTransformer, BaseTransformer)
