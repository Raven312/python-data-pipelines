# Python Data Pipelines by Defining Architecture with LLM

A modular, production-ready ETL pipeline architecture designed with LLM assistance. Built with **Pandas** for data processing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE ORCHESTRATOR                     │
│                         (src/pipeline.py)                        │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┐
        ▼             ▼           ▼           ▼             ▼
   ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐  ┌───────┐
   │ EXTRACT │  │ TRANSFORM │  │ VALIDATE │  │  LOAD  │  │ UTILS │
   │         │  │           │  │          │  │        │  │       │
   │ • Dict  │  │ • Join    │  │ • Unique │  │ • CSV  │  │• Logger│
   │ • CSV   │  │ • Clean   │  │ • NotNull│  │• Parquet│ │• Tracker│
   │ • API   │  │ • Rename  │  │ • Count  │  │ • JSON │  │       │
   └─────────┘  └───────────┘  └──────────┘  └────────┘  └───────┘
```

## Project Structure

```
Python Data Pipelines by Defining Architecture with LLM/
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration dataclasses
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Main orchestrator
│   ├── extract/
│   │   ├── __init__.py
│   │   └── extractor.py       # Data extraction (Dict, CSV, API)
│   ├── transform/
│   │   ├── __init__.py
│   │   └── transformer.py     # Data transformations
│   ├── validate/
│   │   ├── __init__.py
│   │   └── validator.py       # Data quality validation
│   ├── load/
│   │   ├── __init__.py
│   │   └── loader.py          # Data output (CSV, Parquet, JSON)
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging configuration
│       └── tracker.py         # Pipeline metrics tracking
├── data/
│   ├── raw/                   # Input data
│   └── processed/             # Output data
├── logs/                      # Pipeline logs and metrics
└── tests/                     # Unit tests
```

## Installation

```bash
cd "Python Data Pipelines by Defining Architecture with LLM"
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Run with defaults
python main.py

# Custom output path
python main.py --output data/processed/custom_output.csv

# Debug logging
python main.py --log-level DEBUG

# No file logging
python main.py --no-log-file
```

### Programmatic Usage

```python
from src.pipeline import SupplierETLPipeline, create_sample_data
from pathlib import Path

# Create sample data
supplier_data, nation_data = create_sample_data()

# Create and run pipeline
pipeline = SupplierETLPipeline(
    output_path=Path("output.csv"),
    metrics_path=Path("logs/metrics.csv")
)

result = pipeline.run(supplier_data, nation_data)

if result.success:
    print(f"Output: {result.output_path}")
    print(f"Rows: {result.output_rows}")
```

### Using Individual Components

```python
import pandas as pd
from src.extract import DataExtractor
from src.transform import DataTransformer
from src.validate import DataValidator
from src.load import DataLoader

# Extract
df = DataExtractor.from_csv("data.csv")

# Transform with method chaining
result = (
    DataTransformer(df)
    .select(["col1", "col2", "col3"])
    .rename({"col1": "new_name"})
    .uppercase("new_name")
    .drop_nulls()
    .get_result()
)

# Validate
validator = (
    DataValidator()
    .add_uniqueness_check("new_name")
    .add_not_null_check("col2")
)
results = validator.validate(result)

# Load
if validator.all_passed():
    DataLoader.to_csv(result, "output.csv")
```

## Key Components

### Extract (`src/extract/extractor.py`)

| Method | Description |
|--------|-------------|
| `from_dict(data)` | Load from Python dictionary |
| `from_csv(path)` | Load from CSV file |
| `from_parquet(path)` | Load from Parquet file |
| `from_api(url)` | Load from REST API |

### Transform (`src/transform/transformer.py`)

| Method | Description |
|--------|-------------|
| `select(columns)` | Select specific columns |
| `rename(mapping)` | Rename columns |
| `filter(condition)` | Filter rows |
| `join(other, ...)` | Join DataFrames |
| `uppercase(column)` | Convert to uppercase |
| `drop_nulls()` | Remove null rows |
| `drop_duplicates()` | Remove duplicates |

### Validate (`src/validate/validator.py`)

| Method | Description |
|--------|-------------|
| `add_uniqueness_check(col)` | Check column uniqueness |
| `add_not_null_check(col)` | Check for null values |
| `add_distinct_count_check(col, max)` | Check distinct count |
| `add_row_count_check(min, max)` | Check row count range |
| `add_custom_check(...)` | Add custom validation |

### Load (`src/load/loader.py`)

| Method | Description |
|--------|-------------|
| `to_csv(df, path)` | Save to CSV |
| `to_parquet(df, path)` | Save to Parquet |
| `to_json(df, path)` | Save to JSON |
| `auto(df, path)` | Auto-detect format |

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **Factory** | `DataExtractor`, `DataLoader` static methods |
| **Builder** | `DataTransformer` method chaining |
| **Strategy** | Pluggable extractors, loaders, validators |
| **Dataclass** | Configuration and result objects |
| **Template** | Abstract base classes for components |

## Output

```
============================================================
Pipeline completed successfully!
  Run ID:       run_20260126_234500
  Input rows:   {'supplier': 10, 'nation': 9}
  Output rows:  10
  Duration:     0.15s
  Output file:  data/processed/supplier_data.csv
============================================================
```

## License

MIT
