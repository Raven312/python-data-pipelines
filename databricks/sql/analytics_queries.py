"""Databricks SQL analytics queries for dashboards and reporting.

These queries are designed to run against Gold-layer Delta tables via
Databricks SQL warehouses. They power executive dashboards, operational
reports, and ad-hoc analytics.

AE Pathway Coverage:
  - Databricks SQL query patterns
  - Window functions for rankings and running totals
  - CTEs for readability
  - Query optimization with Delta Lake
"""

SUPPLIER_NATION_DISTRIBUTION = """
-- Supplier distribution by nation with rankings
SELECT
    nation_name,
    supplier_count,
    avg_account_balance,
    total_account_balance,
    RANK() OVER (ORDER BY supplier_count DESC) AS nation_rank,
    ROUND(supplier_count * 100.0 / SUM(supplier_count) OVER (), 2) AS pct_of_total,
    ROUND(total_account_balance * 100.0 / SUM(total_account_balance) OVER (), 2) AS pct_of_balance,
    _computed_at
FROM {catalog}.gold.supplier_by_nation
ORDER BY supplier_count DESC
"""

SUPPLIER_KPI_DASHBOARD = """
-- Executive KPI dashboard query
SELECT
    total_suppliers,
    total_nations,
    avg_account_balance,
    total_account_balance,
    ROUND(total_account_balance / NULLIF(total_suppliers, 0), 2) AS balance_per_supplier,
    last_updated,
    _computed_at,
    DATEDIFF(CURRENT_TIMESTAMP(), last_updated) AS days_since_update
FROM {catalog}.gold.supplier_summary
"""

TOP_SUPPLIERS_WITH_PERCENTILE = """
-- Top suppliers with percentile ranking
WITH supplier_stats AS (
    SELECT
        supplier_id,
        supplier_name,
        nation_name,
        account_balance,
        PERCENT_RANK() OVER (ORDER BY account_balance) AS balance_percentile,
        account_balance - AVG(account_balance) OVER () AS balance_vs_avg,
        ROW_NUMBER() OVER (PARTITION BY nation_name ORDER BY account_balance DESC) AS rank_in_nation
    FROM {catalog}.silver.suppliers
    WHERE _is_current = true
)
SELECT
    supplier_id,
    supplier_name,
    nation_name,
    account_balance,
    ROUND(balance_percentile * 100, 1) AS percentile,
    ROUND(balance_vs_avg, 2) AS vs_average,
    rank_in_nation
FROM supplier_stats
ORDER BY account_balance DESC
LIMIT 20
"""

PIPELINE_HEALTH_MONITOR = """
-- Pipeline health monitoring query
WITH table_freshness AS (
    SELECT
        'silver_suppliers' AS table_name,
        MAX(_valid_from) AS last_update,
        COUNT(*) AS row_count,
        COUNT(DISTINCT supplier_id) AS distinct_keys,
        SUM(CASE WHEN _is_current THEN 1 ELSE 0 END) AS current_rows,
        SUM(CASE WHEN NOT _is_current THEN 1 ELSE 0 END) AS historical_rows
    FROM {catalog}.silver.suppliers
    UNION ALL
    SELECT
        'gold_supplier_by_nation',
        MAX(_computed_at),
        COUNT(*),
        COUNT(DISTINCT nation_name),
        COUNT(*),
        0
    FROM {catalog}.gold.supplier_by_nation
)
SELECT
    table_name,
    last_update,
    TIMESTAMPDIFF(MINUTE, last_update, CURRENT_TIMESTAMP()) AS minutes_stale,
    row_count,
    distinct_keys,
    current_rows,
    historical_rows,
    CASE
        WHEN TIMESTAMPDIFF(MINUTE, last_update, CURRENT_TIMESTAMP()) > 60 THEN 'STALE'
        WHEN TIMESTAMPDIFF(MINUTE, last_update, CURRENT_TIMESTAMP()) > 30 THEN 'WARNING'
        ELSE 'FRESH'
    END AS freshness_status
FROM table_freshness
ORDER BY table_name
"""

DATA_QUALITY_AUDIT = """
-- Data quality audit across layers
SELECT
    'bronze_suppliers' AS layer_table,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN s_suppkey IS NULL THEN 1 ELSE 0 END) AS null_keys,
    SUM(CASE WHEN s_name IS NULL THEN 1 ELSE 0 END) AS null_names,
    COUNT(DISTINCT _batch_id) AS batch_count,
    MIN(_ingested_at) AS first_ingestion,
    MAX(_ingested_at) AS last_ingestion
FROM {catalog}.bronze.suppliers
UNION ALL
SELECT
    'silver_suppliers',
    COUNT(*),
    SUM(CASE WHEN supplier_id IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN supplier_name IS NULL THEN 1 ELSE 0 END),
    0,
    MIN(_valid_from),
    MAX(_valid_from)
FROM {catalog}.silver.suppliers
WHERE _is_current = true
"""


def get_query(query_name: str, catalog: str = "supply_chain") -> str:
    """Get a formatted SQL query by name.

    Args:
        query_name: One of the query constants defined in this module.
        catalog: Unity Catalog name to substitute into the query.

    Returns:
        Formatted SQL query string.
    """
    queries = {
        "supplier_nation_distribution": SUPPLIER_NATION_DISTRIBUTION,
        "supplier_kpi_dashboard": SUPPLIER_KPI_DASHBOARD,
        "top_suppliers_with_percentile": TOP_SUPPLIERS_WITH_PERCENTILE,
        "pipeline_health_monitor": PIPELINE_HEALTH_MONITOR,
        "data_quality_audit": DATA_QUALITY_AUDIT,
    }
    template = queries.get(query_name)
    if template is None:
        raise ValueError(f"Unknown query: {query_name}. Available: {list(queries.keys())}")
    return template.format(catalog=catalog)
