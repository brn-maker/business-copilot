"""
Data Tools: Functions for loading, cleaning, and processing agricultural data.

These tools are called by the Data Ingestion Agent and other agents.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


def describe_dataset(df: pd.DataFrame, max_rows: int = 1000) -> Dict[str, Any]:
    """
    Generate descriptive statistics for a dataset.
    
    Args:
        df: Input DataFrame
        max_rows: Max rows to describe (for performance)
        
    Returns:
        Dictionary with statistics
    """
    if df is None or df.empty:
        return {"error": "Empty dataset"}
    
    df = df.head(max_rows) if len(df) > max_rows else df
    
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "missing_percentage": (df.isna().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    # Numeric summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Sample rows
    stats["sample_rows"] = df.head(3).to_dict(orient="records")
    
    return stats


def get_columns_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed information about columns."""
    if df is None or df.empty:
        return {"error": "Empty dataset"}
    
    info = {}
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info.update({
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
            })
        else:
            # Categorical
            col_info["unique_values"] = int(df[col].nunique())
            col_info["top_values"] = df[col].value_counts().head(5).to_dict()
        
        info[col] = col_info
    
    return info


def filter_data(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> pd.DataFrame:
    """
    Filter dataframe by condition.
    
    Args:
        df: Input DataFrame
        column: Column to filter on
        operator: '==', '!=', '>', '<', '>=', '<=', 'in', 'contains'
        value: Filter value
        
    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    if operator == "==":
        return df[df[column] == value]
    elif operator == "!=":
        return df[df[column] != value]
    elif operator == ">":
        return df[df[column] > value]
    elif operator == "<":
        return df[df[column] < value]
    elif operator == ">=":
        return df[df[column] >= value]
    elif operator == "<=":
        return df[df[column] <= value]
    elif operator == "in":
        return df[df[column].isin(value)]
    elif operator == "contains":
        return df[df[column].astype(str).str.contains(str(value), case=False)]
    else:
        raise ValueError(f"Unknown operator: {operator}")


def aggregate_data(
    df: pd.DataFrame,
    group_by: List[str],
    agg_dict: Dict[str, str],
) -> pd.DataFrame:
    """
    Group and aggregate data.
    
    Args:
        df: Input DataFrame
        group_by: Columns to group by
        agg_dict: Dictionary like {'column': 'mean', 'column2': 'sum'}
        
    Returns:
        Aggregated DataFrame
    """
    return df.groupby(group_by, as_index=False).agg(agg_dict)


def resample_timeseries(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    freq: str = "M",  # 'D' for day, 'W' for week, 'M' for month, 'Q' for quarter, 'Y' for year
    agg_func: str = "mean",
) -> pd.DataFrame:
    """
    Resample time-series data to different frequency.
    
    Args:
        df: DataFrame with dates
        date_column: Name of date column
        value_column: Column to aggregate
        freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
        agg_func: Aggregation function ('mean', 'sum', 'first', 'last', 'min', 'max')
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    
    resampled = df.set_index(date_column)[value_column].resample(freq).agg(agg_func)
    return resampled.reset_index()


def calculate_rolling_average(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    window: int = 3,
) -> pd.DataFrame:
    """
    Calculate rolling average for smoothing trends.
    
    Args:
        df: DataFrame with dates
        date_column: Name of date column
        value_column: Column to smooth
        window: Rolling window size
        
    Returns:
        DataFrame with added rolling_average column
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.sort_values(date_column)
    df["rolling_average"] = df[value_column].rolling(window=window, center=True).mean()
    return df


def get_summary_text(df: pd.DataFrame, dataset_name: str = "") -> str:
    """Generate human-readable summary of dataset."""
    lines = []
    
    if dataset_name:
        lines.append(f"📊 {dataset_name}")
    
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    if missing_pct > 0:
        lines.append(f"Missing data: {missing_pct:.1f}%")
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        lines.append(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}")
    
    # Date columns summary
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        lines.append(f"Date columns: {', '.join(date_cols)}")
    
    return "\n".join(lines)


def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def export_to_json(df: pd.DataFrame) -> str:
    """Export DataFrame to JSON string."""
    return df.to_json(orient="records", date_format="iso")


def validate_data_quality(df: pd.DataFrame, strict: bool = False) -> Dict[str, Any]:
    """
    Validate data quality and return issues.
    
    Args:
        df: DataFrame to validate
        strict: If True, flag any missing values as errors
        
    Returns:
        Quality validation report
    """
    issues = []
    warnings = []
    
    # Check for completely empty rows/columns
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        issues.append(f"Found {empty_rows} completely empty rows")
    
    empty_cols = df.isnull().all(axis=0).sum()
    if empty_cols > 0:
        issues.append(f"Found {empty_cols} completely empty columns")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(1)
        for col, pct in missing_pct[missing_pct > 0].items():
            if strict or pct > 30:
                issues.append(f"Column '{col}': {pct}% missing values")
            else:
                warnings.append(f"Column '{col}': {pct}% missing values")
    
    # Check for potential outliers in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_pct = outliers / len(df) * 100
            if outlier_pct > 5:
                warnings.append(f"Column '{col}': {outlier_pct:.1f}% potential outliers")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "summary": f"{'✅ Data valid' if len(issues) == 0 else '❌ Data has issues'} | Issues: {len(issues)}, Warnings: {len(warnings)}",
    }
