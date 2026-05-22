"""
Data Processing Module: Load, validate, clean, and process agricultural data from Excel files.

Handles:
- Excel file parsing with sheet auto-detection
- Date standardization
- Column name normalization
- Missing value handling
- Outlier detection
- Dataset merging by date/month
- Data quality reporting
"""

import io
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json


class DataProcessor:
    """Processes agricultural Excel data for analytics."""
    
    # Sheet name patterns to detect
    SHEET_PATTERNS = {
        "tenant_payments": ["tenant", "payment", "revenue"],
        "coffee_outturn": ["outturn", "output", "yield", "coffee yield"],
        "grading": ["grading", "grade", "quality"],
        "hotel_sales": ["hotel", "accommodation", "room sales", "hospitality"],
        "value_addition": ["value add", "processing", "roasting", "packag"],
        "fertilizer": ["fertilizer", "fert", "input cost", "agro input"],
        "coffee_sales": ["coffee sales", "coffee_sales", "sales"],
        "transactions": ["transactions", "transaction"],
    }
    
    def __init__(self):
        """Initialize DataProcessor."""
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        self.data_quality_report: Dict[str, Any] = {}
    
    def load_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load Excel file and auto-detect relevant sheets.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary of {sheet_type: DataFrame}
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            detected_sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                sheet_type = self._detect_sheet_type(sheet_name)
                if sheet_type:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    detected_sheets[sheet_type] = df
                    self.loaded_data[sheet_type] = df
            
            return detected_sheets
        except Exception as e:
            raise ValueError(f"Error loading Excel file {file_path}: {str(e)}")
    
    def _detect_sheet_type(self, sheet_name: str) -> Optional[str]:
        """Detect sheet type by comparing against known patterns."""
        sheet_lower = sheet_name.lower()
        
        for sheet_type, patterns in self.SHEET_PATTERNS.items():
            for pattern in patterns:
                if pattern in sheet_lower:
                    return sheet_type
        
        return None
    
    def clean_data(self, df: pd.DataFrame, sheet_type: str = "unknown") -> pd.DataFrame:
        """
        Clean dataframe: fix dates, standardize columns, handle missing values.
        
        Args:
            df: Input DataFrame
            sheet_type: Type of data (for context-specific cleaning)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove completely empty rows/columns
        df = df.dropna(how="all")
        df = df.loc[:, (df != "").any(axis=0)]
        
        # Standardize column names: lowercase, remove special chars, strip
        df.columns = [
            col.lower().strip().replace(" ", "_").replace("-", "_")
            for col in df.columns
        ]
        
        # Try to parse date columns
        date_cols = [col for col in df.columns if any(d in col for d in ["date", "month", "year", "period"])]
        for col in date_cols:
            df = self._standardize_date_column(df, col)
        
        # Handle missing values based on data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 0:
                # Fill small gaps with interpolation, otherwise mark
                if missing_pct < 20:
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
                    # Fill any remaining NaNs at edges
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _standardize_date_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Try to parse and standardize date column.
        
        Args:
            df: DataFrame
            col: Column name
            
        Returns:
            DataFrame with standardized date column
        """
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
        except:
            # If parsing fails, leave as-is
            pass
        
        return df
    
    def merge_datasets(
        self,
        dataframes: Dict[str, pd.DataFrame],
        on_date: bool = True,
        on_property: bool = False,
    ) -> pd.DataFrame:
        """
        Intelligently merge multiple datasets.
        
        Args:
            dataframes: Dictionary of {type: DataFrame}
            on_date: Merge by date/month
            on_property: Merge by property/location column if available
            
        Returns:
            Merged DataFrame
        """
        if not dataframes:
            raise ValueError("No dataframes to merge")
        
        dfs = list(dataframes.values())
        merged = dfs[0].copy()
        
        for df in dfs[1:]:
            if on_date:
                # Find date columns in both
                date_cols_merged = [c for c in merged.columns if "date" in c]
                date_cols_df = [c for c in df.columns if "date" in c]
                
                if date_cols_merged and date_cols_df:
                    merged = pd.merge(
                        merged,
                        df,
                        left_on=date_cols_merged[0],
                        right_on=date_cols_df[0],
                        how="outer",
                    )
                else:
                    # Fallback: concatenate columns
                    merged = pd.concat([merged, df], axis=1)
            else:
                merged = pd.concat([merged, df], axis=1)
        
        return merged
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = "iqr") -> pd.DataFrame:
        """
        Detect outliers in numeric column.
        
        Args:
            df: DataFrame
            column: Column name
            method: 'iqr' (interquartile range) or 'zscore'
            
        Returns:
            DataFrame with 'outlier' column added
        """
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df["outlier"] = (df[column] < lower) | (df[column] > upper)
        
        elif method == "zscore":
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df["outlier"] = z_scores > 3
        
        return df
    
    def get_quality_report(self, df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
        """
        Generate data quality report.
        
        Args:
            df: DataFrame
            name: Dataset name
            
        Returns:
            Quality report dictionary
        """
        report = {
            "name": name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isna().sum().to_dict(),
            "missing_percentage": (df.isna().sum() / len(df) * 100).to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "duplicates": len(df[df.duplicated()]),
        }
        
        # Numeric column stats
        numeric_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        
        report["numeric_stats"] = numeric_stats
        
        return report
    
    def format_quality_report(self, report: Dict[str, Any]) -> str:
        """Format quality report as human-readable string."""
        lines = [
            f"📊 Data Quality Report: {report['name']}",
            f"Total rows: {report['total_rows']} | Total columns: {report['total_columns']}",
            f"Duplicate rows: {report['duplicates']}",
        ]
        
        missing = {k: v for k, v in report['missing_percentage'].items() if v > 0}
        if missing:
            lines.append("Missing values:")
            for col, pct in missing.items():
                lines.append(f"  - {col}: {pct:.1f}%")
        
        return "\n".join(lines)


def process_uploaded_file(file_buffer: io.BytesIO, filename: str) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Process an uploaded file and return detected sheets + quality report.
    
    Args:
        file_buffer: File buffer from Streamlit upload
        filename: Original filename
        
    Returns:
        (detected_sheets, quality_report_text)
    """
    processor = DataProcessor()
    
    try:
        # Save buffer to temp location for processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(file_buffer.getvalue())
            tmp_path = tmp.name
        
        # Load and detect sheets
        detected_sheets = processor.load_excel_file(tmp_path)
        
        # Clean each sheet
        cleaned_data = {}
        report_lines = [f"✅ Successfully loaded: {filename}"]
        
        for sheet_type, df in detected_sheets.items():
            cleaned_df = processor.clean_data(df, sheet_type)
            cleaned_data[sheet_type] = cleaned_df
            
            quality_report = processor.get_quality_report(cleaned_df, sheet_type)
            report_lines.append(processor.format_quality_report(quality_report))
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        return cleaned_data, "\n".join(report_lines)
        
    except Exception as e:
        return {}, f"❌ Error processing file: {str(e)}"
