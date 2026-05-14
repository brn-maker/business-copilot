"""
Data Ingestion Agent: Handles loading, parsing, and cleaning agricultural data.

Uses LangChain tools for data operations and interfaces with the supervisor agent.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from tools import data_tools
from utils import DataProcessor, prompts


@tool
def load_excel_data(file_path: str) -> Dict[str, Any]:
    """
    Load Excel file and detect relevant sheets (Tenant Payments, Coffee Outturn, Grading, etc).
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        Dictionary with loaded sheets and quality report
    """
    processor = DataProcessor()
    try:
        detected_sheets = processor.load_excel_file(file_path)
        quality_reports = {}
        
        for sheet_type, df in detected_sheets.items():
            quality_report = processor.get_quality_report(df, sheet_type)
            quality_reports[sheet_type] = quality_report
        
        return {
            "status": "success",
            "sheets_detected": list(detected_sheets.keys()),
            "quality_reports": quality_reports,
            "file_path": file_path,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def clean_dataset(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Clean datasets: handle missing values, standardize formats, detect outliers.
    
    Args:
        df_dict: Dictionary of {sheet_type: DataFrame}
        
    Returns:
        Cleaning results and quality metrics
    """
    processor = DataProcessor()
    results = {}
    
    try:
        for sheet_type, df in df_dict.items():
            cleaned_df = processor.clean_data(df, sheet_type)
            quality_report = processor.get_quality_report(cleaned_df, f"{sheet_type}_cleaned")
            
            results[sheet_type] = {
                "dataframe": cleaned_df,
                "quality_report": quality_report,
                "rows_retained": len(cleaned_df),
                "issues_found": len(quality_report.get("missing_values", {})) > 0,
            }
        
        return {"status": "success", "cleaning_results": results}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def describe_dataset(df: pd.DataFrame, dataset_name: str = "unknown") -> Dict[str, Any]:
    """
    Generate comprehensive descriptive statistics for dataset.
    
    Args:
        df: Input DataFrame
        dataset_name: Name of dataset for reporting
        
    Returns:
        Descriptive statistics
    """
    try:
        stats = data_tools.describe_dataset(df)
        summary = data_tools.get_summary_text(df, dataset_name)
        return {
            "status": "success",
            "statistics": stats,
            "summary": summary,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def get_data_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get detailed information about dataset columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Column information
    """
    try:
        col_info = data_tools.get_columns_info(df)
        return {
            "status": "success",
            "columns_info": col_info,
            "total_columns": len(col_info),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def validate_data_quality(df: pd.DataFrame, strict: bool = False) -> Dict[str, Any]:
    """
    Validate data quality and report issues.
    
    Args:
        df: Input DataFrame
        strict: If True, flag any missing values as errors
        
    Returns:
        Validation report
    """
    try:
        validation = data_tools.validate_data_quality(df, strict=strict)
        return {
            "status": "success",
            "validation": validation,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def merge_multiple_datasets(dataframes_dict: Dict[str, pd.DataFrame], on_date: bool = True) -> Dict[str, Any]:
    """
    Merge multiple datasets by date or other common keys.
    
    Args:
        dataframes_dict: Dictionary of {name: DataFrame}
        on_date: Whether to merge on date column
        
    Returns:
        Merged DataFrame and merge info
    """
    try:
        merged = data_tools.DataProcessor().merge_datasets(dataframes_dict, on_date=on_date)
        
        merge_info = {
            "original_datasets": len(dataframes_dict),
            "merged_shape": merged.shape,
            "columns_in_merged": list(merged.columns),
        }
        
        return {
            "status": "success",
            "merged_data": merged,
            "merge_info": merge_info,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def filter_data_by_condition(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> Dict[str, Any]:
    """
    Filter data by specified condition.
    
    Args:
        df: Input DataFrame
        column: Column to filter on
        operator: '==', '!=', '>', '<', '>=', '<=', 'in', 'contains'
        value: Filter value
        
    Returns:
        Filtered DataFrame and filter info
    """
    try:
        filtered = data_tools.filter_data(df, column, operator, value)
        return {
            "status": "success",
            "filtered_data": filtered,
            "original_rows": len(df),
            "filtered_rows": len(filtered),
            "rows_removed": len(df) - len(filtered),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_data_agent_system_prompt() -> str:
    """Create system prompt for data agent."""
    return prompts.SYSTEM_PROMPTS["data_agent"] + "\n\n" + prompts.CONTEXT_INJECTION


def get_data_agent_tools() -> List:
    """Return list of tools available to data agent."""
    return [
        load_excel_data,
        clean_dataset,
        describe_dataset,
        get_data_columns,
        validate_data_quality,
        merge_multiple_datasets,
        filter_data_by_condition,
    ]
