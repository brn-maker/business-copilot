"""
Analysis Agent: Performs statistical analysis on agricultural data.

Handles correlations, regression, anomalies, and growth analysis.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import json
from langchain_core.tools import tool
from tools import analysis_tools, viz_tools
from utils import prompts


@tool
def analyze_correlations(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze correlations between variables.
    
    Useful for identifying relationships like:
    - Fertilizer usage vs outturn
    - Grading vs value addition sales
    - Hotel occupancy vs revenue
    
    Args:
        df: Input DataFrame
        columns: Specific columns to analyze (default: all numeric)
        
    Returns:
        Correlation analysis with strong relationships identified
    """
    try:
        result = analysis_tools.correlation_analysis(df, columns)
        return {
            "status": "success",
            "analysis": result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def perform_regression(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
) -> Dict[str, Any]:
    """
    Perform linear/multiple regression analysis.
    
    Examples:
    - Coffee outturn vs fertilizer usage, rainfall, processing method
    - Value addition sales vs grading, season, market price
    - Tenant payments vs farm yield, coffee quality
    
    Args:
        df: Input DataFrame
        dependent_var: Y variable (what we predict)
        independent_vars: X variables (predictors)
        
    Returns:
        Regression coefficients, R², p-values, confidence intervals, interpretation
    """
    try:
        result = analysis_tools.regression_analysis(df, dependent_var, independent_vars)
        return {
            "status": "success",
            "regression_results": result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def analyze_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> Dict[str, Any]:
    """
    Decompose time-series into trend, seasonality, and residuals.
    
    Useful for understanding patterns in:
    - Monthly coffee outturn (seasonality)
    - Hotel revenue trends (seasonal peaks/troughs)
    - Fertilizer usage patterns (seasonal purchasing)
    
    Args:
        df: DataFrame with time-series data
        date_column: Name of date column
        value_column: Value column to decompose
        
    Returns:
        Trend, seasonal, and residual components
    """
    try:
        result = analysis_tools.time_series_decomposition(df, date_column, value_column)
        return {
            "status": "success",
            "decomposition": result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
) -> Dict[str, Any]:
    """
    Detect anomalies in data using IQR or Z-score method.
    
    Useful for identifying unusual:
    - Coffee outturn values
    - Revenue spikes/drops
    - Payment delays or unusual amounts
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        method: 'iqr' (interquartile range) or 'zscore'
        
    Returns:
        Anomalies detected with thresholds and counts
    """
    try:
        result = analysis_tools.anomaly_detection(df, column, method)
        return {
            "status": "success",
            "anomalies": result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def analyze_growth_rates(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = 1,
) -> Dict[str, Any]:
    """
    Calculate growth rates and compound annual growth rate (CAGR).
    
    Useful for measuring trends in:
    - Revenue growth over months/years
    - Outturn improvement
    - Sales volume trends
    
    Args:
        df: DataFrame with time-series
        date_column: Date column
        value_column: Value column
        periods: Periods for growth calc (1=period-over-period, 12=year-over-year for monthly data)
        
    Returns:
        Growth rates, CAGR, and statistics
    """
    try:
        result = analysis_tools.growth_rate_analysis(df, date_column, value_column, periods)
        return {
            "status": "success",
            "growth_analysis": result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def get_summary_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics for a single column.
    
    Returns mean, median, std dev, variance, quantiles, skewness, kurtosis.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        
    Returns:
        Summary statistics
    """
    try:
        stats = analysis_tools.get_summary_statistics(df, column)
        return {
            "status": "success",
            "statistics": stats,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def create_correlation_visualization(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create interactive correlation heatmap visualization.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to include
        
    Returns:
        Plotly figure serialized as JSON
    """
    try:
        fig = viz_tools.create_correlation_heatmap(df, columns)
        return {
            "status": "success",
            "chart": fig.to_json(),
            "chart_type": "correlation_heatmap",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def create_regression_visualization(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> Dict[str, Any]:
    """
    Create interactive regression chart with confidence interval.
    
    Args:
        df: Input DataFrame
        x_column: Independent variable (X)
        y_column: Dependent variable (Y)
        
    Returns:
        Plotly figure serialized as JSON
    """
    try:
        fig = viz_tools.create_regression_chart(df, x_column, y_column)
        return {
            "status": "success",
            "chart": fig.to_json(),
            "chart_type": "regression",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def create_distribution_chart(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Create histogram of data distribution.
    
    Args:
        df: Input DataFrame
        column: Column to plot
        
    Returns:
        Plotly figure serialized as JSON
    """
    try:
        fig = viz_tools.create_distribution_chart(df, column)
        return {
            "status": "success",
            "chart": fig.to_json(),
            "chart_type": "distribution",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_analysis_agent_system_prompt() -> str:
    """Create system prompt for analysis agent."""
    return prompts.SYSTEM_PROMPTS["analysis_agent"] + "\n\n" + prompts.CONTEXT_INJECTION


def get_analysis_agent_tools() -> List:
    """Return list of tools available to analysis agent."""
    return [
        analyze_correlations,
        perform_regression,
        analyze_time_series,
        detect_anomalies,
        analyze_growth_rates,
        get_summary_stats,
        create_correlation_visualization,
        create_regression_visualization,
        create_distribution_chart,
    ]
