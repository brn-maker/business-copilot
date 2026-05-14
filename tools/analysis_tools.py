"""
Analysis Tools: Statistical analysis functions for agricultural data.

Includes correlation, regression, time-series analysis, and scenario modeling.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import json


def correlation_analysis(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate correlation matrix and identify strong correlations.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to analyze (default: all numeric)
        
    Returns:
        Correlation analysis results
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[[col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]]
    
    if numeric_df.empty:
        return {"error": "No numeric columns to correlate"}
    
    correlation_matrix = numeric_df.corr()
    
    # Find strong correlations (abs > 0.6)
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.6:
                strong_correlations.append({
                    "variable_1": correlation_matrix.columns[i],
                    "variable_2": correlation_matrix.columns[j],
                    "correlation": float(corr_value),
                    "strength": "strong" if abs(corr_value) > 0.8 else "moderate",
                })
    
    return {
        "correlation_matrix": correlation_matrix.to_dict(),
        "strong_correlations": sorted(strong_correlations, key=lambda x: abs(x["correlation"]), reverse=True),
        "summary": f"Found {len(strong_correlations)} strong correlations (|r| > 0.6)",
    }


def regression_analysis(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
) -> Dict[str, Any]:
    """
    Perform linear/multiple regression analysis.
    
    Args:
        df: Input DataFrame
        dependent_var: Y variable (dependent)
        independent_vars: X variables (independent)
        
    Returns:
        Regression results with coefficients, R², p-values, CI
    """
    # Filter to numeric columns and remove NaNs
    analysis_cols = [dependent_var] + independent_vars
    df_clean = df[analysis_cols].dropna()
    
    if df_clean.empty:
        return {"error": "No valid data for regression"}
    
    X = df_clean[independent_vars].values
    y = df_clean[dependent_var].values
    
    # Add constant for intercept
    X = np.column_stack([np.ones(len(X)), X])
    
    # Calculate regression
    try:
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ coefficients
        
        # Calculate statistics
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Standard error and t-stats
        n = len(y)
        k = len(independent_vars)
        mse = ss_res / (n - k - 1)
        var_covar = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_covar))
        t_stats = coefficients / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        # Confidence intervals (95%)
        t_crit = stats.t.ppf(0.975, n - k - 1)
        ci_lower = coefficients - t_crit * se
        ci_upper = coefficients + t_crit * se
        
        # Format results
        results = {
            "intercept": float(coefficients[0]),
            "intercept_pvalue": float(p_values[0]),
            "coefficients": {},
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - k - 1)),
            "n_observations": int(n),
            "residual_std_error": float(np.sqrt(mse)),
        }
        
        for i, var in enumerate(independent_vars):
            results["coefficients"][var] = {
                "coefficient": float(coefficients[i + 1]),
                "std_error": float(se[i + 1]),
                "t_statistic": float(t_stats[i + 1]),
                "p_value": float(p_values[i + 1]),
                "ci_lower": float(ci_lower[i + 1]),
                "ci_upper": float(ci_upper[i + 1]),
                "significant": p_values[i + 1] < 0.05,
            }
        
        # Interpretation
        interpretation = f"R² = {r_squared:.3f} (explains {r_squared*100:.1f}% of variance). "
        significant_vars = [v for v, c in results["coefficients"].items() if c["significant"]]
        if significant_vars:
            interpretation += f"Significant predictors: {', '.join(significant_vars)}"
        else:
            interpretation += "No significant predictors at α=0.05"
        
        results["interpretation"] = interpretation
        
        return results
    
    except Exception as e:
        return {"error": f"Regression failed: {str(e)}"}


def time_series_decomposition(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> Dict[str, Any]:
    """
    Decompose time-series into trend and seasonality.
    
    Args:
        df: DataFrame with time-series
        date_column: Name of date column
        value_column: Value column to decompose
        
    Returns:
        Decomposition results
    """
    try:
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column], errors="coerce")
        df_ts = df_ts.sort_values(date_column)
        
        # Set as index for seasonal_decompose
        df_ts = df_ts.set_index(date_column)
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Determine period based on data frequency
        period = max(4, len(df_ts) // 4)  # At least 4 observations per cycle
        
        decomposition = seasonal_decompose(
            df_ts[value_column],
            model="additive",
            period=period,
            extrapolate="fill_ea",
        )
        
        return {
            "trend": decomposition.trend.to_dict(),
            "seasonal": decomposition.seasonal.to_dict(),
            "residual": decomposition.resid.to_dict(),
            "seasonal_period": period,
            "method": "additive",
        }
    
    except Exception as e:
        return {"error": f"Decomposition failed: {str(e)}"}


def anomaly_detection(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Detect anomalies in numeric column.
    
    Args:
        df: DataFrame
        column: Column to analyze
        method: 'iqr' (interquartile range) or 'zscore'
        threshold: For z-score method, typically 2-3
        
    Returns:
        Anomaly detection results
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    values = df[column].dropna()
    
    if method == "iqr":
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = values[(values < lower_bound) | (values > upper_bound)]
        
        result = {
            "method": "IQR",
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }
    
    else:  # zscore
        mean = values.mean()
        std = values.std()
        z_scores = np.abs((values - mean) / std)
        anomalies = values[z_scores > threshold]
        
        result = {
            "method": "Z-Score",
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold),
        }
    
    result["anomaly_count"] = len(anomalies)
    result["anomaly_percentage"] = float(len(anomalies) / len(values) * 100)
    result["anomalies"] = anomalies.tolist()
    
    return result


def growth_rate_analysis(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = 1,
) -> Dict[str, Any]:
    """
    Calculate growth rates and compound growth rates.
    
    Args:
        df: DataFrame with time-series
        date_column: Date column
        value_column: Value column
        periods: Periods for growth calculation (1=period-over-period, 12=YoY for monthly)
        
    Returns:
        Growth rate analysis
    """
    df_sorted = df.sort_values(date_column).copy()
    
    # Calculate simple growth rate
    growth_rates = df_sorted[value_column].pct_change(periods=periods) * 100
    
    # Calculate CAGR if possible
    if len(df_sorted) > periods:
        start_value = df_sorted[value_column].iloc[0]
        end_value = df_sorted[value_column].iloc[-1]
        n_years = (len(df_sorted) - 1) / 12 if periods == 12 else (len(df_sorted) - 1)
        
        if start_value > 0 and n_years > 0:
            cagr = (((end_value / start_value) ** (1 / n_years)) - 1) * 100
        else:
            cagr = None
    else:
        cagr = None
    
    return {
        "growth_rates": growth_rates.to_dict(),
        "average_growth_rate": float(growth_rates.mean()),
        "std_growth_rate": float(growth_rates.std()),
        "min_growth_rate": float(growth_rates.min()),
        "max_growth_rate": float(growth_rates.max()),
        "cagr": float(cagr) if cagr else None,
        "periods": periods,
    }


def scenario_analysis(
    df: pd.DataFrame,
    value_column: str,
    scenarios: Dict[str, float],
) -> Dict[str, Any]:
    """
    Run what-if scenarios on data.
    
    Args:
        df: DataFrame
        value_column: Column to apply scenarios to
        scenarios: Dictionary like {'optimistic': 1.15, 'pessimistic': 0.85}
        
    Returns:
        Scenario results
    """
    base_value = df[value_column].sum()
    
    results = {
        "base_case": float(base_value),
        "scenarios": {},
    }
    
    for scenario_name, multiplier in scenarios.items():
        scenario_value = base_value * multiplier
        change = scenario_value - base_value
        change_pct = (change / base_value) * 100 if base_value != 0 else 0
        
        results["scenarios"][scenario_name] = {
            "multiplier": float(multiplier),
            "projected_value": float(scenario_value),
            "change_from_base": float(change),
            "change_percentage": float(change_pct),
        }
    
    return results


def get_summary_statistics(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Get comprehensive summary statistics for a column."""
    values = df[column].dropna()
    
    if values.empty:
        return {"error": "No valid values"}
    
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std_dev": float(values.std()),
        "variance": float(values.var()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "skewness": float(values.skew()),
        "kurtosis": float(values.kurtosis()),
    }
