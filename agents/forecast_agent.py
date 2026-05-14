"""
Forecast Agent: Performs time-series forecasting and predictions.

Uses Prophet for seasonal forecasting and supports scenario analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_core.tools import tool
from tools import viz_tools
from utils import prompts

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


@tool
def forecast_with_prophet(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = 12,
    include_history: bool = True,
) -> Dict[str, Any]:
    """
    Forecast time-series using Facebook Prophet (handles seasonality, trends, holidays).
    
    Ideal for:
    - Coffee outturn forecasting (strong seasonality)
    - Hotel revenue forecasting (seasonal patterns)
    - Tenant payment forecasting
    
    Args:
        df: DataFrame with time-series
        date_column: Date column (must be datetime)
        value_column: Value column to forecast
        periods: Number of periods to forecast (e.g., 12 for 12 months)
        include_history: Whether to include historical data in output
        
    Returns:
        Forecast DataFrame with point estimates and confidence intervals
    """
    if Prophet is None:
        return {"status": "error", "error": "Prophet not installed"}
    
    try:
        # Prepare data for Prophet
        df_prophet = df[[date_column, value_column]].dropna().copy()
        df_prophet.columns = ["ds", "y"]
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
        df_prophet = df_prophet.dropna()
        
        if len(df_prophet) < 10:
            return {"status": "error", "error": "Need at least 10 data points for forecasting"}
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
        )
        
        model.fit(df_prophet)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Prepare output
        forecast_output = forecast[[
            "ds", "yhat", "yhat_lower", "yhat_upper"
        ]].copy()
        forecast_output.columns = ["date", "forecast", "lower_ci", "upper_ci"]
        
        if not include_history:
            forecast_output = forecast_output[len(df_prophet):]
        
        # Calculate metrics
        last_value = df_prophet["y"].iloc[-1]
        avg_forecast = forecast_output["forecast"].iloc[-periods:].mean()
        trend = ((avg_forecast - last_value) / last_value * 100) if last_value != 0 else 0
        
        return {
            "status": "success",
            "forecast": forecast_output.to_dict(orient="records"),
            "metrics": {
                "last_historical_value": float(last_value),
                "average_forecast": float(avg_forecast),
                "trend_percentage": float(trend),
                "confidence_interval_width": 0.95,
            },
            "periods": periods,
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def run_scenario_forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    scenarios: Dict[str, float],
    periods: int = 12,
) -> Dict[str, Any]:
    """
    Run what-if scenario forecasts (e.g., fertilizer cost +15%).
    
    Args:
        df: DataFrame with historical data
        date_column: Date column
        value_column: Value column
        scenarios: Dictionary like {'optimistic': 1.2, 'pessimistic': 0.8}
        periods: Forecast periods
        
    Returns:
        Scenario forecast results
    """
    try:
        # Get base forecast
        base_result = forecast_with_prophet(df, date_column, value_column, periods, include_history=False)
        
        if base_result["status"] != "success":
            return base_result
        
        base_forecast = base_result["forecast"]
        base_avg = base_result["metrics"]["average_forecast"]
        
        # Apply scenarios
        scenario_results = {
            "base_case": float(base_avg),
            "scenarios": {},
        }
        
        for scenario_name, multiplier in scenarios.items():
            scenario_values = [f["forecast"] * multiplier for f in base_forecast]
            scenario_avg = np.mean(scenario_values)
            change = scenario_avg - base_avg
            change_pct = (change / base_avg * 100) if base_avg != 0 else 0
            
            scenario_results["scenarios"][scenario_name] = {
                "average_value": float(scenario_avg),
                "multiplier": float(multiplier),
                "change_from_base": float(change),
                "change_percentage": float(change_pct),
            }
        
        return {
            "status": "success",
            "scenario_analysis": scenario_results,
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def forecast_with_exponential_smoothing(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = 12,
) -> Dict[str, Any]:
    """
    Forecast using exponential smoothing (simpler alternative to Prophet).
    
    Args:
        df: DataFrame with time-series
        date_column: Date column
        value_column: Value column
        periods: Forecast periods
        
    Returns:
        Forecast results
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        df_ts = df[[date_column, value_column]].dropna().copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column], errors="coerce")
        df_ts = df_ts.sort_values(date_column)
        
        values = df_ts[value_column].values
        
        if len(values) < 10:
            return {"status": "error", "error": "Need at least 10 data points"}
        
        # Fit model
        model = ExponentialSmoothing(
            values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        
        # Forecast
        forecast = fit.forecast(steps=periods)
        
        # Simple CI (using fitted values std error)
        residuals = values - fit.fittedvalues
        std_error = np.std(residuals)
        ci_lower = forecast - 1.96 * std_error
        ci_upper = forecast + 1.96 * std_error
        
        # Prepare output
        last_date = pd.to_datetime(df_ts[date_column].iloc[-1])
        forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
        
        forecast_output = []
        for date, value, lower, upper in zip(forecast_dates, forecast, ci_lower, ci_upper):
            forecast_output.append({
                "date": str(date.date()),
                "forecast": float(value),
                "lower_ci": float(lower),
                "upper_ci": float(upper),
            })
        
        return {
            "status": "success",
            "forecast": forecast_output,
            "method": "Exponential Smoothing",
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def create_forecast_chart(
    historical_df: pd.DataFrame,
    date_column: str,
    value_column: str,
    forecast_df: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create interactive forecast visualization with confidence intervals.
    
    Args:
        historical_df: Historical data
        date_column: Date column
        value_column: Value column
        forecast_df: Forecast results dictionary
        
    Returns:
        Plotly figure as JSON
    """
    try:
        # Extract historical values
        hist_sorted = historical_df.sort_values(date_column).copy()
        hist_dates = hist_sorted[date_column].tolist()
        hist_values = hist_sorted[value_column].tolist()
        
        # Extract forecast values
        forecast_records = forecast_df.get("forecast", [])
        forecast_dates = [rec["date"] if isinstance(rec.get("date"), str) else rec["date"] 
                         for rec in forecast_records]
        forecast_values = [rec["forecast"] for rec in forecast_records]
        forecast_lower = [rec["lower_ci"] for rec in forecast_records]
        forecast_upper = [rec["upper_ci"] for rec in forecast_records]
        
        # Create chart
        fig = viz_tools.create_forecast_chart(
            historical_dates=hist_dates,
            historical_values=hist_values,
            forecast_dates=forecast_dates,
            forecast_values=forecast_values,
            forecast_lower_ci=forecast_lower,
            forecast_upper_ci=forecast_upper,
            title="Time-Series Forecast with Confidence Intervals",
        )
        
        return {
            "status": "success",
            "chart": fig.to_json(),
            "chart_type": "forecast",
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def create_scenario_comparison_chart(scenario_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Create scenario comparison visualization.
    
    Args:
        scenario_data: Dictionary of {scenario_name: value}
        
    Returns:
        Plotly figure as JSON
    """
    try:
        fig = viz_tools.create_scenario_comparison_chart(
            scenario_data,
            title="Scenario Comparison Analysis",
        )
        
        return {
            "status": "success",
            "chart": fig.to_json(),
            "chart_type": "scenario_comparison",
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def get_forecast_summary(forecast_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate human-readable forecast summary.
    
    Args:
        forecast_results: Forecast dictionary from forecast_with_prophet
        
    Returns:
        Summary text and key metrics
    """
    try:
        metrics = forecast_results.get("metrics", {})
        last_value = metrics.get("last_historical_value", 0)
        avg_forecast = metrics.get("average_forecast", 0)
        trend = metrics.get("trend_percentage", 0)
        
        summary_lines = [
            "📊 Forecast Summary",
            f"Last historical value: {last_value:,.2f}",
            f"Average forecast: {avg_forecast:,.2f}",
            f"Trend: {trend:+.1f}%",
        ]
        
        if trend > 10:
            summary_lines.append("📈 Strong upward trend")
        elif trend > 0:
            summary_lines.append("📈 Slight upward trend")
        elif trend < -10:
            summary_lines.append("📉 Strong downward trend")
        elif trend < 0:
            summary_lines.append("📉 Slight downward trend")
        else:
            summary_lines.append("➡️ Stable trend")
        
        return {
            "status": "success",
            "summary": "\n".join(summary_lines),
            "metrics": metrics,
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_forecast_agent_system_prompt() -> str:
    """Create system prompt for forecast agent."""
    return prompts.SYSTEM_PROMPTS["forecast_agent"] + "\n\n" + prompts.CONTEXT_INJECTION


def get_forecast_agent_tools() -> List:
    """Return list of tools available to forecast agent."""
    return [
        forecast_with_prophet,
        run_scenario_forecast,
        forecast_with_exponential_smoothing,
        create_forecast_chart,
        create_scenario_comparison_chart,
        get_forecast_summary,
    ]
