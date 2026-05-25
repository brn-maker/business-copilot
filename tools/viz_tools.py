"""
Visualization Tools: Create interactive Plotly charts for agricultural analytics.

Includes time-series, regression, distribution, and scenario visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple


def create_timeseries_chart(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "Time Series",
    include_trend: bool = True,
) -> go.Figure:
    """
    Create interactive time-series chart with trend line.
    
    Args:
        df: DataFrame with time-series data
        date_column: Name of date column
        value_column: Name of value column
        title: Chart title
        include_trend: Whether to include trend line
        
    Returns:
        Plotly Figure
    """
    df_sorted = df.sort_values(date_column).copy()
    df_sorted[date_column] = pd.to_datetime(df_sorted[date_column])
    
    fig = go.Figure()
    
    # Main line (spline = smooth curve)
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=df_sorted[value_column],
        mode="lines+markers",
        name=value_column,
        line=dict(color="#1f77b4", width=3, shape="spline", smoothing=1.3),
        marker=dict(size=5, color="#1f77b4"),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.08)",
    ))
    
    # Trend line (linear regression)
    if include_trend and len(df_sorted) > 2:
        x_numeric = np.arange(len(df_sorted))
        z = np.polyfit(x_numeric, df_sorted[value_column].dropna(), 1)
        p = np.poly1d(z)
        trend_y = p(x_numeric)
        
        fig.add_trace(go.Scatter(
            x=df_sorted[date_column],
            y=trend_y,
            mode="lines",
            name="Trend",
            line=dict(color="#ff7f0e", width=2, dash="dash", shape="spline", smoothing=1.3),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title=value_column,
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_regression_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Regression Analysis",
) -> go.Figure:
    """
    Create scatter plot with regression line and confidence interval.
    
    Args:
        df: DataFrame
        x_column: X-axis column (independent variable)
        y_column: Y-axis column (dependent variable)
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    df_clean = df[[x_column, y_column]].dropna()
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df_clean[x_column],
        y=df_clean[y_column],
        mode="markers",
        name="Data",
        marker=dict(size=6, color="#1f77b4", opacity=0.7),
    ))
    
    # Regression line with confidence interval
    if len(df_clean) > 2:
        x_vals = df_clean[x_column].values
        y_vals = df_clean[y_column].values
        
        # Linear regression
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        
        x_sorted = np.sort(x_vals)
        y_pred = p(x_sorted)
        
        # Confidence interval
        residuals = y_vals - p(x_vals)
        std_residuals = np.std(residuals)
        
        ci_upper = y_pred + 1.96 * std_residuals
        ci_lower = y_pred - 1.96 * std_residuals
        
        # Regression line
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_pred,
            mode="lines",
            name="Regression",
            line=dict(color="#ff7f0e", width=2),
        ))
        
        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=ci_upper,
            fill=None,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ))
        
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=ci_lower,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="95% CI",
            fillcolor="rgba(255,127,14,0.2)",
        ))
        
        # Calculate correlation
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        r_squared = corr ** 2
        
        fig.add_annotation(
            text=f"R² = {r_squared:.3f}",
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        hovermode="closest",
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame
        columns: Specific columns to include (default: all numeric)
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    if columns:
        numeric_df = df[[col for col in columns if col in df.columns]]
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_distribution_chart(
    df: pd.DataFrame,
    column: str,
    title: str = "Distribution",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram with statistics.
    
    Args:
        df: DataFrame
        column: Column to plot
        title: Chart title
        nbins: Number of bins
        
    Returns:
        Plotly Figure
    """
    values = df[column].dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=nbins,
        name="Distribution",
        marker_color="#1f77b4",
        opacity=0.7,
    ))
    
    # Add mean and median lines
    mean_val = values.mean()
    median_val = values.median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", name=f"Mean: {mean_val:.2f}")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green", name=f"Median: {median_val:.2f}")
    
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Frequency",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_forecast_chart(
    historical_dates: List,
    historical_values: List,
    forecast_dates: List,
    forecast_values: List,
    forecast_lower_ci: List,
    forecast_upper_ci: List,
    title: str = "Forecast",
) -> go.Figure:
    """
    Create forecast visualization with confidence intervals.
    
    Args:
        historical_dates: Historical date points
        historical_values: Historical values
        forecast_dates: Forecast date points
        forecast_values: Forecast values (mean)
        forecast_lower_ci: Forecast lower confidence bound
        forecast_upper_ci: Forecast upper confidence bound
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_values,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=4),
    ))
    
    # Upper CI
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_upper_ci,
        fill=None,
        mode="lines",
        line_color="rgba(0,0,0,0)",
        showlegend=False,
    ))
    
    # Lower CI
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_lower_ci,
        fill="tonexty",
        mode="lines",
        line_color="rgba(0,0,0,0)",
        name="95% Confidence Interval",
        fillcolor="rgba(255,127,14,0.2)",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_comparison_chart(
    data_dict: Dict[str, List[float]],
    labels: List[str],
    title: str = "Comparison",
    chart_type: str = "bar",
) -> go.Figure:
    """
    Create comparison chart (bar or box).
    
    Args:
        data_dict: Dictionary of {label: values}
        labels: X-axis labels
        title: Chart title
        chart_type: 'bar' or 'box'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    if chart_type == "box":
        for i, (label, values) in enumerate(data_dict.items()):
            fig.add_trace(go.Box(
                y=values,
                name=label,
                marker_color=colors[i % len(colors)],
            ))
    else:  # bar
        for i, (label, values) in enumerate(data_dict.items()):
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                name=label,
                marker_color=colors[i % len(colors)],
            ))
    
    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_scenario_comparison_chart(
    scenarios: Dict[str, float],
    title: str = "Scenario Analysis",
) -> go.Figure:
    """
    Create scenario comparison chart.
    
    Args:
        scenarios: Dictionary of {scenario_name: value}
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    scenarios_sorted = sorted(scenarios.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in scenarios_sorted]
    values = [s[1] for s in scenarios_sorted]
    
    # Color based on value (green for positive, red for negative relative to first)
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]
    
    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Value",
        hovermode="x",
        height=500,
        template="plotly_white",
        showlegend=False,
    )
    
    return fig


def create_dashboard_summary(
    charts: Dict[str, go.Figure],
    title: str = "Analytics Dashboard",
) -> go.Figure:
    """
    Create multi-chart dashboard.
    
    Args:
        charts: Dictionary of {name: Figure}
        title: Dashboard title
        
    Returns:
        Plotly Figure with subplots
    """
    n_charts = len(charts)
    rows = (n_charts + 1) // 2
    
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=list(charts.keys()),
    )
    
    for idx, (name, chart) in enumerate(charts.items()):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        for trace in chart.data:
            fig.add_trace(trace, row=row, col=col)
    
    fig.update_layout(
        title_text=title,
        height=300 * rows,
        showlegend=True,
    )
    
    return fig
