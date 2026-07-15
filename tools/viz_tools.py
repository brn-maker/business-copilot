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
    Create a crypto/forex TradingView-style time-series chart.
    Dark theme, gradient area fill, range slider, trend-direction coloring.
    """
    df_sorted = df.sort_values(date_column).copy()
    df_sorted[date_column] = pd.to_datetime(df_sorted[date_column], errors="coerce")
    df_sorted = df_sorted.dropna(subset=[date_column, value_column])

    values = df_sorted[value_column]
    first_val = values.iloc[0] if len(values) > 0 else 0
    last_val  = values.iloc[-1] if len(values) > 0 else 0
    is_up = last_val >= first_val

    line_color   = "#00c896" if is_up else "#ff4d6d"   # green up / red down
    fill_color   = "rgba(0,200,150,0.12)" if is_up else "rgba(255,77,109,0.12)"
    trend_color  = "rgba(255,200,0,0.85)"               # gold trend line

    fig = go.Figure()

    # --- Area fill (below the line) ---
    fig.add_trace(go.Scatter(
        x=df_sorted[date_column],
        y=values,
        mode="lines",
        name=value_column,
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.2f}<extra></extra>",
    ))

    # --- Trend line (linear regression) ---
    if include_trend and len(df_sorted) > 2:
        x_numeric = np.arange(len(df_sorted))
        try:
            z = np.polyfit(x_numeric, values.values, 1)
            p = np.poly1d(z)
            trend_y = p(x_numeric)

            fig.add_trace(go.Scatter(
                x=df_sorted[date_column],
                y=trend_y,
                mode="lines",
                name="Trend",
                line=dict(color=trend_color, width=1.5, dash="dot"),
                hoverinfo="skip",
            ))
        except Exception:
            pass

    # --- High / Low annotations ---
    if len(values) > 0:
        max_idx = values.idxmax()
        min_idx = values.idxmin()
        fig.add_annotation(
            x=df_sorted.loc[max_idx, date_column], y=values[max_idx],
            text=f"▲ {values[max_idx]:,.0f}",
            showarrow=True, arrowhead=0, arrowcolor=line_color,
            font=dict(color=line_color, size=11), bgcolor="rgba(0,0,0,0.5)",
            bordercolor=line_color, borderwidth=1, arrowwidth=1,
            ay=-30,
        )
        fig.add_annotation(
            x=df_sorted.loc[min_idx, date_column], y=values[min_idx],
            text=f"▼ {values[min_idx]:,.0f}",
            showarrow=True, arrowhead=0, arrowcolor="#ff4d6d",
            font=dict(color="#ff4d6d", size=11), bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#ff4d6d", borderwidth=1, arrowwidth=1,
            ay=30,
        )

    # --- Layout: TradingView dark style ---
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=15), x=0.01),
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(
            font=dict(color="#aaaaaa"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=None,
            color="#555e70",
            gridcolor="#1e2433",
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=True, thickness=0.04, bgcolor="#1a1f2e"),
            rangeselector=dict(
                bgcolor="#1a1f2e",
                activecolor="#2a3150",
                font=dict(color="#aaaaaa"),
                buttons=[
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
        yaxis=dict(
            title=None,
            color="#555e70",
            gridcolor="#1e2433",
            showgrid=True,
            zeroline=False,
            side="right",
        ),
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


def create_category_distribution_chart(
    df: pd.DataFrame,
    category_column: str,
    chart_type: str = "pie",
    title: str = "Distribution",
) -> go.Figure:
    """
    Create a pie or bar chart for categorical distribution (counts).
    """
    counts = df[category_column].value_counts().reset_index()
    counts.columns = [category_column, 'Count']
    
    if len(counts) > 15:
        counts = counts.head(15)
        title += " (Top 15)"
        
    if chart_type.lower() == "pie":
        fig = px.pie(counts, names=category_column, values='Count', title=title)
    else:
        fig = px.bar(counts, x=category_column, y='Count', title=title, color=category_column)
        
    fig.update_layout(
        template="plotly_white",
        height=500,
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
