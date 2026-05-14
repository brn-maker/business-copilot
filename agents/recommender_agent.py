"""
Recommender Agent: Generates business intelligence and actionable recommendations.

Synthesizes insights from all other agents to provide strategic guidance.
"""

from typing import Any, Dict, List, Optional
import json
from langchain_core.tools import tool
from utils import prompts


@tool
def generate_yield_recommendations(
    analysis_results: Dict[str, Any],
    forecast_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate recommendations for improving coffee yield and outturn.
    
    Based on:
    - Fertilizer regression analysis
    - Outturn trends
    - Grading performance
    
    Args:
        analysis_results: Statistical analysis outputs
        forecast_results: Forecast predictions
        
    Returns:
        Actionable recommendations for yield improvement
    """
    recommendations = {
        "category": "Yield Optimization",
        "priority": "high",
        "recommendations": [],
    }
    
    try:
        # Check fertilizer impact
        if "regression_results" in analysis_results:
            reg_results = analysis_results["regression_results"]
            coefficients = reg_results.get("coefficients", {})
            
            fert_coef = coefficients.get("fertilizer", {})
            if fert_coef.get("significant"):
                coef_value = fert_coef.get("coefficient", 0)
                if coef_value > 0:
                    recommendations["recommendations"].append({
                        "action": "Increase strategic fertilizer application",
                        "rationale": f"Strong positive correlation (coef={coef_value:.3f}) between fertilizer and outturn",
                        "expected_impact": "2-5% outturn improvement",
                        "timeframe": "Next growing season",
                        "effort": "Medium",
                    })
                else:
                    recommendations["recommendations"].append({
                        "action": "Review fertilizer application timing and method",
                        "rationale": f"Negative fertilizer coefficient suggests optimization opportunity",
                        "expected_impact": "Better ROI on fertilizer spend",
                        "timeframe": "Immediate",
                        "effort": "Low",
                    })
        
        # Check outturn trend
        if "growth_analysis" in analysis_results:
            growth = analysis_results["growth_analysis"]
            avg_growth = growth.get("average_growth_rate", 0)
            
            if avg_growth < -5:
                recommendations["recommendations"].append({
                    "action": "Investigate declining outturn quality",
                    "rationale": f"Average outturn declining at {avg_growth:.1f}% per period",
                    "expected_impact": "Identify and address quality issues",
                    "timeframe": "This month",
                    "effort": "Medium",
                })
            elif avg_growth > 5:
                recommendations["recommendations"].append({
                    "action": "Document and replicate current best practices",
                    "rationale": f"Strong outturn growth trend at {avg_growth:.1f}% per period",
                    "expected_impact": "Sustain momentum and identify success factors",
                    "timeframe": "Ongoing",
                    "effort": "Low",
                })
        
        recommendations["summary"] = f"Generated {len(recommendations['recommendations'])} actionable recommendations"
        return {"status": "success", "recommendations": recommendations}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def generate_revenue_recommendations(
    analysis_results: Dict[str, Any],
    forecast_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate recommendations for revenue growth and margin improvement.
    
    Based on:
    - Grading vs price analysis
    - Value addition opportunity
    - Sales trends
    
    Args:
        analysis_results: Analysis outputs
        forecast_results: Forecast outputs
        
    Returns:
        Revenue improvement recommendations
    """
    recommendations = {
        "category": "Revenue Growth",
        "priority": "high",
        "recommendations": [],
    }
    
    try:
        # Value addition opportunity
        if "correlation_results" in analysis_results:
            correlations = analysis_results["correlation_results"].get("strong_correlations", [])
            
            for corr in correlations:
                if "grading" in corr.get("variable_1", "").lower() or \
                   "grading" in corr.get("variable_2", "").lower():
                    if "value" in corr.get("variable_1", "").lower() or \
                       "value" in corr.get("variable_2", "").lower():
                        corr_value = corr.get("correlation", 0)
                        if corr_value > 0.7:
                            recommendations["recommendations"].append({
                                "action": "Expand value addition operations",
                                "rationale": f"Strong correlation (r={corr_value:.2f}) between grading and value addition sales",
                                "expected_impact": "20-40% margin increase vs raw coffee sales",
                                "timeframe": "Next quarter",
                                "effort": "High",
                            })
        
        # Forecast-based recommendations
        if "forecast" in forecast_results:
            forecast = forecast_results.get("forecast", [])
            if forecast:
                last_forecast = forecast[-1] if isinstance(forecast, list) else None
                if last_forecast:
                    upper_ci = last_forecast.get("upper_ci", 0)
                    lower_ci = last_forecast.get("lower_ci", 0)
                    
                    if upper_ci > lower_ci * 1.2:
                        recommendations["recommendations"].append({
                            "action": "Plan for demand variability",
                            "rationale": f"Forecast shows wide confidence interval (CI width: {(upper_ci-lower_ci):.1%})",
                            "expected_impact": "Better inventory and pricing strategy",
                            "timeframe": "Before next season",
                            "effort": "Medium",
                        })
        
        recommendations["summary"] = f"Generated {len(recommendations['recommendations'])} revenue recommendations"
        return {"status": "success", "recommendations": recommendations}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def generate_risk_mitigation_recommendations(
    analysis_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate risk mitigation recommendations.
    
    Based on:
    - Anomalies detected
    - Volatility analysis
    - Payment consistency
    
    Args:
        analysis_results: Analysis results including anomalies
        
    Returns:
        Risk mitigation strategies
    """
    recommendations = {
        "category": "Risk Mitigation",
        "priority": "medium",
        "recommendations": [],
    }
    
    try:
        # Check for anomalies
        if "anomalies" in analysis_results:
            anomalies = analysis_results["anomalies"]
            anomaly_pct = anomalies.get("anomaly_percentage", 0)
            
            if anomaly_pct > 5:
                recommendations["recommendations"].append({
                    "action": "Review and address data quality issues",
                    "rationale": f"{anomaly_pct:.1f}% anomalies detected in dataset",
                    "expected_impact": "Improved data reliability for decisions",
                    "timeframe": "This week",
                    "effort": "Low",
                })
        
        # Volatility warning
        if "statistics" in analysis_results:
            stats = analysis_results["statistics"]
            if "numeric_summary" in stats:
                for col, col_stats in stats["numeric_summary"].items():
                    mean = col_stats.get("mean", 1)
                    std = col_stats.get("std", 0)
                    
                    if mean != 0:
                        cv = std / abs(mean)  # Coefficient of variation
                        if cv > 0.5:
                            recommendations["recommendations"].append({
                                "action": f"Create contingency plans for {col} volatility",
                                "rationale": f"High variability (CV={cv:.2f}) in {col}",
                                "expected_impact": "Better prepared for market fluctuations",
                                "timeframe": "Next planning cycle",
                                "effort": "Medium",
                            })
        
        recommendations["summary"] = f"Generated {len(recommendations['recommendations'])} risk mitigation strategies"
        return {"status": "success", "recommendations": recommendations}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def synthesize_executive_summary(
    analysis_summary: str,
    forecast_summary: str,
    yield_recommendations: List[Dict[str, str]],
    revenue_recommendations: List[Dict[str, str]],
    risk_recommendations: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Create executive summary combining all insights and recommendations.
    
    Args:
        analysis_summary: Summary from analysis agent
        forecast_summary: Summary from forecast agent
        yield_recommendations: Yield improvement recommendations
        revenue_recommendations: Revenue growth recommendations
        risk_recommendations: Risk mitigation recommendations
        
    Returns:
        Formatted executive summary
    """
    try:
        summary_parts = [
            "# Executive Summary: Agricultural Business Intelligence Report",
            "",
            "## Key Insights",
            analysis_summary or "Analysis pending",
            "",
            "## Forecasts & Predictions",
            forecast_summary or "Forecasts pending",
            "",
            "## Strategic Recommendations",
            "",
            "### Yield Optimization Priority",
            f"**Recommendations:** {len(yield_recommendations or [])} actions identified",
            "",
            "### Revenue Growth Opportunities",
            f"**Recommendations:** {len(revenue_recommendations or [])} actions identified",
            "",
            "### Risk Mitigation Strategies",
            f"**Recommendations:** {len(risk_recommendations or [])} actions identified",
            "",
            "## Next Steps",
            "1. Review prioritized recommendations",
            "2. Assign owners and timelines",
            "3. Track implementation progress",
            "4. Monitor forecast accuracy",
        ]
        
        return {
            "status": "success",
            "executive_summary": "\n".join(summary_parts),
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def rank_recommendations_by_impact(
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Rank recommendations by expected business impact and effort.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Prioritized recommendations with impact scores
    """
    try:
        effort_scores = {"Low": 1, "Medium": 2, "High": 3}
        impact_scores = {"2-5%": 2, "5-15%": 5, "15-40%": 15, "20-40%": 20}
        
        ranked = []
        
        for rec in recommendations:
            effort = effort_scores.get(rec.get("effort", "Medium"), 2)
            impact_str = rec.get("expected_impact", "Unknown")
            
            impact = 0
            for impact_desc, score in impact_scores.items():
                if impact_desc in impact_str:
                    impact = score
                    break
            
            score = (impact / effort) if effort > 0 else impact
            
            ranked.append({
                **rec,
                "effort_score": effort,
                "impact_score": impact,
                "roi_score": score,
            })
        
        # Sort by ROI score descending
        ranked.sort(key=lambda x: x["roi_score"], reverse=True)
        
        return {
            "status": "success",
            "ranked_recommendations": ranked,
            "top_recommendation": ranked[0] if ranked else None,
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_recommender_system_prompt() -> str:
    """Create system prompt for recommender agent."""
    return prompts.SYSTEM_PROMPTS["recommender_agent"] + "\n\n" + prompts.CONTEXT_INJECTION


def get_recommender_tools() -> List:
    """Return list of tools available to recommender agent."""
    return [
        generate_yield_recommendations,
        generate_revenue_recommendations,
        generate_risk_mitigation_recommendations,
        synthesize_executive_summary,
        rank_recommendations_by_impact,
    ]
