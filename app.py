"""
Main Streamlit Application: Coffee/Agriculture Business Intelligence System

Provides web interface for:
- Uploading Excel files with agricultural data
- Natural language queries via agentic chat
- Statistical analysis and forecasting
- Interactive visualizations and business recommendations
"""

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file (local dev only — no-op on Streamlit Cloud)
except ImportError:
    pass  # python-dotenv not available on Streamlit Cloud; secrets come from st.secrets

import streamlit as st
import pandas as pd
import os

# Streamlit Cloud: fall back to st.secrets if env var not set via .env
if not os.getenv("OPENROUTER_API_KEY"):
    try:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass  # Key not found — app will show the warning banner
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import model_router, DataProcessor, process_uploaded_file
from agents import create_supervisor
from tools import data_tools, analysis_tools, viz_tools


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="☕ Coffee Analytics Copilot",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .main-header {
            font-size: 2.5em;
            font-weight: bold;
            color: #8B4513;
            margin-bottom: 0.3em;
        }
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            color: #6F4E37;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            border-bottom: 2px solid #DEB887;
            padding-bottom: 0.3em;
        }
        .info-box {
            background-color: #f0f0f0;
            border-left: 4px solid #8B4513;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .error-box {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {}
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "supervisor" not in st.session_state:
        try:
            st.session_state.supervisor = create_supervisor()
        except Exception as e:
            st.warning(f"Could not initialize supervisor: {str(e)}")
            st.session_state.supervisor = None
    
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    
    if "model_router" not in st.session_state:
        try:
            st.session_state.model_router = model_router.create_model_router()
        except Exception as e:
            st.warning(f"⚠️ OpenRouter API not configured. Set OPENROUTER_API_KEY environment variable.")
            st.session_state.model_router = None


initialize_session_state()


# ============================================================================
# SIDEBAR: FILE UPLOAD & CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render sidebar with file upload and configuration."""
    st.sidebar.markdown("# ⚙️ Configuration")
    
    # File upload section
    st.sidebar.markdown("## 📁 Data Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload Excel files with agricultural data",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="Supported sheets: Tenant Payments, Coffee Outturn, Grading, Hotel Sales, Value Addition Sales, Fertilizer Sales",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    cleaned_data, quality_report = process_uploaded_file(
                        uploaded_file,
                        uploaded_file.name,
                    )
                    
                    if cleaned_data:
                        st.session_state.uploaded_files[uploaded_file.name] = uploaded_file
                        st.session_state.processed_data[uploaded_file.name] = cleaned_data
                        st.sidebar.success(f"✅ {uploaded_file.name}")
                    else:
                        st.sidebar.error(f"❌ {uploaded_file.name}")
    
    # Show uploaded files
    if st.session_state.processed_data:
        st.sidebar.markdown("### Loaded Datasets")
        for filename, data in st.session_state.processed_data.items():
            sheet_count = len(data)
            st.sidebar.text(f"📊 {filename} ({sheet_count} sheets)")
    
    # Model configuration
    st.sidebar.markdown("## 🤖 Model Settings")
    
    # Quick actions
    st.sidebar.markdown("## 🎯 Quick Actions")
    
    if st.sidebar.button("🔄 Clear All Data", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.processed_data = {}
        st.session_state.chat_history = []
        st.session_state.analysis_cache = {}
        st.session_state.uploader_key += 1  # Reset the file uploader widget
        st.rerun()
    
    # Info section
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.info(
        "**Coffee Analytics Copilot**\n\n"
        "Agentic AI system for agricultural analytics:\n"
        "- Statistical analysis\n"
        "- Time-series forecasting\n"
        "- Business recommendations\n\n"
        "Supports: Coffee, hotel, fertilizer data"
    )


# ============================================================================
# MAIN CONTENT: CHAT INTERFACE & RESULTS
# ============================================================================

def render_data_summary():
    """Render summary of uploaded data."""
    if not st.session_state.processed_data:
        st.info("📁 Upload Excel files in the sidebar to get started")
        
        # Show example usage
        st.markdown("""
        ### 🚀 Quick Start
        
        1. **Upload your data** - Excel files with these sheet types:
           - Tenant Payments
           - Coffee Outturn & Yield
           - Grading / Quality
           - Hotel Sales
           - Value Addition Sales
           - Fertilizer Sales
        
        2. **Ask questions** using natural language:
           - "Show correlation between fertilizer and outturn"
           - "Forecast next quarter coffee revenue"
           - "Give me recommendations to improve yield"
           - "What are the trends in grading?"
        
        3. **View results** - Charts, statistics, and insights
        
        ### 📊 Sample Analysis
        Try these after uploading data:
        - Statistical analysis (correlations, regression)
        - Time-series forecasting with confidence intervals
        - Scenario analysis ("what-if" planning)
        - Anomaly detection
        - Business recommendations
        """)
        return
    
    st.markdown("### 📊 Loaded Datasets")
    
    for filename, data_dict in st.session_state.processed_data.items():
        with st.expander(f"📁 {filename}", expanded=False):
            cols = st.columns(len(data_dict))
            
            for col, (sheet_type, df) in zip(cols, data_dict.items()):
                with col:
                    st.metric(sheet_type, f"{len(df)} rows")
                    st.caption(f"{len(df.columns)} columns")


def render_chat_interface():
    """Render chat interface for user queries."""
    st.markdown("### 💬 Analytics Assistant Chat")
    
    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your data...")
    
    if user_input:
        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Process query
        with st.spinner("🤔 Thinking..."):
            response = process_user_query(user_input)
        
        # Add response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
        st.rerun()


def process_user_query(user_query: str) -> str:
    """
    Process user query and return response.
    
    This is where the agentic system gets invoked.
    """
    if not st.session_state.processed_data:
        return "Please upload data first using the sidebar."
    
    # Extract data for analysis
    combined_data = {}
    for filename, sheets_dict in st.session_state.processed_data.items():
        for sheet_type, df in sheets_dict.items():
            if sheet_type not in combined_data:
                combined_data[sheet_type] = df
            # Could merge multiple files here if needed
    
    # Check what kind of analysis is requested
    query_lower = user_query.lower()
    
    response_lines = []
    
    try:
        # CORRELATION ANALYSIS
        if any(word in query_lower for word in ["correlation", "relate", "relationship", "between"]):
            response_lines.append("**📈 Correlation Analysis**\n")
            
            numeric_cols = combined_data[list(combined_data.keys())[0]].select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                df = pd.concat(combined_data.values(), axis=1)
                corr_result = analysis_tools.correlation_analysis(df, numeric_cols)
                
                if "strong_correlations" in corr_result:
                    correlations = corr_result["strong_correlations"]
                    if correlations:
                        for corr in correlations[:5]:
                            response_lines.append(
                                f"- **{corr['variable_1']}** ↔️ **{corr['variable_2']}**: "
                                f"r = {corr['correlation']:.3f} ({corr['strength']})"
                            )
                    else:
                        response_lines.append("No strong correlations found (|r| > 0.6)")
        
        # FORECASTING
        elif any(word in query_lower for word in ["forecast", "predict", "next", "future", "trend"]):
            response_lines.append("**📊 Forecasting Analysis**\n")
            
            # Find time-series data
            for sheet_type, df in combined_data.items():
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if date_cols and numeric_cols:
                    response_lines.append(f"📈 {sheet_type}:")
                    response_lines.append(f"- Date range: {df[date_cols[0]].min()} to {df[date_cols[0]].max()}")
                    response_lines.append(f"- Data points: {len(df)}")
                    response_lines.append(f"- Ready for forecasting: {numeric_cols[0]}")
        
        # STATISTICS / SUMMARY
        elif any(word in query_lower for word in ["summary", "stats", "statistics", "overview", "describe"]):
            response_lines.append("**📊 Data Summary**\n")
            
            for sheet_type, df in combined_data.items():
                response_lines.append(f"**{sheet_type}**")
                response_lines.append(f"- Rows: {len(df)}")
                response_lines.append(f"- Columns: {len(df.columns)}")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    response_lines.append(f"- Numeric: {', '.join(numeric_cols[:3])}")
        
        # RECOMMENDATIONS
        elif any(word in query_lower for word in ["recommend", "suggest", "improve", "action", "strategy"]):
            response_lines.append("**💡 Business Recommendations**\n")
            response_lines.append("Based on available data, here are key areas to explore:")
            response_lines.append("- Correlation analysis between inputs (fertilizer) and outputs (outturn)")
            response_lines.append("- Time-series trends in quality grades and pricing")
            response_lines.append("- Seasonal patterns in coffee production and hotel occupancy")
            response_lines.append("- Revenue opportunities through value addition")
        
        # GENERAL QUERY
        else:
            response_lines.append("**📊 Analysis Results**\n")
            
            df = pd.concat(combined_data.values(), axis=1)
            
            response_lines.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            response_lines.append(f"Date range: {df.iloc[:, 0].min()} to {df.iloc[:, -1].max()}")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                response_lines.append(f"\nNumeric variables: {', '.join(numeric_cols[:5])}")
                
                for col in numeric_cols[:2]:
                    mean = df[col].mean()
                    std = df[col].std()
                    response_lines.append(f"- **{col}**: mean={mean:.2f}, std={std:.2f}")
        
        response = "\n".join(response_lines)
    
    except Exception as e:
        response = f"⚠️ Analysis error: {str(e)}\n\nTry asking about specific data or correlations."
    
    return response


def render_analysis_section():
    """Render quick analysis section."""
    st.markdown("### 🔍 Quick Analysis")
    
    if not st.session_state.processed_data:
        st.info("Upload data to enable analysis")
        return
    
    # Get combined data
    combined_data = {}
    for filename, sheets_dict in st.session_state.processed_data.items():
        for sheet_type, df in sheets_dict.items():
            if sheet_type not in combined_data:
                combined_data[sheet_type] = df
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Correlations", use_container_width=True):
            # Show correlation heatmap
            if combined_data:
                df = pd.concat(combined_data.values(), axis=1)
                numeric_df = df.select_dtypes(include=['number'])
                
                if len(numeric_df.columns) >= 2:
                    fig = viz_tools.create_correlation_heatmap(numeric_df)
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.button("📊 Data Summary", use_container_width=True):
            for sheet_type, df in combined_data.items():
                st.markdown(f"**{sheet_type}**")
                st.dataframe(df.head(10), use_container_width=True)
    
    with col3:
        if st.button("📉 Trends", use_container_width=True):
            # Find and plot time-series
            for sheet_type, df in combined_data.items():
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if date_cols and numeric_cols:
                    fig = viz_tools.create_timeseries_chart(
                        df,
                        date_cols[0],
                        numeric_cols[0],
                        title=f"{sheet_type} Over Time",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    break


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app function."""
    # Header
    st.markdown('<div class="main-header">☕ Coffee Analytics Copilot</div>', unsafe_allow_html=True)
    st.markdown("*Agentic AI system for agricultural business intelligence*")
    
    # Sidebar
    render_sidebar()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning(
            "⚠️ **OpenRouter API key not configured**\n\n"
            "Please set the `OPENROUTER_API_KEY` environment variable in your `.env` file.\n\n"
            "Get a free API key at https://openrouter.ai"
        )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Chat", "📁 Data"])
    
    with tab1:
        st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            render_data_summary()
        
        with col2:
            render_analysis_section()
    
    with tab2:
        st.markdown('<div class="section-header">💬 Chat with Analytics Assistant</div>', unsafe_allow_html=True)
        render_chat_interface()
    
    with tab3:
        st.markdown('<div class="section-header">📁 Data Management</div>', unsafe_allow_html=True)
        
        if st.session_state.processed_data:
            for filename, sheets_dict in st.session_state.processed_data.items():
                st.markdown(f"### {filename}")
                
                for sheet_type, df in sheets_dict.items():
                    st.markdown(f"**{sheet_type}**")
                    st.dataframe(df, use_container_width=True)
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {sheet_type} as CSV",
                            data=csv,
                            file_name=f"{sheet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
        else:
            st.info("No data uploaded yet")


if __name__ == "__main__":
    main()
