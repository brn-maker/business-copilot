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
        help="Supported sheets: Tenant Payments, Coffee Outturn, Grading, Hotel Sales, Value Addition Sales, Fertilizer Sales, Coffee Sales / Exports, Transactions",
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
           - Coffee Sales / Exports
           - Transactions / Sales Logs
        
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
    Process user query and return response using the LLM.
    """
    if not st.session_state.processed_data:
        return "Please upload data first using the sidebar."
    
    if getattr(st.session_state, "model_router", None) is None:
        api_key_set = bool(os.getenv("OPENROUTER_API_KEY"))
        return f"⚠️ AI models not initialized. Please set your OPENROUTER_API_KEY environment variable.\n\nAPI Key status: {'Set' if api_key_set else 'Not set'}"
        
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # 1. Build a contextual summary of the uploaded data
        data_context = []
        for filename, sheets_dict in st.session_state.processed_data.items():
            for sheet_type, df in sheets_dict.items():
                data_context.append(f"### Sheet: {sheet_type}")
                data_context.append(f"- Columns: {', '.join(df.columns.tolist())}")
                data_context.append(f"- Rows: {len(df)}")
                
                # Add sample numerical statistics
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    # Keep it concise to save tokens
                    means = df[numeric_cols].mean().round(2).to_dict()
                    data_context.append(f"- Averages: {means}")
                data_context.append("")
                
        context_str = "\n".join(data_context)
        
        # 2. Setup the prompt
        system_prompt = f"""You are an expert Agricultural Business Intelligence Agent named Coffee Analytics Copilot.
You have access to the following dataset context from the user's uploaded files:

{context_str}

Analyze the user's query and provide a specific, professional answer based ONLY on the data context provided above. 
If the query asks for complex calculations (like regression or complex forecasting) that you cannot compute directly in your head, 
explain what the data suggests based on the averages and trends, and recommend using the dashboard's built-in 'Quick Analysis' buttons (Correlations, Trends, etc.) to generate precise charts.
Use markdown formatting (bolding, bullet points) to make your response clear and easy to read."""

        # 3. Invoke the LLM
        model = st.session_state.model_router.get_synthesis_model()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        
        response = model.invoke(messages)
        return response.content
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"⚠️ Error processing query with AI: {str(e)}\n\nDetails:\n{error_details}\n\nPlease ensure your OpenRouter API key is valid."


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
            trend_plotted = False
            for sheet_type, df in combined_data.items():
                df = df.copy()

                # Step 1: Look for already-typed datetime columns
                date_cols = [
                    c for c in df.columns
                    if pd.api.types.is_datetime64_any_dtype(df[c])
                ]

                # Step 2: Try columns with date-like keywords
                if not date_cols:
                    for c in df.columns:
                        if any(kw in c.lower() for kw in ["date", "month", "year", "period", "time", "week", "quarter"]):
                            try:
                                parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
                                if parsed.notna().sum() >= len(df) * 0.5:  # at least 50% parseable
                                    df[c] = parsed
                                    date_cols.append(c)
                                    break
                            except Exception:
                                pass

                # Step 3: Try ALL object/string columns for date parsing
                if not date_cols:
                    for c in df.select_dtypes(include=["object"]).columns:
                        try:
                            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
                            if parsed.notna().sum() >= len(df) * 0.5:
                                df[c] = parsed
                                date_cols.append(c)
                                break
                        except Exception:
                            pass

                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

                if not numeric_cols:
                    continue

                if date_cols:
                    # Plot with a real date axis
                    fig = viz_tools.create_timeseries_chart(
                        df,
                        date_cols[0],
                        numeric_cols[0],
                        title=f"{sheet_type} — {numeric_cols[0]} Over Time",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    trend_plotted = True
                    break
                else:
                    # Fallback: plot numeric column over row index
                    st.info(
                        f"ℹ️ No date column detected in **{sheet_type}**. "
                        f"Plotting **{numeric_cols[0]}** by row order instead."
                    )
                    temp_df = df[[numeric_cols[0]]].reset_index(drop=True)
                    temp_df["Row"] = temp_df.index + 1
                    fig = viz_tools.create_timeseries_chart(
                        temp_df,
                        "Row",
                        numeric_cols[0],
                        title=f"{sheet_type} — {numeric_cols[0]} Trend",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    trend_plotted = True
                    break

            if not trend_plotted:
                all_cols = []
                for sheet_type, df in combined_data.items():
                    all_cols += [f"`{c}`" for c in df.columns[:8]]
                st.warning(
                    "⚠️ Could not find any plottable data. "
                    f"Columns found: {', '.join(all_cols[:15])}"
                )


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
                            key=f"download_{filename}_{sheet_type}",
                        )
        else:
            st.info("No data uploaded yet")


if __name__ == "__main__":
    main()
