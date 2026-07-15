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
    load_dotenv()  # Local dev only — Streamlit Cloud uses App secrets
except ImportError:
    pass

import streamlit as st
import pandas as pd
import os
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
# STREAMLIT PAGE CONFIG (must be the first Streamlit command)
# ============================================================================

st.set_page_config(
    page_title="☕ Coffee Analytics Copilot",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_openrouter_api_key() -> Optional[str]:
    """
    Resolve OPENROUTER_API_KEY from env (local .env) or Streamlit secrets (Cloud).

    Returns the key if found, else None. Always sets os.environ when found so
    model_router / LangChain can read it the same way locally and on Cloud.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if key and str(key).strip():
        return str(key).strip()

    # Streamlit Cloud / local secrets.toml
    try:
        secrets = st.secrets
    except Exception:
        return None

    candidates = []
    try:
        candidates.append(secrets.get("OPENROUTER_API_KEY"))
    except Exception:
        pass
    try:
        # Nested: [openrouter] api_key = "..."
        candidates.append(secrets["openrouter"]["api_key"])
    except Exception:
        pass
    try:
        candidates.append(secrets.get("openrouter_api_key"))
    except Exception:
        pass

    for candidate in candidates:
        if candidate and str(candidate).strip():
            value = str(candidate).strip()
            os.environ["OPENROUTER_API_KEY"] = value
            return value
    return None


def ensure_model_router():
    """Create or recover the model router after secrets are available."""
    if st.session_state.get("model_router") is not None:
        return st.session_state.model_router

    api_key = ensure_openrouter_api_key()
    if not api_key:
        st.session_state.model_router = None
        return None

    try:
        st.session_state.model_router = model_router.create_model_router()
        return st.session_state.model_router
    except Exception as e:
        st.session_state.model_router = None
        st.session_state.model_router_error = str(e)
        return None

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
    # Secrets must load before model router (esp. Streamlit Cloud)
    ensure_openrouter_api_key()

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
            st.session_state.supervisor = None
            st.session_state.supervisor_error = str(e)

    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "model_router" not in st.session_state:
        st.session_state.model_router = None
        ensure_model_router()


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


def _message_text(content) -> str:
    """Normalize LLM response content (string or content blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def process_user_query(user_query: str) -> str:
    """
    Process user query and return response using the LLM.
    """
    if not st.session_state.processed_data:
        return "Please upload data first using the sidebar."

    # Re-resolve secrets / router on every query (Cloud sessions can start
    # before secrets are visible, or after a deploy with old state).
    ensure_openrouter_api_key()
    router = ensure_model_router()
    if router is None:
        api_key_set = bool(ensure_openrouter_api_key())
        init_err = st.session_state.get("model_router_error", "")
        return (
            "⚠️ AI models not initialized.\n\n"
            f"**API key status:** {'Found' if api_key_set else 'Not set'}\n\n"
            "On **Streamlit Cloud**, set the secret in *App settings → Secrets*:\n"
            "```toml\n"
            'OPENROUTER_API_KEY = "sk-or-v1-..."\n'
            "```\n"
            "Then click **Reboot** so the new code and secrets load.\n\n"
            "Locally, put the same key in your `.env` file.\n"
            f"{('Init error: ' + init_err) if init_err else ''}"
        )

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

        # 3. Invoke free models with automatic fallback (skips retired / rate-limited)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]
        return router.invoke_with_fallback(messages, temperature=0.7, max_tokens=2000)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        err = str(e)
        hint = ""
        if "No endpoints found" in err or "gemma-2" in err:
            hint = (
                "\n\n**Hint:** A free model ID was retired. Reboot the Streamlit app so it "
                "loads the latest fallback list (gemma-4, gpt-oss, etc.)."
            )
        elif "credit" in err.lower() or "402" in err or "max_tokens" in err.lower():
            hint = (
                "\n\n**Hint:** Free-tier credit/token limit. Use only free models "
                "(already configured) or add credits at https://openrouter.ai/settings/credits"
            )
        elif "401" in err or "auth" in err.lower() or "api key" in err.lower():
            hint = (
                "\n\n**Hint:** Check `OPENROUTER_API_KEY` in Streamlit Cloud "
                "App settings → Secrets, then reboot."
            )
        elif "429" in err or "rate" in err.lower() or "TooManyRequests" in err:
            hint = (
                "\n\n**Hint:** Free-model rate limit (~50/day without credits). "
                "Wait and retry, or add a small credit balance on OpenRouter."
            )
        elif "All free OpenRouter models failed" in err:
            hint = (
                "\n\n**Hint:** Every free model was offline or rate-limited. "
                "Retry in a few minutes, or top up OpenRouter credits for higher limits."
            )
        return (
            f"⚠️ Error processing query with AI: {err}\n\n"
            f"Details:\n```\n{error_details}\n```"
            f"{hint}"
        )


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
            st.session_state["show_trends"] = True

    # Render trends panel (persists after button click)
    if st.session_state.get("show_trends") and combined_data:
        st.markdown("---")
        st.markdown("#### 📉 Trend Analysis")

        # --- Build list of plottable sheets ---
        plottable = {}
        for sheet_type, df in combined_data.items():
            df = df.copy()
            date_col = None

            # Check already-typed datetime columns
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                date_col = dt_cols[0]
            else:
                # Try keyword-named columns
                for c in df.columns:
                    if any(kw in c.lower() for kw in ["date", "month", "year", "period", "time", "week", "quarter"]):
                        try:
                            parsed = pd.to_datetime(df[c], errors="coerce")
                            if parsed.notna().sum() >= max(1, len(df) * 0.5):
                                df[c] = parsed
                                date_col = c
                                break
                        except Exception:
                            pass
                # Try all object columns as last resort
                if not date_col:
                    for c in df.select_dtypes(include=["object"]).columns:
                        try:
                            parsed = pd.to_datetime(df[c], errors="coerce")
                            if parsed.notna().sum() >= max(1, len(df) * 0.5):
                                df[c] = parsed
                                date_col = c
                                break
                        except Exception:
                            pass

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                plottable[sheet_type] = {
                    "df": df,
                    "date_col": date_col,  # None means use row index
                    "numeric_cols": numeric_cols,
                }

        if not plottable:
            st.warning("⚠️ No numeric data found to plot trends for.")
        else:
            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                selected_sheet = st.selectbox(
                    "Select Dataset",
                    options=list(plottable.keys()),
                    key="trend_sheet",
                )
            with sel_col2:
                available_metrics = plottable[selected_sheet]["numeric_cols"]
                selected_metric = st.selectbox(
                    "Select Metric",
                    options=available_metrics,
                    key="trend_metric",
                )

            sheet_info = plottable[selected_sheet]
            df_plot = sheet_info["df"]
            date_col = sheet_info["date_col"]

            if date_col:
                fig = viz_tools.create_timeseries_chart(
                    df_plot,
                    date_col,
                    selected_metric,
                    title=f"{selected_sheet} — {selected_metric} Over Time",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"ℹ️ No date column found in **{selected_sheet}** — showing trend by row order.")
                temp_df = df_plot[[selected_metric]].reset_index(drop=True)
                temp_df["Row"] = temp_df.index + 1
                fig = viz_tools.create_timeseries_chart(
                    temp_df, "Row", selected_metric,
                    title=f"{selected_sheet} — {selected_metric} Trend",
                )
                st.plotly_chart(fig, use_container_width=True)


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
    
    # Check API key (env locally, secrets on Streamlit Cloud)
    if not ensure_openrouter_api_key():
        st.warning(
            "⚠️ **OpenRouter API key not configured**\n\n"
            "**Streamlit Cloud:** App settings → Secrets → add:\n"
            "```toml\n"
            'OPENROUTER_API_KEY = "sk-or-v1-your-key"\n'
            "```\n"
            "Then **Reboot** the app.\n\n"
            "**Local:** set `OPENROUTER_API_KEY` in your `.env` file.\n\n"
            "Get a free API key at https://openrouter.ai"
        )
    elif st.session_state.get("model_router") is None:
        ensure_model_router()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Chat", "📁 Data"])
    
    with tab1:
        st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            render_data_summary()
        
        with col2:
            render_analysis_section()
            
        if st.session_state.processed_data:
            st.markdown("---")
            st.markdown("### 📊 Distribution Analysis")
            
            combined_data = {}
            for filename, sheets_dict in st.session_state.processed_data.items():
                for sheet_type, df in sheets_dict.items():
                    if sheet_type not in combined_data:
                        combined_data[sheet_type] = df
                        
            if combined_data:
                dist_col1, dist_col2, dist_col3 = st.columns(3)
                with dist_col1:
                    dataset_name = st.selectbox("Select Dataset", list(combined_data.keys()), key="dist_dataset")
                
                df_dist = combined_data[dataset_name]
                # Fallback to number if no objects exist to allow some visualization
                cat_cols = df_dist.select_dtypes(include=['object', 'category']).columns.tolist()
                if not cat_cols:
                    cat_cols = df_dist.columns.tolist()
                    
                if cat_cols:
                    with dist_col2:
                        cat_col = st.selectbox("Select Category", cat_cols, key="dist_category")
                    with dist_col3:
                        chart_type = st.radio("Chart Type", ["Pie Chart", "Bar Graph"], horizontal=True, key="dist_chart_type")
                    
                    chart_mode = "pie" if chart_type == "Pie Chart" else "bar"
                    fig = viz_tools.create_category_distribution_chart(
                        df_dist, 
                        cat_col, 
                        chart_type=chart_mode,
                        title=f"Distribution of {cat_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No columns found in {dataset_name} for distribution analysis.")
    
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
