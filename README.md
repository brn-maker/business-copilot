# ☕ Coffee Analytics Copilot

**Agentic AI System for Coffee/Agriculture/Hospitality Business Intelligence**

A complete, low-cost analytics platform powered by LangGraph multi-agent orchestration and LLMs. Analyze agricultural data (coffee, hotel, fertilizer) with statistical analysis, time-series forecasting, and intelligent business recommendations.

## 🎯 Features

### Core Analytics
- **Statistical Analysis**: Correlations, regression, anomaly detection, growth rates
- **Time-Series Forecasting**: Prophet-based seasonal forecasting with confidence intervals
- **Scenario Analysis**: "What-if" planning for business decisions
- **Data Quality**: Automatic validation, outlier detection, missing value handling

### Agentic System
- **5 Specialized Agents**: Data Ingestion, Analysis, Forecasting, Recommender, Supervisor
- **LangGraph Orchestration**: Multi-step workflows with intelligent routing
- **Cost-Optimized**: Cheap models for computation, expensive models only for synthesis
- **Domain Knowledge**: Coffee-specific context (outturn %, grading impact, fertilizer ROI)

### User Interface
- **Natural Language Chat**: Ask questions in plain English
- **File Upload**: Support multiple Excel files with auto-detection
- **Interactive Charts**: Plotly visualizations with Streamlit
- **Session Persistence**: Maintain analysis context across conversations

## 📋 Requirements

- **Python**: 3.11+
- **Linux/macOS/Windows**: Tested on Linux Mint XFCE
- **RAM**: 4GB minimum (8GB recommended)
- **API Key**: OpenAI account (paid API required)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download project
cd biz-copilot

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Create .env file with your API key
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

Get an OpenAI API key at: https://platform.openai.com/api-keys

### 3. Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# App will open at: http://localhost:8501
```

## 📊 Data Format

### Supported Excel Sheets

Upload Excel files with these sheet names (auto-detected):

| Sheet Type | Description | Key Columns |
|-----------|-------------|------------|
| **Tenant Payments** | Monthly tenant income | Date, Amount, Tenant |
| **Coffee Outturn** | Coffee yield/output | Date, Outturn (%), Quantity |
| **Grading** | Coffee quality grades | Date, Grade, Percentage |
| **Hotel Sales** | Hotel/hospitality revenue | Date, Revenue, Occupancy |
| **Value Addition Sales** | Processed coffee sales | Date, Revenue, Type |
| **Fertilizer Sales** | Input costs & usage | Date, Amount, Type |
| **Coffee Sales / Exports** | Coffee bean exports | Export Date, Coffee Bean Type, Grade, Quantity KG, Total Value USD |
| **Transactions** | Retail shop transactions | Transaction ID, Transaction Date, Unit Price, Product Category |

### Example Data Structure

```
Tenant Payments
├─ Date (yyyy-mm-dd)
├─ Tenant Name
└─ Amount

Coffee Outturn
├─ Month (yyyy-mm)
├─ Outturn % (10-15% typical)
├─ Kg Harvested
└─ Kg Parchment

Grading
├─ Date
├─ Grade A (%)
├─ Grade B (%)
└─ Grade C (%)
```

## 💬 Example Queries

After uploading data, try these natural language queries:

```
# Statistical Analysis
"Show correlation between fertilizer and outturn"
"What's the relationship between grading and value addition sales?"

# Forecasting
"Predict coffee outturn for next 3 months"
"Forecast hotel revenue for next quarter"

# Business Insights
"Give me recommendations to improve yield"
"What are the trends in coffee grading quality?"
"Analyze tenant payment consistency"

# Scenario Planning
"What if fertilizer costs increase 15%?"
"Show impact of 10% price increase on revenue"
```

## 📁 Project Structure

```
biz-copilot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── README.md              # This file
│
├── agents/                 # Multi-agent system
│   ├── supervisor.py       # LangGraph orchestrator
│   ├── data_agent.py       # Data ingestion & processing
│   ├── analysis_agent.py   # Statistical analysis
│   ├── forecast_agent.py   # Time-series forecasting
│   └── recommender_agent.py # Business recommendations
│
├── tools/                  # Reusable tool functions
│   ├── data_tools.py       # Data manipulation
│   ├── analysis_tools.py   # Statistical functions
│   └── viz_tools.py        # Plotly visualizations
│
├── utils/                  # Utilities & helpers
│   ├── model_router.py     # LLM cost optimization
│   ├── prompts.py          # Agent system prompts
│   └── data_processor.py   # Data loading & cleaning
│
├── models/                 # Saved Prophet models (auto-created)
├── data/                   # Sample data & SQLite cache
└── .gitignore             # Git ignore rules
```

## 🔧 Architecture

### Multi-Agent Orchestration

```
User Input (Natural Language)
    ↓
Supervisor Agent (LangGraph)
    ├─ Routes to appropriate agent(s)
    ├─ Manages state and context
    └─ Synthesizes final response
    ↓
┌───────────────────────────────────────┐
│  Specialized Agents (Parallel Ready)   │
├───────────────────────────────────────┤
│ Data Agent    → Load/Clean/Validate   │
│ Analysis Ag   → Stats/Regression      │
│ Forecast Ag   → Time-Series Predict   │
│ Recommender   → Business Insights     │
└───────────────────────────────────────┘
    ↓
Results + Visualizations + Chat Response
```

### Cost Optimization Strategy

1. **Cheap Models**: Haiku, Llama 3.1 70B for data processing & analysis
2. **Expensive Models Only When**: Final synthesis, business recommendations
3. **Computation in Python**: Stats, forecasting, charts done locally (not via LLM)
4. **LLM Receives**: Summarized results, interpretation requests

**Typical Cost**: < $0.01 per analysis (vs $0.10+ for naive LLM approach)

## 🎓 Domain Knowledge

### Coffee Metrics Explained

- **Outturn %**: Percentage of harvested cherry that becomes sellable product
  - Typical: 10-15% of cherry weight → parchment
  - Affected by: Processing method, rainfall, fertilizer
  
- **Grading**: Coffee quality classification (Grade A/B/C)
  - Grade A: Premium (20-30% price premium)
  - Grade B: Standard
  - Grade C: Lower grade/reprocessed
  
- **Value Addition**: Local processing/roasting/packaging
  - Increases margin: 20-40% vs selling raw coffee
  - Dependent on: Grading quality, market size, processing capacity

- **Fertilizer ROI**: Nitrogen fertilizer impact on yield
  - Typical: 4-5 kg coffee cherry per 1 kg N applied
  - Diminishing returns after certain levels
  - Timing critical: Application before/during flowering

## 🔍 Analysis Examples

### 1. Correlation Analysis
Find relationships between variables:
```
Upload data → Chat: "Show fertilizer vs outturn correlation"
Result: Correlation coefficient, significance, interpretation
```

### 2. Regression Analysis
Predict one variable from others:
```
Chat: "Predict outturn from fertilizer and rainfall"
Result: Regression coefficients, R², confidence intervals
```

### 3. Forecasting
Predict future values with confidence:
```
Chat: "Forecast coffee revenue for next 12 months"
Result: Point forecast + 95% confidence interval
```

### 4. Scenario Analysis
Model business "what-ifs":
```
Chat: "What if fertilizer cost increases 15%?"
Result: Revenue impact analysis
```

## ⚙️ Configuration & Customization

### Change Model Defaults

Edit `utils/model_router.py` (use only concrete `:free` model IDs — avoid `openrouter/free`,
which can resolve to retired models):
```python
FREE_MODEL_CHAIN = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
    "cohere/north-mini-code:free",
    # ... more free fallbacks
]
```

Or set env: `OPENROUTER_FREE_MODEL_CHAIN=model1:free,model2:free`.

### Customize Agent Behavior

Edit system prompts in `utils/prompts.py`:
```python
SYSTEM_PROMPTS["data_agent"] = "Your custom instructions..."
```

### Add New Analysis Tools

Create functions in `tools/analysis_tools.py`:
```python
def custom_analysis(df):
    # Your analysis logic
    return results
```

## 🐛 Troubleshooting

### "OpenAI API key not found"
```bash
# Solution 1: Set environment variable
export OPENAI_API_KEY=sk-...
streamlit run app.py

# Solution 2: Add to .env file
# OPENAI_API_KEY=sk-...

# Solution 3: Enter in app sidebar
```

### "No module named 'prophet'"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Slow forecasting
- Prophet can be slow with large datasets (>1000 rows)
- Use `forecast_with_exponential_smoothing()` as faster alternative
- Check RAM usage with `top` or Task Manager

### Out of memory errors
- Reduce data size (filter to relevant date range)
- Process files one at a time
- Increase swap space or upgrade RAM

## 📖 Usage Tips

1. **Start Simple**: Begin with data summary queries
2. **Add Complexity**: Ask for correlations after understanding data
3. **Forecast Last**: Forecasting needs sufficient historical data (20+ points)
4. **Save Charts**: Right-click plotly charts → Download as PNG
5. **Export Data**: Use "Data" tab to export filtered results as CSV

## 🚀 Performance Optimization

For production use:

1. **Enable Caching**:
   ```python
   @st.cache_data
   def load_data():
       return process_uploaded_file(...)
   ```

2. **Use SQLite for Persistence**:
   ```python
   import sqlite3
   conn = sqlite3.connect("analytics.db")
   # Cache processed data
   ```

3. **Parallel Agent Execution**:
   - LangGraph supports parallel branch execution
   - Modify `supervisor.py` to invoke multiple agents simultaneously

4. **Model Quantization**:
   - Use quantized Prophet models for faster inference
   - Consider ONNX export for production

## 📝 API Documentation

### Main Functions

#### Data Processing
```python
from utils import DataProcessor, process_uploaded_file

processor = DataProcessor()
sheets = processor.load_excel_file("file.xlsx")
cleaned = processor.clean_data(sheets["coffee_outturn"])
```

#### Analysis
```python
from tools import analysis_tools

corr = analysis_tools.correlation_analysis(df)
reg = analysis_tools.regression_analysis(df, "y", ["x1", "x2"])
forecast = analysis_tools.time_series_decomposition(df, "date", "value")
```

#### Visualization
```python
from tools import viz_tools

fig = viz_tools.create_timeseries_chart(df, "date", "value")
fig = viz_tools.create_regression_chart(df, "x", "y")
fig.show()  # or st.plotly_chart(fig)
```

## 🤝 Contributing

To extend the system:

1. **Add new analysis**: Create function in `tools/`
2. **Add new agent**: Create agent file in `agents/`
3. **Update supervisor**: Add routing logic in `supervisor.py`
4. **Test thoroughly**: Verify with sample data

## 📄 License

This project is provided as-is for agricultural analytics use.

## 🔗 Resources

- **OpenAI**: https://platform.openai.com
- **Streamlit**: https://streamlit.io
- **LangChain**: https://langchain.com
- **Prophet**: https://facebook.github.io/prophet
- **Plotly**: https://plotly.com

## 📞 Support

For issues or questions:

1. Check `.env.example` configuration
2. Verify OpenAI API key is valid
3. Ensure Python 3.11+ installed
4. Check data format matches examples
5. Review logs in terminal output

## 🎯 Future Enhancements

- [ ] Database persistence (SQLite/PostgreSQL)
- [ ] Multi-user support with authentication
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (Docker, AWS, GCP)
- [ ] Real-time data streaming
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Custom metric definitions
- [ ] Export reports (PDF, Excel)
- [ ] Scheduled automated analysis

---

**Last Updated**: May 2026
**Version**: 1.0.0
**Author**: Coffee Analytics Team
