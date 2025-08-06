import streamlit as st

# Configure the main page

st.set_page_config(
    page_title="RetailMind",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header

st.title("RetailMind")
st.markdown("### AI-Powered Retail Analytics Platform")

# Welcome section

st.markdown("""
Welcome to **RetailMind**, your comprehensive retail analytics solution. This platform combines advanced data analysis 
with conversational AI to help you make informed business decisions based on your sales data.
""")

# How to get started guide

st.markdown("## How to Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1. Upload Your Data
    **Data Hub**
    
    Start by uploading your retail datasets:
    - Sales training data (train.csv)
    - External features (features.csv) 
    - Daily sales data (optional)
    
    Our platform will validate and process your data automatically.
    """)

with col2:
    st.markdown("""
    ### 2. Explore Insights  
    **EDA Dashboard**
    
    View comprehensive analytics including:
    - Sales trends over time
    - Holiday impact analysis
    - Store performance rankings
    - Economic factor correlations
    
    All visualizations are generated automatically.
    """)

with col3:
    st.markdown("""
    ### 3. Ask Questions
    **AI Agent**
    
    Interact with our intelligent assistant:
    - Ask complex business questions
    - Get data-driven recommendations
    - Explore custom analyses
    - Receive strategic insights
    
    Natural language interface for deep analysis.
    """)

# Feature highlights

st.markdown("##  Key Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    ** Advanced Analytics**
    - Time series analysis
    - Performance benchmarking
    - Economic impact assessment
    - Event correlation analysis
    
    ** Interactive Exploration**
    - Dynamic filtering by stores
    - Custom date range analysis  
    - Department-level insights
    - Holiday vs non-holiday comparisons
    """)

with feature_col2:
    st.markdown("""
    **AI-Powered Insights**
    - Natural language queries
    - Automated report generation
    - Strategic recommendations
    - Conversational analysis
    
    **Professional Visualizations**
    - Publication-ready charts
    - Interactive dashboards
    - Exportable analytics
    - Real-time data processing
    """)

# Navigation instructions
st.markdown("##Navigation")
st.info("""
 **Use the sidebar** to navigate between different sections of the application. 
Start with the **Data Hub** to upload your files, then explore the **EDA Dashboard** 
for automated insights, and finally interact with the **AI Agent** for custom analysis.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RetailMind v1.0 | Built with Streamlit & LangChain | Powered by OpenAI</p>
</div>
""", unsafe_allow_html=True)