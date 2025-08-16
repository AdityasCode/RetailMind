import streamlit as st

# Configure the main page

st.set_page_config(
    page_title="RetailMind | Your AI Retail Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HEADER ---

st.title("Welcome to RetailMind 🧠")
st.markdown("### Stop Guessing. Start Doing. Turn Your Sales Data Into Actionable Insights In <5 Minutes.")

# --- WELCOME & PROBLEM STATEMENT ---

st.markdown("""
---
You're sitting on a mountain of sales data. Every transaction, every customer, every day tells a story. But who has the time to listen?  Hiring a data analyst is expensive.

**That's why we built RetailMind.** It's an expert analyst customized for your business needs, and it can support you in all aspects. We bridge the gap between the data you *have* and the insights you *need*, turning confusion into confidence.
""")

# --- HOW TO GET STARTED ---

st.markdown("## Get Answers in 3 Simple Steps")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    ### 1. Connect Your Data
    **Go to the `Data Hub`**
    
    Getting started is as simple as uploading a few files. Just provide your:
    - Core Sales History (`train.csv`)
    - External Features Information (`features.csv`) 
    
    That's it. Our system securely analyzes your data in seconds. No complex setup required.
    """)

with col2:
    st.markdown("""
    ### 2. See the Big Picture
    **Visit your `EDA Dashboard`**
    
    The moment your data is in, your business dashboard comes to life. Instantly see:
    - Which stores are your true superstars
    - How holidays *really* impact your bottom line
    - Sales trends and where they're heading
    - The link between fuel prices and your sales
    
    This is your command center, fully automated.
    """)

with col3:
    st.markdown("""
    ### 3. Ask Anything
    **Chat with the `AI Agent`**
    
    This is your MVP analyst. Go beyond the charts and talk to your data as you would with anyone. Ask questions like:
    - *"Why were sales down last Tuesday?"*
    - *"Show me my most profitable department."*
    - *"Forecast sales for next month in Store 10."*
    
    Get immediate, intelligent answers.
    """)

# --- KEY FEATURES ---

st.markdown("## Your New Superpowers")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    #### 💡 Powerful Insights, Made Simple
    - **See the Future:** Get accurate sales forecasts to plan inventory and staffing with confidence.
    - **Connect the Dots:** Finally understand how factors like holidays, promotions, or even the weather impact your sales.
    - **Benchmark Performance:** See clear rankings of your top-performing stores and departments.
    """)

with feature_col2:
    st.markdown("""
    #### 🤖 Your Personal AI Analyst
    - **Plain English, Smart Answers:** No need to learn code or technical jargon. If you can ask a question, you can use RetailMind.
    - **Deeper Dives, On Demand:** The AI can create any chart or analysis you need, right in the chat.
    - **Strategic Recommendations:** Get suggestions on where to focus your efforts for maximum impact.
    """)

# --- NAVIGATION ---

st.markdown("---")
st.success("""
 **Ready to begin?** Use the sidebar on the left to navigate. 
Start with the **Data Hub** to connect your files, then explore the **EDA Dashboard** for your instant overview.
""")

# --- FOOTER ---

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RetailMind v1.0 | Built with Streamlit & LangChain | Powered by OpenAI | Built by Aditya Gandhi</p>
</div>
""", unsafe_allow_html=True)