import streamlit as st
import os
import re

st.set_page_config(page_title="AI Agent - RetailMind", layout="wide")

st.title("AI Agent")
st.markdown("### Ask questions about your retail data in natural language")

# Check if agent is loaded
if 'agent' not in st.session_state or st.session_state.agent is None:
    st.warning("AI Agent not initialized. Please upload and process your data in the **Data Hub** first.")
    if st.button("Go to Data Hub"):
        st.switch_page("pages/1_Data_Hub.py")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current data context
with st.sidebar:
    st.header("Current Data Context")
    try:
        crud = st.session_state.crud
        gen_df = crud.gen_df

        st.metric("Total Records", f"{len(gen_df):,}")
        st.metric("Stores", len(crud.storeIDs))
        st.metric("Date Range", f"{gen_df['Date'].min().date()} to {gen_df['Date'].max().date()}")
        st.metric("Total Sales", f"${gen_df['Weekly_Sales'].sum():,.0f}")

        st.markdown("### Stores Being Analyzed")
        st.write(f"Stores: {', '.join(map(str, crud.storeIDs))}")

        if hasattr(st.session_state, 'event_log') and st.session_state.event_log.log:
            st.markdown("### Events Available for Analysis")
            for event in st.session_state.event_log.log:
                st.write(f"• {event.description}")
                st.caption(f"  {event.start_date.date()} to {event.end_date.date()}")

    except Exception as e:
        st.error(f"Error displaying context: {str(e)}")

    # Sample questions

    st.markdown("---")
    st.header("Sample Questions")
    st.markdown("""
    **Performance Analysis:**
    - "Which store performed best last quarter?"
    - "Show me sales trends for store 1"
    - "How did sales change during holidays?"
    
    **Strategic Insights:**
    - "What factors affect our sales the most?"
    - "Which departments should we focus on?"
    - "How does unemployment impact our performance?"
    
    **Event Impact Analysis:**
    - "What was the impact of the [event name] on sales?"
    - "Analyze the effect of our marketing campaign"
    - "How did the promotion affect store performance?"
    
    **Custom Analysis:**
    - "Compare sales between stores 1 and 5"
    - "What was our peak sales period?"
    - "Analyze the impact of fuel price changes"
    """)

def display_response_with_images(response_text):
    """Display response text and any embedded chart references"""
    st.markdown(response_text)

    chart_patterns = [
        r'chart.*?\.png',
        r'graph.*?\.png',
        r'plot.*?\.png',
        r'./charts/\w+\.png',
        r'./assets/charts/\w+\.png'
    ]

    charts_found = []
    for pattern in chart_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        charts_found.extend(matches)

    # common_charts = [
    #     "./charts/sales.png",
    #     "./charts/holiday.png",
    #     "./charts/top_performers.png",
    #     "./charts/department.png",
    #     "./charts/department_holiday.png",
    #     "./charts/propensity.png",
    #     "./assets/charts/sales.png",
    #     "./assets/charts/holiday.png",
    #     "./assets/charts/top_performers.png"
    # ]

    # Display any charts that exist
    displayed_charts = set()

    # for chart_path in charts_found + common_charts:
    for chart_path in charts_found:
        if chart_path not in displayed_charts and os.path.exists(chart_path):
            try:
                st.image(chart_path, caption=f"Generated Chart: {os.path.basename(chart_path)}", use_container_width=True)
                displayed_charts.add(chart_path)
            except Exception as e:
                st.error(f"Error displaying chart {chart_path}: {str(e)}")

st.subheader("Conversation")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                display_response_with_images(message["content"])
prompt = st.chat_input("Ask me anything about your retail data...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data and generating insights..."):
            try:
                agent = st.session_state.agent
                response = agent.ask(prompt)
                display_response_with_images(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_message = f"I encountered an error while processing your request: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.expander("Technical Details"):
                    st.code(str(e))

# some controls

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("View Dashboard", type="primary"):
        st.switch_page("pages/2_EDA_Dashboard.py")

# some tips

st.markdown("---")
st.subheader("Tips for Better Analysis")

tip_col1, tip_col2 = st.columns(2)

with tip_col1:
    st.markdown("""
    **Effective Questions:**
    - Be specific about time periods
    - Mention specific stores or departments
    - Ask for comparisons and trends
    - Request actionable insights
    
    **Analysis Types:**
    - Performance comparisons
    - Seasonal trend analysis  
    - Economic factor correlations
    - Holiday impact assessment
    - Business event impact analysis
    """)

with tip_col2:
    st.markdown("""
    **Example Queries:**
    - "Compare Q1 vs Q2 sales performance"
    - "Which economic factors hurt sales most?"
    - "Show me the best performing departments during holidays"
    - "How do stores 1-5 compare in terms of seasonal patterns?"
    
    **Event Analysis:**
    - "What was the impact of [event name] on sales?"
    - "Analyze our marketing campaign effectiveness"
    - "How did the promotion affect different stores?"
    - "Compare sales before and during the event"
    """)

st.markdown("---")
st.info("""
**AI Capabilities:** I can analyze sales trends, compare store performance, assess economic impacts, 
examine holiday effects, rank departments, analyze business event impacts, and provide strategic business 
recommendations based on your data. Ask me anything about your retail performance!
""")