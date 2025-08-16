import streamlit as st
import os
import re
from PIL import Image
from src.utils import display_chart_and_summary

# --- Page Configuration ---
st.set_page_config(page_title="AI Agent - RetailMind", layout="wide")

# --- ALWAYS INITIALIZE SESSION STATE AT THE TOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Display ---
st.title("🤖 AI Agent")
st.markdown("### Ask questions about your retail data in natural language")

# --- Tips Section ---
with st.expander("💡 Tips for Getting the Best Insights"):
    tip_col1, tip_col2 = st.columns(2)
    with tip_col1:
        st.markdown("""
        **Effective Questions:**
        - Be specific about time periods ("last quarter", "December 2010").
        - Mention stores or departments by ID.
        - Ask for comparisons ("Compare store 5 and store 10").
        - Request actionable insights.
        """)
    with tip_col2:
        st.markdown("""
        **Example Queries:**
        - "Compare Q1 vs Q2 sales performance for store 2."
        - "Which economic factors hurt sales the most last year?"
        - "What was the impact of the *[event name]* on sales?"
        """)
    st.info("""
    **AI Capabilities:** I can analyze trends, compare performance, assess economic impacts, examine holiday effects, rank departments, and provide strategic recommendations based on your data.
    """)

# --- Agent and Data Validation ---
if 'agent' not in st.session_state or st.session_state.agent is None:
    st.warning("AI Agent not initialized. Please upload and process your data in the **Data Hub** first.")
    if st.button("Go to Data Hub"):
        st.switch_page("pages/1_Data_Hub.py")
    st.stop()

# --- Sidebar with Data Context ---
with st.sidebar:
    st.header("Current Data Context")
    try:
        crud = st.session_state.crud
        gen_df = crud.gen_df

        st.metric("Total Records", f"{len(gen_df):,}")
        st.metric("Stores Analyzed", len(crud.storeIDs))
        st.metric("Date Range", f"{gen_df['Date'].min().date()} to {gen_df['Date'].max().date()}")
        st.metric("Total Sales", f"${gen_df['Weekly_Sales'].sum():,.0f}")

        st.markdown("### Stores Being Analyzed")
        st.write(f"IDs: {', '.join(map(str, crud.storeIDs))}")

        if hasattr(st.session_state, 'event_log') and st.session_state.event_log.log:
            st.markdown("### Logged Business Events")
            for event in st.session_state.event_log.log:
                st.write(f"• {event.description}")

    except Exception as e:
        st.error(f"Error displaying context: {str(e)}")

# --- Helper Function for Displaying Responses ---
def display_response_with_images(response_text):
    """Generalized function to display content which may include chart references."""
    if not isinstance(response_text, str):
        st.write(response_text)
        return

    cleaned_text = response_text.replace("$", "\\$").replace(" 00:00:00", "")
    st.markdown(cleaned_text)

    chart_paths = re.findall(r'charts/[^\s]+\.png', response_text, re.IGNORECASE)
    for chart_path in set(chart_paths):
        if os.path.exists(chart_path):
            try:
                image = Image.open(chart_path)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading chart {chart_path}: {str(e)}")

# --- Main Chat Interface Logic ---
st.subheader("Conversation")

# Display the entire chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["content"] is not None:
            display_chart_and_summary(message["content"])

# Check if the AI is currently "thinking"
is_ai_thinking: bool = False
if (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] is None):
    is_ai_thinking = True

# --- Recommended Questions Buttons ---
st.markdown("---")
st.markdown("##### Quick-Start Questions")

prompt = None  # Initialize prompt to handle button clicks

colA, colB, colC = st.columns(3)
recommended_questions_1 = "Which were the top 3 performing stores last year?"
recommended_questions_2 = "How did holiday weeks impact sales on average?"
recommended_questions_3 = "Analyze the effect of our Summer Sale Campaign"

# Disable buttons while the AI is thinking
with colA:
    if st.button(recommended_questions_1, use_container_width=True, disabled=is_ai_thinking):
        prompt = recommended_questions_1
with colB:
    if st.button(recommended_questions_2, use_container_width=True, disabled=is_ai_thinking):
        prompt = recommended_questions_2
with colC:
    if st.button(recommended_questions_3, use_container_width=True, disabled=is_ai_thinking, help="Note: Requires a relevant event to be logged."):
        prompt = recommended_questions_3

# --- Chat Input Box ---
user_input = st.chat_input("Ask me anything about your retail data...", disabled=is_ai_thinking)
if user_input:
    prompt = user_input

# --- Processing Logic (Robust Multi-Rerun Pattern) ---
# 1. If a prompt was submitted, add it to history and rerun
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": None})
    st.rerun()

# 2. If the last message is an assistant placeholder, generate the real response
if is_ai_thinking:
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                last_user_prompt = st.session_state.messages[-2]["content"]
                agent = st.session_state.agent
                response = agent.ask(last_user_prompt)
                st.session_state.messages[-1]["content"] = response
                st.rerun()

            except Exception as e:
                error_message = f"I encountered an error: {str(e)}"
                st.session_state.messages[-1]["content"] = error_message
                st.rerun()

# --- Controls ---
st.markdown("---")
if st.button("Clear Conversation", type="secondary"):
    st.session_state.messages = []
    st.rerun()