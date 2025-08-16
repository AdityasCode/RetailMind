import datetime
import streamlit as st
import os
import tempfile
import sys
from autogluon.timeseries import TimeSeriesPredictor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.crud import CRUD, EventLog, Event
from src.eda import EDAFeatures
from src.agent import RetailAgent
from src.utils import parse_pdf, default_storeIDs

# --- PAGE CONFIGURATION ---

st.set_page_config(page_title="Data Hub - RetailMind", layout="wide")

# --- HEADER ---

st.title("🔗 Your Data Command Center")
st.markdown("### Securely connect your business data to unlock powerful, AI-driven insights.")

# --- DATA LOADED STATE ---

if hasattr(st.session_state, 'agent') and st.session_state.agent is not None:
    st.success("Excellent! Your data is connected and the system is ready for analysis.")

    st.markdown("#### Current Data Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales Records", f"{st.session_state.crud.gen_df.shape[0]:,} rows")
    with col2:
        st.metric("Economic Data Points", f"{st.session_state.crud.spec_df.shape[0]:,} rows")
    with col3:
        st.metric("Number of Stores", len(st.session_state.crud.storeIDs))

    st.markdown("#### Data Preview")
    st.dataframe(st.session_state.crud.gen_df.head(), use_container_width=True)

    if st.button("Disconnect Data and Start Over", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.stop()

# --- FILE UPLOAD SECTION ---

st.markdown("## Step 1: Upload Your Core Business Data")
st.info("Upload your sales and store information below. All files must be in CSV format.", icon="📁")

col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader(
        "Your Sales History (Required)",
        type=['csv'],
        help="This is the heart of your data. It should contain columns for Store, Department, Date, Weekly_Sales, and IsHoliday.",
        key="train_upload"
    )

    features_file = st.file_uploader(
        "Store & Economic Data (Required)",
        type=['csv'],
        help="This file adds context. It needs Store, Date, and data like local Temperature, Fuel_Price, and Unemployment rates.",
        key="features_upload"
    )

with col2:
    daily_file = st.file_uploader(
        "Daily Sales Figures (Optional)",
        type=['csv'],
        help="For a more granular, day-by-day analysis, provide your daily sales numbers here.",
        key="daily_upload"
    )
    store_ids_input = st.text_input(
        "Focus on Specific Stores (Optional)",
        placeholder="e.g., 1, 5, 12 or leave empty for all",
        help="Enter a comma-separated list of store numbers to analyze only a subset of your locations.",
    )

# --- EVENT UPLOAD SECTION ---

st.markdown("---")
st.markdown("## Step 2: Tell Us About Business Events (Optional but Recommended!)")
st.info("Did you run a big promotion? Launch a new product line? Upload the details here, and RetailMind can connect those events directly to your sales performance.", icon="💡")

event_col1, event_col2 = st.columns(2)
with event_col1:
    event_name = st.text_input(
        "Event Name",
        placeholder="e.g., Summer Sizzler Sale",
        help="Give your event a clear, memorable name."
    )
    event_start_date = st.date_input(
        "Event Start Date",
        help="The day the event or campaign kicked off.",
        min_value=datetime.datetime(1970, 1, 1)
    )

with event_col2:
    event_end_date = st.date_input(
        "Event End Date",
        help="The day the event concluded.",
        min_value=datetime.datetime(1970, 1, 1)
    )
    event_pdf = st.file_uploader(
        "Upload Event Details (PDF)",
        type=['pdf'],
        help="Upload a PDF with details about the event, like a marketing plan or press release.",
        key="event_pdf_upload"
    )

# --- DISPLAY CURRENT EVENTS ---

if hasattr(st.session_state, 'event_log') and st.session_state.event_log.log:
    st.markdown("#### Logged Business Events")
    for i, event in enumerate(st.session_state.event_log.log):
        with st.expander(f"**{event.description}** ({event.start_date.date()} to {event.end_date.date()})"):
            st.write(f"**Text Preview:** *'{event.text[:200]}...'*")

# --- ADD EVENT BUTTON ---

if st.button("Add Event to Log", disabled=not (event_name and event_pdf)):
    try:
        with st.spinner("Analyzing your event document..."):
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, "event_document.pdf")
            with open(pdf_path, "wb") as f:
                f.write(event_pdf.getbuffer())
            event_text = parse_pdf(pdf_path)
            new_event = Event(name=event_name, start_date=str(event_start_date), end_date=str(event_end_date), text=event_text)
            if not hasattr(st.session_state, 'event_log'):
                st.session_state.event_log = EventLog()
            st.session_state.event_log.add_event_from_event(new_event)
            st.session_state.event_log.save_log("./events_log.json")
        st.success(f"Event '{event_name}' has been successfully logged!")
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred while adding the event: {str(e)}")

# --- PROCESS BUTTON ---

st.markdown("---")
required_files = train_file is not None and features_file is not None
if not required_files:
    st.warning("Please upload both required files (Sales History and Store & Economic Data) to proceed.")

process_button = st.button(
    "Connect and Analyze My Data",
    disabled=not required_files,
    type="primary"
)

if process_button and required_files:
    try:
        store_ids = [int(x.strip()) for x in store_ids_input.split(',')] if store_ids_input.strip() else default_storeIDs
        with st.spinner("Building your analytics dashboard... This may take a few moments."):
            temp_dir = tempfile.mkdtemp()
            train_path = os.path.join(temp_dir, "train.csv")
            features_path = os.path.join(temp_dir, "features.csv")
            daily_path = None
            with open(train_path, "wb") as f: f.write(train_file.getbuffer())
            with open(features_path, "wb") as f: f.write(features_file.getbuffer())
            if daily_file is not None:
                daily_path = os.path.join(temp_dir, "daily.csv")
                with open(daily_path, "wb") as f: f.write(daily_file.getbuffer())
            os.makedirs("assets/charts", exist_ok=True)
            crud = CRUD(sales_path=train_path, features_path=features_path, daily_path=daily_path if daily_path else None, storeIDs=store_ids)
            event_log = st.session_state.event_log if hasattr(st.session_state, 'event_log') else EventLog()
            eda_analyzer = EDAFeatures(gen_df=crud.gen_df, spec_df=crud.spec_df, event_log=event_log, daily_df=crud.daily_df, storeIDs=crud.storeIDs, predictor=TimeSeriesPredictor.load("models/autogluon-m4-hourly"))
            agent = RetailAgent(crud_obj=crud, eda_obj=eda_analyzer)
            st.session_state.crud, st.session_state.eda_analyzer, st.session_state.agent, st.session_state.event_log = crud, eda_analyzer, agent, event_log

        st.success("Brilliant! Your analysis is ready.")

        st.markdown("#### Initial Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Sales Records", f"{len(crud.gen_df):,}")
        with col2: st.metric("Stores Analyzed", len(crud.storeIDs))
        with col3: st.metric("Time Periods Covered", f"{len(crud.gen_df['Date'].unique())}")
        with col4: st.metric("Total Sales Volume", f"${crud.gen_df['Weekly_Sales'].sum():,.0f}")

        st.dataframe(crud.gen_df.head(10), use_container_width=True)
        st.balloons()
        st.info("Navigate to the **EDA Dashboard** to see your automated insights, or chat with the **AI Agent** for custom analysis.")

    except Exception as e:
        st.error(f"Error Processing Data: {str(e)}")
        st.warning("Please check that your file formats match the requirements below. Ensure column names are correct.")
        with st.expander("Show Technical Error Details"): st.code(str(e))

# --- DATA FORMAT REQUIREMENTS ---

st.markdown("---")
st.markdown("### Data Format Requirements")

with st.expander("Sales Training Data Format"):
    st.markdown("""
    **Required columns:**
    - `Store`: Store identifier (integer)
    - `Date`: Date in YYYY-MM-DD format
    - `Weekly_Sales`: Sales amount (numeric)
    - `IsHoliday`: Boolean indicating if the week contains a holiday
    - `Dept`: Department identifier (integer)
    """)

with st.expander("External Features Data Format"):
    st.markdown("""
    **Required columns:**
    - `Store`: Store identifier (integer)
    - `Date`: Date in YYYY-MM-DD format  
    - `Temperature`: Temperature (numeric)
    - `Fuel_Price`: Fuel price (numeric)
    - `CPI`: Consumer Price Index (numeric)
    - `Unemployment`: Unemployment rate (numeric)
    """)

with st.expander("Daily Sales Data Format (Optional)"):
    st.markdown("""
    **Required columns:**
    - `Store`: Store identifier (integer)
    - `Date`: Date in YYYY-MM-DD format
    - `Daily_Sales`: Daily sales amount (numeric)
    """)

with st.expander("Event Documentation Guidelines"):
    st.markdown("""
    **Event Upload Requirements:**
    - **Name**: Descriptive name for the business event
    - **Start/End Dates**: Must align with your sales data period
    - **PDF Document**: Should contain detailed event information
    - **Content**: Marketing campaigns, product launches, promotions, etc.
    
    **Usage**: Events can be analyzed for their impact on sales performance using the AI Agent.
    """)