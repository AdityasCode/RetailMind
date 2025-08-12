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

st.set_page_config(page_title="Data Hub - RetailMind", layout="wide")

st.title(f"Data Hub")
st.markdown(f"### Upload and validate your retail datasets")

# Check if data is already loaded
if hasattr(st.session_state, 'agent') and st.session_state.agent is not None:
    st.success("Data is already loaded and processed!")

    # Show current data status
    st.markdown("### Current Data Status")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("General Data Shape", f"{st.session_state.crud.gen_df.shape[0]} rows")
    with col2:
        st.metric("Spec Data Shape", f"{st.session_state.crud.spec_df.shape[0]} rows")
    with col3:
        st.metric("Store Count", len(st.session_state.crud.storeIDs))

    # Preview current data
    st.markdown("### Data Preview")
    st.dataframe(st.session_state.crud.gen_df.head(), use_container_width=True)

    # Clear data option
    if st.button("Clear Data & Restart", type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.stop()

# File upload section
st.markdown("## File Upload")
st.markdown("Upload your retail datasets below. All files should be in CSV format.")

# Create file uploaders
col1, col2 = st.columns(2)

with col1:
    train_file = st.file_uploader(
        "Sales Training Data (Required)",
        type=['csv'],
        help="Main sales dataset containing Store, Date, Weekly_Sales, and IsHoliday columns",
        key="train_upload"
    )

    features_file = st.file_uploader(
        "External Features Data (Required)",
        type=['csv'],
        help="Features dataset with Store, Date, Temperature, Fuel_Price, CPI, Unemployment columns",
        key="features_upload"
    )

with col2:
    daily_file = st.file_uploader(
        "Daily Sales Data (Optional)",
        type=['csv'],
        help="Daily sales data for more granular analysis",
        key="daily_upload"
    )

    # Store selection
    store_ids_input = st.text_input(
        "Store IDs (Optional)",
        placeholder="e.g., 1,2,3,4,5 or leave empty for all stores",
        help="Comma-separated list of store IDs to analyze. Leave empty to include all stores.",
    )

# Event upload section
st.markdown("## Event Documentation Upload")
st.markdown("Upload PDF documents related to business events (campaigns, promotions, etc.) for impact analysis.")

event_col1, event_col2 = st.columns(2)

with event_col1:
    event_name = st.text_input(
        "Event Name",
        placeholder="e.g., Summer Sale Campaign",
        help="Descriptive name for the business event"
    )

    event_start_date = st.date_input(
        "Event Start Date",
        help="When the event began",
        min_value=datetime.datetime(1970, 1, 1)
    )

with event_col2:
    event_end_date = st.date_input(
        "Event End Date",
        help="When the event ended",
        min_value=datetime.datetime(1970, 1, 1)
    )

    event_pdf = st.file_uploader(
        "Event Documentation (PDF)",
        type=['pdf'],
        help="PDF document describing the event details",
        key="event_pdf_upload"
    )

# Display current events if any exist
if hasattr(st.session_state, 'event_log') and st.session_state.event_log.log:
    st.markdown("### Current Events in Log")
    for i, event in enumerate(st.session_state.event_log.log):
        with st.expander(f"{event.description} ({event.start_date.date()} to {event.end_date.date()})"):
            st.write(f"**Text Preview:** {event.text[:200]}...")

# Add event button
if st.button("Add Event to Log", disabled=not (event_name and event_pdf)):
    try:
        with st.spinner("Processing event PDF and adding to log..."):
            # save PDF temporarily
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, "event_document.pdf")

            with open(pdf_path, "wb") as f:
                f.write(event_pdf.getbuffer())

            # parse PDF text
            event_text = parse_pdf(pdf_path)

            # create Event object
            new_event = Event(
                name=event_name,
                start_date=str(event_start_date),
                end_date=str(event_end_date),
                text=event_text
            )

            # add to event log create if doesn't exist
            if not hasattr(st.session_state, 'event_log'):
                st.session_state.event_log = EventLog()

            st.session_state.event_log.add_event_from_event(new_event)

            # save the updated log
            st.session_state.event_log.save_log("./events_log.json")

        st.success(f"Event '{event_name}' successfully added to log!")
        st.rerun()

    except Exception as e:
        st.error(f"Error processing event: {str(e)}")
        with st.expander("Technical Details"):
            st.code(str(e))

# process store IDs

store_ids = None
if store_ids_input.strip():
    try:
        store_ids = [int(x.strip()) for x in store_ids_input.split(',')]
        st.info(f"Will analyze stores: {store_ids}")
    except ValueError:
        st.error("Invalid store IDs format. Please use comma-separated integers.")
else:
    store_ids = default_storeIDs

required_files = train_file is not None and features_file is not None
if not required_files:
    st.warning("Please upload both required files (Sales Training Data and External Features Data) to proceed.")

# process button

process_button = st.button(
    "Process and Validate Data",
    disabled=not required_files,
    type="primary"
)

if process_button and required_files:
    try:
        with st.spinner("Processing your data... This may take a few moments."):

            # save uploaded files temporarily

            temp_dir = tempfile.mkdtemp()
            train_path = os.path.join(temp_dir, "train.csv")
            features_path = os.path.join(temp_dir, "features.csv")
            daily_path = None

            with open(train_path, "wb") as f:
                f.write(train_file.getbuffer())

            with open(features_path, "wb") as f:
                f.write(features_file.getbuffer())

            if daily_file is not None:
                daily_path = os.path.join(temp_dir, "daily.csv")
                with open(daily_path, "wb") as f:
                    f.write(daily_file.getbuffer())
            os.makedirs("assets/charts", exist_ok=True)

            # initialize instances for all classes

            if store_ids:
                crud = CRUD(
                    sales_path=train_path,
                    features_path=features_path,
                    daily_path=daily_path if daily_path else "./test_data/train_daily_1.csv",
                    storeIDs=store_ids
                )
            else:
                crud = CRUD(
                    sales_path=train_path,
                    features_path=features_path,
                    daily_path=daily_path if daily_path else "./test_data/train_daily_1.csv"
                )
            if hasattr(st.session_state, 'event_log'):
                event_log = st.session_state.event_log
            else:
                event_log = EventLog()
            # if event_pdf and event_name:



            eda_analyzer = EDAFeatures(
                gen_df=crud.gen_df,
                spec_df=crud.spec_df,
                event_log=event_log,
                daily_df=crud.daily_df,
                storeIDs=crud.storeIDs,
                predictor=TimeSeriesPredictor.load("models/autogluon-m4-hourly")
            )
            agent = RetailAgent(crud_obj=crud, eda_obj=eda_analyzer)

            st.session_state.crud = crud
            st.session_state.eda_analyzer = eda_analyzer
            st.session_state.agent = agent
            st.session_state.event_log = event_log

        st.success("Data processed successfully!")


        st.markdown("### Data Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(crud.gen_df):,}")
        with col2:
            st.metric("Stores Analyzed", len(crud.storeIDs))
        with col3:
            st.metric("Date Range", f"{len(crud.gen_df['Date'].unique())} periods")
        with col4:
            st.metric("Total Sales", f"${crud.gen_df['Weekly_Sales'].sum():,.0f}")

        st.markdown("### Data Preview")
        st.dataframe(crud.gen_df.head(10), use_container_width=True)

        st.info("Ready to explore! Navigate to the **EDA Dashboard** to view automated insights, or try the **AI Agent** for custom analysis and event impact queries.")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please check your file formats and try again. Ensure your CSV files have the required columns.")

        with st.expander("Technical Details"):
            st.code(str(e))

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