import streamlit as st
import os
import re
from PIL import Image

st.set_page_config(page_title="EDA Dashboard - RetailMind", layout="wide")

st.title("EDA Dashboard")
st.markdown("### Comprehensive analytics and insights from your retail data")

if 'eda_analyzer' not in st.session_state:
    st.warning(" No data loaded. Please upload and process your data in the **Data Hub** first.")
    if st.button("Go to Data Hub"):
        st.switch_page("pages/1_Data_Hub.py")
    st.stop()

eda_analyzer = st.session_state.eda_analyzer

def extract_image_path(summary_text):
    chart_patterns = [
        r'./charts/(\w+)\.png',
        r'./assets/charts/(\w+)\.png',
        r'charts/(\w+)\.png',
        r'assets/charts/(\w+)\.png'
    ]

    for pattern in chart_patterns:
        match = re.search(pattern, summary_text)
        if match:
            return match.group(0)
    return None

def display_chart_and_summary(summary_text, default_chart_path=None):
    st.markdown(re.sub(r'\s*Chart saved to:\s*.*\.png\s*$', '', (summary_text.replace("$", "\$").replace(" 00:00:00", ""))))
    image_path = extract_image_path(summary_text) or default_chart_path
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading chart: {str(e)}")
    elif default_chart_path and os.path.exists(default_chart_path):
        try:
            image = Image.open(default_chart_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading chart: {str(e)}")

tab1, tab2, tab3 = st.tabs(["Sales Overview", "Performance Analysis", "External Factors"])

with tab1:
    st.header("Sales Trends & Patterns")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sales Over Time")
        try:
            with st.spinner("Generating sales trend analysis..."):
                sales_summary = eda_analyzer.sales_t()
            display_chart_and_summary(sales_summary, "./charts/sales.png")
        except Exception as e:
            st.error(f"Error generating sales trend: {str(e)}")

    with col2:
        st.subheader("Key Metrics")
        try:
            gen_df = eda_analyzer.get_gen_df()
            total_sales = gen_df['Weekly_Sales'].sum()
            avg_weekly_sales = gen_df['Weekly_Sales'].mean()
            total_weeks = len(gen_df['Date'].unique())
            stores_count = len(gen_df['Store'].unique())

            st.metric("Total Sales", f"${total_sales:,.0f}")
            st.metric("Avg Weekly Sales", f"${avg_weekly_sales:,.0f}")
            st.metric("Analysis Period", f"{total_weeks} weeks")
            st.metric("Stores Analyzed", stores_count)
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")

    col3, col4 = st.columns([2, 1])

    with col3:
        st.subheader("Forecasted Sales")
        try:
            with st.spinner("Generating forecasted sales..."):
                sales_summary = eda_analyzer.forecast_weekly_sales()
            display_chart_and_summary(sales_summary, "./charts/forecasted.png")
        except Exception as e:
            st.error(f"Error generating sales trend: {str(e)}")

    if len(eda_analyzer.pred_df) is not 0:
        with col4:
            st.subheader("Key Metrics")
            try:
                gen_df = eda_analyzer.pred_df
                total_sales = gen_df['Weekly_Sales'].sum()
                avg_weekly_sales = gen_df['Weekly_Sales'].mean()
                total_weeks = len(gen_df['Date'].unique())
                stores_count = len(gen_df['Store'].unique())

                st.metric("Total Sales", f"${total_sales:,.0f}")
                st.metric("Avg Weekly Sales", f"${avg_weekly_sales:,.0f}")
                st.metric("Analysis Period", f"{total_weeks} weeks")
                st.metric("Stores Analyzed", stores_count)
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

    st.subheader("Holiday Impact Analysis")
    try:
        with st.spinner("Analyzing holiday impact..."):
            holiday_summary = eda_analyzer.holiday_impact()
        display_chart_and_summary(holiday_summary, "./charts/holiday.png")
    except Exception as e:
        st.error(f"Error analyzing holiday impact: {str(e)}")

with tab2:
    st.header("Store & Department Performance")

    if eda_analyzer.storeIDs and len(eda_analyzer.storeIDs) > 1:
        st.subheader("Top Performing Stores")
        try:
            with st.spinner("Analyzing store performance..."):
                store_summary = eda_analyzer.top_performing_stores()
            display_chart_and_summary(store_summary, "./charts/top_performers.png")
        except Exception as e:
            st.error(f"Error analyzing store performance: {str(e)}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Department Performance (All Time)")
        try:
            with st.spinner("Analyzing department performance..."):
                dept_summary = eda_analyzer.department_analysis_no_holiday()
            display_chart_and_summary(dept_summary, "./charts/department.png")
        except Exception as e:
            st.error(f"Error analyzing departments: {str(e)}")

    with col2:
        st.subheader("Department Performance (Holidays)")
        try:
            with st.spinner("Analyzing holiday department performance..."):
                holiday_dept_summary = eda_analyzer.department_analysis_holiday()
            display_chart_and_summary(holiday_dept_summary, "./charts/department_holiday.png")
        except Exception as e:
            st.error(f"Error analyzing holiday departments: {str(e)}")

with tab3:
    st.header("Economic Factors & Market Conditions")

    st.subheader("Economic Headwinds Analysis")
    try:
        with st.spinner("Analyzing economic factors..."):
            economic_summary = eda_analyzer.analyze_economic_headwinds()
        display_chart_and_summary(economic_summary, "./charts/propensity.png")

        st.subheader("Economic Factor Insights")

        spec_df = eda_analyzer.get_spec_df()
        if not spec_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Key Economic Indicators:**")
                avg_temp = spec_df['Temperature'].mean()
                avg_unemployment = spec_df['Unemployment'].mean()
                avg_cpi = spec_df['CPI'].mean()
                avg_fuel = spec_df['Fuel_Price'].mean()

                st.write(f"• Average Temperature: {avg_temp:.1f}°F")
                st.write(f"• Average Unemployment: {avg_unemployment:.1f}%")
                st.write(f"• Average CPI: {avg_cpi:.1f}")
                st.write(f"• Average Fuel Price: ${avg_fuel:.2f}")

            with col2:
                st.markdown("**Market Volatility:**")
                temp_std = spec_df['Temperature'].std()
                unemployment_std = spec_df['Unemployment'].std()
                cpi_std = spec_df['CPI'].std()
                fuel_std = spec_df['Fuel_Price'].std()

                st.write(f"• Temperature Volatility: ±{temp_std:.1f}°F")
                st.write(f"• Unemployment Volatility: ±{unemployment_std:.2f}%")
                st.write(f"• CPI Volatility: ±{cpi_std:.1f}")
                st.write(f"• Fuel Price Volatility: ±${fuel_std:.2f}")

    except Exception as e:
        st.error(f"Error analyzing economic factors: {str(e)}")

st.markdown("---")
st.header("Executive Summary")

try:
    with st.spinner("Generating executive summary..."):
        gen_df = eda_analyzer.get_gen_df()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Performance Highlights")
            peak_sales = gen_df.groupby('Date')['Weekly_Sales'].sum().max()
            peak_date = gen_df.groupby('Date')['Weekly_Sales'].sum().idxmax()

            st.write(f"**Peak Sales Period:** ${peak_sales:,.0f}")
            st.write(f"**Peak Date:** {peak_date.strftime('%B %Y')}")

            total_sales = gen_df['Weekly_Sales'].sum()
            st.write(f"**Total Sales:** ${total_sales:,.0f}")

        with col2:
            st.subheader("Store Insights")
            if len(eda_analyzer.storeIDs) > 1:
                store_performance = gen_df.groupby('Store')['Weekly_Sales'].sum()
                top_store = store_performance.idxmax()
                top_store_sales = store_performance.max()

                st.write(f"**Top Performing Store:** #{top_store}")
                st.write(f"**Top Store Sales:** ${top_store_sales:,.0f}")
            else:
                st.write(f"**Analyzing Store:** #{eda_analyzer.storeIDs[0]}")

            avg_store_sales = gen_df.groupby('Store')['Weekly_Sales'].sum().mean()
            st.write(f"**Avg Store Performance:** ${avg_store_sales:,.0f}")

        with col3:
            st.subheader("Data Coverage")
            date_range = gen_df['Date'].max() - gen_df['Date'].min()
            unique_weeks = len(gen_df['Date'].unique())

            st.write(f"**Analysis Period:** {date_range.days} days")
            st.write(f"**Data Points:** {unique_weeks} weeks")
            st.write(f"**Stores:** {len(eda_analyzer.storeIDs)} locations")

except Exception as e:
    st.error(f"Error generating summary: {str(e)}")

st.subheader("Recommended Next Steps")
st.info("""
1. **Explore Custom Analysis:** Use the AI Agent to ask specific questions about your data
2. **Deep Dive:** Investigate any unusual patterns or trends identified in the dashboard  
3. **Compare Periods:** Ask the AI Agent to compare different time periods or store performances
4. **Strategic Planning:** Use insights to inform inventory, staffing, and marketing decisions
""")