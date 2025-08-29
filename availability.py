import streamlit as st
import pandas as pd
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Staff Availability Dashboard",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sample Data ---
# In a real-world scenario, you would load your data from a CSV or database.
# For this example, we'll create a DataFrame based on the provided image.
@st.cache_data
def load_data():
    """
    Creates a sample DataFrame mimicking the WaitWell data structure.
    This function is cached to improve performance.
    """
    file_path = r'C:\Users\e5016177\Documents\data\waitwell\Availabilities.xlsx'  # Path to your Excel file
    df = pd.read_excel(file_path)

    # --- Data Cleaning and Feature Engineering ---
    # Create a more readable 'Time Slot' column
    df['Time Slot'] = df['Start time'].str.slice(0, 5) + ' - ' + df['End time'].str.slice(0, 5)

    # Define the order of days for sorting
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Melt the DataFrame to transform it from a wide to a long format.
    # This makes it easier to plot and aggregate.
    id_vars = ['Staff', 'Time Slot', 'Service Type(s)']
    value_vars = days_of_week
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Day', value_name='Is Available')

    # Filter for only available slots
    df_available = df_melted[df_melted['Is Available'] == 1].copy()
    
    # Convert 'Day' to a categorical type to ensure correct sorting in visualizations
    df_available['Day'] = pd.Categorical(df_available['Day'], categories=days_of_week, ordered=True)
    
    return df_available

# Load the data
df_available = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

# Get unique values for filters
all_staff = sorted(df_available['Staff'].dropna().astype(str).unique())
all_services = sorted(df_available['Service Type(s)'].dropna().astype(str).unique())



selected_staff = st.sidebar.multiselect(
    "Select Staff Member",
    options=all_staff,
    default=all_staff
)

selected_services = st.sidebar.multiselect(
    "Select Service",
    options=all_services,
    default=all_services
)

# --- Filtering Logic ---
# Apply filters to the DataFrame. If a filter is empty, select all.
if not selected_staff:
    selected_staff = all_staff
if not selected_services:
    selected_services = all_services

filtered_data = df_available[
    (df_available['Staff'].isin(selected_staff)) &
    (df_available['Service Type(s)'].isin(selected_services))
]

# --- Main Dashboard ---
st.title("📅 Success Coach Availability Dashboard")
st.markdown("Use the filters on the left to analyze staff availability across different campuses and services.")

if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    # --- Visualization 1: Availability Heatmap/Grid ---
    st.header("Weekly Availability Schedule")
    st.markdown("This grid shows which staff members are available during specific time slots each day.")

    # Create a pivot table to structure the data for the grid view
    # The aggregation function joins the names of all available staff in a given slot.
    availability_pivot = pd.pivot_table(
        filtered_data,
        values='Staff',
        index=['Time Slot'],
        columns='Day',
        aggfunc=lambda x: ', '.join(sorted(x.unique()))
    ).fillna('') # Use an empty string for unavailable slots

    # Style the pivot table to improve readability
    def highlight_available(val):
        """Highlights cells that have an available staff member."""
        color = '#D4EDDA' if val else 'white' # Light green for available
        return f'background-color: {color}; color: #155724'

    # Display the styled DataFrame
    st.dataframe(availability_pivot.style.applymap(highlight_available), use_container_width=True)


    # --- Visualization 2: Availability by Campus and Day ---
    st.header("Availability Analysis")
    st.markdown("These charts break down the number of available time slots by campus and day of the week.")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: Total available slots per queue name
        st.subheader("Total Available Slots by Queue Name")
        campus_counts = filtered_data.groupby('Queue Name')['Time Slot'].count().reset_index()
        campus_counts = campus_counts.rename(columns={'Time Slot': 'Number of Available Slots'})
        st.bar_chart(campus_counts, x='Queue Name', y='Number of Available Slots', color='#007bff')

    with col2:
        # Bar chart: Total available slots per day of the week
        st.subheader("Total Available Slots by Day")
        day_counts = filtered_data.groupby('Day')['Time Slot'].count().reset_index()
        day_counts = day_counts.rename(columns={'Time Slot': 'Number of Available Slots'})
        st.bar_chart(day_counts, x='Day', y='Number of Available Slots', color='#28a745')

# --- Raw Data Expander ---
with st.expander("Show Raw Filtered Data"):
    st.markdown("The table below shows the detailed availability data based on your current filter selection.")
    st.dataframe(filtered_data, use_container_width=True)

