# WaitWell Analytics Dashboard

A comprehensive, multi-page Streamlit application for analyzing and optimizing queue performance with WaitWell data.

## 🚀 App Features

### 📁 Data Management
- **File Upload:** Upload cleaned CSV datasets
- **Location Filter:** Filter all visualizations by specific locations
- **Data Validation:** Automatic preprocessing and error handling

### 📱 Multi-Page Navigation
- **Overview Dashboard:** Key metrics and status distributions
- **Operational Metrics:** Enhanced chart with call volume, staff, and wait times
- **Time Analysis:** Hourly and daily wait time patterns
- **Staff Performance:** Productivity and performance metrics
- **Service Analysis:** Service types and queue performance

### 🎯 Key Highlights
- **Integrated Plotly Chart:** With location filtering and summary insights
- **Interactive Filtering:** Location dropdown affects all pages in real time
- **Professional Styling:** Custom CSS, responsive design, and consistent branding
- **Dashboard Elements:** KPIs, error handling, progress indicators, and status messages

## 🛠️ Getting Started

1. Save the code as `waitwell_dashboard.py`
2. Install required packages:
    ```bash
    pip install streamlit plotly pandas numpy
    ```
3. Run the application:
    ```bash
    streamlit run waitwell_dashboard.py
    ```

## 📊 Data Requirements

- Expects standard WaitWell columns
- Handles date/time conversions, missing columns (hour, DayOfWeek), and status-based filtering for completed tickets

## 📈 Scalability

The modular design allows easy addition of new pages or visualizations, making it ideal for leadership to analyze queue performance across locations and time periods.

## License

MIT License