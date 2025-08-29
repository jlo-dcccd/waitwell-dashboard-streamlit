import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
from streamlit_option_menu import option_menu
from waitwell_function import *

# Configure page
st.set_page_config(
    page_title="WaitWell Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #23235b;
        text-align: center;
        margin-bottom: 2.2rem;
        letter-spacing: 1.5px;
        background: linear-gradient(90deg, #e0e7ff 0%, #f8fafc 100%);
        border-radius: 1.2rem;
        box-shadow: 0 4px 24px 0 rgba(56,56,171,0.08);
        padding: 1.2rem 0 1.2rem 0;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        left: 50%;
        bottom: 0;
        transform: translateX(-50%);
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #3838ab 0%, #ff6b6b 100%);
        border-radius: 2px;
        margin-top: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    .stSelectbox > div > div > select {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# Sidebar for navigation and controls
st.sidebar.title("🎛️ Dashboard Controls")

# Data upload section
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload WaitWell Dataset (CSV)",
    type=['csv'],
    help="Upload your cleaned WaitWell dataset"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.toast("✅ Data loaded successfully.")
        
        # Data preprocessing
        if 'CreatedLocalTime' in df.columns:
            df['CreatedLocalTime'] = pd.to_datetime(df['CreatedLocalTime'])
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        if 'hour' not in df.columns and 'CreatedLocalTime' in df.columns:
            df['hour'] = df['CreatedLocalTime'].dt.hour
        if 'DayOfWeek' not in df.columns and 'CreatedLocalTime' in df.columns:
            df['DayOfWeek'] = df['CreatedLocalTime'].dt.day_name()
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        st.session_state.data = None

# Location filter
location_filter = None
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Location filter dropdown
    locations = ['All Locations'] + sorted(df['Location Name'].unique().tolist())
    selected_location = st.sidebar.selectbox(
        "\U0001F3E2 Select Location:",
        locations,
        help="Filter data by specific location"
    )
    
    # Apply location filter
    if selected_location != 'All Locations':
        filtered_df = df[df['Location Name'] == selected_location].copy()
        location_filter = selected_location
    else:
        filtered_df = df.copy()
        location_filter = None

    # Queue Name filter
    if 'Queue Name' in filtered_df.columns:
        queues = ['All Queues'] + sorted(filtered_df['Queue Name'].dropna().unique().tolist())
        selected_queue = st.sidebar.selectbox(
            "\U0001F39F\uFE0F Select Queue:",
            queues,
            help="Filter data by specific queue"
        )
        if selected_queue != 'All Queues':
            filtered_df = filtered_df[filtered_df['Queue Name'] == selected_queue].copy()
            queue_filter = selected_queue
        else:
            queue_filter = None
    else:
        queue_filter = None

    # Week filter (using WeekStart column)
    if 'WeekStart' in filtered_df.columns:
        weeks = ['All Weeks'] + sorted(filtered_df['WeekStart'].dropna().unique().tolist())
        selected_week = st.sidebar.selectbox(
            "\U0001F4C5 Select Week:",
            weeks,
            help="Filter data by specific week (WeekStart)"
        )
        if selected_week != 'All Weeks':
            # Apply week filter
            filtered_df = filtered_df[filtered_df['WeekStart'] == selected_week].copy()
            week_filter = selected_week
        else:
            week_filter = None
    else:
        week_filter = None

    st.session_state.filtered_data = filtered_df

# Page navigation
pages = {
    "\U0001F4CA Overview Dashboard": "overview",
    "\u23F0 Time Analysis": "time_analysis",
    "\U0001F465 Staff Performance": "staff_performance",
    "\U0001F4CB Service Analysis": "service_analysis",
    "\U0001F4C8 Forcasting Staff": "forcasting_staff"
}

# Use a dark style for the option menu that works in both dark and light modes
menu_styles = {
    "container": {"padding": "0!important", "background-color": "#797575"},
    "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "color": "#f0f2f6", "--hover-color": "#3838ab33"},
    "nav-link-selected": {"background-color": "#3838ab", "color": "#fff"},
    # Set a fixed color for the menu title that works in both light and dark mode
    "menu-title": {"font-size": "15px", "color": "#f0f2f6", "font-weight": "bold", "background": "#797575"}
}

with st.sidebar:
    selected_page = option_menu(
        "Select Dashboard:",
        list(pages.keys()),
        menu_icon="clipboard-data",
        default_index=0,
        styles=menu_styles
    )
current_page = pages[selected_page]

# Main content area
if st.session_state.data is None:
    st.markdown('<div class="main-header">WaitWell Analytics Dashboard</div>', unsafe_allow_html=True)
    st.info("👆 Please upload your WaitWell dataset using the sidebar to begin analysis.")
    
    # Show sample data structure
    st.subheader("📋 Expected Data Structure")
    sample_columns = [
        'id', 'Status', 'Source', 'Service Type', 'Date', 'Customer ID',
        'CreatedLocalTime', 'Wait time start', 'Completed', 'ActualWaitTimeMins',
        'Initial Position', 'Location Name', 'Queue Name', 'Staff', 'Meet Method',
        'DayOfWeek', 'hour', 'Week', 'WeekStart', 'WeekEnd', 'ServiceStartTime',
        'ServiceTimeMins', 'TotalTimeMins'
    ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Core Fields:**")
        for col in sample_columns[:8]:
            st.write(f"• {col}")
    with col2:
        st.write("**Time Fields:**")
        for col in sample_columns[8:16]:
            st.write(f"• {col}")
    with col3:
        st.write("**Calculated Fields:**")
        for col in sample_columns[16:]:
            st.write(f"• {col}")

else:
    df = st.session_state.filtered_data
    df_completed = df[df['Status'] == 'Completed'].copy() if 'Status' in df.columns else df.copy()
    
    # Header with location info
    header_text = "WaitWell Analytics Dashboard"
    if location_filter:
        header_text += f" - {location_filter}"
    st.markdown(f'<div class="main-header">{header_text}</div>', unsafe_allow_html=True)
    
    # Overview Dashboard
    if current_page == "overview":
        # Show header with latest week value
        if 'WeekStart' in df.columns:
            latest_week = df['WeekStart'].max()
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['WeekStart']):
                latest_week_dt = pd.to_datetime(latest_week)
            else:
                latest_week_dt = latest_week
            st.header(f"📊 Overview for Week Starting {latest_week_dt.strftime('%Y-%m-%d')}")
        else:
            st.header(f"📊 This Week's Overview ")
        
        # Key metrics row
        # Use base_df for metrics (filtered by location and queue only)
        base_df = st.session_state.data.copy()
        if location_filter:
            base_df = base_df[base_df['Location Name'] == location_filter]
        if queue_filter:
            base_df = base_df[base_df['Queue Name'] == queue_filter]
        # Filter to only latest week
        if 'WeekStart' in base_df.columns:
            latest_week = base_df['WeekStart'].max()
            prev_week = base_df['WeekStart'].sort_values().unique()
            prev_week = prev_week[-2] if len(prev_week) > 1 else None
            latest_df = base_df[base_df['WeekStart'] == latest_week]
            prev_df = base_df[base_df['WeekStart'] == prev_week] if prev_week is not None else pd.DataFrame(columns=base_df.columns)
        else:
            latest_df = base_df.copy()
            prev_df = pd.DataFrame(columns=base_df.columns)
        latest_completed = latest_df[latest_df['Status'] == 'Completed'] if 'Status' in latest_df.columns else latest_df
        prev_completed = prev_df[prev_df['Status'] == 'Completed'] if 'Status' in prev_df.columns else prev_df
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_tickets = len(latest_df)
            prev_total_tickets = len(prev_df)
            completed_tickets = len(latest_completed)
            prev_completed_tickets = len(prev_completed)
            delta = ((completed_tickets - prev_completed_tickets) / prev_completed_tickets * 100) if prev_completed_tickets > 0 else None
            st.metric("Completed Tickets", f"{completed_tickets:,}", delta=f"{delta:.1f}%" if delta is not None else None)
        with col2:
            completed_tickets = len(latest_completed)
            prev_completed_tickets = len(prev_completed)
            completion_rate = (completed_tickets / total_tickets * 100) if total_tickets > 0 else 0
            prev_completion_rate = (prev_completed_tickets / prev_total_tickets * 100) if prev_total_tickets > 0 else 0
            delta = completion_rate - prev_completion_rate if prev_total_tickets > 0 else None
            st.metric("Completion Rate", f"{completion_rate:.1f}%", delta=f"{delta:.1f}%" if delta is not None else None)
        with col3:
            if len(latest_completed) > 0 and 'ActualWaitTimeMins' in latest_completed.columns:
                median_wait = latest_completed['ActualWaitTimeMins'].median()
                prev_median_wait = prev_completed['ActualWaitTimeMins'].median() if len(prev_completed) > 0 and 'ActualWaitTimeMins' in prev_completed.columns else None
                delta = (median_wait - prev_median_wait) if prev_median_wait is not None else None
                st.metric("Median Wait Time", f"{median_wait:.1f} min", delta=f"{delta:+.1f} min" if delta is not None else None, delta_color='inverse')
            else:
                st.metric("Median Wait Time", "N/A")
        with col4:
            # Peak Queue Length for the latest week with delta vs previous week
            if 'Initial Position' in latest_df.columns:
                peak_queue_length = latest_df['Initial Position'].max()
                prev_peak_queue_length = prev_df['Initial Position'].max() if 'Initial Position' in prev_df.columns and not prev_df.empty else None
                if prev_peak_queue_length is not None and pd.notnull(prev_peak_queue_length) and prev_peak_queue_length > 0:
                    delta = ((peak_queue_length - prev_peak_queue_length) / prev_peak_queue_length) * 100
                    st.metric("Peak Queue Length", int(peak_queue_length) if pd.notnull(peak_queue_length) else "N/A", delta=f"{delta:+.1f}%", delta_color='off')
                else:
                    st.metric("Peak Queue Length", int(peak_queue_length) if pd.notnull(peak_queue_length) else "N/A")
            else:
                st.metric("Peak Queue Length", "N/A")

        if 'WeekStart' in st.session_state.data.columns and 'Staff' in st.session_state.data.columns:
            # Filter by location and queue only (not week)
            base_df = st.session_state.data.copy()
            if location_filter:
                base_df = base_df[base_df['Location Name'] == location_filter]
            if queue_filter:
                base_df = base_df[base_df['Queue Name'] == queue_filter]
            base_df = base_df[base_df['Status'] == 'Completed'] if 'Status' in base_df.columns else base_df
            # Use WeekStart instead of Week
            staff_per_week = base_df.groupby('WeekStart')['Staff'].nunique().reset_index()
            staff_per_week.columns = ['WeekStart', 'Number of Staff']

            queue_volume_per_week = base_df.groupby('WeekStart').size().reset_index(name='Queue Volume')
            merged_weekly = pd.merge(staff_per_week, queue_volume_per_week, on='WeekStart', how='outer').sort_values('WeekStart')

            if 'ActualWaitTimeMins' in base_df.columns:
                median_wait_time_per_week = base_df.groupby('WeekStart')['ActualWaitTimeMins'].median().reset_index()
                median_wait_time_per_week.columns = ['WeekStart', 'Median Wait Time (min)']
                merged_weekly = merged_weekly.merge(median_wait_time_per_week, on='WeekStart', how='outer')
            
            # Create the plotly figure
            fig = go.Figure()
            
            # Add call volume as a bar (left y-axis)
            fig.add_trace(go.Bar(
                x=merged_weekly['WeekStart'],
                y=merged_weekly['Queue Volume'],
                name='Queue Volume',
                marker_color='blue',
                opacity=0.7,
                yaxis='y1'
            ))
            
            # Add staff count as a line (right y-axis)
            fig.add_trace(go.Scatter(
                x=merged_weekly['WeekStart'],
                y=merged_weekly['Number of Staff'],
                name='Number of Staff',
                mode='lines+markers',
                marker=dict(size=7, color='orange'),
                line=dict(width=2, color='orange'),
                yaxis='y2'
            ))

            # Add median wait time if available (also on y2)
            if 'Median Wait Time (min)' in merged_weekly.columns:
                fig.add_trace(go.Scatter(
                    x=merged_weekly['WeekStart'],
                    y=merged_weekly['Median Wait Time (min)'],
                    name='Median Wait Time (min)',
                    mode='lines+markers',
                    marker=dict(size=7, color='red'),
                    line=dict(width=2, color='red', dash='dash'),
                    yaxis='y2'
                ))
                
            fig.update_layout(
                title='Queue Volume, Number of Staff, and Median Wait Time Per Week',
                xaxis_title='Week Start',
                yaxis=dict(
                    title='Queue Volume',
                    side='left'
                ),
                yaxis2=dict(
                    title='Number of Staff / Median Wait Time (min)',
                    overlaying='y',
                    side='right',
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified',
                height=450,
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Source' in df.columns:
                source_volume = df.groupby('Source').size().reset_index(name='Volume')
                fig_source = px.pie(
                    source_volume, values='Volume', names='Source',
                    title="Ticket Source Distribution"
                    )
                fig_source.update_layout(height=400)
                st.plotly_chart(fig_source, use_container_width=True)

        
        with col2:
            if 'Service Type' in df.columns:
                service_counts = df['Service Type'].value_counts().head(10)
                fig_service = px.bar(
                    x=service_counts.values,
                    y=service_counts.index,
                    orientation='h',
                    title="Top 10 Service Types"
                )
                fig_service.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_service, use_container_width=True)

        if 'Service Type' in df.columns:
            service_metrics = df.groupby('Service Type').agg({
                'id': 'count',
                'ActualWaitTimeMins': 'mean' if 'ActualWaitTimeMins' in df.columns else 'count'
            }).round(2)
            service_metrics = service_metrics.reset_index().sort_values('id', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_service_volume = px.bar(
                    service_metrics.head(15),
                    x='id',
                    y='Service Type',
                    orientation='h',
                    title="Top 15 Services by Volume"
                )
                fig_service_volume.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_service_volume, use_container_width=True)
            
            with col2:
                if 'Queue Name' in df.columns:
                    queue_metrics = df.groupby('Queue Name').size().reset_index(name='Volume')
                    queue_metrics = queue_metrics.sort_values('Volume', ascending=False)
                    
                    fig_queue = px.pie(
                        queue_metrics.head(10),
                        values='Volume',
                        names='Queue Name',
                        title="Queue Distribution (Top 10)"
                    )
                    fig_queue.update_layout(height=500)
                    st.plotly_chart(fig_queue, use_container_width=True)
    
    # Time Analysis
    elif current_page == "time_analysis":
        st.header("⏰ Time Analysis")

        # --- Plotly Heatmap of ticket volume by day of week and hour of day ---
        import plotly.graph_objects as go
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        if 'hour' in df.columns and 'DayOfWeek' in df.columns:
            # Prepare heatmap data
            heatmap_data = df.groupby(['hour', 'DayOfWeek']).size().unstack(fill_value=0).reindex(columns=day_order, fill_value=0)
            z = heatmap_data.values
            x = heatmap_data.columns.tolist()
            y = heatmap_data.index.tolist()
            min_date = df['CreatedLocalTime'].min().date() if 'CreatedLocalTime' in df.columns else ''
            max_date = df['CreatedLocalTime'].max().date() if 'CreatedLocalTime' in df.columns else ''
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='YlOrRd',
                colorbar=dict(title='Number of Tickets'),
                hovertemplate='Day: %{x}<br>Hour: %{y}<br>Tickets: %{z}<extra></extra>'
            ))
            fig_heatmap.update_layout(
                title=f'Ticket Volume from {min_date} to {max_date}',
                yaxis_title='Hour of Day',
                yaxis_nticks=24,
                height=500,
                margin=dict(t=60, b=40, l=60, r=20)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # --- Hourly Staff, Tickets Completed, New Tickets Created, and Median Wait Time Chart (Wide Plotly Chart) ---
        if (
            'hour' in df_completed.columns and 
            'Staff' in df_completed.columns and 
            'ActualWaitTimeMins' in df_completed.columns and
            'CreatedLocalTime' in df.columns
        ):
            staff_per_hour = df_completed.groupby('hour')['Staff'].nunique().reset_index(name='Number of Staff')
            tickets_completed_per_hour = df_completed.groupby('hour_completed').size().reset_index(name='Tickets Completed')
            median_wait_per_hour = df_completed.groupby('hour')['ActualWaitTimeMins'].median().reset_index(name='Median Wait Time')
            # New tickets created per hour (from all tickets, not just completed)
            new_tickets_per_hour = df.groupby('hour').size().reset_index(name='New Tickets Created')

            # Merge all summaries
            hourly_summary = staff_per_hour.merge(tickets_completed_per_hour, left_on='hour', right_on='hour_completed', how='left')
            hourly_summary = hourly_summary.merge(median_wait_per_hour, on='hour', how='outer')
            hourly_summary = hourly_summary.merge(new_tickets_per_hour, on='hour', how='outer')
            hourly_summary = hourly_summary.sort_values('hour')

            # drop 'hour_completed' column as it's not needed anymore
            hourly_summary = hourly_summary.drop(columns=['hour_completed'], errors='ignore')

            fig_hourly_summary = go.Figure()
            fig_hourly_summary.add_trace(go.Bar(
            x=hourly_summary['hour'],
            y=hourly_summary['Number of Staff'],
            name='Number of Staff',
            marker_color='orange',
            opacity=0.7,
            yaxis='y1'
            ))
            fig_hourly_summary.add_trace(go.Bar(
            x=hourly_summary['hour'],
            y=hourly_summary['Tickets Completed'],
            name='Tickets Completed',
            marker_color='blue',
            opacity=0.7,
            yaxis='y1'
            ))
            fig_hourly_summary.add_trace(go.Bar(
            x=hourly_summary['hour'],
            y=hourly_summary['New Tickets Created'],
            name='New Tickets Created',
            marker_color='green',
            opacity=0.7,
            yaxis='y1'
            ))
            fig_hourly_summary.add_trace(go.Scatter(
            x=hourly_summary['hour'],
            y=hourly_summary['Median Wait Time'],
            name='Median Wait Time',
            mode='lines+markers',
            marker=dict(color='red', size=10),
            line=dict(color='red', width=3, dash='dash'),
            yaxis='y2'
            ))
            fig_hourly_summary.update_layout(
            title='Hourly Staff Coverage, Tickets Completed, New Tickets Created, and Median Wait Time',
            xaxis=dict(title='Hour of Day', tickmode='array', tickvals=hourly_summary['hour'], ticktext=[str(h) for h in hourly_summary['hour']]),
            yaxis=dict(title='Count (Staff, Tickets)', showgrid=False),
            yaxis2=dict(title='Median Wait Time (min)', overlaying='y', side='right', showgrid=False, color='red'),
            legend=dict(x=0.01, y=0.99),
            height=500,
            barmode='group',
            margin=dict(t=40, b=40, l=60, r=60)
            )
            st.plotly_chart(fig_hourly_summary, use_container_width=True)

            # --- Backlog Estimate Visualization ---
            if 'New Tickets Created' in hourly_summary.columns and 'Tickets Completed' in hourly_summary.columns:
                hourly_summary['Backlog Estimate'] = hourly_summary['New Tickets Created'].fillna(0) - hourly_summary['Tickets Completed'].fillna(0)
                fig_backlog = go.Figure()
                fig_backlog.add_trace(go.Scatter(
                    x=hourly_summary['hour'],
                    y=hourly_summary['Backlog Estimate'],
                    name='Backlog Estimate',
                    mode='lines+markers',
                    marker=dict(color='purple', size=10),
                    line=dict(color='purple', width=3),
                ))
                # Add dashed line at y=0
                fig_backlog.add_trace(go.Scatter(
                    x=hourly_summary['hour'],
                    y=[0]*len(hourly_summary['hour']),
                    name='Zero Backlog',
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    showlegend=True
                ))
                fig_backlog.update_layout(
                    title='Hourly Backlog Estimate (Tickets Created - Tickets Completed)',
                    xaxis=dict(title='Hour of Day', tickmode='array', tickvals=hourly_summary['hour'], ticktext=[str(h) for h in hourly_summary['hour']]),
                    yaxis=dict(title='Backlog Estimate'),
                    legend=dict(x=0.01, y=0.99),
                    height=350,
                    margin=dict(t=40, b=40, l=60, r=60)
                )
                st.plotly_chart(fig_backlog, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            if 'hour' in df.columns:
                hourly_volume = df.groupby('hour').size().reset_index(name='Volume')
                # Sort by volume to find top 3
                sorted_idx = hourly_volume['Volume'].sort_values(ascending=False).index[:3]
                # Normalize the volume for color intensity
                max_volume = hourly_volume['Volume'].max()
                min_volume = hourly_volume['Volume'].min()
                if max_volume > min_volume:
                    norm_volume = (hourly_volume['Volume'] - min_volume) / (max_volume - min_volume)
                else:
                    norm_volume = np.zeros_like(hourly_volume['Volume'])
                # Default color for all bars
                colors = [f'rgba(255, {int(200*(1-v))}, {int(200*(1-v))}, 1)' for v in norm_volume]
                # Set top 3 bars to a much deeper red
                top3_idx = hourly_volume['Volume'].sort_values(ascending=False).index[:3]
                for idx in top3_idx:
                    colors[idx] = 'rgba(139,0,0,1)'
                fig_hourly = px.bar(
                    hourly_volume, x='hour', y='Volume',
                    title="Hourly Ticket Volume",
                    labels={'hour': 'Hour of Day', 'Volume': 'Number of Tickets'},
                    color_discrete_sequence=colors
                )
                fig_hourly.update_traces(marker_color=colors)
                fig_hourly.update_layout(
                    height=500,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=hourly_volume['hour'],
                        ticktext=[str(h) for h in hourly_volume['hour']],
                        tickangle=0
                    )
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            if 'DayOfWeek' in df.columns:
                # Compute daily stats
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                daily_stats = df.groupby('DayOfWeek').agg({
                    'ActualWaitTimeMins': 'mean' if 'ActualWaitTimeMins' in df.columns else 'count',
                    'id': 'count',
                    'Staff': 'nunique' if 'Staff' in df.columns else 'count'
                }).reset_index()
                # Add day_num for sorting
                daily_stats['day_num'] = daily_stats['DayOfWeek'].map({day: i for i, day in enumerate(day_order)})
                daily_stats = daily_stats.sort_values('day_num')
                # Calculate y-axis max with 10% headroom
                y1_max = daily_stats['id'].max() * 1.4 if not daily_stats['id'].empty else None
                y2_max = daily_stats['ActualWaitTimeMins'].max() * 1.4 if 'ActualWaitTimeMins' in daily_stats.columns and not daily_stats['ActualWaitTimeMins'].empty else None
                # Plotly: bar for ticket volume, line for avg wait time
                import plotly.graph_objects as go
                fig_daily = go.Figure()
                # Bar for ticket volume
                fig_daily.add_trace(go.Bar(
                    x=daily_stats['DayOfWeek'],
                    y=daily_stats['id'],
                    name='Ticket Volume',
                    marker_color='lightblue',
                    yaxis='y1',
                    opacity=0.7
                ))
                # Line for avg wait time
                if 'ActualWaitTimeMins' in daily_stats.columns:
                    fig_daily.add_trace(go.Scatter(
                        x=daily_stats['DayOfWeek'],
                        y=daily_stats['ActualWaitTimeMins'],
                        name='Avg Wait Time',
                        mode='lines+markers',
                        marker=dict(color='red', size=10),
                        line=dict(color='red', width=3),
                        yaxis='y2'
                    ))
                fig_daily.update_layout(
                    title='Daily Performance Overview',
                    xaxis=dict(title='Day of Week', tickangle=45, categoryorder='array', categoryarray=day_order),
                    yaxis=dict(title='Number of Tickets', showgrid=False, range=[0, y1_max]),
                    yaxis2=dict(title='Average Wait Time (minutes)', overlaying='y', side='right', showgrid=False, color='red', range=[0, y2_max]),
                    legend=dict(x=0.01, y=0.99),
                    height=500,
                    margin=dict(t=40, b=40, l=60, r=60)
                )
                st.plotly_chart(fig_daily, use_container_width=True)
        
        # Wait time analysis
        if 'ActualWaitTimeMins' in df_completed.columns and len(df_completed) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'hour' in df_completed.columns and 'ActualWaitTimeMins' in df_completed.columns:
                    # Hourly wait time stats
                    hourly_stats = df_completed.groupby('hour')['ActualWaitTimeMins'].agg(['mean', 'median', 'count']).reset_index()
                    # Plotly: bar for mean, line for median
                    import plotly.graph_objects as go
                    fig_wait_hourly = go.Figure()
                    # Bar for average wait time
                    fig_wait_hourly.add_trace(go.Bar(
                        x=hourly_stats['hour'],
                        y=hourly_stats['mean'],
                        name='Avg Wait Time',
                        marker_color='#FF6B6B',
                        opacity=0.7,
                        yaxis='y1'
                    ))
                    # Line for median wait time
                    fig_wait_hourly.add_trace(go.Scatter(
                        x=hourly_stats['hour'],
                        y=hourly_stats['median'],
                        name='Median Wait Time',
                        mode='lines+markers',
                        marker=dict(color='red', size=8),
                        line=dict(color='red', width=2),
                        yaxis='y1'
                    ))
                    fig_wait_hourly.update_layout(
                        title='Average Wait Time by Hour',
                        xaxis=dict(title='Hour of Day', tickmode='array', tickvals=hourly_stats['hour'], ticktext=[str(h) for h in hourly_stats['hour']]),
                        yaxis=dict(title='Wait Time (minutes)'),
                        legend=dict(x=0.01, y=0.99),
                        height=500,
                        margin=dict(t=40, b=40, l=60, r=60),
                        bargap=0.2
                    )
                    st.plotly_chart(fig_wait_hourly, use_container_width=True)
            
            with col2:
                wait_stats = df_completed['ActualWaitTimeMins'].describe()
                st.subheader("📊 Wait Time Statistics")
                st.write(f"• **Mean:** {wait_stats['mean']:.1f} minutes")
                st.write(f"• **Median:** {wait_stats['50%']:.1f} minutes")
                st.write(f"• **75th Percentile:** {wait_stats['75%']:.1f} minutes")
                st.write(f"• **Max:** {wait_stats['max']:.1f} minutes")
                st.write(f"• **Std Dev:** {wait_stats['std']:.1f} minutes")
    
    # Staff Performance
    elif current_page == "staff_performance":
        st.header("👥 Staff Performance")
        
        # Agent Efficiency Plot
        if 'Staff' in df_completed.columns and 'ServiceTimeMins' in df_completed.columns and 'id' in df_completed.columns:
            agent_efficiency = df_completed.groupby('Staff').agg({
                'ServiceTimeMins': 'mean',
                'id': 'count'
            }).rename(columns={'id': 'tickets_handled'})
            # Filter agents with at least x tickets for meaningful analysis
            agent_efficiency = agent_efficiency[agent_efficiency['tickets_handled'] >= 1]
            if not agent_efficiency.empty:
                agent_eff_plot = agent_efficiency.reset_index()
                agent_eff_plot['Agent Name'] = agent_eff_plot['Staff'].apply(lambda x: x.split('_')[-1] if '_' in x else x)
                fig = px.scatter(
                    agent_eff_plot,
                    x='ServiceTimeMins',
                    y='tickets_handled',
                    color='tickets_handled',
                    hover_data=['Agent Name', 'ServiceTimeMins', 'tickets_handled'],
                    title='Agent Efficiency: Service Time vs Tickets Handled',
                    labels={
                        'ServiceTimeMins': 'Average Service Time (Minutes)',
                        'tickets_handled': 'Tickets Handled'
                    },
                    color_continuous_scale='Viridis'
                )
                fig.update_traces(marker=dict(size=12, opacity=0.7), textposition='top center')
                fig.update_layout(
                    xaxis_title='Average Service Time (Minutes)',
                    yaxis_title='Tickets Handled',
                    title_font_size=16,
                    coloraxis_colorbar=dict(title='Tickets Handled')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if 'Staff' in df_completed.columns and len(df_completed) > 0:
            staff_metrics = df_completed.groupby('Staff').agg({
                'id': 'count',
                'ServiceTimeMins': 'mean'
            }).round(2)
            
            # Flatten column names
            staff_metrics.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in staff_metrics.columns]
            staff_metrics = staff_metrics.reset_index()
            # rename id to 'Tickets Created'
            staff_metrics.rename(columns={'id': 'Tickets'}, inplace=True)
            staff_metrics = staff_metrics.sort_values('Tickets', ascending=False)

            # Calculate average number of tickets created per hour per staff
            staff_hourly = df_completed.groupby(['Staff', 'hour']).size().reset_index(name='Tickets')
            avg_tickets_per_hour = staff_hourly.groupby('Staff')['Tickets'].mean().round(1).reset_index(name='Avg Tickets Per Hour')
            avg_service_time = df_completed.groupby('Staff')['ServiceTimeMins'].mean().round(1).reset_index(name='Average Service Time Mins')
            staff_metrics = pd.merge(staff_metrics, avg_tickets_per_hour, on='Staff', how='left')
            staff_metrics = pd.merge(staff_metrics, avg_service_time, on='Staff', how='left')

            col1, col2 = st.columns(2)
            
            with col1:
                # Top performers by volume
                fig_staff_volume = px.bar(
                    staff_metrics.head(15), 
                    x='Tickets', 
                    y='Staff',
                    orientation='h',
                    title="Top 15 Staff by Ticket Volume"
                )
                fig_staff_volume.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_staff_volume, use_container_width=True)
            
            with col2:
                # Staff Coverage vs Ticket Volume Analysis
                datetime_cols = ['Date', 'CreatedLocalTime', 'Wait time start', 'Completed']
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

                completed_df = df[df['Status'] == 'Completed'].copy() if 'Status' in df.columns else df.copy()

                staff_availability = completed_df.groupby(['DayOfWeek', 'hour'])['Staff'].nunique().reset_index()
                staff_availability.columns = ['DayOfWeek', 'hour', 'staff_count']

                ticket_demand = df_completed.groupby(['DayOfWeek', 'hour']).size().reset_index()
                ticket_demand.columns = ['DayOfWeek', 'hour', 'ticket_count']

                # Median wait time per (DayOfWeek, hour)
                if 'ActualWaitTimeMins' in completed_df.columns:
                    median_wait = completed_df.groupby(['DayOfWeek', 'hour'])['ActualWaitTimeMins'].median().reset_index()
                    median_wait.columns = ['DayOfWeek', 'hour', 'median_wait_time']
                else:
                    median_wait = None
                supply_demand = pd.merge(staff_availability, ticket_demand, on=['DayOfWeek', 'hour'], how='outer').fillna(0)
                if median_wait is not None:
                    supply_demand = pd.merge(supply_demand, median_wait, on=['DayOfWeek', 'hour'], how='left')
                else:
                    supply_demand['median_wait_time'] = np.nan
                supply_demand['tickets_per_staff'] = np.where(
                    supply_demand['staff_count'] > 0,
                    supply_demand['ticket_count'] / supply_demand['staff_count'],
                    np.nan
                )

                supply_demand = supply_demand.replace([np.inf, -np.inf], np.nan).dropna(subset=['tickets_per_staff'])
                fig_supply_demand = px.scatter(
                    supply_demand,
                    x='staff_count',
                    y='ticket_count',
                    color='tickets_per_staff',
                    color_continuous_scale='Reds',
                    size_max=18,
                    hover_data=['DayOfWeek', 'hour', 'staff_count', 'ticket_count', 'tickets_per_staff', 'median_wait_time'],
                    title='Staff Coverage vs Ticket Volume',
                    labels={
                        'staff_count': 'Staff Available',
                        'ticket_count': 'Tickets Created',
                        'tickets_per_staff': 'Tickets per Staff',
                        'median_wait_time': 'Median Wait Time (min)'
                    }
                )
                # Add trend line if enough data
                if len(supply_demand) > 1:
                    z = np.polyfit(supply_demand['staff_count'], supply_demand['ticket_count'], 1)
                    supply_demand['trend'] = z[0] * supply_demand['staff_count'] + z[1]
                    fig_supply_demand.add_traces([
                        go.Scatter(
                            x=supply_demand['staff_count'],
                            y=supply_demand['trend'],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Trend Line'
                        )
                    ])
                fig_supply_demand.update_layout(
                    xaxis_title='Staff Available',
                    yaxis_title='Tickets Created',
                    coloraxis_colorbar=dict(title='Tickets per Staff'),
                    height=500
                )
                st.plotly_chart(fig_supply_demand, use_container_width=True)
            
            # Staff performance table
            st.subheader("📊 Staff Performance Summary")
            display_cols = ['Staff', 'Tickets', 'Avg Tickets Per Hour', 'Average Service Time Mins']
            st.dataframe(staff_metrics[display_cols], use_container_width=True)

            # --- Staff Hourly Summary Table ---
            st.subheader("📊 Staff Hourly Summary (by Date)")
            def staff_hourly_summary(df, date):
                filtered = df[
                    (df['Date'] == date) &
                    (df['Status'] == 'Completed')
                ].copy()
                filtered['Completed'] = pd.to_datetime(filtered['Completed'])
                summary = filtered.groupby('Staff').agg(
                    total_tickets=('id', 'count'),
                    first_completed=('Completed', 'min'),
                    last_completed=('Completed', 'max'),
                ).reset_index()
                filtered['hour_completed'] = filtered['Completed'].dt.hour
                tickets_per_hour = filtered.groupby(['Staff', 'hour_completed']).size().unstack(fill_value=0)
                summary = summary.merge(tickets_per_hour, left_on='Staff', right_index=True, how='left')
                return summary

            # Streamlit control for date only
            available_dates = np.sort(df_completed['Date'].dt.strftime('%Y-%m-%d').unique())
            selected_date = st.selectbox("Select Date:", available_dates, index=0 if len(available_dates) > 0 else None)

            if selected_date:
                hourly_summary_df = staff_hourly_summary(df_completed, selected_date)
                if not hourly_summary_df.empty:
                    # Styling: color scale from red to green for only hour_completed columns
                    def color_scale(val, vmin, vmax):
                        # Discrete color categories
                        if val == 0:
                            return 'background-color: rgb(195,83,84)'
                        elif 1 <= val < 10:
                            return 'background-color: rgb(188,176,99)'
                        elif val >= 10:
                            return 'background-color: rgb(99,190,123)'
                        else:
                            return ''
                    # Find only hour_completed columns (int type)
                    style_cols = [col for col in hourly_summary_df.columns if isinstance(col, int)]
                    if style_cols:
                        vmin = hourly_summary_df[style_cols].min().min()
                        vmax = hourly_summary_df[style_cols].max().max()
                        styled_df = hourly_summary_df.style.applymap(lambda v: color_scale(v, vmin, vmax) if pd.notnull(v) and isinstance(v, (int, float)) else '', subset=style_cols)
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        st.dataframe(hourly_summary_df, use_container_width=True)
                else:
                    st.info("No completed tickets for the selected date.")
    
    # Service Analysis
    elif current_page == "service_analysis":
        st.header("📋 Service Analysis")
        
        # Location Performance Chart (added at the top)
        if 'Location Name' in df.columns and 'ActualWaitTimeMins' in df.columns and 'id' in df.columns:
            location_stats = df.groupby('Location Name').agg({
                'id': 'count',
                'ActualWaitTimeMins': 'mean'
            }).rename(columns={'id': 'ticket_count', 'ActualWaitTimeMins': 'avg_wait_time'}).reset_index()
            fig = px.scatter(
                location_stats,
                x='avg_wait_time',
                y='Location Name',
                size='ticket_count',
                color='avg_wait_time',
                color_continuous_scale='viridis',
                size_max=40,
                title='Location Performance: Wait Time vs Volume',
                labels={
                    'avg_wait_time': 'Average Wait Time (Minutes)',
                    'Location Name': 'Location',
                    'ticket_count': 'Ticket Volume'
                },
                hover_data=['ticket_count']
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                title_font_size=16,
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Initial Position vs Actual Wait Time")
            # Dropdown to select hue/category for color
            hue_options = ['Service Type']
            if 'Location Name' in df.columns:
                hue_options.append('Location Name')
            if 'Queue Name' in df.columns:
                hue_options.append('Queue Name')
            if 'Staff' in df.columns:
                hue_options.append('Staff')
            legend_cat = st.selectbox("Color by:", hue_options, index=0)

            # Prepare data for scatter plot
            dfo = df.copy()
            # Remove rows with missing values in required columns
            dfo = dfo.dropna(subset=['Initial Position', 'ActualWaitTimeMins', legend_cat])

            # Get location name for title
            location_name = location_filter if location_filter else "All Locations"

            fig = px.scatter(
                dfo,
                x='Initial Position',
                y='ActualWaitTimeMins',
                color=legend_cat,
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set1,
                title=f'Initial Position vs Actual Wait Time - {location_name} - Completed Tickets',
                labels={
                    'Initial Position': 'Initial Position in Queue',
                    'ActualWaitTimeMins': 'Actual Wait Time (Minutes)',
                    'Status': 'Status',
                    'Staff': 'Staff Member',
                    'id': 'Ticket ID'
                },
                hover_data=['id']
            )

            fig.update_layout(
                legend_title_text=legend_cat,
                legend=dict(x=1.05, y=1, bordercolor="Black", borderwidth=1),
                width=1200,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            service_metrics = df.groupby('Service Type').agg({
                'id': 'count',
                'ActualWaitTimeMins': 'mean' if 'ActualWaitTimeMins' in df.columns else 'count'
            }).round(2)
            service_metrics = service_metrics.reset_index().sort_values('id', ascending=False)
            # Rename columns
            service_metrics = service_metrics.rename(columns={
                'id': 'Count',
                'ActualWaitTimeMins': 'Average ActualWaitTimeMins'
            })
            
            # Service performance table
            st.subheader("📊 Service Performance Summary")
            st.dataframe(service_metrics.head(20), use_container_width=True)
        else:
            st.warning("Service Type data not available in the dataset.")

    elif current_page == "forcasting_staff":
        st.header("📈 Forcasting Staff")
        st.subheader("🔮 Find Optimal Staff Count")
        # Input fields for user selection
        day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        selected_day = st.selectbox("Select Day of Week:", day_options, index=0, width=200)
        # hour = st.number_input("Select Hour (0-23):", min_value=0, max_value=23, value=11, width=200)
        import streamlit as st

        # Create a list of hours from 8 AM to 11 PM
        hours = [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}" for h in range(8, 24)]

        # Dropdown for selecting hour
        hour = st.selectbox("Select Hour:", hours, index=3, width=200)  # Default to 11 AM

        # Convert selected label back to 24-hour format if needed
        selected_hour_24 = hours.index(hour) + 8

        sla_target_minutes = st.number_input("Target Average Waiting Time (minutes):", min_value=1, max_value=120, value=15, width=200)
        location_options = df_completed['Location Name'].unique().tolist() if 'Location Name' in df_completed.columns else []
        queue_options = df_completed['Queue Name'].unique().tolist() if 'Queue Name' in df_completed.columns else []

        selected_location = st.selectbox("Select Location:", location_options, index=0, width=400) if location_options else None
        selected_queue = st.selectbox("Select Queue:", queue_options, index=0, width=400) if queue_options else None
        if st.button("Calculate Optimal Staff Count"):
            result = find_optimal_staff_count(
            df_completed,
            location_name=selected_location,
            queue_name=selected_queue,
            day_of_week=selected_day,
            hour=selected_hour_24,
            sla_target_minutes=sla_target_minutes
            )
            if result is not None:
                st.write('')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**WaitWell Analytics Dashboard**")
st.sidebar.image("dallas-1.webp", use_container_width=False)