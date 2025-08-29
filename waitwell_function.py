import pandas as pd
import numpy as np
import datetime
import math
import streamlit as st
import time

def calculate_avg_wait_time(df, hour, location_name='Online Services', queue_name='Success Coaching Queue', day_of_week=None):
    """
    Calculate wait time statistics for a specific date and hour.
    Calculates the average wait time by filtering the last 4 weeks of data for a specific hour, location, queue, and day of week.
    
    Parameters:
        df (DataFrame): The DataFrame containing ticket data.
        hour (int): The hour to filter (0-23).
        location_name (str): The name of the location to filter.
        queue_name (str): The name of the queue to filter.
    
    Returns:
        dict: A dictionary with calculated statistics.
    """
    # Only completed tickets
    filtered_df = df[df['Status'] == 'Completed'].copy()

    if location_name:
        filtered_df = filtered_df[filtered_df['Location Name'] == location_name]
    if queue_name:
        filtered_df = filtered_df[filtered_df['Queue Name'] == queue_name]
    if day_of_week:
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]

        # Find the last 4 weeks for this day
        recent_weeks = (
            filtered_df['WeekStart']
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(4)
        )
        filtered_df = filtered_df[filtered_df['WeekStart'].isin(recent_weeks)]

    if hour is not None:
        filtered_df = filtered_df[filtered_df['hour'] == hour]

    # get the date and actual wait time values and avg by date
    wait_time_data = filtered_df[['Date', 'ActualWaitTimeMins']].copy()
    wait_time_data['Date'] = wait_time_data['Date'].astype(str)
    wait_time_data = wait_time_data.groupby('Date').mean().reset_index()
    wait_time_data['ActualWaitTimeMins'] = wait_time_data['ActualWaitTimeMins'].round(1)

    wait_times = filtered_df['ActualWaitTimeMins'].dropna()
    avg_wait_time = wait_times.mean().round(0).astype(int)
    
    return avg_wait_time, wait_time_data

def calculate_service_rate(df, location_name=None, queue_name=None, day_of_week=None, hour=None):
    """
    Calculate the number of students a single staff member can serve per hour,
    using the last 4 weeks of the specified day of week.
    """
    # Only completed tickets
    filtered_df = df[df['Status'] == 'Completed'].copy()

    if location_name:
        filtered_df = filtered_df[filtered_df['Location Name'] == location_name]
    if queue_name:
        filtered_df = filtered_df[filtered_df['Queue Name'] == queue_name]
    if day_of_week:
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]

        # Find the last 4 weeks for this day
        recent_weeks = (
            filtered_df['WeekStart']
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(4)
        )
        filtered_df = filtered_df[filtered_df['WeekStart'].isin(recent_weeks)]

    if hour is not None:
        filtered_df = filtered_df[filtered_df['hour'] == hour]

    # Ensure 'ServiceTimeMins' is numeric
    filtered_df['ServiceTimeMins'] = pd.to_numeric(filtered_df['ServiceTimeMins'], errors='coerce')
    service_rate_date = filtered_df[['Date', 'ServiceTimeMins']].copy()
    service_rate_date['Date'] = service_rate_date['Date'].astype(str)
    service_rate_data = service_rate_date.groupby('Date').mean().reset_index()
    service_rate_data['service_rate'] = (60 / service_rate_data['ServiceTimeMins']).round(1)
    # drop service_time column
    service_rate_data = service_rate_data.drop(columns=['ServiceTimeMins'])

    service_times = filtered_df['ServiceTimeMins'].dropna()
    if len(service_times) == 0:
        return np.nan

    avg_service_time = service_times.mean()
    service_rate = (60 / avg_service_time).round(1)  # Convert minutes to hours

    return service_rate, service_rate_data

def calculate_arrival_rate(df, location_name=None, queue_name=None, day_of_week=None, hour=None):
    """
    Calculate the student arrival rate.
    λ = number of students entering a queue per unit time (per hour).

    Parameters:
        df (pd.DataFrame): DataFrame containing ticket data.
        location_name (str, optional): Filter by Location Name.
        queue_name (str, optional): Filter by Queue Name.
        date (str or datetime, optional): Filter by Date (YYYY-MM-DD).
        hour (int, optional): Filter by hour (0-23).

    Returns:
        float: Arrival rate (students per hour) for the specified parameters.
    """
    filtered_df = df.copy()
    # only completed tickets
    filtered_df = filtered_df[filtered_df['Status'] == 'Completed']

    if location_name:
        filtered_df = filtered_df[filtered_df['Location Name'] == location_name]
    if queue_name:
        filtered_df = filtered_df[filtered_df['Queue Name'] == queue_name]
    if day_of_week:
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]

        # Find the last 4 weeks for this day
        recent_weeks = (
            filtered_df['WeekStart']
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(4)
        )
        filtered_df = filtered_df[filtered_df['WeekStart'].isin(recent_weeks)]
        

    if hour is not None:
        filtered_df = filtered_df[filtered_df['hour'] == hour]

    # Group by Location Name, Queue Name, Date, hour and count tickets
    group = filtered_df.groupby(['Location Name', 'Queue Name', 'Date', 'hour']).size().reset_index(name='arrival_count')
    arrival_rate_data = group[['Date', 'arrival_count']].copy()
    arrival_rate_data['Date'] = arrival_rate_data['Date'].astype(str)
    if group.empty:
        return np.nan
    
    # Calculate average students per hour over the last 4 weeks
    avg_arrival_rate = group['arrival_count'].mean()

    return avg_arrival_rate, arrival_rate_data

def calculate_staff_count(df, location_name=None, queue_name=None, day_of_week=None, hour=None):
    """
    Calculate the number of staff members working in a specific queue at a specific hour.

    Parameters:
        df (pd.DataFrame): DataFrame containing ticket data.
        location_name (str, optional): Filter by Location Name.
        queue_name (str, optional): Filter by Queue Name.
        date (str or datetime, optional): Filter by Date (YYYY-MM-DD).
        hour (int, optional): Filter by hour (0-23).

    Returns:
        int or pd.DataFrame: Staff count for the specified parameters, or grouped DataFrame if not all parameters are provided.
    """
    filtered_df = df.copy()
    # only completed tickets
    filtered_df = filtered_df[filtered_df['Status'] == 'Completed']

    if location_name:
        filtered_df = filtered_df[filtered_df['Location Name'] == location_name]
    if queue_name:
        filtered_df = filtered_df[filtered_df['Queue Name'] == queue_name]
    if day_of_week:
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]

        # Find the last 4 weeks for this day
        recent_weeks = (
            filtered_df['WeekStart']
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(4)
        )
        filtered_df = filtered_df[filtered_df['WeekStart'].isin(recent_weeks)]

    if hour is not None:
        filtered_df = filtered_df[filtered_df['hour'] == hour]

    # Group by Location Name, Queue Name, Date, hour and count unique staff
    group = filtered_df.groupby(['Location Name', 'Queue Name', 'Date', 'hour'])['Staff'].nunique().reset_index(name='staff_count')
    staff_count_data = group[['Date', 'staff_count']].copy()
    staff_count_data['Date'] = staff_count_data['Date'].astype(str)

    if group.empty:
        return np.nan

    # Calculate average staff count per hour over the last 4 weeks
    staff_count = group['staff_count'].mean()

    return staff_count, staff_count_data

def calculate_wait_metrics(lambda_rate, mu_rate, c_servers):
    """
    Calculates key metrics for an M/M/c queue.
    Returns a dictionary with rho, Pw, Wq_minutes, and W80_minutes.
    """
    # Check for system stability
    rho = lambda_rate / (c_servers * mu_rate)
    if rho >= 1.0:
        return {
            "rho": rho, "Pw": 1.0, "Wq_minutes": float('200'), "status": "Unstable"
        }

    R = lambda_rate / mu_rate

    # Calculate the first part of the Pw denominator (the summation)
    sum_term = 0
    for n in range(c_servers):
        sum_term += (R**n) / math.factorial(n)

    # Calculate the second part of the Pw denominator
    second_term = (R**c_servers) / math.factorial(c_servers)

    # Calculate Pw (Probability of Waiting)
    erlang_c = second_term / (sum_term + second_term)
    Pw = erlang_c / (1 - rho)
    
    # Calculate average wait time in queue (Wq)
    Wq_hours = Pw * (1 / (c_servers * mu_rate - lambda_rate))
    Wq_minutes = Wq_hours * 60

    return {
        "rho": rho,
        "Pw": Pw,
        "Wq_minutes": Wq_minutes,
        "status": "Stable"
    }

# ultimate function to calculate the optimal staff count
def find_optimal_staff_count(df, location_name, queue_name, day_of_week, hour, sla_target_minutes):
    service_rate, service_rate_data = calculate_service_rate(df, location_name=location_name, queue_name=queue_name, day_of_week=day_of_week, hour=hour)
    arrival_rate, arrival_rate_data = calculate_arrival_rate(df, location_name=location_name, queue_name=queue_name, day_of_week=day_of_week, hour=hour)
    staff_count, staff_count_data = calculate_staff_count(df, location_name=location_name, queue_name=queue_name, day_of_week=day_of_week, hour=hour)
    lambda_rate = arrival_rate
    mu_rate = service_rate
    if pd.isna(lambda_rate) or pd.isna(mu_rate):
        return st.error("Insufficient data to calculate optimal staff count.")

    # --- 2. Finding the Optimal Staff Count ---
    # Set the SLA target
    sla_target_minutes = sla_target_minutes

    # Minimum staff needed to keep the system stable
    min_staff = math.ceil(lambda_rate / mu_rate) 

    hour_converted = f"{(hour % 12 or 12)} {'AM' if hour < 12 else 'PM'}"

    st.subheader(f"Finding Optimal Staff for {day_of_week} at {hour_converted} \n (Average Wait Time < {sla_target_minutes} mins)")
    with st.spinner("Analyzing historical data...", show_time=True):
        time.sleep(2)
        st.success(f"Minimum possible staff to handle load: {min_staff}")

    with st.status("Analyzing historical data...", expanded=False):
        time.sleep(1)
        col1, col2 = st.columns(2)
        with col1:
            avg_wait_time, wait_time_data = calculate_avg_wait_time(df, hour, location_name, queue_name, day_of_week)
            # create a bar chart for date and actual wait time
            wait_time_df = pd.DataFrame(wait_time_data)
            st.write(f"Average Wait Time during this hour: {avg_wait_time} minutes")
            st.bar_chart(wait_time_df.set_index("Date"), y_label="Average Wait Time (mins)", height=350, width=350, use_container_width=False)
        with col2:
            # create a bar chart for date and actual service time
            service_time_df = pd.DataFrame(service_rate_data)
            st.write(f"Number of staff members typically working during this hour: {staff_count:.0f}")
            st.bar_chart(staff_count_data.set_index("Date"), y_label="Staff Count", height=350, width=350, use_container_width=False)
        time.sleep(1)

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Single staff member typically handles an average of {service_rate:.0f} students during this hour.")
            st.bar_chart(service_rate_data.set_index("Date"), y_label="Service Rate (students)", height=350, width=350, use_container_width=False)
        with col2:
            st.write(f"Number of students typically entering this queue during this hour: {lambda_rate:.0f}")
            st.bar_chart(arrival_rate_data.set_index("Date"), y_label="Arrival Rate (students)", height=350, width=350, use_container_width=False)
        
    
    # Iterate from minimum staff upwards
    for c in range(min_staff, 50): 
        
        # Calculate metrics for the current staff count 'c'
        current_metrics = calculate_wait_metrics(lambda_rate, mu_rate, c)
        wt = current_metrics["Wq_minutes"]

        with st.spinner(f"Testing with {c} staff..."):
            time.sleep(1.5)
            st.write(f"Testing with {c} staff... Estimated Average Wait Time: {wt:.2f} minutes")

        # Check if the SLA is met
        if wt <= sla_target_minutes:
            st.success(f"The optimal number of staff for {day_of_week} at {hour_converted}: {c}")
            break