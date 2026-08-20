"""
Wind Data Averager — Streamlit web app
Fetches hourly wind data from Open-Meteo with hourly detail as priority.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from io import StringIO

# ── Config ──
st.set_page_config(page_title="Wind Data Averager", layout="wide")
st.title("Wind Data Averager")

# ── Helper functions ──

def avg_wind_dir(group):
    """Average wind direction using vector mean."""
    rads = np.radians(group["wind_dir_deg"])
    mean_u = np.sin(rads).mean()
    mean_v = np.cos(rads).mean()
    return round(np.degrees(np.arctan2(mean_u, mean_v)) % 360, 1)


def deg_to_compass(deg):
    """Convert degrees to 16-point compass direction."""
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    return dirs[int((deg + 11.25) / 22.5) % 16]


def fmt_hour(h):
    """Format 0-23 hour as '10 PM' style."""
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


@st.cache_data(ttl=3600)
def fetch_openmeteo(lat, lon, start_date, end_date):
    """Fetch Open-Meteo data with retry logic for rate limits."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join([
                    "wind_speed_10m",
                    "wind_direction_10m",
                ]),
                "wind_speed_unit": "mph",
                "timezone": "America/Los_Angeles",
            }

            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            hourly = data["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "wind_speed_mph": hourly["wind_speed_10m"],
                "wind_dir_deg": hourly["wind_direction_10m"],
            })

            df["hour"] = df["datetime"].dt.hour
            df["date"] = df["datetime"].dt.date

            return df
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            raise


def process_data(df, start_hour, end_hour, location_name, start_date, end_date):
    """Process hourly data with hourly detail as priority."""
    
    if df.empty:
        return None, "No data to process."
    
    # Filter to time window
    if start_hour > end_hour:
        hour_mask = (df["hour"] >= start_hour) | (df["hour"] <= end_hour)
        early_hours = df["hour"] <= end_hour
    else:
        hour_mask = (df["hour"] >= start_hour) & (df["hour"] <= end_hour)
        early_hours = pd.Series(False, index=df.index)

    filtered = df[hour_mask].copy()

    if filtered.empty:
        return None, "No data found for the specified hours."

    filtered["night_of"] = filtered["date"] - pd.to_timedelta(
        early_hours[hour_mask].astype(int), unit="D"
    )

    # Wind direction
    filtered["wind_dir_rad"] = np.radians(filtered["wind_dir_deg"])
    filtered["wind_u"] = np.sin(filtered["wind_dir_rad"])
    filtered["wind_v"] = np.cos(filtered["wind_dir_rad"])

    grouped = filtered.groupby("night_of")

    # ── HOURLY DATA (Priority) ──
    # Wind speed as separate columns per hour
    speed_pivot = filtered.sort_values("datetime").pivot_table(
        index="night_of",
        columns="hour",
        values="wind_speed_mph",
        aggfunc="first"
    ).reset_index()
    
    speed_pivot.columns = [f"wind_speed_{col:02d}" if isinstance(col, int) else col 
                           for col in speed_pivot.columns]
    
    # Wind direction as separate columns per hour
    dir_pivot = filtered.sort_values("datetime").pivot_table(
        index="night_of",
        columns="hour",
        values="wind_dir_deg",
        aggfunc="first"
    ).reset_index()
    
    dir_pivot.columns = [f"wind_dir_{col:02d}" if isinstance(col, int) else col 
                         for col in dir_pivot.columns]
    
    # Compass direction for each hour
    compass_data = []
    for hour in sorted(filtered["hour"].unique()):
        hour_data = filtered[filtered["hour"] == hour]
        if not hour_data.empty:
            avg_dir = avg_wind_dir(hour_data)
            compass = deg_to_compass(avg_dir)
            compass_data.append({
                "night_of": None,  # Will be filled in merge
                f"wind_compass_{hour:02d}": compass
            })
    
    # ── NIGHTLY AVERAGES (Secondary) ──
    agg = grouped.agg(
        avg_wind_speed_mph=("wind_speed_mph", "mean"),
    ).reset_index()
    agg["avg_wind_speed_mph"] = agg["avg_wind_speed_mph"].round(1)
    
    # Vector-averaged wind direction
    wind_dirs = grouped.apply(avg_wind_dir).reset_index()
    wind_dirs.columns = ["night_of", "avg_wind_dir_deg"]
    agg = agg.merge(wind_dirs, on="night_of")
    agg["avg_wind_dir_compass"] = agg["avg_wind_dir_deg"].apply(deg_to_compass)
    
    # Merge hourly data
    agg = agg.merge(speed_pivot, on="night_of", how="left")
    agg = agg.merge(dir_pivot, on="night_of", how="left")
    
    # Build compass columns for display
    compass_dict = {}
    for hour in sorted(filtered["hour"].unique()):
        hour_data = filtered[filtered["hour"] == hour]
        if not hour_data.empty:
            avg_dir = avg_wind_dir(hour_data)
            compass = deg_to_compass(avg_dir)
            for night in agg["night_of"]:
                hour_vals = filtered[(filtered["night_of"] == night) & (filtered["hour"] == hour)]
                if not hour_vals.empty:
                    compass_dict[(night, hour)] = hour_vals["wind_dir_deg"].iloc[0]
    
    # Add compass direction from direction values
    for col in agg.columns:
        if col.startswith("wind_dir_"):
            hour_num = int(col.split("_")[-1])
            compass_col = f"wind_compass_{hour_num:02d}"
            agg[compass_col] = agg[col].apply(deg_to_compass)

    # Column order: nightly summary first, then hourly
    col_order = ["night_of", "avg_wind_speed_mph", "avg_wind_dir_deg", "avg_wind_dir_compass"]
    
    # Add hourly wind speed columns
    speed_cols = sorted([c for c in agg.columns if c.startswith("wind_speed_")])
    col_order.extend(speed_cols)
    
    # Add hourly wind direction columns with compass
    dir_cols = sorted([c for c in agg.columns if c.startswith("wind_dir_")])
    compass_cols = sorted([c for c in agg.columns if c.startswith("wind_compass_")])
    for dc, cc in zip(dir_cols, compass_cols):
        col_order.extend([dc, cc])
    
    agg = agg[[c for c in col_order if c in agg.columns]]

    # Summary stats
    fmt_s = fmt_hour(start_hour)
    fmt_e = fmt_hour(end_hour)
    avg_wind = agg["avg_wind_speed_mph"].mean()
    max_wind = agg["avg_wind_speed_mph"].max()
    min_wind = agg["avg_wind_speed_mph"].min()

    # Build CSV
    access_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_name = location_name.lower().replace(" ", "_").replace(",", "")
    date_s = start_date.replace("-", "_")
    date_e = end_date.replace("-", "_")
    filename = f"{safe_name}_wind_{date_s}_to_{date_e}.csv"

    csv_header = (
        f"# {location_name} wind summary\n"
        f"# Source: Open-Meteo archive API (open-meteo.com)\n"
        f"# Date range: {start_date} to {end_date}\n"
        f"# Hours averaged: {fmt_s} to {fmt_e} (America/Los_Angeles)\n"
        f"# Data accessed: {access_date}\n"
        f"# Average wind speed: {avg_wind:.1f} mph\n"
        f"# Max nightly avg: {max_wind:.1f} mph\n"
        f"# Min nightly avg: {min_wind:.1f} mph\n"
        f"# Wind direction averaged using vector (sin/cos) method\n"
        f"#\n"
    )

    csv_buffer = StringIO()
    csv_buffer.write(csv_header)
    agg.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    return {
        "agg": agg,
        "avg_wind": avg_wind,
        "max_wind": max_wind,
        "min_wind": min_wind,
        "fmt_s": fmt_s,
        "fmt_e": fmt_e,
        "csv_content": csv_content,
        "filename": filename,
    }, None


# ── Sidebar inputs ──
with st.sidebar:
    st.header("Settings")
    
    location_name = st.text_input("Location name", value="Madera CA")
    lat = st.number_input("Latitude", value=36.9613, format="%.4f")
    lon = st.number_input("Longitude", value=-120.0607, format="%.4f")
    start_date = st.date_input("Start date", value=datetime(2026, 4, 14))
    end_date = st.date_input("End date", value=datetime(2026, 5, 7))
    start_hour = st.slider("Start hour (0-23)", min_value=0, max_value=23, value=22)
    end_hour = st.slider("End hour (0-23)", min_value=0, max_value=23, value=5)

# ── Main ──
if st.button("Fetch Data", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            df = fetch_openmeteo(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            result, error = process_data(df, start_hour, end_hour, location_name, 
                                        start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"))
            
            if error:
                st.error(error)
            else:
                agg = result["agg"]
                
                # Summary stats
                st.subheader(f"Wind Data ({result['fmt_s']} – {result['fmt_e']}), {location_name}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Wind Speed", f"{result['avg_wind']:.1f} mph")
                col2.metric("Highest Nightly Avg", f"{result['max_wind']:.1f} mph")
                col3.metric("Lowest Nightly Avg", f"{result['min_wind']:.1f} mph")
                
                # Hourly data table (priority)
                st.subheader("Hourly Wind Speed (mph)")
                speed_cols = ["night_of"] + sorted([c for c in agg.columns if c.startswith("wind_speed_")])
                st.dataframe(agg[speed_cols], use_container_width=True, hide_index=True)
                
                st.subheader("Hourly Wind Direction (degrees & compass)")
                dir_compass_cols = ["night_of"]
                for col in sorted(agg.columns):
                    if col.startswith("wind_dir_"):
                        dir_compass_cols.append(col)
                        hour_num = col.split("_")[-1]
                        compass_col = f"wind_compass_{hour_num}"
                        if compass_col in agg.columns:
                            dir_compass_cols.append(compass_col)
                st.dataframe(agg[dir_compass_cols], use_container_width=True, hide_index=True)
                
                # Nightly summary
                st.subheader("Nightly Averages")
                nightly_cols = ["night_of", "avg_wind_speed_mph", "avg_wind_dir_deg", "avg_wind_dir_compass"]
                st.dataframe(agg[nightly_cols], use_container_width=True, hide_index=True)
                
                # Download button
                st.download_button(
                    label="Download CSV",
                    data=result["csv_content"],
                    file_name=result["filename"],
                    mime="text/csv",
                )
        
        except Exception as e:
            st.error(f"Error: {e}")

st.info("Data source: Open-Meteo (open-meteo.com) — Free historical weather data for any location")
