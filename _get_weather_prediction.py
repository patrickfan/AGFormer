import geopandas as gpd
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

from intp_latlon_from_clatlon import intp_clatlon_bilinear


reservoir_info = {
    "BSR": 1990,
    "CAU": 1999,
    "CRY": 1982,
    "DCR": 1987,
    "DIL": 1985,
    "ECH": 1982,
    "ECR": 1992,
    "FGR": 1982,
    "FON": 1990,
    "GMR": 1982,
    "HYR": 1999,
    "JOR": 1997,
    "JVR": 1996,
    "LCR": 1998,
    "LEM": 1982,
    "MCP": 1991,
    "MCR": 1998,
    "NAV": 1986,
    "PIN": 1990,
    "RFR": 1989,
    "RID": 1990,
    "ROC": 1982,
    "RUE": 1982,
    "SCO": 1996,
    "SJR": 1992,
    "STA": 1982,
    "STE": 1982,
    "TPR": 1982,
    "USR": 1991,
    "VAL": 1986,
}

# Extract reservoir coordinates
rsrs = reservoir_info.keys()
map = gpd.read_file("./data/map/ReservoirElevations.shp")
map = map[['Initials', 'Lat', 'Lon', 'RASTERVALU']]
map.loc[map['Initials'] == 'TAY', 'Initials'] = 'TPR'
map = map[map['Initials'].isin(rsrs)]
map.reset_index(drop=True, inplace=True)
map.columns = ['RSR', 'LAT', 'LON', 'ELEV']

# Create reservoir coordinates dictionary
reservoir_coords = {}
for _, row in map.iterrows():
    reservoir_coords[row['RSR']] = {'lat': row['LAT'], 'lon': row['LON']}
    
print(f"Extracted coordinates for {len(reservoir_coords)} reservoirs:")
for rsr, coords in reservoir_coords.items():
    print(f"{rsr}: lat={coords['lat']:.4f}, lon={coords['lon']:.4f}")

def is_leap_year(year):
    """Check if a year is a leap year"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def day_of_year_to_date(day_of_year, year):
    """Convert day of year to date string"""
    base_date = datetime(year, 1, 1)
    target_date = base_date + timedelta(days=int(day_of_year) - 1)
    return target_date.strftime('%Y-%m-%d')

def process_weather_data(years, reservoir_coords, data_path_template):
    """Process weather data for multiple years and reservoirs"""
    all_data = []
    
    for year in years:
        print(f"Processing year {year}...")
        year_data = []
        
        # Process all 20 files for this year (suffix 0-19)
        for suffix in range(20):
            data_file = data_path_template.format(year=year, suffix=suffix)
            
            if not os.path.exists(data_file):
                print(f"Warning: File {data_file} not found, skipping...")
                continue
                
            print(f"  Loading {data_file}...")
            data = np.load(data_file)
            
            # Get grid information
            ddeg = 0.25  
            lat = data["latitude"][0,0,:,0]
            nlon = int(360/ddeg)
            lon = np.zeros(nlon)
            for i in range(nlon):
                lon[i] = i * ddeg
            
            # Get time variables
            days_of_year = data['days_of_year']  
            hours_of_day = data['time_of_day']   
            precipitation = data['total_precipitation_24hr']  
            temperature = data['2m_temperature']  
            
            # Print shapes for debugging
            if suffix == 0:
                print(f"    Grid info - Lat range: [{lat.min():.2f}, {lat.max():.2f}], Lon range: [{lon.min():.2f}, {lon.max():.2f}]")
                print(f"    Grid shape: lat={len(lat)}, lon={len(lon)}")
                print(f"    Data shapes:")
                print(f"      days_of_year: {days_of_year.shape}")
                print(f"      time_of_day: {hours_of_day.shape}")  
                print(f"      precipitation: {precipitation.shape}")
                print(f"      temperature: {temperature.shape}")
                
                # Check if days_of_year and time_of_day are consistent across spatial dimensions
                print(f"    Checking data consistency...")
                day_sample = days_of_year[0, 0, :, :]
                time_sample = hours_of_day[0, 0, :, :]
                
                day_uniform = np.all(day_sample == day_sample[0, 0])
                time_uniform = np.all(time_sample == time_sample[0, 0])
                
                print(f"      days_of_year uniform across space: {day_uniform}")
                print(f"      time_of_day uniform across space: {time_uniform}")
                
                if day_uniform:
                    print(f"      Sample day_of_year value: {day_sample[0, 0]}")
                if time_uniform:
                    print(f"      Sample time_of_day value: {time_sample[0, 0]}")
            
            nt = temperature.shape[0]  # Number of time frames (438)
            
            # Process each reservoir  
            for rsr_name, coords in reservoir_coords.items():
                clat, clon = coords['lat'], coords['lon']
                
                # Extract data for this reservoir at all time points
                rsr_temp = np.zeros(nt)
                rsr_precip = np.zeros(nt)
                rsr_days = np.zeros(nt)
                rsr_hours = np.zeros(nt)
                
                for n in range(nt):
                    # Use interpolation for temperature and precipitation 
                    rsr_temp[n] = intp_clatlon_bilinear(clat, clon, lat, lon, temperature[n,0,:,:])
                    rsr_precip[n] = intp_clatlon_bilinear(clat, clon, lat, lon, precipitation[n,0,:,:])
                    
                    # For time variables, just take the value at [0,0] since they should be uniform
                    rsr_days[n] = days_of_year[n, 0, 0, 0]
                    rsr_hours[n] = hours_of_day[n, 0, 0, 0]
                
                # Store the data for this reservoir and file
                for n in range(nt):
                    year_data.append({
                        'reservoir': rsr_name,
                        'year': year,
                        'file_suffix': suffix,
                        'time_index': n,
                        'day_of_year': int(round(rsr_days[n])),
                        'hour_of_day': int(round(rsr_hours[n])),
                        'temperature': rsr_temp[n],
                        'precipitation': rsr_precip[n]
                    })
        
        all_data.extend(year_data)
        print(f"  Completed year {year}: {len(year_data)} records")
    
    return all_data

def aggregate_daily_data(raw_data):
    """Aggregate hourly data to daily averages"""
    df = pd.DataFrame(raw_data)
    
    # Group by reservoir, year, and day_of_year to calculate daily averages
    daily_data = df.groupby(['reservoir', 'year', 'day_of_year']).agg({
        'temperature': 'mean',
        'precipitation': 'mean'
    }).reset_index()
    
    # Generate date column
    daily_data['date'] = daily_data.apply(
        lambda row: day_of_year_to_date(row['day_of_year'], row['year']), axis=1
    )
    
    return daily_data

def check_missing_dates(df, years):
    """Check for missing dates in the dataset"""
    print("\nChecking for missing dates...")
    
    for year in years:
        expected_days = 366 if is_leap_year(year) else 365
        year_data = df[df['year'] == year]
        
        for reservoir in df['reservoir'].unique():
            reservoir_year_data = year_data[year_data['reservoir'] == reservoir]
            actual_days = len(reservoir_year_data)
            
            if actual_days != expected_days:
                print(f"Warning: {reservoir} in {year} has {actual_days} days, expected {expected_days}")
                missing_days = set(range(1, expected_days + 1)) - set(reservoir_year_data['day_of_year'])
                if missing_days:
                    print(f"  Missing days: {sorted(missing_days)}")
            else:
                print(f"{reservoir} in {year}: Complete ({actual_days} days)")

# Main processing
if __name__ == "__main__":
    # Years to process
    years_to_process = [2009, 2010, 2011]
    
    # Data path template
    data_path_template = '/lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/train/{year}_{suffix}.npz'
    
    # Process all weather data
    print("Starting weather data processing...")
    raw_data = process_weather_data(years_to_process, reservoir_coords, data_path_template)
    print(f"\nProcessed {len(raw_data)} total records")
    
    # Aggregate to daily data
    print("\nAggregating to daily averages...")
    daily_weather_data = aggregate_daily_data(raw_data)
    print(f"Generated {len(daily_weather_data)} daily records")
    
    # Check for missing dates
    check_missing_dates(daily_weather_data, years_to_process)
    
    # Create output directory
    output_dir = './data/weather'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to pickle
    output_file = os.path.join(output_dir, 'weather_data_2009_2011.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(daily_weather_data, f)
    
    print(f"\nData saved to {output_file}")
    print(f"Final dataset shape: {daily_weather_data.shape}")
    print("\nSample data:")
    print(daily_weather_data.head(10))
