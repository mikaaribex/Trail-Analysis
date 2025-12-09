import io
import numpy as np
import pandas as pd
from fitparse import FitFile


def load_fit(file):
    """Load a FIT file (path or file-like) and return a cleaned DataFrame."""
    # Accept either a path or a file-like object (e.g., uploaded BytesIO)
    if hasattr(file, 'read'):
        fitfile = FitFile(file)
    else:
        fitfile = FitFile(str(file))

    data_points = []
    for record in fitfile.get_messages('record'):
        record_data = {}
        for entry in record:
            record_data[entry.name] = entry.value
        data_points.append(record_data)

    df = pd.DataFrame(data_points)

    # Convert positions
    if 'position_lat' in df.columns and 'position_long' in df.columns:
        conversion_factor = 180 / (2**31)
        df['latitude'] = df['position_lat'] * conversion_factor
        df['longitude'] = df['position_long'] * conversion_factor
        df.drop(['position_lat', 'position_long'], axis=1, inplace=True)

    # Timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Interpolate common columns if present
    cols_to_fix = [c for c in ['heart_rate', 'enhanced_speed', 'accumulated_power', 'power',
                               'enhanced_altitude', 'step_length', 'cadence', 'latitude', 'longitude'] if c in df.columns]
    for col in cols_to_fix:
        df[col] = df[col].interpolate(method='linear').ffill().bfill()

    # Feature engineering
    if 'enhanced_speed' in df.columns:
        df['speed_kmh'] = df['enhanced_speed'] * 3.6

    if 'enhanced_altitude' in df.columns:
        df['delta_alt'] = df['enhanced_altitude'].diff()

    if 'distance' in df.columns:
        df['delta_dist'] = df['distance'].diff()
        df['delta_dist'] = df['delta_dist'].replace(0, np.nan)
        df['slope_raw'] = (df['delta_alt'] / df['delta_dist']) * 100
        df['slope_smooth'] = df['slope_raw'].rolling(window=25, center=True).mean()
        df['slope_smooth'] = df['slope_smooth'].clip(-50, 30).fillna(0)
    
    return df


def calculate_energy_cost(slope_percent):
    g = slope_percent / 100
    cost = 155.4 * g**5 - 30.4 * g**4 - 43.3 * g**3 + 46.3 * g**2 + 19.5 * g + 3.6
    return cost / 3.6


def enrich_energy(df):
    if 'slope_smooth' in df.columns:
        df['energy_cost'] = df['slope_smooth'].apply(calculate_energy_cost)
    if 'enhanced_speed' in df.columns and 'energy_cost' in df.columns:
        df['gap_speed'] = df['enhanced_speed'] * df['energy_cost']


def ensure_cadence_spm(df):
    if 'cadence' in df.columns:
        if df['cadence'].median() < 100:
            df['cadence'] = df['cadence'] * 2


def preprocess_all(file):
    df = load_fit(file)
    enrich_energy(df)
    ensure_cadence_spm(df)
    return df
