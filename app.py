import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from trail_utils import preprocess_all


st.set_page_config(page_title='Trail Analysis', layout='wide')


@st.cache_data
def _cached_preprocess(uploaded_file):
    # internal cached function that may raise; wrapper handles user-friendly errors
    return preprocess_all(uploaded_file)


def load_and_preprocess(uploaded_file):
    try:
        return _cached_preprocess(uploaded_file)
    except Exception as e:
        # Show a friendly error to the user in the app
        st.error(f"Failed to preprocess FIT file: {e}")
        return None


def plot_slope_speed_scatter(df):
    df_graph = df.copy()
    if 'speed_kmh' not in df_graph.columns:
        df_graph['speed_kmh'] = df_graph['enhanced_speed'] * 3.6

    fig, ax = plt.subplots(figsize=(10, 6))
    points = ax.scatter(
        x=df_graph['slope_smooth'],
        y=df_graph['speed_kmh'],
        c=df_graph['cadence'],
        cmap='Spectral_r',
        s=8,
        alpha=0.6,
        vmin=120,
        vmax=180,
    )
    ax.axvspan(0, 8, color='green', alpha=0.12)
    plt.colorbar(points, label='Cadence (SPM)')
    ax.set_title('Slope vs Speed (colored by cadence)')
    ax.set_xlabel('Slope (%)')
    ax.set_ylabel('Speed (km/h)')
    ax.grid(True, alpha=0.2)
    return fig


def plot_effort_vs_terrain(df):
    df_eff = df.copy()
    if 'time_min' not in df_eff.columns:
        df_eff['time_min'] = (df_eff.index - df_eff.index[0]).total_seconds() / 60
    if 'ef_smooth' not in df_eff.columns:
        df_eff['delta_alt'] = df_eff['enhanced_altitude'].diff()
        df_eff['gain_pos'] = df_eff['delta_alt'].clip(lower=0)
        real_speed_m_min = df_eff['enhanced_speed'] * 60
        effort_bonus_m_min = (df_eff['gain_pos'] * 10) * 60
        df_eff['effort_speed_m_min'] = real_speed_m_min + effort_bonus_m_min
        mask_active = (df_eff['heart_rate'] > 40) & (df_eff['effort_speed_m_min'] > 50)
        df_eff = df_eff[mask_active]
        df_eff['efficiency_effort'] = df_eff['effort_speed_m_min'] / df_eff['heart_rate']
        df_eff['ef_smooth'] = df_eff['efficiency_effort'].rolling(window=900, center=True).mean()

    total_time = df_eff['time_min'].max()
    avg_first = df_eff.loc[df_eff['time_min'] <= 60, 'efficiency_effort'].mean()
    avg_last = df_eff.loc[df_eff['time_min'] >= (total_time - 60), 'efficiency_effort'].mean()
    loss_pct = (avg_first - avg_last) / avg_first * 100

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax2.fill_between(df_eff['time_min'], df_eff['enhanced_altitude'], color='gray', alpha=0.2)
    sns.lineplot(data=df_eff, x='time_min', y='ef_smooth', ax=ax1, color='#8e44ad')
    ax1.hlines(avg_first, 0, 60, color='#27ae60', linestyle='--')
    ax1.hlines(avg_last, total_time - 60, total_time, color='#c0392b', linestyle='--')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Effort Meters / Beat')
    ax2.set_ylabel('Altitude (m)', color='gray')
    ax1.set_title(f"Effort Km & Terrain — Drift: -{loss_pct:.1f}%")
    return fig


def plot_heat_buildup(df):
    time_peak = df['heart_rate'].idxmax()
    start_window = time_peak - pd.Timedelta(minutes=45)
    df_heat = df.loc[start_window:time_peak].copy()
    df_heat['rolling_corr'] = df_heat['gap_speed'].rolling(window='5min').corr(df_heat['heart_rate'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(df_heat.index, df_heat['heart_rate'], color='#e74c3c', label='Heart Rate')
    ax1_speed = ax1.twinx()
    ax1_speed.plot(df_heat.index, df_heat['gap_speed'] * 3.6, color='#2980b9', alpha=0.6)
    ax2.plot(df_heat.index, df_heat['rolling_corr'], color='purple')
    ax2.fill_between(df_heat.index, df_heat['rolling_corr'], 0, where=(df_heat['rolling_corr'] < 0), color='red', alpha=0.3)
    ax2.axhline(0.8, color='green', linestyle=':')
    ax1.set_title('Heat Stress: Last 45 minutes')
    return fig


def plot_pace_heatmap(df):
    df_pace = df.copy()
    df_pace['gap_kmh'] = df_pace['gap_speed'] * 3.6
    df_pace = df_pace[df_pace['gap_kmh'] > 3]
    df_pace['gap_pace'] = 60 / df_pace['gap_kmh']
    df_pace['slope_bin'] = df_pace['slope_smooth'].round().astype(int)
    df_viz = df_pace[(df_pace['slope_bin'] >= -40) & (df_pace['slope_bin'] <= 40)]
    heatmap_data = df_viz.groupby('slope_bin')[['gap_pace']].median().T
    fig, ax = plt.subplots(figsize=(14, 2))
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=True, fmt='.1f', cbar_kws={'label': 'GAP Pace (min/km)'}, ax=ax)
    ax.set_title('GAP Pace by Gradient')
    ax.set_yticks([])
    return fig


def main():
    st.title('Trail Analysis — Interactive')
    st.sidebar.header('Controls')
    uploaded = st.sidebar.file_uploader('Upload a .fit file', type=['fit'])
    analysis = st.sidebar.selectbox('Analysis', ['Scatter Slope/Speed', 'Effort Km', 'Heat Buildup', 'Pace Heatmap', 'Data Preview'])

    if uploaded is None:
        st.info('Upload a FIT file on the left to get started.')
        return

    df = load_and_preprocess(uploaded)

    if df is None:
        st.info('No valid data available from the uploaded file.')
        return

    st.sidebar.markdown(f"Data points: **{len(df)}**")

    if analysis == 'Scatter Slope/Speed':
        fig = plot_slope_speed_scatter(df)
        if fig is not None:
            st.pyplot(fig)
    elif analysis == 'Effort Km':
        fig = plot_effort_vs_terrain(df)
        if fig is not None:
            st.pyplot(fig)
    elif analysis == 'Heat Buildup':
        fig = plot_heat_buildup(df)
        if fig is not None:
            st.pyplot(fig)
    elif analysis == 'Pace Heatmap':
        fig = plot_pace_heatmap(df)
        if fig is not None:
            st.pyplot(fig)
    else:
        st.dataframe(df.head(200))


if __name__ == '__main__':
    main()
