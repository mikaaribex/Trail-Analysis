import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    if 'speed_kmh' not in df_graph.columns and 'enhanced_speed' in df_graph.columns:
        df_graph['speed_kmh'] = df_graph['enhanced_speed'] * 3.6

    if 'slope_smooth' not in df_graph.columns or 'speed_kmh' not in df_graph.columns or 'cadence' not in df_graph.columns:
        return None

    fig = px.scatter(
        df_graph,
        x='slope_smooth',
        y='speed_kmh',
        color='cadence',
        color_continuous_scale='Spectral',
        labels={'slope_smooth': 'Slope (%)', 'speed_kmh': 'Speed (km/h)', 'cadence': 'Cadence (SPM)'},
        title='Slope vs Speed (colored by cadence)',
        opacity=0.7,
        height=600,
    )

    # Add transition zone shading (0-8%) as a rectangle
    fig.add_vrect(x0=0, x1=8, fillcolor='green', opacity=0.12, line_width=0)
    fig.update_layout(coloraxis_colorbar=dict(title='Cadence (SPM)'))
    return fig


def plot_effort_vs_terrain(df):
    df_eff = df.copy()
    if 'time_min' not in df_eff.columns:
        df_eff['time_min'] = (df_eff.index - df_eff.index[0]).total_seconds() / 60
    if 'efficiency_effort' not in df_eff.columns:
        df_eff['delta_alt'] = df_eff['enhanced_altitude'].diff()
        df_eff['gain_pos'] = df_eff['delta_alt'].clip(lower=0)
        real_speed_m_min = df_eff['enhanced_speed'] * 60
        effort_bonus_m_min = (df_eff['gain_pos'] * 10) * 60
        df_eff['effort_speed_m_min'] = real_speed_m_min + effort_bonus_m_min
        mask_active = (df_eff['heart_rate'] > 40) & (df_eff['effort_speed_m_min'] > 50)
        df_eff = df_eff[mask_active]
        df_eff['efficiency_effort'] = df_eff['effort_speed_m_min'] / df_eff['heart_rate']
        df_eff['ef_smooth'] = df_eff['efficiency_effort'].rolling(window=900, center=True).mean()

    if df_eff.empty or 'ef_smooth' not in df_eff.columns or 'enhanced_altitude' not in df_eff.columns:
        return None

    total_time = df_eff['time_min'].max()
    avg_first = df_eff.loc[df_eff['time_min'] <= 60, 'efficiency_effort'].mean()
    avg_last = df_eff.loc[df_eff['time_min'] >= (total_time - 60), 'efficiency_effort'].mean()
    loss_pct = (avg_first - avg_last) / avg_first * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Effort line on primary y
    fig.add_trace(go.Scatter(x=df_eff['time_min'], y=df_eff['ef_smooth'], mode='lines', name='Effort (Smoothed)', line=dict(color='#8e44ad', width=3)), secondary_y=False)
    # Altitude as filled area on secondary y
    fig.add_trace(go.Scatter(x=df_eff['time_min'], y=df_eff['enhanced_altitude'], fill='tozeroy', name='Altitude', line=dict(color='gray'), opacity=0.25), secondary_y=True)

    # Mean lines
    fig.add_shape(type='line', x0=0, x1=60, y0=avg_first, y1=avg_first, line=dict(color='#27ae60', dash='dash'), xref='x', yref='y')
    fig.add_shape(type='line', x0=total_time - 60, x1=total_time, y0=avg_last, y1=avg_last, line=dict(color='#c0392b', dash='dash'), xref='x', yref='y')

    fig.update_xaxes(title_text='Time (min)')
    fig.update_yaxes(title_text='Effort Meters / Beat', secondary_y=False)
    fig.update_yaxes(title_text='Altitude (m)', secondary_y=True)
    fig.update_layout(title_text=f"Effort Km & Terrain — Drift: -{loss_pct:.1f}%", height=500)
    return fig


def plot_heat_buildup(df):
    if 'heart_rate' not in df.columns or 'gap_speed' not in df.columns:
        return None
    time_peak = df['heart_rate'].idxmax()
    start_window = time_peak - pd.Timedelta(minutes=45)
    df_heat = df.loc[start_window:time_peak].copy()
    df_heat['rolling_corr'] = df_heat['gap_speed'].rolling(window='5min').corr(df_heat['heart_rate'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, specs=[[{}],[{}]])

    # Top: HR and GAP speed (secondary y)
    fig.add_trace(go.Scatter(x=df_heat.index, y=df_heat['heart_rate'], name='Heart Rate', line=dict(color='#e74c3c', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_heat.index, y=df_heat['gap_speed'] * 3.6, name='GAP Speed (km/h)', line=dict(color='#2980b9', width=2), opacity=0.6), row=1, col=1, secondary_y=True)

    # Bottom: rolling correlation
    fig.add_trace(go.Scatter(x=df_heat.index, y=df_heat['rolling_corr'], name='Rolling Corr (5min)', line=dict(color='purple', width=2)), row=2, col=1)
    # Fill negative correlation
    neg_mask = df_heat['rolling_corr'] < 0
    if neg_mask.any():
        fig.add_trace(go.Scatter(x=df_heat.index[neg_mask], y=df_heat['rolling_corr'][neg_mask], fill='tozeroy', name='Decoupling', line=dict(color='red'), opacity=0.3), row=2, col=1)

    fig.update_layout(height=700, title_text='Heat Stress Analysis: Last 45 Minutes')
    fig.update_yaxes(title_text='HR (bpm)', row=1, col=1)
    fig.update_yaxes(title_text='Corr', row=2, col=1)
    return fig


def plot_pace_heatmap(df):
    if 'gap_speed' not in df.columns or 'slope_smooth' not in df.columns:
        return None
    df_pace = df.copy()
    df_pace['gap_kmh'] = df_pace['gap_speed'] * 3.6
    df_pace = df_pace[df_pace['gap_kmh'] > 3]
    df_pace['gap_pace'] = 60 / df_pace['gap_kmh']
    df_pace['slope_bin'] = df_pace['slope_smooth'].round().astype(int)
    df_viz = df_pace[(df_pace['slope_bin'] >= -40) & (df_pace['slope_bin'] <= 40)]
    heatmap_data = df_viz.groupby('slope_bin')[['gap_pace']].median().T
    # Plotly heatmap
    z = heatmap_data.values
    x = heatmap_data.columns.astype(str)
    y = heatmap_data.index
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='RdYlGn_r', colorbar=dict(title='GAP Pace (min/km)')))
    fig.update_layout(title='GAP Pace by Gradient', height=250)
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
            st.plotly_chart(fig, use_container_width=True)
    elif analysis == 'Effort Km':
        fig = plot_effort_vs_terrain(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    elif analysis == 'Heat Buildup':
        fig = plot_heat_buildup(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    elif analysis == 'Pace Heatmap':
        fig = plot_pace_heatmap(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df.head(200))


if __name__ == '__main__':
    main()
