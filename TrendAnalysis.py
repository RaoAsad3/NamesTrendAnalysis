import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Baby Name Trend Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv'
    df = pd.read_csv(url)
    df.columns = ['year', 'name', 'percent', 'sex']
    df_grouped = df.groupby(['year', 'name', 'sex']).sum().reset_index()
    return df_grouped

df_grouped = load_data()
available_names = sorted(df_grouped['name'].unique())

st.title("📈 Baby Name Trend Dashboard")
st.markdown("Explore trends and forecasts of baby names over the years using real-world data.")

# Sidebar Inputs
st.sidebar.header("🎛️ Filter Options")
selected_name = st.sidebar.selectbox("Select a Name", available_names, index=available_names.index('James'))
name_data = df_grouped[df_grouped['name'].str.lower() == selected_name.lower()]

# Suggest default gender
if not name_data.empty:
    suggested_gender = name_data.groupby('sex')['percent'].sum().idxmax()
else:
    suggested_gender = 'boy'

gender = st.sidebar.radio("Select Gender", ['boy', 'girl'], index=0 if suggested_gender == 'boy' else 1)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Name Trend & Forecast", 
    "📊 Total Name Usage", 
    "🏆 Top 10 Names by Year",
    "⚖️ Compare Multiple Names", 
    "📉 Gender-wise Forecast"
])

# ----------- TAB 1: Name Trend & Forecast -----------
with tab1:
    st.subheader(f"Trend of the Name '{selected_name}' Over the Years")
    fig_trend = px.line(name_data, x='year', y='percent', color='sex', markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    arima_data = name_data[name_data['sex'] == gender]
    ts = arima_data.set_index('year')['percent']

    st.subheader("🔮 ARIMA Forecast (Next 10 Years)")
    if len(ts) >= 3:
        model = ARIMA(ts, order=(1, 1, 1))
        results = model.fit()
        future_years = np.arange(ts.index.max() + 1, ts.index.max() + 11)
        forecast = results.get_forecast(steps=10)
        conf_int = forecast.conf_int()

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Actual'))
        fig_forecast.add_trace(go.Scatter(x=future_years, y=forecast.predicted_mean, mode='lines', name='Forecast', line=dict(color='red')))
        fig_forecast.add_trace(go.Scatter(x=future_years, y=conf_int.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False))
        fig_forecast.add_trace(go.Scatter(x=future_years, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', name='Confidence Interval', line=dict(width=0), fillcolor='rgba(255,0,0,0.1)'))
        fig_forecast.update_layout(xaxis_title='Year', yaxis_title='Frequency (%)')
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.download_button("📥 Download Forecast CSV", forecast.summary_frame().to_csv().encode(), f"{selected_name}_{gender}_forecast.csv", "text/csv")
    else:
        st.warning("Not enough data to generate forecast.")

# ----------- TAB 2: Total Name Usage -----------
with tab2:
    st.subheader("🧮 Total Baby Name Frequency Over Time")
    total_trend = df_grouped.groupby('year')['percent'].sum().reset_index()
    fig_total = px.line(total_trend, x='year', y='percent', title="Total Name Usage Over Time")
    st.plotly_chart(fig_total, use_container_width=True)

    ts_total = total_trend.set_index('year')['percent']
    if len(ts_total) >= 3:
        model_total = ARIMA(ts_total, order=(1, 1, 1)).fit()
        forecast_total = model_total.get_forecast(steps=10)
        future_years = np.arange(ts_total.index.max() + 1, ts_total.index.max() + 11)
        conf_int = forecast_total.conf_int()

        fig_forecast_total = go.Figure()
        fig_forecast_total.add_trace(go.Scatter(x=ts_total.index, y=ts_total, mode='lines', name='Actual'))
        fig_forecast_total.add_trace(go.Scatter(x=future_years, y=forecast_total.predicted_mean, mode='lines', name='Forecast', line=dict(color='orange')))
        fig_forecast_total.add_trace(go.Scatter(x=future_years, y=conf_int.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False))
        fig_forecast_total.add_trace(go.Scatter(x=future_years, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', name='Confidence Interval', line=dict(width=0), fillcolor='rgba(255,165,0,0.1)'))
        st.plotly_chart(fig_forecast_total, use_container_width=True)

# ----------- TAB 3: Top 10 Names by Year -----------
with tab3:
    st.subheader("🏅 Top 10 Baby Names")
    selected_year = st.slider("Select Year", min_value=int(df_grouped['year'].min()), max_value=int(df_grouped['year'].max()), value=2000)
    selected_gender = st.radio("Select Gender", ['boy', 'girl'], horizontal=True)

    top_10 = df_grouped[(df_grouped['year'] == selected_year) & (df_grouped['sex'] == selected_gender)]
    top_10 = top_10.sort_values('percent', ascending=False).head(10)
    fig_top10 = px.bar(top_10, x='name', y='percent', title=f"Top 10 Names in {selected_year} ({selected_gender})")
    st.plotly_chart(fig_top10, use_container_width=True)

# ----------- TAB 4: Compare Multiple Names -----------
with tab4:
    st.subheader("📈 Compare Multiple Names")
    compare_names = st.multiselect("Select Names to Compare", available_names, default=['James', 'Mary'])
    if compare_names:
        compare_data = df_grouped[df_grouped['name'].isin(compare_names)]
        fig_compare = px.line(compare_data, x='year', y='percent', color='name', title="Name Popularity Comparison")
        st.plotly_chart(fig_compare, use_container_width=True)

# ----------- TAB 5: Gender-wise Forecast -----------
with tab5:
    st.subheader("🔮 Forecast by Gender (Total Usage)")
    for gen in ['boy', 'girl']:
        st.markdown(f"#### Forecast for {gen.title()} Names")
        ts_gender = df_grouped[df_grouped['sex'] == gen].groupby('year')['percent'].sum()
        if len(ts_gender) >= 3:
            model = ARIMA(ts_gender, order=(1, 1, 1)).fit()
            forecast = model.get_forecast(steps=10)
            conf_int = forecast.conf_int()
            future_years = np.arange(ts_gender.index.max() + 1, ts_gender.index.max() + 11)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_gender.index, y=ts_gender, mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast.predicted_mean, mode='lines', name='Forecast', line=dict(color='blue' if gen == 'boy' else 'pink')))
            fig.add_trace(go.Scatter(x=future_years, y=conf_int.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_years, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', name='Confidence Interval', line=dict(width=0), fillcolor='rgba(0,0,255,0.1)' if gen == 'boy' else 'rgba(255,105,180,0.1)'))
            st.plotly_chart(fig, use_container_width=True)
