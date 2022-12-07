import os.path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64
from adtk.detector import ThresholdAD, QuantileAD, GeneralizedESDTestAD


def function_plot(df, col, detect_model):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name='normal'))
    fig.add_trace(go.Scatter(x=df.index[df[detect_model]], y=df[df[detect_model]][col], mode='markers', name='anomaly'))
    fig.update_layout(showlegend=False, xaxis=dict(rangeslider=dict(visible=True)))
    return fig


def export(df):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown('### Download output CSV File')
    href = f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)


def app():
    # title
    st.title('Cleanits-MEDetect')
    st.subheader('Multiple Errors Detection for timeseries data with Cleanits')

    # upload function
    st.sidebar.markdown('# 1 Upload Dataset')
    upload_file = st.sidebar.file_uploader('Upload your data Here')
    if upload_file is None:
        st.sidebar.markdown('Status: **Not uploaded**')
    else:
        st.sidebar.markdown('Status: **Upload Success**')
        if upload_file.name.startswith('train'):
            train_df = pd.read_csv(upload_file)
            train_df.to_csv('./data/train.csv', index=False)
        elif upload_file.name.startswith('test'):
            test_df = pd.read_csv(upload_file)
            test_df.to_csv('./data/test.csv', index=False)

    # Simple Detection Models
    st.sidebar.markdown('# 2 Simple Model Selection')
    model_selected = st.sidebar.selectbox('Select a model for simple detection',
                                          ('QuantileAD',
                                           'ThresholdAD',
                                           'GeneralizedESDTestAD'))
    st.markdown('## ' + model_selected)
    # load data
    if os.path.exists('./data/test.csv'):
        df = pd.read_csv('./data/test.csv')
        df.index = pd.date_range(start='1/1/2010', periods=len(df))
        cols = df.columns[1:]
    else:
        df = pd.read_csv('./data/demo.csv', index_col='timestamp', parse_dates=True, squeeze=True)
        df = pd.DataFrame(df)
        cols = df.columns
    # select attribute
    col = st.selectbox('Select an attribute for detection', cols)
    series = df[col]

    if model_selected == 'ThresholdAD':
        # parameters
        col1, col2 = st.columns(2)
        with col1:
            ThresholdAD_low = float(st.text_input('low', value=60))
        with col2:
            ThresholdAD_high = float(st.text_input('high', value=100))
        threshold_ad = ThresholdAD(low=ThresholdAD_low, high=ThresholdAD_high)
        anomalies = threshold_ad.detect(series)
        df[model_selected] = anomalies.values
    elif model_selected == 'QuantileAD':
        # parameters
        col1, col2 = st.columns(2)
        with col1:
            QuantileAD_low = float(st.text_input('low', value=0.05))
        with col2:
            QuantileAD_high = float(st.text_input('high', value=0.95))
        quantile_ad = QuantileAD(low=QuantileAD_low, high=QuantileAD_high)
        anomalies = quantile_ad.fit_detect(series)
        df[model_selected] = anomalies.values
    elif model_selected == 'GeneralizedESDTestAD':
        # parameters
        GeneralizedESDTestAD_alpha = float(st.text_input('alpha', value=0.3))
        esd_ad = GeneralizedESDTestAD(alpha=GeneralizedESDTestAD_alpha)
        anomalies = esd_ad.fit_detect(series)
        df[model_selected] = anomalies.values

    # Plot
    st.markdown('## Data Plot')
    fig = function_plot(df, col, model_selected)
    st.plotly_chart(fig)

    # export anomalies
    st.markdown('## Export Anomalies')
    button_export = st.button('Export')
    if button_export:
        export(df)
