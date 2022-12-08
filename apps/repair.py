import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64


def repair_df(df, pred, alpha=0.2):
    repaired_df = pd.DataFrame()
    cols = df.columns
    for col in cols:
        series = df[col].values
        exp = [series[0]]
        for i in range(1, len(series)):
            exp.append(alpha * series[i] + (1- alpha) * series[i-1])
        combined = [series[i] if pred[i] == 0 else exp[0] for i in range(len(series))]
        repaired_df[col] = combined
    return repaired_df


def export(df):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown('### Download output CSV File')
    href = f'<a href="data:file/csv;base64,{b64}" download="repaired.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)


def app():
    # Initial session
    if 'repair' not in st.session_state:
        st.session_state['repair'] = False
    # title
    st.title('Cleanits-MEDetect')
    st.subheader('Multiple Errors Detection for timeseries data with Cleanits')

    # upload constraints
    st.sidebar.markdown('# 1 Upload Constraints')
    upload_file = st.sidebar.file_uploader('Upload your data Here')
    if upload_file is None:
        st.sidebar.markdown('Status: **Not uploaded**')
    else:
        st.sidebar.markdown('Status: **Upload Success**')
        with open('./data/constraints.txt', 'w') as w:
            data = str(upload_file.read(), 'utf-8')
            w.write(data)

    # Repair
    st.sidebar.markdown('# 2 Start Repairing')
    repair = st.sidebar.button('Repair')

    # Repair Results
    st.markdown('## Repair Results')
    if repair or st.session_state['repair']:
        if not st.session_state['repair']:
            df = pd.read_csv('./data/test.csv', index_col='timestamp_(min)')
            pred = np.load('./data/pred.npy')
            pred = pred.tolist()
            for i in range(len(pred), len(df)):
                pred.append(pred[-1])
            repaired_df = repair_df(df, pred)
            cols = repaired_df.columns
            cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[0:2])
            fig, ax = plt.subplots()
            for col in cols_selected:
                ax.plot(repaired_df.index, repaired_df[col])
            st.pyplot(fig)
            repaired_df.to_csv('./data/repaired.csv', index=False)
            st.session_state['repair'] = True
        else:
            repaired_df = pd.read_csv('./data/repaired.csv')
            cols = repaired_df.columns
            cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[0:2])
            fig, ax = plt.subplots()
            for col in cols_selected:
                ax.plot(repaired_df.index, repaired_df[col])
            st.pyplot(fig)

    # export Repair Results
    if st.session_state['repair']:
        st.markdown('## Export Repair Results')
        button_export = st.button('Export')
        if button_export:
            export(repaired_df)
