import pandas as pd
import streamlit as st
from detect.dataset import get_loader_segment
import torch
import torch.nn as nn
from detect.model import AnomalyDetect
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class Solver(object):
    def __init__(self, input_size, num_epochs, batch_size, win_size, lr, ae_size, rnn_size, tcn_size, anomaly_ratio):
        self.optimizer = None
        self.model = None
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.win_size = win_size
        self.lr = lr
        self.ae_size = ae_size
        self.rnn_size = rnn_size
        self.tcn_size = tcn_size
        self.anomaly_ratio = anomaly_ratio

        self.train_dataloader = get_loader_segment(batch_size, win_size, step=1, mode='train')
        self.valid_dataloader = get_loader_segment(batch_size, win_size, step=1, mode='valid')
        self.test_dataloader = get_loader_segment(batch_size, win_size, step=1, mode='test')
        self.thre_dataloader = get_loader_segment(batch_size, win_size, step=1, mode='thre')

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyDetect(self.win_size, self.input_size, self.ae_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.model.cuda()

    def train(self):
        train_bar = st.progress(0)
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []

            self.model.train()
            for i, (input_data, _) in enumerate(self.train_dataloader):
                iter_count += 1
                x = input_data.float().to(self.device)
                x_recon = self.model(x)
                loss = self.criterion(x_recon, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(float(loss.item()))
            loss_np = np.array(loss_list)
            np.save('./loss/epoch_{}.npy'.format(epoch), loss_np)
            train_bar.progress((epoch + 1) / self.num_epochs)

        torch.save(self.model.state_dict(), './model.pth')

    def test(self):
        test_bar = st.progress(0)
        # load pretrained model
        self.model.load_state_dict(torch.load('./model.pth'))

        self.model.eval()
        criterion = nn.MSELoss(reduce=False)
        # compute training anomaly score
        anomaly_scores = []
        for i, (input_data, _) in enumerate(self.train_dataloader):
            x = input_data.float().to(self.device)
            x_recon = self.model(x)
            loss = torch.mean(criterion(x, x_recon), dim=-1)

            cri = loss
            cri = cri.detach().cpu().numpy()
            anomaly_scores.append(cri)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0).reshape(-1)
        train_scores = np.array(anomaly_scores)
        # compute threshold scores
        anomaly_scores = []
        for i, (input_data, _) in enumerate(self.thre_dataloader):
            x = input_data.float().to(self.device)
            x_recon = self.model(x)
            loss = torch.mean(criterion(x, x_recon), dim=-1)

            cri = loss
            cri = cri.detach().cpu().numpy()
            anomaly_scores.append(cri)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0).reshape(-1)
        test_scores = np.array(anomaly_scores)
        # compute threshold
        combined_scores = np.concatenate([train_scores, test_scores], axis=0)
        threshold = np.percentile(combined_scores, 100 - self.anomaly_ratio)

        # predict labels
        test_labels = []
        anomaly_scores = []
        for i, (input_data, labels) in enumerate(self.thre_dataloader):
            x = input_data.float().to(self.device)
            x_recon = self.model(x)
            loss = torch.mean(criterion(x, x_recon), dim=-1)
            cri = loss
            cri = cri.detach().cpu().numpy()
            anomaly_scores.append(cri)
            test_labels.append(labels)
            test_bar.progress((i + 1) / len(self.thre_dataloader))
        anomaly_scores = np.concatenate(anomaly_scores, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_scores = np.array(anomaly_scores)
        test_labels = np.array(test_labels)

        pred = (test_scores > threshold).astype(int)
        gt = test_labels.astype(int)
        # adjust
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        pred = np.array(pred)
        gt = np.array(gt)
        np.save('./data/pred.npy', pred)
        np.save('./data/anomaly_score.npy', test_scores)


def plot_loss(num_epochs):
    y = []
    for epoch in range(num_epochs):
        enc = np.load('./loss/epoch_{}.npy'.format(epoch))
        y_tmp = list(enc)
        y += y_tmp
    st.line_chart(y)


def app():
    # Initial session
    if 'train' not in st.session_state:
        st.session_state['train'] = False
    if 'test' not in st.session_state:
        st.session_state['test'] = False
    # title
    st.title('Cleanits-MEDetect')
    st.subheader('Multiple Errors Detection for timeseries data with Cleanits')

    # hyper parameters
    st.sidebar.markdown('# Data Config')
    input_size = st.sidebar.number_input('Input size', value=25)

    st.sidebar.markdown('# Exp Config')
    num_epochs = st.sidebar.number_input('Num epochs', value=20)
    batch_size = st.sidebar.select_slider('Batch size', [8, 16, 32, 64, 128])
    win_size = st.sidebar.select_slider('Sliding Window size', [32, 64, 128])
    lr = float(st.sidebar.text_input('Learning Rate', 1e-5))

    st.sidebar.markdown('# Model Config')
    ae_size = st.sidebar.select_slider('AutoEncoder hidden size', [8, 16, 32, 64])
    rnn_size = st.sidebar.select_slider('RNN hidden size', [8, 16, 32, 64])
    tcn_size = st.sidebar.select_slider('TCN hidden size', [8, 16, 32, 64])

    # Start Training
    st.sidebar.markdown('# Start Training')
    start_train = st.sidebar.button('Train')

    # Start Detection
    st.sidebar.markdown('# Start Detection')
    anomaly_ratio = int(st.sidebar.number_input('Anomaly Ratio', value=1))
    start_detect = st.sidebar.button('Detect')

    # Solver
    solver = Solver(input_size, num_epochs, batch_size, win_size, lr, ae_size, rnn_size, tcn_size, anomaly_ratio)

    # Show training results
    st.markdown('## Training Results')
    if start_train or st.session_state['train']:
        if not st.session_state['train']:
            # Train model
            st.markdown('### Training...')
            solver.train()
            plot_loss(num_epochs)
            st.session_state['train'] = True
        else:
            st.markdown('### Training Results')
            plot_loss(num_epochs)
    else:
        st.markdown('You can try different configuration for training.')

    # Show detect results
    st.markdown('## Detect Results')
    if start_detect or st.session_state['test']:
        if not st.session_state['test']:
            solver.test()
            df = pd.read_csv('./data/test.csv', index_col='timestamp_(min)')
            pred = np.load('./data/pred.npy')
            pred = pred.tolist()
            for i in range(len(pred), len(df)):
                pred.append(pred[-1])
            pred = np.array(pred)
            cols = df.columns
            cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[0:2])
            fig, ax = plt.subplots()
            for col in cols_selected:
                ax.plot(df.index, df[col])
            ax.fill_between(df.index, 0, 1, where=(pred == 1), color='red', alpha=0.1)
            st.pyplot(fig)
            st.session_state['test'] = True
        else:
            df = pd.read_csv('./data/test.csv', index_col='timestamp_(min)')
            pred = np.load('./data/pred.npy')
            pred = pred.tolist()
            for i in range(len(pred), len(df)):
                pred.append(pred[-1])
            pred = np.array(pred)
            cols = df.columns
            cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[0:2])
            fig, ax = plt.subplots()
            for col in cols_selected:
                ax.plot(df.index, df[col])
            ax.fill_between(df.index, 0, 1, where=(pred == 1), color='red', alpha=0.1)
            st.pyplot(fig)
    else:
        df = pd.read_csv('./data/test.csv', index_col='timestamp_(min)')
        cols = df.columns
        cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[0:2])
        fig = go.Figure()
        for col in cols_selected:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(showlegend=False, xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig)

    # show DataFrame
    st.markdown('## DataFrame')
    df = pd.read_csv('./data/test.csv', index_col='timestamp_(min)')
    left, right = st.slider('Dataframe Index: ', min_value=0, max_value=len(df)-1, value=(10000, 12000))
    if st.session_state['test']:
        scores = np.load('./data/anomaly_score.npy')
        scores = scores.tolist()
        for i in range(len(scores), len(df)):
            scores.append(scores[-1])
        scores = np.array(scores)
        df['scores'] = scores
        pred = np.load('./data/pred.npy')
        pred = pred.tolist()
        for i in range(len(pred), len(df)):
            pred.append(pred[-1])
        pred = np.array(pred)
        df['pred'] = pred
        cols = df.columns
        cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[-4:])
        df = df[cols_selected]
        st.dataframe(df.iloc[left:right, :].style.background_gradient(axis=0, gmap=df['scores'], cmap='YlOrRd'))
    else:
        cols = df.columns
        cols_selected = st.multiselect('Select series: ', options=cols, default=cols.tolist()[0:3])
        df = df[cols_selected]
        st.dataframe(df.iloc[left:right, :])
