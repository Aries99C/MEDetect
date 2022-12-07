import streamlit as st
from detect.dataset import get_loader_segment
import torch
import torch.nn as nn
from detect.model import AnomalyDetect
import time
import numpy as np


class Solver(object):
    def __init__(self, input_size, num_epochs, batch_size, win_size, lr, ae_size, rnn_size, tcn_size):
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

        self.train_dataloader = get_loader_segment(batch_size, win_size, step=1, mode='train')
        self.valid_dataloader = get_loader_segment(batch_size, win_size, step=1,  mode='valid')
        self.test_dataloader = get_loader_segment(batch_size, win_size, step=1, mode='test')

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


def plot_loss(num_epochs):
    y = []
    for epoch in range(num_epochs):
        enc = np.load('./loss/epoch_{}.npy'.format(epoch))
        y_tmp = list(enc)
        y += y_tmp
    st.line_chart(y)


def app():
    # title
    st.title('Cleanits-MEDetect')
    st.subheader('Multiple Errors Detection for timeseries data with Cleanits')

    # hyper parameters
    st.sidebar.markdown('# Data Config')
    input_size = st.sidebar.number_input('Input size', value=25)

    st.sidebar.markdown('# Exp Config')
    num_epochs = st.sidebar.number_input('Num epochs', value=20)
    batch_size = st.sidebar.slider('Batch size', min_value=8, max_value=64)
    win_size = st.sidebar.slider('Sliding Window size', min_value=32, max_value=128)
    lr = float(st.sidebar.text_input('Learning Rate', 1e-4))

    st.sidebar.markdown('# Model Config')
    ae_size = st.sidebar.slider('AutoEncoder hidden size', min_value=32, max_value=256)
    rnn_size = st.sidebar.slider('RNN hidden size', min_value=32, max_value=256)
    tcn_size = st.sidebar.slider('TCN hidden size', min_value=32, max_value=256)

    # Start Training
    st.sidebar.markdown('# Start Training')
    start = st.sidebar.button('Train')

    if start:
        # Solver
        solver = Solver(input_size, num_epochs, batch_size, win_size, lr, ae_size, rnn_size, tcn_size)
        # Train model
        st.markdown('## Training...')
        solver.train()
        # Show training results
        st.markdown('## Training Results')
        plot_loss(num_epochs)
