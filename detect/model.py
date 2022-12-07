import torch.nn as nn


class AnomalyDetect(nn.Module):
    def __init__(self, win_size, input_size, ae_size):
        super(AnomalyDetect, self).__init__()

        self.win_size = win_size
        self.input_size = input_size
        self.ae_size = ae_size

        self.encoder = nn.Sequential(
            nn.Linear(win_size * input_size, ae_size * 4),
            nn.ReLU(),
            nn.Linear(ae_size * 4, ae_size * 2),
            nn.ReLU(),
            nn.Linear(ae_size * 2, ae_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(ae_size, ae_size * 2),
            nn.ReLU(),
            nn.Linear(ae_size * 2, ae_size * 4),
            nn.ReLU(),
            nn.Linear(ae_size * 4, win_size * input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.win_size * self.input_size)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, self.win_size, self.input_size)
        return x
