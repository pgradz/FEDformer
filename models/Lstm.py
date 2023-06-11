import torch
import torch.nn as nn


# this code defines encoder decoder network with LSTM layers

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers=1, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(input_size = n_features, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc):
        # x_enc : [batch_size, seq_len, n_features]

        outputs, (hidden, cell) = self.rnn(x_enc)
        # outputs = [batch_size, seq_len,  hid_dim ]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell



class Decoder(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers=1, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.lstm_decoder = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_input, encoder_hidden_states):
       
       # x_input : [batch_size, 1, n_features], n_features = 1
       # encoder_hidden_states is a tuple with hidden state and cell, both of dim: [n_layers * n_direction, batch_size, hidden_size] => [1, batch_size, hidden_size]

        lstm_out, self.hidden = self.lstm_decoder(x_input, encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, self.hidden


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_size = configs.input_size
        self.hidden_size = configs.hidden_size
        self.output_size = configs.pred_len
        self.batch_size = configs.batch_size

        self.encoder = Encoder(n_features = self.input_size, hidden_size = self.hidden_size)
        self.decoder = Decoder(n_features = self.input_size, hidden_size = self.hidden_size)


    def forward(self, x_enc, yb=None):
        '''
        x_enc : [batch_size, seq_len, n_features]
        yb : [batch_size, output_size] only needed for leaning with teacher forcing
        '''

        encoder_hidden = self.encoder(x_enc)

        outputs = torch.zeros(self.batch_size, self.output_size, 1) 
        decoder_input = x_enc[ :, -1, 3] # hard coded (3) for now column index. We take the last value of the target column of the encoder input
        decoder_input = decoder_input.unsqueeze(1).unsqueeze(1) # since above we take the last value of the target column, we need to add 2 dimensions to make it compatible with the decoder input
        decoder_hidden = encoder_hidden
        for i in range(self.output_size):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden) # decoder_hidden is tuple of hidden and cell
            outputs[:, i, :] = decoder_output.squeeze(1)
            decoder_input = decoder_output

        return outputs

