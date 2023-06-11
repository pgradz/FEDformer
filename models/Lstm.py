import torch
import torch.nn as nn


# this code defines encoder decoder network with LSTM layers

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers=1, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(input_size = n_features, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
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
        
        self.lstm_decoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_input, encoder_hidden_states):
       
        lstm_out, self.hidden = self.lstm_decoder(x_input, encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, self.hidden


class EncoderDecoderWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, encoder, decoder_cell, output_size=12, teacher_forcing=0.3, sequence_len=96, decoder_input=True, device='cpu'):
        super().__init__()
        self.encoder = Encoder(n_features = input_size, hidden_size = hidden_size)
        self.decoder = Decoder(n_features = input_size, hidden_size = hidden_size)
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_length = sequence_len
        self.decoder_input = decoder_input
        self.device = device

    def forward(self, xb, yb=None):

        encoder_output, encoder_hidden = self.encoder(input_batch)



        if self.decoder_input:
            decoder_input = xb[-1]
            input_seq = xb[0]
            if len(xb) > 2:
                encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
            else:
                encoder_output, encoder_hidden = self.encoder(input_seq)
        else:
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        return outputs

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)