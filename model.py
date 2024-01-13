import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

class Multi_Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, batch_size, device, num_layers=5, dropout=0.3):
        super(Multi_Transformer, self).__init__()
        hidden_dim = 128
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, norm_first=True, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.input = nn.Linear(in_dim, hidden_dim, device=device)
        self.decoder = nn.Linear(hidden_dim, out_dim, device=device)
        
        #self.omni = nn.Linear(hidden_dim, 1, device=device)
        '''
        self.omni = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2, device=device),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1, device=device)
        )
        '''
        self.device = device
        self.batch_size = batch_size
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tec):
        output = F.relu(self.input(tec))
        mask = self._generate_square_subsequent_mask(output.size(1)).to(self.device)
        output = self.transformer_encoder(output, mask)
        tec_output = F.relu(self.decoder(output))
        #omni_output = F.relu(self.omni(output))
        return tec_output#, omni_output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device, num_layers=5, dropout=0.3):
        super(Transformer, self).__init__()
        hidden_dim = 128
        self.encoder = PositionalEncoding(d_model=hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, norm_first=True, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.input = nn.Linear(in_dim, hidden_dim, device=device)
        self.decoder = nn.Linear(hidden_dim, out_dim, device=device)
        self.device = device
        self.batch_size = batch_size
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.input.bias.data.zero_()
        self.input.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tec):
        output = self.input(tec)
        output = self.encoder(output)
        mask = self._generate_square_subsequent_mask(output.size(1)).to(self.device)
        output = self.transformer_encoder(output, mask)
        output = F.relu(self.decoder(output))
        return output

class TEC_LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device):
        super(TEC_LSTM, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.hidden_dim = 128
        self.fc_input = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.lstm1 =  nn.LSTM(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.lstm2 =  nn.LSTM(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, tec):
        h0 = torch.zeros(2, self.batch_size, self.hidden_dim//2).float().to(self.device)
        c0 = torch.zeros(2, self.batch_size, self.hidden_dim//2).float().to(self.device)
        output = self.fc_input(tec)
        output = self.relu(output)
        output = self.dropout(output)
        output, (hn, cn) = self.lstm1(output, (h0, c0))
        output = self.relu(output)
        output = self.dropout(output)
        output, (hn, cn) = self.lstm2(output, (hn, cn))
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_output(output)
        return output

class TEC_GRU(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device):
        super(TEC_GRU, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = 128
        self.fc_input = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.gru1 =  nn.GRU(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.gru2 =  nn.GRU(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, tec, times=None):
        output = self.relu(self.fc_input(tec))
        output = self.dropout(output)
        output, _ = self.gru1(output)
        output = self.relu(self.layer_norm(output))
        output = self.dropout(output)
        output, _ = self.gru2(output)
        output = self.relu(self.layer_norm(output))
        output = self.relu(self.fc_output(output))
        return output

class TEC_CNNGRU(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device):
        super(TEC_CNNGRU, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = 64
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,71), padding=0)  
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1,71), padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(1,71), padding=0)
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim//2)
        self.fc_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.gru1 =  nn.GRU(82, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.gru2 =  nn.GRU(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, tec):
        output = F.relu(self.bn1(self.conv1(tec)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.conv3(output))
        output = output.squeeze(1)
        output, _ = self.gru1(output)
        output = F.relu(self.ln(output))
        output, _ = self.gru2(output)
        output = F.relu(self.ln(output))
        output = F.relu(self.fc_output(output))
        return output

if __name__ == "__main__":
    from dataloader import TecDataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    window_sz = 4
    tmp_model = ConvLSTM_Encoder(channels=1, num_layers=1)#Transformer(72*window_sz, 72, 16, device)
    tmp_model = tmp_model.float().to(device)    
    tmpdata = TecDataset('txt/test/', data_type='dir', window_size=window_sz, to_sequence=True)
    tmpdataloader = DataLoader(tmpdata, batch_size = 16, shuffle = False)
    batch = next(iter(tmpdataloader))
    b_input, b_target = tuple(b.to(device) for b in batch[:2])
    b_information = batch[3].to(device)
    #tec_output, omni_output = tmp_model(torch.cat((b_input, b_information), 2))
    
    output = tmp_model(b_input) 
    print(b_input.size())   
    print(b_target.size())
    print(output.size())
    input()
    criterion = torch.nn.MSELoss()
    print(torch.sqrt(criterion(output, b_target.float().to(device))))
    #print(torch.sqrt(criterion(omni_output, omni.float().to(device))))

