import torch
import torch.nn as nn


class Seq2SeqLSTM_v0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0.2):
        super(Seq2SeqLSTM_v0, self).__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_list):
        all_predictions = []
        num_nodes = graph_list[0].x.size(0)  # Number of reservoirs
        for node_idx in range(num_nodes):
            node_sequence = torch.stack([data.x[node_idx] for data in graph_list], dim=0)  # [seq_len, features]
            node_sequence = node_sequence.unsqueeze(0)  # [1, seq_len, features] - add batch dimension
            node_sequence = self.input_encoder(node_sequence)  # [1, seq_len, hidden_dim]
            encoder_outputs, (hidden, cell) = self.encoder(node_sequence)
            decoder_input = encoder_outputs[:, -1:, :]  # Shape: [1, 1, hidden_dim]
            decoder_outputs = []
            for _ in range(7):  # Predict 7 days
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
                out = self.relu(self.fc1(decoder_output))
                out = self.dropout(out)
                out = self.fc2(out)  # Shape: [1, 1, 1]
                decoder_outputs.append(out.squeeze(-1))  # Shape: [1, 1]
            # Concatenate predictions for this node
            node_predictions = torch.cat(decoder_outputs, dim=1)  # Shape: [1, 7]
            all_predictions.append(node_predictions.squeeze(0))  # Shape: [7]
        # Stack all node predictions
        predictions = torch.stack(all_predictions, dim=0)  # Shape: [nodes, 7]
        return predictions


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0.):
        super(Seq2SeqLSTM, self).__init__()
        self.input_encoder = nn.Sequential(                
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_list):
        # node_seq: [nodes, seq_len, feats] - process all nodes in parallel
        node_seq = torch.stack([g.x for g in graph_list], dim=1)
        
        # Apply input encoder
        node_seq = self.input_encoder(node_seq)  # [nodes, seq_len, hidden_dim]  
        
        # Encode all nodes at once
        encoder_outputs, (hidden, cell) = self.encoder(node_seq)  # h,c: [num_layers, nodes, hidden]  
        
        # Use last encoder output as decoder input
        decoder_input = encoder_outputs[:, -1:, :]  # Shape: [nodes, 1, hidden_dim]
        
        decoder_outputs = []
        for _ in range(7):  # Predict 7 days
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.relu(self.fc1(decoder_output))
            out = self.dropout(out)
            out = self.fc2(out)  # Shape: [nodes, 1, 1]
            decoder_outputs.append(out.squeeze(-1))  # Shape: [nodes, 1]
        
        # Concatenate all predictions
        predictions = torch.cat(decoder_outputs, dim=1)  # Shape: [nodes, 7]
        return predictions

# Autoregressive Decoding: Error propagation ->  Prediction for day 2 depends on day 1 prediction
class Seq2SeqLSTM_new(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.encoder = nn.LSTM(hidden_dim, hidden_dim,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)

        self.decoder = nn.LSTM(hidden_dim, hidden_dim,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, graph_list):
        # node_seq: [nodes, seq_len, feats]
        node_seq = torch.stack([g.x for g in graph_list], dim=1)
        node_seq = self.input_encoder(node_seq)  # [nodes, seq_len, hidden_dim]
        _, (h, c) = self.encoder(node_seq)  # h,c: [num_layers, nodes, hidden]
        dec_input = torch.zeros(node_seq.size(0), 1, h.size(-1),
                                device=node_seq.device)  # [nodes, 1, hidden]
        preds = []
        for _ in range(7):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            step = self.fc(dec_out.squeeze(1))      # [nodes, 1]
            preds.append(step)
            dec_input = dec_out

        return torch.cat(preds, dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 input_dim:  int,    # Original feature dimension
                 hidden_dim: int,    # Transformer hidden dimension (d_model)
                 pred_days:  int = 7,    # Number of prediction days
                 num_layers: int = 2,    # Number of Transformer stacked layers
                 n_heads:    int = 4,    # Number of multi-head attention heads
                 dropout:    float = 0.2,  # Dropout rate
                 input_days: int = 30):  # Input sequence length in days
        super().__init__()

        self.d_model = hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.ReLU()
        )

        self.pos_src = nn.Embedding(input_days, self.d_model)
        self.pos_tgt = nn.Embedding(pred_days, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True)

        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers)
        self.transformer_dec = nn.TransformerDecoder(
            dec_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(self.d_model, 1)

        self.sos = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pred_days = pred_days
    
    def forward(self, graph_list):
        # h_seq: [N, d_model]
        h_seq = [self.input_encoder(g.x) for g in graph_list]
        # src: [N, input_days, d_model]
        src = torch.stack(h_seq, dim=1)
        # self.pos_src.weight: [input_days, d_model] -> [None, input_days, d_model]
        src = src + self.pos_src.weight[None, :, :]
        # memory: [N, input_days, d_model]
        memory = self.transformer_enc(src)
        # tgt: [N, 1, d_model]
        tgt = self.sos.repeat(src.size(0), 1, 1)

        preds = []
        for t in range(self.pred_days):
            tgt_pe = tgt + self.pos_tgt.weight[:t+1][None, :, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=src.device)
            dec = self.transformer_dec(
                tgt=tgt_pe,
                memory=memory,
                tgt_mask=tgt_mask)
            step = self.fc_out(dec[:, -1])          # [N, 1]
            preds.append(step)
            step_embed = dec[:, -1:].detach()       # [N, 1, d_model]
            tgt = torch.cat([tgt, step_embed], dim=1)

        y_hat = torch.cat(preds, dim=-1)  # [N, pred_days]

        return y_hat
