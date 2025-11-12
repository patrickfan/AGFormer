import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=True)
        # self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=True)
        # self.dropout = nn.Dropout(dropout)
        # self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        # h = F.relu(self.conv1(x, edge_index, edge_weight))
        # h = self.dropout(h)
        # h = self.conv2(h, edge_index, edge_weight)
        # return h + self.residual(x)
        h = self.conv1(x, edge_index, edge_weight)
        return h


class GATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.2):
        """
        Two-layer GAT with skip connection.
        Attention weights can be retrieved by setting
        `return_attention=True` in forward().
        """
        super().__init__()
        # First hop
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True)                     # -> hidden_dim
        # Second hop
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=output_dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True)                     # -> output_dim
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None, return_attention=False):
        h1, att1 = self.conv1(x, edge_index, edge_weight,
                              return_attention_weights=True)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        h2, att2 = self.conv2(h1, edge_index, edge_weight,
                              return_attention_weights=True)
        out = h2 + self.residual(x)
        # out = h1 + self.residual(x)
        if return_attention:
            _, alpha1 = att1
            _, alpha2 = att2
            return out, (alpha1, alpha2)
            # return out, alpha1
        return out


class ReservoirNet(nn.Module):
    def __init__(self, in_dim, hid_dim, gnn_dim, lstm_dim, pred_days, dropout=0.4, gnn_type='gcn'):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU())
        
        if gnn_type == 'gcn':
            self.gnn = GCNLayer(hid_dim, gnn_dim, gnn_dim, dropout)
        elif gnn_type == 'gat':
            self.gnn = GATLayer(hid_dim, gnn_dim, gnn_dim, dropout=dropout)
        else:
            raise ValueError(f"Invalid GNN type: {gnn_type}")
        
        self.lstm = nn.LSTM(gnn_dim, lstm_dim, batch_first=True)
        self.decoder = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(lstm_dim, pred_days))

    def forward(self, graph_list):
        h_seq = []
        for data in graph_list:                    # 30 days
            x = self.encoder(data.x)               # [nodes, hid]
            h = self.gnn(x, data.edge_index)       # [nodes, gnn_dim]
            h_seq.append(h)
        h_seq = torch.stack(h_seq, dim=1)          # [nodes, 30 days, gnn_dim]
        out, _ = self.lstm(h_seq)                  # [nodes, 30 days, lstm_dim]
        pred = self.decoder(out[:, -1])            # [nodes, 7]
        return pred


class ReservoirNetSeq2Seq(nn.Module):
    def __init__(self, in_dim, hid_dim, gnn_dim, lstm_dim, pred_days, dropout=0.2, gnn_type='gcn'):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU())
        
        if gnn_type == 'gcn':
            self.gnn = GCNLayer(hid_dim, gnn_dim, gnn_dim, dropout)
        elif gnn_type == 'gat':
            self.gnn = GATLayer(hid_dim, gnn_dim, gnn_dim, dropout=dropout)
        else:
            raise ValueError(f"Invalid GNN type: {gnn_type}")
        
        self.enc_lstm = nn.LSTM(gnn_dim, lstm_dim, batch_first=True)
        self.dec_lstm = nn.LSTM(lstm_dim, lstm_dim, batch_first=True)
        self.out_fc   = nn.Linear(lstm_dim, 1)
        self.pred_days = pred_days
        self.register_buffer('sos', torch.zeros(1, 1, lstm_dim))

    def forward(self, graph_list):
        h_seq = []
        
        for data in graph_list:                         # 30 天
            x = self.encoder(data.x)                   # [nodes, hid_dim]
            h = self.gnn(x, data.edge_index)           # [nodes, gnn_dim]
            h_seq.append(h)
        
        h_seq = torch.stack(h_seq, dim=1)              # [nodes, 30, gnn_dim]
        _, (h, c) = self.enc_lstm(h_seq)               # h,c: [1, nodes, lstm_dim]
        dec_in = self.sos.repeat(h_seq.size(0), 1, 1)  # [nodes,1,lstm_dim]
        preds  = []
        
        for _ in range(self.pred_days):
            dec_out, (h, c) = self.dec_lstm(dec_in, (h, c))
            step = self.out_fc(dec_out.squeeze(1))     # [nodes,1]
            preds.append(step)
            dec_in = dec_out

        return torch.cat(preds, dim=-1)                # [nodes, 7]


class ReservoirAttentionNet(nn.Module):
    def __init__(self,
                 in_dim:     int,    # original feature dim
                 hid_dim:    int,    # MLP hidden
                 gnn_dim:    int,    # GAT hidden (= previous gnn_dim)
                 lstm_dim:   int,    # will be used as Transformer d_model
                 pred_days:  int,
                 dropout:    float = 0.2,
                 nheads:     int = 4,
                 n_layers:   int = 2):
        super().__init__()
        
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU())
        self.gnn = GATLayer(hid_dim, gnn_dim, gnn_dim, dropout=dropout)
        # ---- learned positional encoding for 30 days & 7 preds ----
        self.pos_src = nn.Embedding(30, lstm_dim)
        self.pos_tgt = nn.Embedding(pred_days, lstm_dim)
        # ---- Transformer encoder & decoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=lstm_dim,
            nhead=nheads,
            dim_feedforward=lstm_dim * 4,
            dropout=dropout,
            batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=lstm_dim,
            nhead=nheads,
            dim_feedforward=lstm_dim * 4,
            dropout=dropout,
            batch_first=True)
        
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers)
        self.transformer_dec = nn.TransformerDecoder(
            dec_layer, num_layers=n_layers)
        
        self.fc_out = nn.Linear(lstm_dim, 1)
        
        # learnable <SOS> token (shape 1×1×d_model)
        self.sos = nn.Parameter(torch.zeros(1, 1, lstm_dim))
        self.pred_days = pred_days
    
    def forward(self, graph_list, return_gat_att: bool = False):
        # ------- spatial + day-wise embedding -------
        h_seq, att_scores = [], []
        for g in graph_list:                               # 30 days
            x = self.encoder_mlp(g.x)                      # [N, hid_dim]
            if return_gat_att:
                h, att = self.gnn(x, g.edge_index,
                                  return_attention=True)
                att_scores.append(att)                     # list[(α1,α2), ...]
            else:
                h = self.gnn(x, g.edge_index)              # [N, gnn_dim]
            h_seq.append(h)
        # h_seq → [N, 30, gnn_dim]
        src = torch.stack(h_seq, dim=1)
        src = src + self.pos_src.weight[None]              # add pos encoding

        # ------- Transformer encoder -------
        memory = self.transformer_enc(src)                 # [N, 30, d_model]

        # ------- auto-regressive decoder -------
        # start with <SOS>
        tgt = self.sos.repeat(src.size(0), 1, 1)           # [N,1,d_model]
        preds = []
        for t in range(self.pred_days):
            tgt_pe = tgt + self.pos_tgt.weight[:t+1][None]
            # causal mask for t+1 length
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                t + 1, device=src.device)
            dec = self.transformer_dec(
                tgt_pe, memory, tgt_mask=tgt_mask)
            step = self.fc_out(dec[:, -1])                 # [N,1]
            preds.append(step)
            # project step to d_model and append to tgt
            step_embed = dec[:, -1:].detach()              # use decoder output
            tgt = torch.cat([tgt, step_embed], dim=1)

        y_hat = torch.cat(preds, dim=-1)                   # [N, pred_days]

        if return_gat_att:
            return y_hat, att_scores  # list of 30 tuples -> ((E1,H),(E2,H))
        return y_hat
