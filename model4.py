import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from embed import DataEmbedding
from utils import*
from MMD import mmd
from knn_sim import*


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__() 
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        for attn_layer in self.attn_layers:
            x, series = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list
    

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, series = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out, series


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.01, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        

class MLP(nn.Module):
    def __init__(self, hidden, dropout=0.005):
        super(MLP, self).__init__()
        encoders = []
        for i,j in zip(hidden[:-2],hidden[1:-1]):
            encoders.append(nn.Linear(in_features=i, out_features=j))
            encoders.append(nn.GELU())
            encoders.append(nn.Dropout(p=dropout))
        
        encoders.append(nn.Linear(hidden[-2], hidden[-1]))
        self.encoders = nn.Sequential(*encoders)  

    def forward(self, x):
        return self.encoders(x)
    

class Discriminator(nn.Module):
    def __init__(self, seq_len, d_model, num_class=2, discrim_hidden=[512, 128], dropout=0.005):
        super(Discriminator, self).__init__()
        self.hiddens = [seq_len*d_model] + discrim_hidden + [num_class]
        self.dropout = dropout
        self.projection = MLP(self.hiddens, self.dropout)
    
    def forward(self, x):
        B, L, d = x.shape
        output = self.projection(x.reshape((B, -1)))
        return output


class DiscriminatorAug(nn.Module):
    def __init__(self, seq_len, n_heads, d_model, num_class=2, discrim_hidden=[512, 128], dropout=0.005):
        super(DiscriminatorAug, self).__init__()
        self.hiddens = [seq_len*(d_model+n_heads*seq_len)] + discrim_hidden + [num_class]
        self.dropout = dropout
        self.projection = MLP(self.hiddens, self.dropout)
    
    def forward(self, x, atts):
        B, L, d = x.shape
        
        aug = torch.stack(atts, dim=0).squeeze(dim=0) 
        # print(aug.shape)
        aug = aug.permute(0,2,1,3).contiguous()
        aug = aug.reshape((B, L, -1)) 

        fea = torch.cat((x, aug), dim=-1) 
        output = self.projection(fea.reshape((B, -1)))
        return output


class Decoder(nn.Module):
    def __init__(self, d_model, c_out, decode_hidden=[512, 128], dropout=0.005):
        super(Decoder, self).__init__()
        self.hiddens = [d_model] + decode_hidden + [c_out]
        self.dropout = dropout
        self.projection = MLP(self.hiddens, self.dropout)
    
    def forward(self, x):
        B, L, d = x.shape
        output = self.projection(x)
        return output
    

class Model(nn.Module):
    def __init__(self, win_size, c_in, num_class, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                 discrim_hidden=[512, 128], decode_hidden=[512, 128], dropout=0.0, activation='gelu', output_attention=True, device='cpu'):
        super(Model, self).__init__()

        self.device = device
        self.num_class = num_class

        self.embedding = DataEmbedding(c_in, d_model, dropout)

        self.attention = FullAttention(attention_dropout=dropout, output_attention=output_attention)
        self.attention_layer = AttentionLayer(self.attention, d_model, n_heads)
        self.encoder_layer = EncoderLayer(self.attention_layer, d_model, d_ff, dropout=dropout, activation=activation)

        self.encoder = Encoder([self.encoder_layer for l in range(e_layers)], norm_layer=torch.nn.LayerNorm(d_model))

        self.discriminator2 = DiscriminatorAug(win_size, n_heads, d_model, num_class=num_class, discrim_hidden=discrim_hidden, dropout=dropout)
        self.decode = Decoder(d_model, c_in, decode_hidden=decode_hidden, dropout=dropout)
        self.soft_ce_loss = SoftCELoss()
        self.mse_loss = nn.MSELoss()
        self.soft_ce_loss_k = SoftCELoss_topk()
        self.cts_loss = KnnCtsLoss3_FNC()

    def forward(self, x):
        enc_input = self.embedding(x)
        enc_out, att_list = self.encoder(enc_input) # [B, h, L, L]*I
        y_hat = self.discriminator2(enc_out, att_list) # [B, num_class]
        x_hat = self.decode(enc_out) # [B, L, d]

        return y_hat, x_hat, att_list
    
    def predict(self, x):
        enc_input = self.embedding(x)
        enc_out, att_list = self.encoder(enc_input)
        y_hat = self.discriminator2(enc_out, att_list)
        y_hat_t = F.log_softmax(y_hat, dim=-1)
        x_hat = self.decode(enc_out) 

        return y_hat_t, x_hat, att_list
    
    def encode(self, x):
        enc_input = self.embedding(x)
        enc_out, att_list = self.encoder(enc_input) 
        return enc_out, att_list
     

    def update_warm(self, minibatch, opt, sch, beta=0.):
        inputs, labels, noi_labels = minibatch
        inputs = inputs.to(self.device)
        noi_labels = noi_labels.long().to(self.device)
        
        enc_out, att_list = self.encode(inputs)
        y_hat = self.discriminator2(enc_out, att_list) 
        x_hat = self.decode(enc_out) # [B, L, d]
        one_hot = torch.zeros(noi_labels.shape[0], self.num_class).to(self.device).scatter_(1, noi_labels.view(-1, 1), 1)
        disc_loss = self.soft_ce_loss(y_hat, one_hot)
        mse_loss = self.mse_loss(x_hat, inputs)
        loss = disc_loss + beta*mse_loss
        # print('ce_loss:', disc_loss)
        # print('mse_loss:', mse_loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'loss': loss.item()}
    
    def update_with_lc(self, minibatch, opt, sch, topk=1., sigma=5, temperature=0.1, lmbda=1.):
        inputs, noi_labels, lc = minibatch

        inputs = inputs.to(self.device)
        noi_labels = noi_labels.long().to(self.device)
        w_x = lc.to(self.device).view(-1,1)


        enc_out, att_list = self.encode(inputs)
        y_hat = self.discriminator2(enc_out, att_list) 

        y_hat_t = F.softmax(y_hat, dim=-1)
        y_hat_t = y_hat_t/y_hat_t.sum(dim=1, keepdim=True)
        y_hat_t = y_hat_t.detach()
        labels = torch.argmax(y_hat_t, dim=1)
        
        one_hot = torch.zeros(noi_labels.shape[0], self.num_class).to(inputs.device).scatter_(1, noi_labels.view(-1, 1), 1)
        targets = one_hot*w_x + y_hat_t*(1-w_x) # 矫正后标签
        targets = targets.detach()

        disc_loss = self.soft_ce_loss_k(y_hat, targets, top_k=topk)
        cts_loss = self.cts_loss(enc_out, labels, sigma, temperature)
        loss = disc_loss + lmbda*cts_loss
        # print(f"ce:{disc_loss.item()} cts:{cts_loss.item()}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'loss': loss.item()}
    





        



        