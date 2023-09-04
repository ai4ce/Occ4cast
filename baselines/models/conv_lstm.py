import torch.nn as nn
import torch
from .conv2d import Encoder, Decoder


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.input_conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.hiden_conv = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        ic = self.input_conv(input_tensor)
        hc = self.hiden_conv(h_cur)
        ic_i, ic_f, ic_o, ic_g = torch.split(ic, self.hidden_dim, dim=1)
        hc_i, hc_f, hc_o, hc_g = torch.split(hc, self.hidden_dim, dim=1)
        i = torch.sigmoid(ic_i + hc_i)
        f = torch.sigmoid(ic_f + hc_f)
        o = torch.sigmoid(ic_o + hc_o)
        g = torch.tanh(ic_g + hc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, p_pre, p_post, n_height):
        super(ConvLSTM, self).__init__()

        self.p_pre = p_pre
        self.p_post = p_post
        self.n_height = n_height

        _in_channels = self.n_height
        self.encoder = Encoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        _out_channels = self.n_height
        self.decoder = Decoder(self.encoder.out_channels, _out_channels)
        self.criterion = nn.BCEWithLogitsLoss()

        self.lstm_cell = (ConvLSTMCell(
            input_dim=self.encoder.out_channels,
            hidden_dim=self.encoder.out_channels,
            kernel_size=(3, 3),
            bias=True
        ))

    def forward(self, input_occ=None, gt_occ=None, invalid_mask=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        N, T_in, L, W, H = input_occ.shape
        assert T_in == self.p_pre, f"T_in {T_in} mismatch p_pre {self.p_pre}"

        input_occ = torch.movedim(input_occ, 4, 2).contiguous()
        feature = self.encoder(input_occ.reshape(-1, H, L, W))
        H_f, L_f, W_f = feature.shape[-3:]
        feature = feature.reshape(N, T_in, H_f, L_f, W_f)
        h, c = self.lstm_cell.init_hidden(N, (L_f, W_f))

        out_feature_list = []

        for t in range(self.p_pre):
            h, c = self.lstm_cell(feature[:, t, ...], cur_state=[h, c])
        
        out_feature_list.append(h)

        for t in range(self.p_post-1):
            h, c = self.lstm_cell(h, cur_state=[h, c])
            out_feature_list.append(h)

        out_feature = torch.stack(out_feature_list, dim=1)
        output = self.decoder(out_feature.reshape(-1, H_f, L_f, W_f)).reshape(N, self.p_post, H, L, W)
        output = torch.movedim(output, 2, 4)

        if self.training:
            valid_output = output[~invalid_mask]
            gt_occ[gt_occ > 0] = 1
            valid_gt = gt_occ[~invalid_mask]
            loss = self.criterion(valid_output, valid_gt.to(torch.float32))
            return loss
        else:
            return output