import torch
import torch.nn as nn

from torch.nn import init
from utils import UpsampleBlock, CBAM, SimplifiedChannelAttention, MultiScaleBlock, SCBAM, BaseBlock


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        # padding参数设置为使得输入和输出维度相同
        self.padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 用于遗忘门的卷积操作
        self.conv_f = nn.Conv3d(self.input_dim + self.hidden_dim,
                                self.hidden_dim,
                                kernel_size,
                                padding=self.padding)
        # 用于输入门的卷积操作
        self.conv_i = nn.Conv3d(self.input_dim + self.hidden_dim,
                                self.hidden_dim,
                                kernel_size,
                                padding=self.padding)
        # 用于候选状态的卷积操作
        self.conv_c = nn.Conv3d(self.input_dim + self.hidden_dim,
                                self.hidden_dim,
                                kernel_size,
                                padding=self.padding)
        # 用于输出门的卷积操作
        self.conv_o = nn.Conv3d(self.input_dim + self.hidden_dim,
                                self.hidden_dim,
                                kernel_size,
                                padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 将输入和前一时刻的状态拼接
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 遗忘门
        f = torch.sigmoid(self.conv_f(combined))
        # 输入门
        i = torch.sigmoid(self.conv_i(combined))
        # 候选状态
        c = torch.tanh(self.conv_c(combined))
        # 新状态
        c_next = f * c_cur + i * c
        # 输出门
        o = torch.sigmoid(self.conv_o(combined))
        # 新状态的输出
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, volume_size):
        depth, height, width = volume_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv_i.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv_i.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.return_all_layers = return_all_layers
        # 创建多层ConvLSTMCell
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim[i], kernel_size))

    def forward(self, input_tensor, hidden_state=None):
        # 如果没有提供初始状态，就初始化为全0
        b, _, c, d, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (d, h, w))

        # 获取序列长度
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # 存储每一层的状态
        # hidden_states = []
        layer_output_list = []
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # 单步运行ConvLSTMCell，得到当前时间步的输出和新状态
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            # 更新
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            # hidden_states.append([h, c])

        output = layer_output
        if self.return_all_layers:
            output = layer_output_list

        return output

    def _init_hidden(self, batch_size, volume_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, volume_size))
        return init_states


class ConvLSTMVSR(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=4, kernel_size=3, num_layers=3, factor=4):
        super(ConvLSTMVSR, self).__init__()

        # 输入通道数
        self.input_dim = input_dim
        # 输出通道数
        self.output_dim = output_dim
        # 隐层通道数
        self.hidden_dim = [hidden_dim]
        for i in range(1, num_layers):
            self.hidden_dim.append(self.hidden_dim[-1] * 2)
        # 卷积核大小
        self.kernel_size = kernel_size
        # ConvLSTM层数
        self.num_layers = num_layers
        # 放大倍数
        self.factor = factor

        # 定义ConvLSTM模块
        self.conv_lstm = ConvLSTM(input_dim=self.input_dim,
                                  hidden_dim=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  num_layers=self.num_layers)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.hidden_dim[-1], out_channels=self.output_dim, kernel_size=1),
        )

        # 上采样模块
        self.upsample = UpsampleBlock(self.output_dim, factor, use_attn=True)

        self.out_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.output_dim, out_channels=self.output_dim * 2, kernel_size=3, padding=1),
            nn.Conv3d(in_channels=self.output_dim * 2, out_channels=self.output_dim * 4, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv3d(in_channels=self.output_dim * 4, out_channels=self.output_dim * 2, kernel_size=3, padding=1),
            nn.Conv3d(in_channels=self.output_dim * 2, out_channels=self.output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        batch_size, seq_length, input_dims, depth, height, width = x.size()

        # 初始化隐层状态
        hidden_state = None

        # 将卷积后的结果输入到ConvLSTM模块中
        lstm_output_list = self.conv_lstm(x, hidden_state)

        # 上采样操作
        high_list = []
        for i in range(seq_length):
            y = lstm_output_list[:, i, :, :, :, :]
            y = self.conv1(y)
            res = x[:, i, :, :, :, :]

            y = self.upsample(y + res)
            y = self.out_conv(y)
            high_list.append(y)
        high_res = torch.stack(high_list, dim=1)
        return high_res


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv3d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv3d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width).type(self.dtype)

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.reset_gate = nn.Conv3d(input_channels + hidden_channels,
                                    hidden_channels,
                                    kernel_size=kernel_size,
                                    padding=self.padding)
        self.update_gate = nn.Conv3d(input_channels + hidden_channels,
                                     hidden_channels,
                                     kernel_size=kernel_size,
                                     padding=self.padding)
        self.out_gate = nn.Conv3d(input_channels + hidden_channels,
                                  hidden_channels,
                                  kernel_size=kernel_size,
                                  padding=self.padding)
        self.init_weight()

    def forward(self, input_, prev_state):
        if prev_state is None:
            h_t = torch.zeros(input_.shape[0], self.hidden_channels, input_.shape[2], input_.shape[3], input_.shape[4],
                              device=self.out_gate.weight.device)
        else:
            h_t = prev_state

        stacked_inputs = torch.cat([input_, h_t], dim=1)

        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out = torch.tanh(self.out_gate(torch.cat([input_, reset * h_t], dim=1)))
        h_t = update * h_t + (1 - update) * out

        return h_t

    def init_weight(self):
        # init conv using xavier
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding, num_layers):
        super(Decoder, self).__init__()
        self.input_channels = input_channels


class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvGRU, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.convgru_cells = nn.ModuleList()

        for i in range(self.num_layers):
            cell = ConvGRUCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            self.convgru_cells.append(cell)

    def forward(self, input_seq, hidden_states=None):
        current_layer_input = input_seq

        for layer_idx in range(self.num_layers):
            h = hidden_states[layer_idx] if hidden_states is not None else None
            output_inner = []

            for t in range(input_seq.shape[1]):
                h = self.convgru_cells[layer_idx](input_=current_layer_input[:, t, :, :, :], prev_state=h)
                output_inner.append(h)

            current_layer_input = torch.stack(output_inner, dim=1)

        return current_layer_input


class ConvGRUVSR(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=[4], kernel_size=3, num_layers=3, block_num=3, factor=4,
                 use_attn=True):
        super(ConvGRUVSR, self).__init__()
        assert len(hidden_channels) == num_layers
        # Conv GRU 提取时序特征
        nf = hidden_channels[-1]
        self.block_num = block_num
        self.conv_gru = ConvGRU(input_channels, hidden_channels, kernel_size, num_layers)

        self.blocks = nn.ModuleList([BaseBlock(nf, use_attn) for _ in range(self.block_num)])
        # self.multi_scale_blocks = nn.ModuleList([MultiScaleBlock(nf) for _ in range(4)])
        # 上采样
        self.upsample = UpsampleBlock(nf, scale_factor=factor, use_attn=False)

        self.out_conv = nn.Conv3d(nf, input_channels, kernel_size=1)

    def forward(self, input_seq):
        seq_len = input_seq.size(1)
        gru_output = self.conv_gru(input_seq)
        output_seq = []
        for t in range(seq_len):
            x = gru_output[:, t, :, :, :, :] + input_seq[:, t, :, :, :, :]
            res = x
            for i in range(self.block_num):
                x = self.blocks[i](x) + res
                res = x
            y = self.upsample(x)
            y = self.out_conv(y)
            output_seq.append(y)
        return torch.stack(output_seq, dim=1)
