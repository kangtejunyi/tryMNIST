import torch.nn
from config import nets, preset, acts

class NN(torch.nn.Module):
    r"""
    A cnn built on top of 2 convolution, 2 dropout, 2 fc
    """
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nets.net1(
            preset.n_input_conv1, 
            preset.n_output_conv1,
            preset.n_kernel_conv1,
            preset.n_stride_conv1,
        )
        self.conv2 = nets.net1(
            preset.n_input_conv2,
            preset.n_output_conv2,
            preset.n_kernel_conv2,
            preset.n_stride_conv2,
        )
        self.dropout1 = nets.net2(
            preset.p_dropout_1,
        )
        self.dropout2 = nets.net2(
            preset.p_dropout_2,
        )
        self.fc1 = nets.net3(
            preset.n_input_fc1,
            preset.n_output_fc1,
        )
        self.fc2 = nets.net3(
            preset.n_input_fc2,
            preset.n_output_fc2,
        )

    def forward(self, x):
        r"""
        make feed-forward structure as torchvision mnist example
        """
        x = self.conv1(x)
        x = acts.act1(x)
        x = self.conv2(x)
        x = acts.act1(x)
        x = acts.act2(x, 2)
        x = self.dropout1(x)
        x = acts.act0(x, 1)
        x = self.fc1(x)
        x = acts.act1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = acts.act99(x, dim=1)
        return output