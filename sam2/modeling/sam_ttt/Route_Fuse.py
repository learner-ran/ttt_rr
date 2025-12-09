import torch
import torch.nn as nn

class routefuse(nn.Module):
    def __init__(self, dense_channels, sparse_channels, dense_kernel_size=3, sparse_kernel_size=3, dilation=2):
        super(routefuse, self).__init__()
        self.conv_dense = nn.Conv2d(dense_channels, dense_channels, kernel_size=dense_kernel_size, padding=dilation,
                                    dilation=dilation)

    def forward(self, dense_1, sparse_1, dense_2, sparse_2):
        # 使用空洞卷积层处理密集张量
        dense_1 = self.conv_dense(dense_1)
        dense_2 = self.conv_dense(dense_2)
        dense_product = dense_1 + dense_2

        sparse_product = sparse_1 + sparse_2

        return dense_product, sparse_product
