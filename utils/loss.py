import torch
import torch.nn as nn

#import

class ELocalLoss(nn.Module):
    def __init__(self, hamiltonian):
        super(self).__init__()
        self.hamiltonian = hamiltonian

    def forward(self, inputs):
        # 假设你的损失函数是输入和Hamiltonian的某种函数
        # 这里只是一个示例，你需要根据实际情况来定义损失计算
        loss = torch.sum(inputs - self.hamiltonian(inputs))
        return loss