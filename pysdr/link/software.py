import torch
import math


class Link:

    def __init__(self, args):
        self.args = args
        self.data = None

    def push(self, data):
        self.data = torch.as_tensor(data, dtype=torch.complex64).reshape(-1).clone()

    def pull(self):
        h = (torch.randn(()) + 1j * torch.randn(())) / math.sqrt(2.0)
        sigma = 10.0 ** (-0.5 * float(self.args.snr_db) / 10.0)
        n = self.data.numel()
        noise = (
            torch.randn(n) + 1j * torch.randn(n)
        ).to(torch.complex64) * (sigma / math.sqrt(2.0))
        return (h * self.data + noise).to(torch.complex64)
