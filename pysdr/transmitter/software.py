import torch


class Transmitter:

    def __init__(self, args, link):
        self.args = args
        self.link = link

    def prepare(self, s):
        self.link.push(torch.as_tensor(s, dtype=torch.complex64).reshape(-1))

    def transmit(self):
        pass

    def stop(self):
        pass
