from . import software
from . import hardware


def create(args, link=None):
    if getattr(args, 'transceiver', 'software') == 'hardware':
        return hardware.Transmitter(args.tx_uri, args)
    return software.Transmitter(args, link)
