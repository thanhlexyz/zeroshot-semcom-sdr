from . import software
from . import hardware


def create(args, link=None):
    if args.transceiver == 'hardware':
        return hardware.Receiver(args.rx_uri, args)
    return software.Receiver(args, link)
