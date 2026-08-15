from . import ofdm, link, transmitter, receiver, util


def create(args):
    if getattr(args, 'transceiver', 'software') == 'hardware':
        tx = transmitter.create(args)
        rx = receiver.create(args)
        return tx, rx
    ch = link.create(args)
    tx = transmitter.create(args, ch)
    rx = receiver.create(args, ch)
    return tx, rx
