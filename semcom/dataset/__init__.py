from . import cifar10, tsrd


def create(args, transform=None):
    if args.dataset == 'cifar10':
        return cifar10.create(args, transform=transform)
    if args.dataset == 'tsrd':
        return tsrd.create(args, transform=transform)
    raise ValueError(f'unknown dataset: {args.dataset}')


def create_pil(args):
    if args.dataset == 'cifar10':
        return cifar10.create_pil(args)
    if args.dataset == 'tsrd':
        return tsrd.create_pil(args)
    raise ValueError(f'unknown dataset: {args.dataset}')


def get_labels(args):
    if args.dataset == 'cifar10':
        return cifar10.get_labels()
    if args.dataset == 'tsrd':
        return tsrd.get_labels()
    raise ValueError(f'unknown dataset: {args.dataset}')


def get_tokens(args):
    if args.dataset == 'cifar10':
        return cifar10.get_tokens(args)
    if args.dataset == 'tsrd':
        return tsrd.get_tokens(args)
    raise ValueError(f'unknown dataset: {args.dataset}')
