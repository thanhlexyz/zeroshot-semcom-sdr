from torch.utils.data import DataLoader

import torchvision.transforms as T
import torchvision
import os


def create(args, transform=None):
    root = os.path.join(args.data_dir, args.dataset)
    if transform is None:
        transform = T.Compose([T.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=False, transform=transform,
    )
    return DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=os.cpu_count(),
    )


def create_pil(args):
    root = os.path.join(args.data_dir, args.dataset)
    return torchvision.datasets.CIFAR10(
        root=root, train=False, download=False, transform=None,
    )


def get_labels():
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]


def get_tokens(args):
    from semcom import model as semcom_model
    prompts = [f'a photo of a {c}' for c in get_labels()]
    return semcom_model.tokenize(prompts, args)
