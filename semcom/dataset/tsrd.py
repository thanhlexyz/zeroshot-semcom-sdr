import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class TSRDDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        files = sorted(f for f in os.listdir(root) if f.endswith('.png'))
        samples = []
        targets = []
        for f in files:
            # clean split: {class}_{i}.png with class in 0..5
            class_id = int(f.split('_', 1)[0])
            samples.append(os.path.join(root, f))
            targets.append(class_id)
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


def create(args, transform=None):
    root = os.path.join(args.data_dir, 'tsrd', 'clean')
    if transform is None:
        transform = T.Compose([T.ToTensor()])
    dataset = TSRDDataset(root=root, transform=transform)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=os.cpu_count(),
    )


def create_pil(args):
    root = os.path.join(args.data_dir, 'tsrd', 'clean')
    return TSRDDataset(root=root, transform=None)


def get_labels():
    return [
        'speed limit is 30',
        'speed limit is 50',
        'speed limit is 60',
        'speed limit is 80',
        'pedestrian crossing',
        'do not enter',
    ]


def get_tokens(args):
    from semcom import model as semcom_model
    return semcom_model.tokenize(get_labels(), args)
