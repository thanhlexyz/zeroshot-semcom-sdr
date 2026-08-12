import copy

import torch

_tokenizer = None


# BEGIN: AI code
def _reparameterize_model(model):
    # fuse MobileOne multi-branch blocks into single conv (Apple MobileCLIP inference path)
    model = copy.deepcopy(model)
    # walk every submodule; open_clip MobileCLIP2 exposes .reparameterize() on fused blocks
    for module in model.modules():
        # skip modules that are not reparameterizable
        if hasattr(module, 'reparameterize'):
            # in-place fuse of that block
            module.reparameterize()
    return model
# END: AI code


def load(args):
    # load vision-language backbone; sets args.n_symbol from embed dim
    global _tokenizer
    if args.model == 'clip':
        import clip
        model, preprocess = clip.load(
            'ViT-B/32', device=args.device, download_root=args.model_dir,
        )
        _tokenizer = clip.tokenize
    elif args.model == 'mobileclip':
        import open_clip
        # MobileCLIP2-S0 (dfndr2b): fast on Pi; 512-d embedding
        model, _, preprocess = open_clip.create_model_and_transforms(
            'MobileCLIP2-S0', pretrained='dfndr2b',
            cache_dir=args.model_dir,
        )
        model = model.to(args.device)
        model.eval()
        # BEGIN: AI code
        # fuse MobileOne branches so Pi encode matches laptop speed (no ml-mobileclip dep)
        model = _reparameterize_model(model)
        # END: AI code
        _tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')
    else:
        raise ValueError(f'unknown model: {args.model}')
    model.eval()
    # embedding size → OFDM complex symbol count
    if hasattr(model.visual, 'output_dim'):
        dim = int(model.visual.output_dim)
    else:
        # MobileCLIP (timm trunk): probe one forward
        size = getattr(model.visual, 'image_size', 224)
        if isinstance(size, (tuple, list)):
            size = int(size[0])
        dummy = torch.zeros(1, 3, size, size, device=args.device)
        with torch.no_grad():
            dim = int(model.encode_image(dummy).shape[-1])
    assert dim % 2 == 0, f'embed dim must be even, got {dim}'
    args.n_symbol = dim // 2
    return model, preprocess


def tokenize(prompts, args):
    assert _tokenizer is not None, 'call semcom.model.load(args) first'
    tokens = _tokenizer(prompts)
    return tokens.to(args.device)


@torch.no_grad()
def encode_image(model, images):
    # image → L2-norm → complex OFDM data symbols
    feat = model.encode_image(images).float()
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return torch.complex(feat[..., 0::2].contiguous(), feat[..., 1::2].contiguous())


@torch.no_grad()
def encode_text(model, tokens):
    feat = model.encode_text(tokens).float()
    return feat / feat.norm(dim=-1, keepdim=True)


def predict(s_hat, text_feats):
    # complex symbols → unit-norm embedding → nearest text prompt
    s = torch.as_tensor(s_hat, dtype=torch.complex64).reshape(-1)
    feat = torch.empty(s.numel() * 2, dtype=torch.float32, device=s.device)
    feat[0::2] = s.real
    feat[1::2] = s.imag
    feat = feat / (feat.norm() + 1e-12)
    text_feats = text_feats.to(device=s.device, dtype=torch.float32)
    return int(torch.argmax(text_feats @ feat).item())
