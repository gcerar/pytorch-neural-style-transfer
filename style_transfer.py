import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim import optimizer
from torchvision import models
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm
from datetime import datetime as dt


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Gatys et al. variant?
STYLE_LAYERS_DEFAULT = {
    'conv1_1': 0.75,
    'conv2_1': 0.5,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2,
}

CONTENT_LAYERS_DEFAULT = ('conv5_2', )

CONTENT_WEIGHT = 8  # "alpha" in the literature (default: 8)
STYLE_WEIGHT = 70  # "beta" in the literature (default: 70)
TV_WEIGHT = 10 # (default: 10)

IMG_SIZE = 512
LEARNING_RATE = 0.004


class NormalizeInverse(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def get_features(image: Tensor, model:nn.Module, layers=None):
    if layers is None:
        layers = tuple(STYLE_LAYERS_DEFAULT) + CONTENT_LAYERS_DEFAULT

    features = {}
    block_num = 1
    conv_num = 0

    x = image

    for layer in model:
        x = layer(x)

        if isinstance(layer, nn.Conv2d):
            # produce layer name to find matching convolutions from the paper
            # and store their output for further processing.
            conv_num += 1
            name = f'conv{block_num}_{conv_num}'
            if name in layers:
                features[name] = x

        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            # In VGG, each block ends with max/avg pooling layer.
            block_num += 1
            conv_num = 0

        elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU)):
            pass

        else:
            raise Exception(f'Unknown layer: {layer}')

    return features


def gram_matrix(input: Tensor, normalize=False) -> Tensor:
    (b, ch, h, w) = input.size()

    # resise F_XL into \hat F_XL
    features = input.view(b * ch, h * w)

    # compute the gram product
    gram = torch.mm(features, features.t())

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    if normalize:
        #gram = gram.div(b * ch * h * w)
        gram /= input.nelement()  # equivalent to: gram = gram.div(b * ch * h * w)

    return gram


transform = T.Compose([
    # Smaller edge of the image will be matched to `IMG_SIZE`
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

inv_transform = T.Compose([
    NormalizeInverse(IMAGENET_MEAN, IMAGENET_STD),
])

inv_transform_preview = T.Compose([
    inv_transform,
    T.CenterCrop((IMG_SIZE, IMG_SIZE)),
])


def load_image(img_path: Union[str, Path]) -> Tensor:
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def content_loss_func(target_features, content_features):
    content_loss = 0.0
    for layer in content_features:
        target_feature = target_features[layer]
        content_feature = content_features[layer]

        content_layer_loss = F.mse_loss(target_feature, content_feature)
        content_loss += content_layer_loss

    return content_loss


def style_loss_func(target_features, style_features, precomputed_style_grams):
    style_loss = 0.0
    for layer in style_features:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)

        style_gram = precomputed_style_grams[layer]

        _, d, h, w = target_feature.shape

        layer_style_loss = STYLE_LAYERS_DEFAULT[layer] * F.mse_loss(target_gram, style_gram)
        style_loss += layer_style_loss / (d * h * w)

    return style_loss

def total_variance_loss_func(target):
    tv_loss = \
        F.l1_loss(target[:, :, :, :-1], target[:, :, :, 1:]) + \
        F.l1_loss(target[:, :, :-1, :], target[:, :, 1:, :])

    return tv_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', metavar='PATH', required=True)
    parser.add_argument('--style', metavar='PATH', required=True)
    parser.add_argument('--epochs', type=int, metavar='N', default=7000, help='number of train epochs (default: 7000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA acceleration')
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'lbfgs', 'sgd'], default='adam', help='select optimizer (default: adam)')
    parser.add_argument('--init', choices=['input', 'noise'], default='input', help='select start image (default: input)')
    #parser.add_argument('--output', '-o', help='image output')
    #parser.add_argument('--animation', action='store_true', default=False, help='intermediate images into animated GIF')
    #parser.add_argument('--no-normalization', action='save_true', default=False, help='disable intermediate image color normalization')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        # Allow CuDNN internal benchmarking for architecture-specific optimizations
        torch.backends.cudnn.benchmark = True

    # We will use frozen pretrained VGG neural network for feature extraction
    # In the original paper, authors have used VGG19 (without bn)
    model = models.vgg19(pretrained=True).features

    # Authors in the original paper suggested use of AvgPool instead of MaxPool for more pleasing results.
    # However changing the pooling also affects activation, so the input needs to be scaled (not implemented).
    #for i, layer in enumerate(model):
    #    if isinstance(layer, torch.nn.MaxPool2d):
    #        model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    model = model.eval().requires_grad_(False).to(device)

    # The "content" image on which we apply style
    content = load_image(args.input).to(device)

    # The "style" image from which we obtain style
    style = load_image(args.style).to(device)

    # The "target" image to store outcome
    if args.init == 'input':
        target = content.clone().requires_grad_(True).to(device)
    elif args.init == 'noise':
        target = torch.rand_like(content).requires_grad_(True).to(device)
    else:
        raise Exception(f'Init type "{args.init}" is not implemented!')

    # Precompute content features, style features, and style gram matrices.
    content_features = get_features(content, model, CONTENT_LAYERS_DEFAULT)
    style_features = get_features(style, model, STYLE_LAYERS_DEFAULT)

    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in style_features
    }


    if args.optimizer == 'lbfgs':
        # LBFGS optimizer has a bit different API from others where it uses closure()
        optimizer = optim.LBFGS([target], max_iter=args.epochs, line_search_fn='strong_wolfe')

        def closure():
            target_features = get_features(target, model)

            content_loss = content_loss_func(target_features, content_features)
            style_loss = style_loss_func(target_features, style_features, style_grams)
            tv_loss = total_variance_loss_func(target)

            total_loss = \
                CONTENT_WEIGHT * content_loss + \
                STYLE_WEIGHT * style_loss + \
                TV_WEIGHT * tv_loss

            if torch.is_grad_enabled():
                optimizer.zero_grad(set_to_none=True)

            if total_loss.requires_grad:
                total_loss.backward()

            return total_loss

        optimizer.step(closure)

    else:
        if args.optimizer == 'adam':
            optimizer = optim.Adam([target], lr=LEARNING_RATE)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD([target], lr=LEARNING_RATE)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW([target], lr=LEARNING_RATE)
        else:
            raise Exception(f'Use of optimizer "{args.optimizer}" not implemented!')


        pbar = tqdm(range(args.epochs))

        for _ in pbar:
            optimizer.zero_grad(set_to_none=True)

            target_features = get_features(target, model)

            content_loss = CONTENT_WEIGHT * content_loss_func(target_features, content_features)
            style_loss = STYLE_WEIGHT * style_loss_func(target_features, style_features, style_grams)
            tv_loss = TV_WEIGHT * total_variance_loss_func(target)

            total_loss = content_loss + style_loss + tv_loss

            total_loss.backward(retain_graph=True) # do we need `retain_graph=True`?
            optimizer.step()

            pbar.set_postfix_str(
                f'total_loss={total_loss.item():.2f} '
                f'content_loss={content_loss.item():.2f} '
                f'style_loss={style_loss.item():.2f} '
                f'tv_loss={tv_loss.item():.2f} '
            )


        #with torch.no_grad():
        #    target = torch.clamp(target, 0.0, 1.0)


    timestamp = dt.now().strftime('%Y%m%dT%H%M%S')

    # Store the outcome
    save_image(inv_transform(target.detach().squeeze().cpu()), f'./{timestamp}-output.jpg')

    # Store content + style + target image for impression
    save_image([
        inv_transform_preview(content.detach().squeeze().cpu()),
        inv_transform_preview(style.detach().squeeze().cpu()),
        inv_transform_preview(target.detach().squeeze().cpu()),
    ], f'./{timestamp}-transition.jpg')
