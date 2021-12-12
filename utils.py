import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
from regularizers.utils_reg import Params
from regularizers import models
from math import log10, pi
import numpy as np


def load_image_tensor(path):
    """
    return a tensor from an image path
    :param path: image path
    :return: image tensor of dimension (1,C,H,W)
    """
    img = Image.open(path)
    return transforms.ToTensor()(img).unsqueeze(0)


def crop_mod(x, m):
    """
    spatially crop input tensor x such as spatial shape of x are dividable by m
    :param x: tensor of dim (B,C,H,W)
    :return: cropped tensor
    """
    d = len(x.shape)
    assert d == 4, 'wrong dimensions, expecting a tensor of dimension 4, got {}'.format(d)
    _, _, H, W = x.shape
    return x[:, :, 0:H - H % m, 0:W - W % m]


def load_regularizer(name, root=None, device=torch.device('cpu')):
    """
    load a trained regularizer network'
    :param name: name of the directory where the model was saved. Must contains a 'params.json' file and 'checkpoint_last.pt' file
    :return: loaded regularizer
    """
    if root is not None:
        path = os.path.join(root, "regularizers", "models_zoo", name)
    else :
        path = os.path.join("regularizers", "models_zoo", name)
    # load model configuration file
    json_path = os.path.join(path, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    checkpoint = torch.load(os.path.join(path, 'checkpoint_last.pt'), map_location=device)
    regularizer = models.load_model(params, checkpoint)
    return regularizer


class WeightedL2(nn.Module):
    """
    compute the MSE loss between x and y, weighting pixel by the number of time they appear in the set of all
     the image patches of dim p x p (N_{i,j}).
     WeightedL2(x, y) = sum_(i,j) N_{i,j}(x_{i,j} - y_{i,j})^2 / p^2
     (centered pixel get more weight)
    """
    def __init__(self, p, x_shape, k=None, square=True):
        """
        return weighted data fidelity term ||k*x - y||^2
        param :
        p : (int) size of the receptive field of the critic discriminator
        x : shape of the input image x.shape
        k : linear degradation operator (default : Id)
        """
        super(WeightedL2, self).__init__()
        self.k = k
        p = int(p)
        # compute 2D weight matrix
        H, W = x_shape[-2:]
        C1 = torch.cat((torch.arange(1, p + 1, dtype=torch.float32),
                        p * torch.ones(H - 2 * p, dtype=torch.float32),
                        torch.arange(p, 0, -1, dtype=torch.float32)))
        L1 = torch.cat((torch.arange(1, p + 1, dtype=torch.float32),
                        p * torch.ones(W - 2 * p, dtype=torch.float32),
                        torch.arange(p, 0, -1, dtype=torch.float32)))
        self.N_2d = torch.ger(C1, L1)
        self.N_3d = self.N_2d.expand(x_shape)
        self.N_3d = torch.nn.Parameter(self.N_3d)
        self.square = square

    def forward(self, x, y):
        if self.k is not None:
            kx = self.k(x)
            l22 = torch.sum((self.N_3d * (self.k(x)-y)**2)) / torch.sum(self.N_2d)
            #return torch.mean((kx - y) ** 2)
        else:
            #k = Id
            l22 = torch.sum((self.N_3d * (x-y)**2)) / torch.sum(self.N_2d)
        if not self.square:
            return torch.sqrt(l22)
        else:
            return l22


def psnr_tensor(x, y):
    if not x.shape == y.shape:
        raise ValueError("image tensor must have the same dimensions")
    mse = torch.mean((x - y) ** 2)
    if 'Float' in x.type():
        res = 10 * (log10(1) - log10(mse))
    else:
        res = 10 * (log10(255 ** 2) - log10(mse))
    return res


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3, stride=1, device=None):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) /2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2. * pi * variance)) * torch.exp(-torch.sum((xy_grid - mean )**2., dim=-1) / (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_filter = np_kernel_to_conv(gaussian_kernel)
    return gaussian_filter


def load_np_kernel(i):
    path = os.path.join('kernels', '{}.npy'.format(i))
    assert i in range(1, 13), 'kernel index i must be in range [1, 12], got {}'.format(i)
    with open(path, 'rb') as f:
        k = np.load(f)
    #k64 = k32.astype(np.float64)
    return k


def np_kernel_to_conv(k, channels=3, stride=1, pad=False):
    """
    :param k: 2d numpy kernel of dimension (H,W), assume H==W:=p
           pad: If true, add replicate padding of len p//2
    :return: torch.nn.Conv2d module with spatial weight equal to k
    """
    H, W = k.shape
    assert H==W, 'numpy kernel shape (H,W) must be equals, got H={} and W={}'.format(H, W)
    # convert kernel to torch
    k = torch.tensor(k)
    k = k.view(1, 1, H, H)
    k = k.repeat(channels, 1, 1, 1)

    if pad:
        filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=H, stride=stride, groups=channels,
                                padding=H//2, padding_mode='replicate', bias=False)
    else:
        filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                 kernel_size=H, stride=stride, groups=channels,
                                 padding=0, bias=False)
    filter.weight.data = k
    filter.weight.requires_grad = False

    return filter


def load_conv_kernel(i, pad=False):
    np_kernel = load_np_kernel(i)
    conv_kernel = np_kernel_to_conv(np_kernel, pad=pad)
    return conv_kernel


class SubSampling(nn.Module):
    def __init__(self, dsf):
        """
        :param dsf: downsmampling factor (int)
        """
        super(SubSampling, self).__init__()
        self.dsf = dsf

    def forward(self, x):
        """
        :param x: input tensor of dimension (B,C,H,W)
        :return: spatially sub-sampled tensor of dim (B,C,h,w), with h=(H//dsf) and w=W//dsf
        """
        return x[:, :, 0::self.dsf, 0::self.dsf]


def upsample_torch(x, f):
    """
    spatially upsample an image tensor with bicubic interpolation
    :param x: tensor of dim (H,C,H,W)
    :param f: upsampling factor (int)
    :return: upsampled tensor of dimension (B, C, F*H, F*W)
    """
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    assert x.dim() == 4, 'argument x have 4 dimension, got {}'.format(x.dim())
    B, C, H, W = x.shape
    out = torch.empty(B, C, H*f, W*f)
    for i, u in enumerate(x):
        temp_pil = toPIL(u)
        up_pil = temp_pil.resize((W*f, H*f))
        up_torch = toTensor(up_pil)
        out[i] = up_torch
    return out


def extract_rgb_patches(tensor, patch_size, stride):
    """
    :param tensor: input tensor of shape [B,C,H,W]
    :return: tensor of RGB patches of shape [B, n_patches, C, patch_size, patch_size]
    """
    B, C, H, W = tensor.shape
    my_unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    flat_patches = my_unfold(tensor).permute(0, 2, 1) # shape = [B, n_patches, C * patch_size**2, n_patches]
    # fps = flat patch size, n = number of patches
    B, n, fps = flat_patches.shape
    patches = flat_patches.view(B, n, C, patch_size, patch_size)
    return patches
