"""
MR imaging reconstruction transforms

Code adapted from Jo Schlemper
"""
import torch
import numpy as np
from torchvision.transforms import Compose, Lambda

import data.reconstruction.deep_med_lib.my_pytorch.myImageTransformations as myit
from utils.tensor_transforms import complex_abs


def _to_torch_tensor():
    return Lambda(lambda x: x.transpose((2, 0, 1)).astype(np.float32))


def train_transform(cs_params, image_size, downscale, augmentation=None):
    scaled_image_size = image_size // downscale

    transforms = []

    # Data augmentation
    if augmentation is not None:
        alpha = augmentation.get('elastic_transform_alpha', None)
        sigma = augmentation.get('elastic_transform_sigma', None)
        if alpha is not None and sigma is not None:
            transforms.append(myit.ElasticTransform(alpha=alpha, sigma=sigma))

        shift = augmentation.get('shift', None)
        if shift is not None:
            transforms.append(myit.RandomTranslate(shift))

        rot = augmentation.get('rotate', None)
        if rot is not None:
            transforms.append(myit.RandomRotate(angle_range=(-rot, rot),
                                                axes=(0, 1),
                                                mode='reflect'))

        zoom_range = augmentation.get('scale', None)
        if zoom_range is not None:
            transforms.append(myit.RandomZoom(zoom=zoom_range))

    # Transform just for input data (i.e. apply forward model)
    transforms += [
        myit.CenterCropInKspace(scaled_image_size),
        Lambda(lambda x: x / np.max(np.abs(x))),
        # Image is real, in (0, 1) range
        myit.Undersample(cs_params['sampling_scheme'],
                         (1, scaled_image_size, scaled_image_size),
                         cs_params['acceleration_factor'],
                         cs_params['variable_acceleration']),
        # Image is complex
        _to_torch_tensor()
    ]

    return Compose(transforms)


def test_transform(cs_params, image_size, downscale, num_images=1):
    # Transform just for input data (i.e. apply forward model)
    transform = [
        myit.CenterCropInKspace(image_size // downscale),
        Lambda(lambda x: x / np.max(np.abs(x))),
        # Image is real, in (0, 1) range
        myit.Undersample(cs_params['sampling_scheme'],
                         (1, image_size // downscale, image_size // downscale),
                         cs_params['acceleration_factor'],
                         variable=False,
                         fixed_mask=True,
                         num_fixed_masks=num_images),
        # Image is complex
        _to_torch_tensor()
    ]

    return Compose(transform)


def output_transform():
  def transform(pred, target):
    pred = torch.clamp(complex_abs(pred), min=0.0, max=1.0)
    target = torch.clamp(complex_abs(target), min=0.0, max=1.0)
    return pred, target

  return transform


if __name__ == '__main__':
    # Run from main directory with python -m data.reconstruction.rec_transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from data.reconstruction.scar_seg import (get_train_set, get_val_set)
    from utils.config import Configuration

    conf = Configuration()
    conf.input_mode = '2d'
    conf.dataset_mode = 'reconstruction'
    conf.downscale = 1
    conf.undersampling = {
        'sampling_scheme': "varden",
        'acceleration_factor': 8,
        'variable_acceleration': False
    }
    conf.augmentation = {
        #'elastic_transform_sigma': 30,
        #'elastic_transform_alpha': 1000,
        'shift': (0, 10),
        'rotate': 10,
        'scale': (0.9, 1.1)
    }

    # TRAINING
    train_set = get_train_set(conf, '../../data')
    loader = DataLoader(dataset=train_set, num_workers=1,
                        batch_size=2, shuffle=True)

    # apply the transform
    plt.ion()
    f, ax_arr = plt.subplots(1, 4, figsize=(20, 5))
    for i, batch in enumerate(loader):
        inp = batch['inp'].numpy()
        kspace = batch['kspace'].numpy()
        mask = batch['mask'].numpy()
        target = batch['target'].numpy()

        # visualise the augmented input and padded target image (padded and
        # normalised) convert it to a numpy array and visualise it
        im_u = inp[0, 0] + 1j * inp[0, 1]
        k_u = kspace[0, 0] + 1j * kspace[0, 1]
        im_ku = np.fft.ifft2(np.fft.fftshift(k_u))
        mask = mask[0, 0]
        gnd = target[0, 0] + 1j * target[0, 1]

        ax_arr[0].imshow(abs(im_u), cmap='gray')
        ax_arr[1].imshow(abs(im_ku), cmap='gray')
        ax_arr[2].imshow(abs(mask), cmap='gray')
        ax_arr[3].imshow(abs(gnd), cmap='gray')
        plt.pause(1)
        for j in range(4):
            ax_arr[j].clear()

        if i > 20:
            break

    # VALIDATION
#    validation_set = get_val_set(conf, '../data')
#    loader = DataLoader(dataset=validation_set, num_workers=1,
#                        batch_size=2, shuffle=True)
#
#    # apply the transform
#    plt.ion()
#    f, ax_arr = plt.subplots(1, 4, figsize=(20, 5))
#    for i, batch in enumerate(loader):
#        inp = batch['inp'].numpy()
#        kspace = batch['kspace'].numpy()
#        mask = batch['mask'].numpy()
#        target = batch['target'].numpy()
#
#        # visualise the augmented input and padded target image (padded and
#        # normalised) convert it to a numpy array and visualise it
#        im_u = inp[0, 0] + 1j * inp[0, 1]
#        k_u = kspace[0, 0] + 1j * kspace[0, 1]
#        im_ku = np.fft.ifft2(np.fft.fftshift(k_u))
#        mask = mask[0, 0]
#        gnd = target[0, 0] + 1j * target[0, 1]
#
#        ax_arr[0].imshow(abs(im_u), cmap='gray')
#        ax_arr[1].imshow(abs(im_ku), cmap='gray')
#        ax_arr[2].imshow(abs(mask), cmap='gray')
#        ax_arr[3].imshow(abs(gnd), cmap='gray')
#
#        plt.pause(1)
#        for j in range(4):
#            ax_arr[j].clear()
#
#        if i > 20:
#            break
