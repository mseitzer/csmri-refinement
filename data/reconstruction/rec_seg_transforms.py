"""
MR imaging reconstruction transforms for datasets with segmentation labels

Code adapted from Jo Schlemper
"""
import numpy as np
from torchvision.transforms import Compose, Lambda

import data.reconstruction.deep_med_lib.my_pytorch.myImageTransformations as myit


def _to_torch_tensor():
    return Lambda(lambda x: x.transpose((2, 0, 1)).astype(np.float32))


def train_transform(cs_params, image_size, downscale, augmentation=None,
                    random_crop_size=None, random_crop_bg_reject_ratio=1.0):
    scaled_image_size = image_size // downscale

    # Data augmentation
    if augmentation is not None:
        rot = augmentation.get('rotate', 0)
        shift = augmentation.get('shift', 0)
        zoom_range = augmentation.get('scale', (1, 1))

    # Transform just for input data (i.e. apply forward model)
    input_transform = Compose([
        myit.CenterCropInKspace(scaled_image_size),
        Lambda(lambda x: x / np.max(np.abs(x))),
        # Image is real, in (0, 1) range
        myit.Undersample(cs_params['sampling_scheme'],
                         (1, scaled_image_size, scaled_image_size),
                         cs_params['acceleration_factor'],
                         cs_params['variable_acceleration']),
        # Image is complex
    ])

    # Combined Transform for both input and segmentation label
    transforms = [myit.Merge(axis=-1)]
    if augmentation is not None:
        transforms += [myit.RandomTranslate(shift),
                       myit.RandomRotatePair(angle_range=(-rot, rot),
                                             axes=(0, 1), orders=[2, 0],
                                             nc=1, nk=1, mode='reflect'),
                       myit.RandomZoomPair(zoom=zoom_range,
                                           orders=[2, 0], nc=1)]
    transforms += [
        myit.Split([0, 1], [1, 2]),
        [input_transform, Lambda(lambda x: x[::downscale, ::downscale])]
    ]

    transforms += [[_to_torch_tensor(), _to_torch_tensor()]]

    return myit.EnhancedCompose(transforms)


def test_transform(cs_params, image_size, downscale, num_images=1):
    # Transform just for input data (i.e. apply forward model)
    input_transform = Compose([
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
    ])

    transform = myit.EnhancedCompose([
        [input_transform, Lambda(lambda x: x[::downscale, ::downscale])],
        [_to_torch_tensor(), _to_torch_tensor()]
    ])

    return transform


if __name__ == '__main__':
    # Run from main directory with python -m data.reconstruction.rec_seg_transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from data.reconstruction.scar_seg import (get_train_set, get_val_set)
    from utils.config import Configuration

    conf = Configuration()
    conf.input_mode = '2d'
    conf.dataset_mode = 'all'
    conf.downscale = 1
    conf.undersampling = {
        'sampling_scheme': "varden",
        'acceleration_factor': 8,
        'variable_acceleration': False
    }

    # TRAINING
    train_set = get_train_set(conf, '../../data')
    loader = DataLoader(dataset=train_set, num_workers=1,
                        batch_size=2, shuffle=True)

    # apply the transform
    plt.ion()
    f, ax_arr = plt.subplots(1, 6, figsize=(20, 5))
    for i, batch in enumerate(loader):
        input, label = batch

        # visualise the augmented input and padded target image (padded and
        # normalised) convert it to a numpy array and visualise it
        im_u = input.numpy()[0, 0] + 1j * input.numpy()[0, 1]
        k_u = input.numpy()[0, 2] + 1j * input.numpy()[0, 3]
        im_ku = np.fft.ifft2(np.fft.fftshift(k_u))
        mask = input.numpy()[0, 4]
        gnd = input.numpy()[0, 6] + 1j * input.numpy()[0, 7]

        ax_arr[0].imshow(abs(im_u), cmap='gray')
        ax_arr[1].imshow(abs(im_ku), cmap='gray')
        ax_arr[2].imshow(abs(mask), cmap='gray')
        ax_arr[3].imshow(abs(gnd), cmap='gray')
        ax_arr[4].imshow(label.numpy()[0])
        ax_arr[5].imshow((label.numpy()[0] + 1) * abs(gnd))
        plt.pause(1)
        for j in range(5):
            ax_arr[j].clear()

        if i > 20:
            break

    # VALIDATION
    validation_set = get_val_set(conf, '../data')
    loader = DataLoader(dataset=validation_set, num_workers=1,
                        batch_size=2, shuffle=True)

    # apply the transform
    plt.ion()
    f, ax_arr = plt.subplots(1, 6, figsize=(20, 5))
    for i, batch in enumerate(loader):
        input, label = batch
        # visualise the augmented input and padded target image (padded and
        # normalised) convert it to a numpy array and visualise it
        im_u = input.numpy()[0, 0] + 1j * input.numpy()[0, 1]
        k_u = input.numpy()[0, 2] + 1j * input.numpy()[0, 3]
        im_ku = np.fft.ifft2(np.fft.fftshift(k_u))
        mask = input.numpy()[0, 4]
        gnd = input.numpy()[0, 6] + 1j * input.numpy()[0, 7]

        ax_arr[0].imshow(abs(im_u), cmap='gray')
        ax_arr[1].imshow(abs(im_ku), cmap='gray')
        ax_arr[2].imshow(abs(mask), cmap='gray')
        ax_arr[3].imshow(abs(gnd), cmap='gray')
        ax_arr[4].imshow(label.numpy()[0])
        ax_arr[5].imshow((label.numpy()[0] + 1) * abs(gnd))

        plt.pause(1)
        for j in range(5):
            ax_arr[j].clear()

        if i > 20:
            break
