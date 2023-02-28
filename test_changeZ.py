import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib

matplotlib.use('Agg')

# import ssl          # enable if downloading models gives CERTIFICATE_VERIFY_FAILED error
# ssl._create_default_https_context = ssl._create_unverified_context

import sys

sys.path.append('submodules')  # needed to make imports work in GAN_stability

from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, update_config, get_render_poses
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples
from graf.transforms import ImgToPatch

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config,
)

from external.colmap.filter_points import filter_ply

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--rotation_elevation', action='store_true', help='Generate videos with changing camera pose.')
    parser.add_argument('--shape_appearance', action='store_true',
                        help='Create grid image showing shape/appearance variation.')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model.')
    parser.add_argument('--reconstruction', action='store_true',
                        help='Generate images and run COLMAP for 3D reconstruction.')

    args, unknown = parser.parse_known_args()
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    if args.pretrained:
        config['expname'] = '%s_%s' % (config['data']['type'], config['data']['imsize'])
        out_dir = os.path.join(config['training']['outdir'], config['expname'] + '_from_pretrained')
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    fid_kid = int(args.fid_kid)

    config['training']['nworkers'] = 0

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)  # hwfr is [H,W,dset.focal,dset.radius]
    # in case of orthographic projection replace focal length by far-near
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far'] - config['data']['near'], config['data']['far'] - config['data']['near'])
        hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr  # add for building generator
    print(train_dataset, hwfr, render_poses.shape)

    val_dataset = train_dataset  # evaluate on training dataset for GANs
    if args.fid_kid:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=True, pin_memory=False, sampler=None, drop_last=False  # enable shuffle for fid/kid computation
        )

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Get model file
    if args.pretrained:
        config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
        model_file = config_pretrained[config['data']['type']][config['data']['imsize']]
    else:
        model_file = 'model_best.pt'

    # Distributions 获得符合概率分布的随机的y与z的
    ydist = get_ydist(1, device=device)  # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)  # Dummy to keep GAN training structure in tact
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    # Evaluator
    evaluator = Evaluator(fid_kid, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device)

    # Train
    tstart = t0 = time.time()

    # Load checkpoint
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    if True:
        with torch.nn.no_grad():
            N_samples = 8
            N_poses = 5  # corresponds to number of frames
            ztest = zdist.sample((N_samples,))
            print(ztest.shape)

            # sample from mean radius
            radius_orig = generator_test.radius
            if isinstance(radius_orig, tuple):
                generator_test.radius = 0.5 * (radius_orig[0] + radius_orig[1])

            # output directories
            rec_dir = os.path.join(eval_dir, 'reconstruction')
            image_dir = os.path.join(rec_dir, 'images')
            colmap_dir = os.path.join(rec_dir, 'models')

            # generate samples and run reconstruction
            for i, z_i in enumerate(ztest):
                outpath = os.path.join(image_dir, 'object_{:04d}'.format(i))
                os.makedirs(outpath, exist_ok=True)

                # create samples
                z_i = z_i.reshape(1, -1).repeat(N_poses, 1)
                rgbs, _, _ = evaluator.create_samples(z_i.to(device))
                rgbs = rgbs / 2 + 0.5
                for j, rgb in enumerate(rgbs):
                    save_image(rgb.clone(), os.path.join(outpath, '{:04d}.png'.format(j)))

                # run COLMAP for 3D reconstruction
                colmap_input_dir = os.path.join(image_dir, 'object_{:04d}'.format(i))
                colmap_output_dir = os.path.join(colmap_dir, 'object_{:04d}'.format(i))
                colmap_cmd = './external/colmap/run_colmap_automatic.sh {} {}'.format(colmap_input_dir, colmap_output_dir)
                print(colmap_cmd)
                os.system(colmap_cmd)

                # filter out white points
                filter_ply(colmap_output_dir)

            # reset radius for generator
            generator_test.radius = radius_orig
