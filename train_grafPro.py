import argparse
import os
from os import path

import clip
import numpy as np
import time
import copy
import csv
import torch
from clip.model import CLIP
from torch import nn

torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
from torchvision.transforms import Resize, Compose, Normalize

matplotlib.use('Agg')

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


def save_all_net(net, net_name):
    """保存整个网络"""
    torch.save(net, net_name)


def save_net_parameters(net, net_name):
    """只保存网络中的参数"""
    torch.save(net.state_dict(), net_name)


def restore_net(net_name):
    """提取整个模型"""
    net = torch.load(net_name)
    return net


def restore_parameters(network, net_name):
    """提取网络中的参数"""
    network.load_state_dict(torch.load(net_name))


class graf_pro(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2=nn.Linear(1024,512)
        # self.fc3=nn.Linear(512,1024)
        # self.fc4 = nn.Linear(1024, 2048)
        # self.fc5=nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)

        self.attention = nn.Linear(512, 256)

        self.fc_mu = nn.Linear(512, 256)
        self.fc_sigma = nn.Linear(1024, 256)
        # self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(8, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(64, 8, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(8, 1, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_z, promt_text):
        # print(latent_z.shape)#8,1,256
        # print(promt_text.shape)#1,1,512
        # y = self.conv1(promt_text)
        # y = self.tanh(y)
        # y = self.conv2(y)
        # y = self.tanh(y)
        # y = self.conv3(y)
        # y = self.tanh(y)
        # y = self.conv4(y)
        # y = self.tanh(y)
        y = promt_text
        mul = self.attention(y)
        y = self.fc1(y)
        y = self.tanh(y)
        y = self.fc2(y)
        y = self.tanh(y)
        y = self.fc3(y)
        y = self.tanh(y)
        y = self.fc4(y)
        y = self.tanh(y)
        y = self.fc5(y)
        y = self.tanh(y)
        y = self.fc6(y)
        y = self.tanh(y)
        y = self.fc7(y)
        y = self.tanh(y)
        y = self.fc8(y)
        y = self.tanh(y)
        # y = self.tanh(y)
        mu = self.fc_mu(y) * mul
        # mu=self.tanh(mu)*10
        # sigma = self.fc_sigma(y)
        # sigma=self.sigmoid(sigma)*0
        sigma = 0
        new_z = latent_z * sigma
        new_z = new_z + mu
        return new_z


# class mutil_plane(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.codebook = torch.nn.Parameter(torch.randn(40, 40, 512, requires_grad=True) * 0.01)
#         self.x_axis = nn.Linear(512, 512)
#         self.y_axis = nn.Linear(512, 512)
#
#     def forward(self, latent_z, promt_text):


from external.colmap.filter_points import filter_ply

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')

    args, unknown = parser.parse_known_args()
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])

    config['expname'] = '%s_%s' % (config['data']['type'], config['data']['imsize'])
    out_dir = os.path.join(config['training']['outdir'], config['expname'] + '_from_pretrained')
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

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

    config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
    model_file = config_pretrained[config['data']['type']][config['data']['imsize']]

    # Distributions 获得符合概率分布的随机的y与z的采样函数
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
    evaluator = Evaluator(False, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device)

    # Train
    tstart = t0 = time.time()

    # Load checkpoint
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    N_samples = 1
    N_poses = 1  # corresponds to number of frames
    epochs = 100000

    model, preprocess = clip.load("ViT-B/32", device=device)
    model_graf_pro = graf_pro()
    model_graf_pro = model_graf_pro.to(device)
    optimizer = torch.optim.Adam(model_graf_pro.parameters(), lr=5e-5)
    text_promts = clip.tokenize(
        ["a good photo of a red car", "a good photo of a yellow car", "a good photo of a blue car",
         "a good photo of a green car",
         "a good photo of a orange car", "a good photo of a purple car", "a good photo of a white car",
         "a good photo of a black car", "a good photo of a gray car",
         "a good photo of a brown car"]).to(device)

    # sample from mean radius
    radius_orig = generator_test.radius
    if isinstance(radius_orig, tuple):
        generator_test.radius = 0.5 * (radius_orig[0] + radius_orig[1])

    # output directories
    rec_dir = os.path.join(eval_dir, 'reconstruction')
    os.makedirs(rec_dir, exist_ok=True)
    image_dir = os.path.join(rec_dir, 'images')
    colmap_dir = os.path.join(rec_dir, 'models')

    torch_resize = Resize([224, 224])  # 定义Resize类对象
    preprocess = Compose([
        torch_resize,
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    train_list = [i for i in range(len(text_promts))]
    for epoch in range(epochs):
        print("-----------start epoch {:07d}-----------".format(epoch))
        np.random.shuffle(train_list)
        loss_print = 0
        loss_min = 0
        cnt = 0
        for promt_idx in train_list:
            text_promts_features = model.encode_text(text_promts[promt_idx].unsqueeze(0)).unsqueeze(1).float().to(
                device)  # 不能移出去，会影响backward
            text_promts_features = text_promts_features / text_promts_features.norm(dim=1, keepdim=True)
            loss = 0
            ztest = zdist.sample((N_samples,))
            ztest = ztest.unsqueeze(1)
            ztest = model_graf_pro(ztest, text_promts_features.unsqueeze(0)).squeeze(1)
            # generate samples and run reconstruction
            for i, z_i in enumerate(ztest):
                # create samples
                z_i = z_i.reshape(1, -1).repeat(N_poses, 1)
                rgbs = evaluator.create_samples(z_i.to(device))
                rgbs = rgbs / 2 + 0.5
                # print(rgbs.shape) N_poses*3*128*128

                if epoch % 100 == 0:
                    rgb = rgbs[0]
                    save_image(rgb.clone(), os.path.join(rec_dir,
                                                         'epoch_{:06d}_promt_{:02d}_object_{:04d}.png'.format(epoch,
                                                                                                              promt_idx,
                                                                                                              i)))
                rgbs = preprocess(rgbs).to(device)
                loss1, _ = model(rgbs, text_promts[promt_idx].unsqueeze(0))  # t()是浅复制
                loss1 = loss1.mean()
                loss1 = -loss1
                loss += loss1
                with torch.no_grad():
                    cnt += 1
                    loss_print += loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            generator_test.radius = radius_orig
        loss_print /= cnt
        print(loss_print)
        if loss_min > loss_print:
            save_all_net(model_graf_pro, 'model_graf_pro_best.pkl')
            loss_min = loss_print
        if epoch % 10 == 0:
            save_all_net(model_graf_pro, 'model_graf_pro_last.pkl')
