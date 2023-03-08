# Generate datasets
import sys

sys.path.append('submodules')  # needed to make imports work in GAN_stability
import copy
import os
import time
from multiprocessing import Process
import multiprocessing as mp
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch

import typer
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from clip2latent.models import Clipper, load_sg

import multiprocessing as mp

from graf.config import update_config, build_models, get_data
from graf.gan_training import Evaluator
from graf.transforms import ImgToPatch
from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.config import load_config
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist

try:
    mp.set_start_method('spawn')
except:
    pass

generators = {
    "sg2-ffhq-1024": partial(load_sg,
                             'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl'),
    "sg3-lhq-256": partial(load_sg,
                           'https://huggingface.co/justinpinkney/stylegan3-t-lhq-256/resolve/main/lhq-256-stylegan3-t-25Mimg.pkl'),
}


def load_graf_evaluater(
        cfg: str = 'configs/carla.yaml',
        batch_size: int = 8,
        pretrained: bool = True,
        device:str='cuda'):
    config = load_config(cfg, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    outdir = os.path.join(config['training']['outdir'], config['expname'])
    train_dataset, hwfr, render_poses = get_data(config)  # hwfr is [H,W,dset.focal,dset.radius]
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far'] - config['data']['near'], config['data']['far'] - config['data']['near'])
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr  # add for building generator
    checkpoint_dir = os.path.join(outdir, 'chkpts')
    eval_dir = os.path.join(outdir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    config['training']['nworkers'] = 0
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )
    generator, _ = build_models(config, disc=False)
    generator = generator.to(device)
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )
    if pretrained:
        config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
        model_file = config_pretrained[config['data']['type']][config['data']['imsize']]

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
    generator_test = generator_test.to(device)
    # Evaluator 即为后面用到的生成器
    evaluator = Evaluator(False, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device)

    # Load checkpoint
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    return evaluator

def mix_styles(w_batch, space):
    """Defines a style mixing procedure"""
    space_spec = {
        "w3": (4, 4, 10),
    }
    latent_mix = space_spec[space]

    bs = w_batch.shape[0]
    spec = torch.tensor(latent_mix).to(w_batch.device)

    index = torch.randint(0, bs, (len(spec), bs)).to(w_batch.device)
    return w_batch[index, 0, :].permute(1, 0, 2).repeat_interleave(spec, dim=1), spec


@torch.no_grad()
def run_folder_list(
        device_index,
        out_dir,
        generator_name,
        feature_extractor_name,
        out_image_size,
        batch_size,
        n_save_workers,
        samples_per_folder,
        folder_indexes,
        space="w",
        save_im=True,
):
    """Generate a directory of generated images and correspdonding embeddings and latents"""
    latent_dim = 512
    device = f"cuda:{device_index}" if torch.cuda.is_available() else 'cpu'
    typer.echo(device_index)

    typer.echo("Loading generator")

    # load graf

    evaluator = load_graf_evaluater(batch_size=batch_size,device=device)
    generator_test=evaluator.generator
    zdist=evaluator.zdist

    # G = generators[generator_name]().to(device).eval()

    typer.echo("Loading feature extractor")
    feature_extractor = Clipper(feature_extractor_name,device).to(device)
    feature_extractor.clip.to(device)

    typer.echo("Generating samples")
    typer.echo(f"using space {space}")

    radius_orig = generator_test.radius
    if isinstance(radius_orig, tuple):
        generator_test.radius = 0.5 * (radius_orig[0] + radius_orig[1])

    with Parallel(n_jobs=n_save_workers, prefer="threads") as parallel:
        for i_folder in folder_indexes:
            folder_name = out_dir/f"{i_folder:05d}"
            folder_name.mkdir(exist_ok=True)
            z = zdist.sample((samples_per_folder,))
            # z = torch.randn(samples_per_folder, latent_dim, device=device)
            # w = G.mapping(z, c=None)
            ds = torch.utils.data.TensorDataset(z)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch_idx, batch in enumerate(tqdm(loader, position=device_index)):
                # print(len(batch))
                generator_test.radius = radius_orig
                if space == "w":
                    this_w = batch[0].to(device)

                    # print(this_w.shape)
                    # this_w=this_w.unsqueeze(1)
                    # print(this_w.shape)
                    # latents = this_w[:,0,:].cpu().numpy()
                    latents = this_w.cpu().numpy()
                else:
                    this_w, select_idxs = mix_styles(batch[0].to(device), space)
                    latents = this_w[:, select_idxs, :].cpu().numpy()
                with torch.no_grad():

                    out = evaluator.create_samples(this_w.to(device)).to(device)
                    # out = out / 2 + 0.5
                    out = F.interpolate(out, (out_image_size, out_image_size), mode="area")
                # print(out.device)
                image_features = feature_extractor.embed_image(out)
                image_features = image_features.cpu().numpy()

                if save_im:
                    print(out.shape)
                    out = out.permute(0,2,3,1).clamp(-1,1)
                    # out=out.cpu().numpy()
                    out = (255*(out/2+0.5).cpu().numpy()).astype(np.uint8)
                else:
                    out = [None] * len(latents)
                parallel(
                    delayed(process_and_save)(batch_size, folder_name, batch_idx, idx, latent, im, image_feature,
                                              save_im)
                    for idx, (latent, im, image_feature) in enumerate(zip(latents, out, image_features))
                )

    typer.echo("finished folder")


def process_and_save(batch_size, folder_name, batch_idx, idx, latent, im, image_feature, save_im):
    count = batch_idx * batch_size + idx
    basename = folder_name / f"{folder_name.stem}{count:04}"
    np.save(basename.with_suffix(".latent.npy"), latent)
    np.save(basename.with_suffix(".img_feat.npy"), image_feature)
    if save_im:
        im = Image.fromarray(im)
        im.save(basename.with_suffix(".gen.jpg"), quality=95)


def make_webdataset(in_dir, out_dir):
    import tarfile

    in_folders = [x for x in Path(in_dir).glob("*") if x.is_dir]
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for folder in in_folders:
        filename = out_dir / f"{folder.stem}.tar"
        files_to_add = sorted(list(folder.rglob("*")))

        with tarfile.open(filename, "w") as tar:
            for f in files_to_add:
                tar.add(f)


def main(
        out_dir: Path = 'outTest',
        n_samples: int = 1_000_000,
        generator_name: str = "sg2-ffhq-1024",  # Key into `generators` dict`
        feature_extractor_name: str = "ViT-B/32",
        n_gpus: int = 1,
        out_image_size: int = 128,
        batch_size: int = 4,
        n_save_workers: int = 4,
        space: str = "w",
        samples_per_folder: int = 10_000,
        save_im: bool = False,  # Save the generated images?
):
    typer.echo("starting")

    out_dir.mkdir(parents=True, exist_ok=True)

    n_folders = math.ceil(n_samples / samples_per_folder)
    folder_indexes = range(n_folders)

    sub_indexes = np.array_split(folder_indexes, n_gpus)

    processes = []
    for dev_idx, folder_list in enumerate(sub_indexes):
        p = Process(
            target=run_folder_list,  # 用这个方法
            args=(
                dev_idx,
                out_dir,
                generator_name,
                feature_extractor_name,
                out_image_size,
                batch_size,
                n_save_workers,
                samples_per_folder,
                folder_list,
                space,
                save_im,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    typer.echo("finished all")


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    typer.run(main)
