import copy
import logging
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
sys.path.append('submodules')
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import wandb
from clip2latent.data import load_data
from clip2latent.models import load_models
from clip2latent.train_utils import (compute_val, make_grid,
                                     make_image_val_data, make_text_val_data)
from graf.config import get_data
from graf.config import update_config, build_models, get_data
from graf.gan_training import Evaluator
from graf.transforms import ImgToPatch
from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.config import load_config
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
# from generate_graf_dataset import load_graf_evaluater

logger = logging.getLogger(__name__)
noop = lambda *args, **kwargs: None
logfun = noop

def load_graf_evaluater(
        cfg: str = '../../../configs/carla.yaml',
        batch_size: int = 8,
        pretrained: bool = True,
        device:str='cuda'):
    # print("aaaa")
    config = load_config(cfg, '../../../configs/default.yaml')
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
        config_pretrained = load_config('../../../configs/pretrained_models.yaml', '../../../configs/pretrained_models.yaml')
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

class Checkpointer():
    """A small class to take care of saving checkpoints"""
    def __init__(self, directory, checkpoint_its):
        directory = Path(directory)
        self.directory = directory
        self.checkpoint_its = checkpoint_its
        if not directory.exists():
            directory.mkdir(parents=True)

    def save_checkpoint(self, model, iteration):
        if iteration % self.checkpoint_its:
            return

        k_it = iteration // 1000
        filename = self.directory/f"{k_it:06}.ckpt"
        checkpoint = {"state_dict": model.state_dict()}
        if hasattr(model, "cfg"):
            checkpoint["cfg"] = model.cfg

        print(f"Saving checkpoint to {filename}")
        torch.save(checkpoint, filename)



def validation(current_it, device, diffusion_prior, G, clip_model, val_data, samples_per_text):
    single_im = {"clip_features": val_data["val_im"]["clip_features"][0].unsqueeze(0)}
    captions = val_data["val_caption"]

    for input_data, key, cond_scale, repeats in zip(
        [val_data["val_im"], single_im, val_data["val_text"], val_data["val_text"]],
        ["image-similarity", "image-vars", "text2im", "text2im-super2"],
        [1.0, 1.0, 1.0, 2.0],
        [1, 8, samples_per_text, samples_per_text],
    ):
        tiled_data = input_data["clip_features"].repeat_interleave(repeats, dim=0)
        cos_sim, ims = compute_val(diffusion_prior, tiled_data, G, clip_model, device, cond_scale=cond_scale)
        logfun({f'val/{key}':cos_sim.mean()}, step=current_it)


        if key.startswith("text"):
            num_chunks = int(np.ceil(ims.shape[0]//repeats))
            for idx, (sim, im_chunk) in enumerate(zip(
                cos_sim.chunk(num_chunks),
                ims.chunk(num_chunks)
                )):

                caption = captions[idx]
                im = wandb.Image(make_grid(im_chunk), caption=f'{sim.mean():.2f} - {caption}')
                logfun({f'val/image/{key}/{idx}': im}, step=current_it)
        else:
            for idx, im in enumerate(ims.chunk(int(np.ceil(ims.shape[0]/16)))):
                logfun({f'val/image/{key}/{idx}': wandb.Image(make_grid(im))}, step=current_it)

    logger.info("Validation done.")

def train_step(diffusion_prior, device, batch):
    diffusion_prior.train()
    batch_z, batch_w = batch
    batch_z = batch_z.to(device)
    batch_w = batch_w.to(device)

    loss = diffusion_prior(batch_z, batch_w)
    loss.backward()
    return loss


def train(trainer, loader, device, val_it, validate, save_checkpoint, max_it, print_it=50):

    current_it = 0
    current_epoch = 0

    while current_it < max_it:

        logfun({'epoch': current_epoch}, step=current_it)
        pbar = tqdm(loader)
        for batch in pbar:
            if current_it % val_it == 0:
                validate(current_it, device, trainer)

            trainer.train()
            batch_clip, batch_latent = batch

            input_args = {#用Et找Ei，Et实际是抽取数据中的Ei，Ei实际是latent
                "image_embed": batch_latent.to(device),
                "text_embed": batch_clip.to(device)
            }
            loss = trainer(**input_args)

            if (current_it % print_it == 0):
                logfun({'loss': loss}, step=current_it)

            trainer.update()
            current_it += 1
            pbar.set_postfix({"loss": loss, "epoch": current_epoch, "it": current_it})

            save_checkpoint(trainer, current_it)

        current_epoch += 1





@hydra.main(config_path="config", config_name="car")
def main(cfg):
    # print(os.path)
    if cfg.logging == "wandb":
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg),
            entity=cfg.wandb_entity,
            name=cfg.name,
        )
        global logfun
        logfun = wandb.log
    elif cfg.logging is None:
        logger.info("Not logging")
    else:
        raise NotImplementedError(f"Logging type {cfg.logging} not implemented")
    # print(os.path)
    device = cfg.device
    stats, loader = load_data(cfg.data) #stats包含w的方差、均值信息，loader取w，Ei对
    _, clip_model, trainer = load_models(cfg, device, stats)#trainer是DiffusionPriorTrainer
    evaluator=load_graf_evaluater()

    text_embed, text_samples = make_text_val_data(evaluator, clip_model, hydra.utils.to_absolute_path(cfg.data.val_text_samples))
    val_data = {
        "val_im": make_image_val_data(evaluator, clip_model, cfg.data.val_im_samples, device),
        "val_text": text_embed,
        "val_caption": text_samples,
    }

    if 'resume' in cfg and cfg.resume is not None:
        # Does not load previous iteration count
        logger.info(f"Resuming from {cfg.resume}")
        trainer.load_state_dict(torch.load(cfg.resume, map_location="cpu")["state_dict"])

    checkpoint_dir = f"checkpoints/{datetime.now():%Y%m%d-%H%M%S}"
    checkpointer = Checkpointer(checkpoint_dir, cfg.train.val_it)
    validate = partial(validation,
        G=evaluator,
        clip_model=clip_model,
        val_data=val_data,
        samples_per_text=cfg.data.val_samples_per_text,
        )

    train(trainer, loader, device,
        val_it=cfg.train.val_it,
        max_it=cfg.train.max_it,
        validate=validate,
        save_checkpoint=checkpointer.save_checkpoint,
        )

if __name__ == "__main__":
    main()
