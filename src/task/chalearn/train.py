import sys

sys.path.append("{workspace}/my_project/src")

import yaml
import logging
import os

from datamodule import ChalearnLightningDataModule

from network import ChalearnMultiModalModel
from network import MultiModalEncoder, MultiModalRegressor
from network import DomainClassifier

from loss import FairnessLoss, DomainLoss
from module import ChalearnLightningModule
import torch

from pytorch_lightning import loggers
from omegaconf import DictConfig, OmegaConf

from callback.metric import BaseMetricCallback
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, KendallRankCorrCoef

from callback.metric import FairnessCallback
from metric import DemographicParity, SPDD
from metric import EqualOpportunity, SPEO

from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl


def _set_gpu_environ(cfg: DictConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"])


def _convert_gpus_for_lit_trainer(cfg: DictConfig) -> list[int]:
    gpu_cnt = len(str(cfg["gpus"]).split(","))
    gpus = list(range(gpu_cnt))
    return gpus


def main(cfg) -> None:
    logging.info(cfg)
    _set_gpu_environ(cfg)

    lit_data_module = ChalearnLightningDataModule(
        video_dir=cfg["datamodule"]["video_dir"],
        audio_dir=cfg["datamodule"]["audio_dir"],
        text_dir=cfg["datamodule"]["text_dir"],
        label_dir=cfg["datamodule"]["label_dir"],
        target_list=cfg["datamodule"]["target_list"],
        frame_count=cfg["datamodule"]["frame_count"],
        seed=cfg["datamodule"]["seed"],
        num_workers=cfg["datamodule"]["num_workers"],
        batch_size=cfg["datamodule"]["batch_size"],
    )

    multimodal_encoder = MultiModalEncoder(
        video_channels=cfg["module"]["encoder"]["video_channels"],
        video_input_size=cfg["module"]["encoder"]["video_input_size"],
        video_hidden_size=cfg["module"]["encoder"]["video_hidden_size"],
        video_num_layers=cfg["module"]["encoder"]["video_num_layers"],
        audio_input_size=cfg["module"]["encoder"]["audio_input_size"],
        audio_hidden_size=cfg["module"]["encoder"]["audio_hidden_size"],
        audio_num_layers=cfg["module"]["encoder"]["audio_num_layers"],
        text_input_size=cfg["module"]["encoder"]["text_input_size"],
        text_hidden_size=cfg["module"]["encoder"]["text_hidden_size"],
    )

    regressor = MultiModalRegressor(
        num_class=cfg["module"]["regressor"]["num_class"],
        video_hidden_size=cfg["module"]["encoder"]["video_hidden_size"],
        audio_hidden_size=cfg["module"]["encoder"]["audio_hidden_size"],
        text_hidden_size=cfg["module"]["encoder"]["text_hidden_size"],
        fusion_hidden_size=cfg["module"]["regressor"]["fusion_hidden_size"],
    )

    classifier = DomainClassifier(
        num_class=cfg["module"]["classifier"]["num_class"],
        video_hidden_size=cfg["module"]["encoder"]["video_hidden_size"],
        audio_hidden_size=cfg["module"]["encoder"]["audio_hidden_size"],
        text_hidden_size=cfg["module"]["encoder"]["text_hidden_size"],
        fusion_hidden_size=cfg["module"]["classifier"]["fusion_hidden_size"],
    )

    multimodal_model = ChalearnMultiModalModel(
        multimodal_encoder,
        regressor,
        classifier,
    )

    base_loss_func = FairnessLoss(
        base_loss_function=torch.nn.MSELoss(),
        base_loss_weight=cfg["loss"]["base_loss_weight"],
        l2_loss_weight=cfg["loss"]["l2_loss_weight"],
        wd_loss_weight=cfg["loss"]["wd_loss_weight"],
        mmd_loss_weight=cfg["loss"]["mmd_loss_weight"],
    )

    domain_loss_func = DomainLoss(
        domain_loss_weight=cfg["domain_loss"]["domain_loss_weight"],
    )

    lit_module = ChalearnLightningModule(
        multimodal_model=multimodal_model,
        base_loss_func=base_loss_func,
        domain_loss_func=domain_loss_func,
        dann_gamma=cfg["domain_loss"]["dann_gamma"],
        optim=torch.optim.AdamW,
        lr_regressor=cfg["trainer"]["lr_regressor"],
        lr_classifier=cfg["trainer"]["lr_classifier"],
        lr_scheduler=None,
        label_names=cfg["datamodule"]["target_list"],
    )

    save_dir = os.path.join(cfg["save_root"], cfg["log_dirname"])
    logger = loggers.TensorBoardLogger(
        save_dir=save_dir, name=cfg["trainer"]["logger"]["name"]
    )
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir)
    OmegaConf.save(cfg, f"{logger.log_dir}/config.yaml")

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor=cfg["callbacks"]["checkpoint"]["monitor"],
        save_top_k=cfg["callbacks"]["checkpoint"]["save_top_k"],
        mode=cfg["callbacks"]["checkpoint"]["mode"],
    )
    callbacks = [checkpoint_callback]

    pearson = BaseMetricCallback(PearsonCorrCoef())
    spearman = BaseMetricCallback(SpearmanCorrCoef())
    kendallrank = BaseMetricCallback(KendallRankCorrCoef())
    callbacks += [pearson, spearman, kendallrank]

    dp = FairnessCallback(DemographicParity(cfg["metric"]["tau"]))
    spdd = FairnessCallback(SPDD())
    callbacks += [dp, spdd]

    eo = FairnessCallback(EqualOpportunity(cfg["metric"]["tau"]))
    speo = FairnessCallback(SPEO())
    callbacks += [eo, speo]

    trainer = pl.Trainer(
        devices=_convert_gpus_for_lit_trainer(cfg),
        max_epochs=cfg["trainer"]["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=31,
    )

    trainer.fit(model=lit_module, datamodule=lit_data_module)

    ckpt_path = [f for f in os.listdir(logger.log_dir) if f.endswith(".ckpt")][0]
    ckpt_path = os.path.join(logger.log_dir, ckpt_path)
    print(ckpt_path)
    trainer.test(model=lit_module, datamodule=lit_data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    config_path = "./task/chalearn/config/config.yaml"
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(cfg)
