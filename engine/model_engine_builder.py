import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from engine.model_engine import ModelEngine
from utils.utils import set_experiment_name
from models.raw_transformer import RawTransformer
from dataloaders.cora import load_graph



def model_engine_builder(cfg):
    # Device
    log = logging.getLogger(__name__)
    device = cfg.device

    #Fix seed for reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)

    #Load the data
    graph = load_graph(cfg)
    load_method = cfg.load_method

    # Setup the model
    if cfg.model == 'RawTransformer':
        model = RawTransformer(cfg.n_tokens, cfg.d_model, cfg.n_head, cfg.dim_feedforward,
                               cfg.num_layers, cfg.n_classes, cfg.dropout)
    else:
        raise NotImplementedError('Model not supported')

    model.to(device)

    # The loss function
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # miscellaneous
    num_epochs = cfg.num_epochs
    save_dir = cfg.save_dir
    show_log = cfg.show_log
    save_ckpt = cfg.save_ckpt
    try:
        exp_name = cfg.exp_name
    except omegaconf.errors.MissingMandatoryValue:
        cfg.exp_name = set_experiment_name(
            data=cfg.data,
            model=cfg.model,
            load_method=cfg.load_method,
            epochs=cfg.num_epochs,
            d_model=cfg.d_model,
            n_head=cfg.n_head,
            dim_feedforward=cfg.dim_feedforward,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lr=cfg.lr)
        exp_name = cfg.exp_name

    return ModelEngine(model, loss_fn, optimizer, graph, load_method, log, device, num_epochs, exp_name, save_dir, show_log, save_ckpt)