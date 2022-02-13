import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from engine.model_engine import ModelEngine
from utils.utils import set_experiment_name
from models.raw_transformer import RawTransformer
from models.learn_embedding import LearnEmbedding
from dataloaders.planetoid import load_graph, process_graph
from utils.laplacian_loss import LaplacianLoss
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


def model_engine_builder(cfg):
    # Log device
    log = logging.getLogger(__name__)

    # Fix seed for reproducibility
    torch.manual_seed(cfg.seed)

    # Load the Graph data
    graph = load_graph(cfg.data_root, cfg.data)
    graph = process_graph(graph, cfg.process_method, cfg.save_dir, cfg.idx)
    graph.to(cfg.device)

    # Graph parameters
    max_len, n_tokens = graph.x.shape
    n_classes = graph.y.max().item() + 1

    # Setup the model
    if cfg.model.name == 'RawTransformer':
        model = RawTransformer(n_tokens, cfg.model.d_model, cfg.model.n_head, cfg.model.dim_feedforward,
                               cfg.model.num_layers, n_classes, cfg.model.dropout, max_len)
        model.to(cfg.device)
    elif cfg.model.name == 'LearnEmbedding':
        model = LearnEmbedding(n_tokens, cfg.model.d_model, cfg.model.n_head, cfg.model.dim_feedforward,
                               cfg.model.num_layers, n_classes, cfg.model.dropout, max_len).to(cfg.device)
    else:
        raise NotImplementedError('Model not supported')

    # Loss_fn diff approach


    # The loss function
    if cfg.model.loss_fn == 'NLLLoss':
        loss_fn = nn.NLLLoss()
    elif cfg.model.loss_fn == 'Laplacian':
        loss_fn = LaplacianLoss(graph, cfg.alpha, sigma=1)
    else:
        raise NotImplementedError('Loss function not supported')

    # The optimizer
    if cfg.model.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)
    else:
        raise NotImplementedError('Optimizer not supported')

    try:
        exp_name = cfg.exp_name
    except omegaconf.errors.MissingMandatoryValue:
        cfg.exp_name = set_experiment_name(
            data=cfg.data,
            model=cfg.model.name,
            process_method=cfg.process_method,
            epochs=cfg.num_epochs,
            d_model=cfg.model.d_model,
            n_head=cfg.model.n_head,
            dim_feedforward=cfg.model.dim_feedforward,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            lr=cfg.model.lr)
        exp_name = cfg.exp_name

    return ModelEngine(model, graph, loss_fn, optimizer, exp_name, cfg.num_epochs, cfg.save_dir, cfg.save_ckpt,
                       log, cfg.show_log)
