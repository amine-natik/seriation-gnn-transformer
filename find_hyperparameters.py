import hydra
from utils.hyperparameter_search import HyperparameterSearch
import optuna


@hydra.main(config_path='configs', config_name='config_hyperparameters')
def experiment(cfg):
    cfg.show_log = False
    cfg.save_dir = hydra.utils.to_absolute_path(cfg.save_dir)
    cfg.data_root = hydra.utils.to_absolute_path(cfg.data_root)
    hyper_search = HyperparameterSearch(cfg)
    study = optuna.create_study(direction=cfg.direction)
    study.optimize(hyper_search.objective, n_trials=cfg.n_trials, timeout=cfg.timeout)
    hyper_search.launch_search(study)

if __name__ == '__main__':
    experiment()