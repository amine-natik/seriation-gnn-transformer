import hydra
from engine.model_engine_builder import model_engine_builder


@hydra.main(config_path='configs', config_name='config')
def experiment(cfg):
    cfg.save_dir = hydra.utils.to_absolute_path(cfg.save_dir)
    cfg.data_root = hydra.utils.to_absolute_path(cfg.data_root)
    engine = model_engine_builder(cfg)
    engine.train()
    print(engine.validate(is_test=True))


if __name__ == '__main__':
    experiment()
