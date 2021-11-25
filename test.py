import hydra

@hydra.main(config_path='configs', config_name='config_hyperparameters')
def experiment(cfg):
    print(cfg)

if __name__ == '__main__':
    experiment()