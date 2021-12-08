from engine.model_engine_builder import model_engine_builder
from optuna.trial import TrialState
from omegaconf import OmegaConf


class HyperparameterSearch:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_parameters(self, cfg, trial):
        for (name, parameter) in self.cfg.hyperparameters.items():
            if parameter.type == 'categorical':
                suggestion = trial.suggest_categorical(name, parameter.choices)
            elif parameter.type == 'int':
                suggestion = trial.suggest_int(name, parameter.min, parameter.max)
            elif parameter.type == 'float':
                suggestion = trial.suggest_float(name, parameter.min, parameter.max, log=parameter.log)
            else:
                raise NotImplemented("parameter type not supported")

            if parameter.confdir == 'model':
                cfg.model[name] = suggestion
            elif parameter.confdir == 'configs':
                cfg[name] = suggestion
            else:
                raise NotImplemented("parameter configuration nor supported")
        return cfg

    def objective(self, trial):
        cfg = self.get_parameters(self.cfg, trial)
        engine = model_engine_builder(cfg)
        engine.train(trial)
        accuracy = engine.validate(is_test=True)
        return accuracy

    def launch_search(self, study):
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        best_accuracy = trial.value
        hyperparameters = dict()
        print("  Best test accuracy: ", best_accuracy)
        print("  Params: ")
        for key, value in trial.params.items():
            hyperparameters[key] = value
            print("    {}: {}".format(key, value))

        # Saving the configuration
        OmegaConf.save(hyperparameters, f"hyperparameters_test_accuracy:{best_accuracy:.4f}.yaml")
        OmegaConf.save(self.cfg, "config.yaml")