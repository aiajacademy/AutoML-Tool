from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

class HyperparameterTuner:
    def __init__(self):
        self.space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)
        }

    def tune_model(self, model, features, target):
        def objective(params):
            model.set_params(**params)
            model.fit(features, target)
            loss = -cross_val_score(model, features, target, cv=3).mean()
            return {'loss': loss, 'status': STATUS_OK}

        trials = Trials()
        best_params = fmin(fn=objective, space=self.space, algo=tpe.suggest, max_evals=50, trials=trials)
        model.set_params(**best_params)
        return model
