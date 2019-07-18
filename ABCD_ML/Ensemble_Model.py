import numpy as np


class Ensemble_Model():

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        '''Calls predict on each model and
        returns the averaged prediction. Handling
        different problem types cases.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = np.array([model.predict(X) for model in self.models])

        if preds[0].dtype == 'float':

            # Regression
            if len(preds[0].shape) == 1:

                mean_preds = np.mean(preds, axis=0)
                return mean_preds

            # Multi-label
            else:
                vote_results = np.zeros(preds[0].shape)

                for i in range(len(vote_results)):
                    mx = np.argmax(np.bincount(np.argmax(preds[:, i], axis=1)))
                    vote_results[i][mx] = 1

                return vote_results

        # Binary or multi-class
        else:

            class_preds = [np.argmax(np.bincount(preds[:, i]))
                           for i in range(preds.shape[1])]

            class_preds = np.array(class_preds)
            return class_preds

    def predict_proba(self, X):
        '''Calls predict_proba on each model and
        returns the averaged prediction.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = np.array([model.predict_proba(X) for model in self.models])
        mean_preds = np.mean(preds, axis=0)

        # Multi-label case
        if len(mean_preds.shape) > 2:
            return list(mean_preds)

        return mean_preds
