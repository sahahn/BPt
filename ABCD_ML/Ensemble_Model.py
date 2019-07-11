"""
Ensemble_Model.py
====================================
Wrapper class for ensembling different combinations of trained models.
"""
import numpy as np
from ABCD_ML.Train_Models import train_model


class Ensemble_Model():
    '''Basic model ensemble wrapper'''

    def __init__(self, data, score_key, CV, model_type, int_cv, metric,
                 class_weight='balanced', random_state=None,
                 score_encoder=None, n_jobs=1, extra_params={}):
        '''Init function for training an ensemble model.
           Models are trained and saved when init is called.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD_ML formatted df.

        score_key : str or list
            The name(s) of the column(s) within data to be
            set as score.

        CV : ABCD_ML CV
            A custom ABCD_ML CV object, which stores the
            desired split behavior

        model_type : list of str
            List of str, where each string refers to a type of model to train.
            Assumes final processed model_type names here.
            For a full list of supported options call:
            self.show_model_types(), with optional problem type parameter.

        int_cv : int
            The number of internal folds to use during
            model k-fold parameter selection, if the chosen model requires
            parameter selection. A value greater
            then 2 must be passed.

        metric : str
            Indicator for which metric to use for calculating
            score and during model parameter selection.
            Note, some metrics are only avaliable for certain problem types.
            For a full list of supported metrics call:
            self.show_metrics, with optional problem type parameter.

        class weight : {dict, 'balanced', None}, optional
            Only used for binary and categorical problem types.
            Follows sklearn api class weight behavior. Typically, either use
            'balanced' in the case of class distribution imbalance, or None.
            (default='balanced')

        random_state : int, or None, optional
            Random state, either as int for a specific seed,
            or if None then the random seed is set by np.random.
            (default=None)

        score_encoder : sklearn encoder, optional
            A sklearn api encoder, for optionally transforming the target
            variable. Used in the case of categorical data in converting from
            one-hot encoding to ordinal.
            (default=None)

        n_jobs : int, optional
            Number of processors to use during training.
            (default = 1)

        extra_params : dict, optional
            Any extra params being passed. Typically, extra params are
            added when the user wants to provide a specific model/classifier,
            or data scaler, with updated (or new) parameters.
            These can be supplied by creating another dict within extra_params.
            E.g., extra_params[model_name] = {'model_param' : new_value}
            Where model param is a valid argument for that model,
            and model_name in this case is the str indicator
            passed to model_type.
            (default={})
        '''

        self.models = []

        for m in model_type:
            model = train_model(data, score_key, CV, m, int_cv, metric,
                                class_weight, random_state, score_encoder,
                                n_jobs, extra_params)
            self.models.append(model)

    def predict(self, X):
        '''Calls predict on each model and
        returns the averaged prediction.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

    def predict_proba(self, X):
        '''Calls predict_proba on each model and
        returns the averaged prediction.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = [model.predict_proba(X) for model in self.models]
        return np.mean(preds, axis=0)
