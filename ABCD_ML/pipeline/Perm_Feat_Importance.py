from joblib import dump, load, Parallel, delayed
import os
import copy
import numpy as np


def get_feat_importances(model_loc, scorer, inds, X, y, n_perm):

    results = {}
    model = load(model_loc)

    X_copy = X.copy()

    for ind in inds:

        original = X_copy[:, ind].copy()
        importances = []

        for perm in range(n_perm):
            X_copy[:, ind] = np.random.permutation(X_copy[:, ind])
            importances.append(scorer(model, X_copy, y))

        results[ind] = np.mean(importances)
        X_copy[:, ind] = original

    return results


class Perm_Feat_Importance():

    def __init__(self, n_perm=1, n_jobs=1, temp_dr=''):

        self.n_perm = n_perm
        self.n_jobs = n_jobs
        self.temp_dr = temp_dr

    def get_chunks(self):

        per_chunk = self.X.shape[1] // self.n_jobs
        chunks = [list(range(i * per_chunk, (i+1) * per_chunk))
                  for i in range(self.n_jobs)]

        last = chunks[-1][-1]
        chunks[-1] += list(range(last+1, self.X.shape[-1]))
        return chunks

    def compute(self, model, scorer, X, y):

        self.X = X
        self.y = y

        baseline_score = scorer(model, self.X, self.y)

        if self.n_jobs == 1:
            scores = []
            for ind in range(X.shape[1]):
                scores.append(self.get_feat_importance(ind, scorer, model))
        else:

            changed = None

            # Ensure models n_jobs set to 1
            try:
                original = model.n_jobs
                model.n_jobs = 1
                changed = original

            except AttributeError:
                pass

            rn = str(np.random.random())
            model_loc = os.path.join(self.temp_dr, 'temp' + rn + '.joblib')
            dump(model, model_loc)
            model_locs = [model_loc for i in range(self.n_jobs)]

            scorers = [scorer for i in range(self.n_jobs)]
            inds = self.get_chunks()
            Xs = [X for i in range(self.n_jobs)]
            ys = [y for i in range(self.n_jobs)]
            n_perms = [self.n_perm for i in range(self.n_jobs)]

            imp_dicts =\
                Parallel(n_jobs=self.n_jobs)(delayed(get_feat_importances)
                                                    (ml, s, i, x, y, n) for
                                             ml, s, i, x, y, n in zip(
                                                 model_locs,
                                                 scorers,
                                                 inds, Xs, ys,
                                                 n_perms))

            scores = [0 for i in range(X.shape[1])]
            for imp_dict in imp_dicts:
                for i in imp_dict:
                    scores[i] = imp_dict[i]

            os.remove(model_loc)

            if changed is not None:
                model.n_jobs = changed

        scores = np.array(scores)
        difs = baseline_score - scores
        return difs

    def get_feat_importance(self, ind, scorer, model):

        X_copy = self.X.copy()

        importances = []
        for _ in range(self.n_perm):

            X_copy[:, ind] = np.random.permutation(X_copy[:, ind])
            importances.append(scorer(model, X_copy, self.y))

        return np.mean(importances)
