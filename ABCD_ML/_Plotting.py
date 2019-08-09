"""
_Plotting.py
====================================
Main class extension file for the some plotting functionality.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap


def Show_Targets_Dist(self):
    '''This method displays some summary statistics about
    the loaded targets, as well as plots the distibution.
    For now it only supports loaded binary and regression target types.
    '''

    assert self.targets.shape[1] == 1, \
        "This function is not yet avaliable for categorical targets"

    print(self.targets.describe())
    vals = self.targets[self.targets_key]
    print()
    print('Num. of unique vals:', len(np.unique(vals)))

    sns.distplot(self.targets)
    plt.title('Target distribution')
    plt.show()


def Plot_Base_Feat_Importances(self, top_n=10):

    top_x = self.Get_Base_Feat_Importances(top_n).index
    just_top = self.Model.feature_importances[top_x]

    self._plot_feature_importance(just_top, title='Base Feature Importances',
                                  xlabel='Feature Importance')


def Plot_Shap_Feat_Importances(self, top_n=10):

    top_x = self.Get_Shap_Feat_Importances(top_n).index
    just_top = np.abs(self.Model.shap_df)[top_x]

    self._plot_feature_importance(just_top, title='Shap Feature Importances',
                                  xlabel='Shap Feature Importance')


def _plot_feature_importance(self, data, title='Feature Importances',
                             xlabel='Feature Importance'):

    sns.barplot(orient='h', data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()


def Plot_Shap_Summary(self):

    assert len(self.Model.shap_df) > 0, \
        "calc_shap_feature_importances must be set to True!"

    shap_df = self.Model.shap_df

    shap_cols, shap_inds = list(shap_df), shap_df.index
    X = self.all_data.loc[shap_inds][shap_cols]

    shap.summary_plot(np.array(shap_df), X)
