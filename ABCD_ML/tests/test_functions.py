from unittest import TestCase

from ABCD_ML.Train_Light_GBM import Train_Light_GBM

class Test_LGBM(TestCase):
    
    def test_fake_lgbm(self):
        fake_X, fake_y = np.ones((100, 100)), np.ones(100)
        fake_params = {'int_cv':2, 'regression':True, 'n_params':1, 'test_size':.2, 'n_jobs':1, 'e_stop_rounds':5}
        model = Train_Light_GBM(fake_X, fake_y, **fake_params)
        self.assertTrue(str(type(model)) == "<class 'lightgbm.sklearn.LGBMRegressor'>") 