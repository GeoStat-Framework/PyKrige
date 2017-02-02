import numpy as np
import os
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from pykrige.rk import RegressionKriging

DATA = '/home/sudipta/temp/JamesGaeWskyMedicalImaging/FilesSudipta/Data'

x = np.loadtxt(os.path.join(DATA, 'Source_STL_Landmarks.txt'), delimiter=',')
xp = np.loadtxt(os.path.join(DATA, 'Source_STL_Spine_Point_Cloud.txt'),
                delimiter=',')

y = np.loadtxt(os.path.join(DATA, 'Target_Image_Landmarks.txt'), delimiter=',')
yp = np.empty_like(xp)

ml_algos = {'svr': SVR(),
            'rf': RandomForestRegressor(),
            'gbr': GradientBoostingRegressor(),
            'en': ElasticNet()}

for ml, v in ml_algos.items():
    print('Machine learning regression model: {}'.format(v.__class__.__name__))
    for k in range(y.shape[1]):
        RK = RegressionKriging(regression_model=v,
                               method='ordinary3d',
                               variogram_model='linear',
                               n_closest_points=10)
        RK.fit(x, x, y[:, k])
        yp[:, k] = RK.predict(xp, xp)

    np.savetxt(os.path.join(DATA, 'RK_{}.csv'.format(ml)), X=yp,
               delimiter=',')
