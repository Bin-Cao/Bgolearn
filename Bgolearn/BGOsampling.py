import inspect
import os
import time
import warnings
import copy
from tkinter import N
import numpy as np
import pandas as pd
from .BGOmax import Global_max
from .BGOmin import Global_min
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF

class Bgolearn(object):
    def fit(self,data_matrix, Measured_response, virtual_samples, noise_std = 1e-5, Kriging_model = None, opt_num = 1 ,min_search = True):
        
        """
        PACKAGE: Bayesian global optimization learn .

        10 Jul 2022, version 1, Bin Cao, MGI, SHU, Shanghai, CHINA.

        :param data_matrix: data matrix of training dataset, X .

        :param Measured_response: response of tarining dataset, y.

        :param virtual_samples: designed virtual samples.

        :param noise_std: float or ndarray of shape (n_samples,), default=1e-5
                Value added to the diagonal of the kernel matrix during fitting.
                This can prevent a potential numerical issue during fitting, by
                ensuring that the calculated values form a positive definite matrix.
                It can also be interpreted as the variance of additional Gaussian.
                measurement noise on the training observations.

        :param Kriging_model (default None): a user defined callable Kriging model, has an attribute of <fit_pre>
                if user isn't applied one, Bgolearn will call a pre-set Kriging model
                atribute <fit_pre> : 
                input -> xtrain, ytrain, xtest ; 
                output -> predicted  mean and std of xtest
                e.g. (take GaussianProcessRegressor in sklearn as an example):
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest):
                        # instantiated model
                        kernel = RBF()
                        mdoel = GaussianProcessRegressor(kernel=kernel).fit(xtrain,ytrain)
                        # defined the attribute's outputs
                        mean,std = mdoel.predict(xtest,return_std=True)
                        return mean,std    

        :param opt_num: the number of recommended candidates for next iteration, default 1. 

        :param min_search: default True -> searching the global minimum ;
                                   False -> searching the global maximum.

        :return: the recommended candidates.
        """
        timename = time.localtime(time.time())
        namey, nameM, named, nameh, namem = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min

        

        if Kriging_model == None:
            kernel = RBF() 
            if type(noise_std) == float:
                # call the default model;
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest,):
                        # ret_std is a placeholder for homogenous noise
                        # instantiated mode
                        mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha = noise_std**2).fit(xtrain,ytrain)
                        # defined the attribute's outputs
                        mean,std = mdoel.predict(xtest,return_std=True)
                        return mean,std 
                print('The internal model is instantiated with homogenous noise: %s' % noise_std)  
                
            elif type(noise_std) == np.ndarray:
                # call the default model;
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest,ret_std = 0.0):
                        # instantiated model
                        if len(xtrain) == len(noise_std):
                            mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha = noise_std**2).fit(xtrain,ytrain)
                        elif len(xtrain) == len(noise_std) + 1:
                            new_alpha = np.append(noise_std,ret_std)
                            mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha = new_alpha**2).fit(xtrain,ytrain)
                        else:
                            print('the input data is not muached with heterogenous noise size') 
                        # defined the attribute's outputs
                        mean,std = mdoel.predict(xtest,return_std=True)
                        return mean,std  
                print('The internal model is instantiated with heterogenous noise')
        else: 
            print('The external model is instantiated')
            pass  
        
        
        # position incluse 'self'
        if len(inspect.getargspec(Kriging_model().fit_pre)[0]) == 5:
            ret_noise = True
        elif len(inspect.getargspec(Kriging_model().fit_pre)[0]) == 4:
            ret_noise = False
        else:
            print('type ERROR! -ILLEGAL form of Krigging-')

        # fitting results
        X_true = np.array(data_matrix)
        Y_true = np.array(Measured_response)
        __fea_num = len(X_true[0])
    

        loo = LeaveOneOut()
        loo.get_n_splits(X_true)
        pre = []
        if ret_noise == 0:
            _Y_pre, _ = Kriging_model().fit_pre(X_true , Y_true, X_true.reshape(-1,__fea_num))
            V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true , Y_true, virtual_samples.reshape(-1,__fea_num))
            for train_index, test_index in loo.split(X_true):
                X_train, X_test = X_true[train_index], X_true[test_index]
                y_train, _ = Y_true[train_index], Y_true[test_index]
                Y_pre, _ = Kriging_model().fit_pre( X_train , y_train, X_test)
                pre.append(Y_pre)
                warnings.filterwarnings('ignore')
        else:
            _Y_pre, _ = Kriging_model().fit_pre(X_true , Y_true, X_true.reshape(-1,__fea_num),0.0)
            V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true , Y_true, virtual_samples.reshape(-1,__fea_num),0.0)
            for train_index, test_index in loo.split(X_true):
                X_train, X_test = X_true[train_index], X_true[test_index]
                y_train, _ = Y_true[train_index], Y_true[test_index]
                Y_pre, _ = Kriging_model().fit_pre( X_train , y_train, X_test,0.0)
                pre.append(Y_pre)
                warnings.filterwarnings('ignore')
        


        Y_pre = np.array(pre)
        results_dataset = pd.DataFrame(Y_true)
        results_dataset.columns = ['Y_true']
        results_dataset['Y_pre'] = Y_pre

        _results_dataset = pd.DataFrame(Y_true)
        _results_dataset.columns = ['Y_true']
        _results_dataset['Y_pre'] = _Y_pre

        V_Xmatrix = pd.DataFrame(np.array(virtual_samples))
        V_Xmatrix['Y_pre'] = V_Y_pre
        V_Xmatrix['Y_std'] = V_Y_std

        RMSE = np.sqrt(mean_squared_error(Y_true,Y_pre))
        MAE = mean_absolute_error(Y_true,Y_pre)
        R2 = r2_score(Y_true,Y_pre)

        _RMSE = np.sqrt(mean_squared_error(Y_true,_Y_pre))
        _MAE = mean_absolute_error(Y_true,_Y_pre)
        _R2 = r2_score(Y_true,_Y_pre)

    
        os.makedirs('Bgolearn', exist_ok=True)

        print('Fitting goodness on training dataset: \n' + str('  RMSE = %f' % _RMSE) +' '+ str('  MAE = %f' % _MAE)
            +' '+ str('  R2 = %f' % _R2))

        print('Fitting goodness of LOOCV: \n' + str('  RMSE = %f' % RMSE) +' '+ str('  MAE = %f' % MAE)
            +' '+ str('  R2 = %f' % R2))

      

        results_dataset.to_csv('./Bgolearn/predictionsByLOOCV_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                        minute=namem),encoding='utf-8-sig')
        


        _results_dataset.to_csv('./Bgolearn/predictionsOnTrainingDataset_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                        minute=namem),encoding='utf-8-sig')

        V_Xmatrix.to_csv('./Bgolearn/predictionsOfVirtualSampels_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                        minute=namem),encoding='utf-8-sig')

        if min_search == True:
            BGOmodel = Global_min(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num, ret_noise)
        elif min_search == False: 
            BGOmodel = Global_max(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num, ret_noise)
        else:
            print('type ERROR! -opt_num-')
        return BGOmodel
  

