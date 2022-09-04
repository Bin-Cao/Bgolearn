import numpy as np
from .BGOmax import Global_max
from .BGOmin import Global_min
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF

class Bgolearn(object):
    def fit(self,data_matrix, Measured_response, virtual_samples, noise_std = 1e-5, Kriging_model = None, opt_num = 1 ,min_search = True):
        
        """
        PACKAGE: Bayesian global optimization 

        10 Jul 2022, version 1, Bin Cao, MGI, SHU, Shanghai, CHINA.

        :param data_matrix: data matrix of training dataset, X 

        :param Measured_response: response of tarining dataset, y

        :param virtual_samples: designed virtual samples

        :param noise_std: float or ndarray of shape (n_samples,), default=1e-5
                Value added to the diagonal of the kernel matrix during fitting.
                This can prevent a potential numerical issue during fitting, by
                ensuring that the calculated values form a positive definite matrix.
                It can also be interpreted as the variance of additional Gaussian
                measurement noise on the training observations.

        :param Kriging_model (default None): a user defined callable Kriging model, has an attribute of <fit_pre>
                if user isn't applied one, Bgolearn will call a pre-set Kriging model
                atribute <fit_pre> : 
                input -> xtrain, ytrain, xtest ; output -> predicted  mean and std of xtest
                e.g. (take GaussianProcessRegressor in sklearn as an example):
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest):
                        # instantiated model
                        kernel = RBF()
                        mdoel = GaussianProcessRegressor(kernel=kernel).fit(xtrain,ytrain)
                        # defined the attribute's outputs
                        mean,std = mdoel.predict(xtest,return_std=True)
                        return mean,std    

        :param opt_num: the number of recommended candidates for next iteration, default 1 

        :param min_search: default True -> searching the global minimum ; False -> searching the global maximum

        :return: the recommended candidates
        """

        if Kriging_model == None:
            kernel = RBF() 
            if type(noise_std) == float:
                # call the default model;
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest,ret_std = 0.0):
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

        if min_search == True:
            BGOmodel = Global_min(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num )
        elif min_search == False: 
            BGOmodel = Global_max(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num )
        else:
            print('type ERROR! -opt_num-')
        return BGOmodel
  

