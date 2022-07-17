from .BGOmax import Global_max
from .BGOmin import Global_min

class Bgolearn(object):
    def fit(self,Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num = 1 ,min_search = True):
        
        """
        PACKAGE: Bayesian global optimization 

        10 Jul 2022, version 1, Bin Cao, MGI, SHU, Shanghai, CHINA.

        :param Kriging_model: a defined callable Kriging model, has an attribute of <fit_pre>
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

        :param data_matrix: data matrix of training dataset, X 

        :param Measured_response: response of tarining dataset, y

        :param virtual_samples: designed virtual samples

        :param opt_num: the number of recommended candidates for next iteration, default 1 

        :param min_search: default True -> searching the global minimum ; False -> searching the global maximum

        :return: the recommended candidates
        """
        if min_search == True:
            BGOmodel = Global_min(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num )
        elif min_search == False: 
            BGOmodel = Global_max(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num )
        else:
            print('type ERROR! -opt_num-')
        return BGOmodel
  

