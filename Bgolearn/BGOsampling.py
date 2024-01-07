import inspect
import os
import time
from typing import Any
import warnings
import numpy as np
import pandas as pd
import copy
import datetime
from art import text2art
from sklearn.utils import resample
from .BgolearnFuns.BGOmax import Global_max
from .BgolearnFuns.BGOmin import Global_min
from .BgolearnFuns.BGOclf import Boundary
from .BgolearnFuns.BGO_eval import BGO_Efficient
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

class Bgolearn(object):
    def __init__(self) -> None:
        os.makedirs('Bgolearn', exist_ok=True)
        
        now = datetime.datetime.now()
        formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')
        print(text2art("Bgolearn"))
        print('Package Name : Bgolearn')
        print('Author : Bin CAO, HKUST(GZ)')
        print('Intro : https://bgolearn.netlify.app/')
        print('URL : https://github.com/Bin-Cao/Bgolearn')
        print("Citation Format Suggestion:")
        print('[Bin CAO et al]. "Active learning accelerates the discovery of high strength and high ductility lead-free solder alloys", [2023], [DOI : http://dx.doi.org/10.2139/ssrn.4686075].')
        print('Executed on :',formatted_date_time, ' | Have a great day.')  
        print('='*80)


    def fit(self,data_matrix, Measured_response, virtual_samples, Mission ='Regression', Classifier = 'GaussianProcess',noise_std = None, Kriging_model = None, opt_num = 1 ,min_search = True, CV_test = False, Dynamic_W = False,seed=42):
        
        """
        ================================================================
        PACKAGE: Bayesian global optimization-learn (Bgolearn) package .
        Author: Bin CAO <binjacobcao@gmail.com> 
        Guangzhou Municipal Key Laboratory of Materials Informatics, Advanced Materials Thrust,
        Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511400, Guangdong, China
        ================================================================
        Please feel free to open issues in the Github :
        https://github.com/Bin-Cao/Bgolearn
        or 
        contact Mr.Bin Cao (bcao686@connect.hkust-gz.edu.cn)
        in case of any problems/comments/suggestions in using the code. 
        ==================================================================
        Thank you for choosing Bgolearn for material design. 
        Bgolearn is developed to facilitate the application of machine learning in research.

        Bgolearn is designed for optimizing single-target material properties. 
        The BgoKit package is being developed to facilitate multi-task design.

        If you need to perform multi-target optimization, here are two kind reminders:
        1. Multi-tasks can be converted into a single task using domain knowledge. 
        For example, you can use a weighted linear combination in the simplest situation. That is, y = w*y1 + y2...

        2. Multi-tasks can be optimized using Pareto fronts. 
        Bgolearn will return two arrays based on your dataset: 
        the first array is a evaluation score for each virtual sample, 
        while the second array is the recommended data considering only the current optimized target.

        The first array is crucial for multi-task optimization. 
        For instance, in a two-task optimization scenario, you can evaluate each candidate twice for the two separate targets. 
        Then, plot the score of target 1 for each sample on the x-axis and the score of target 2 on the y-axis. 
        The trade-off consideration is to select the data located in the front of the banana curve.

        I am delighted to invite you to participate in the development of Bgolearn. 
        If you have any issues or suggestions, please feel free to contact me at binjacobcao@gmail.com.
        ================================================================
        Reference : 
        document : https://bgolearn.netlify.app/
        ================================================================

        :param data_matrix: data matrix of training dataset, X .

        :param Measured_response: response of tarining dataset, y.

        :param virtual_samples: designed virtual samples.

        :param Mission: str, default 'Regression', the mission of optimization.  Mission = 'Regression' or 'Classification'

        :param Classifier: if  Mission == 'Classification', classifier is used.
                if user isn't applied one, Bgolearn will call a pre-set classifier.
                default, Classifier = 'GaussianProcess', i.e., Gaussian Process Classifier.
                five different classifiers are pre-setd in Bgolearn:
                'GaussianProcess' --> Gaussian Process Classifier (default)
                'LogisticRegression' --> Logistic Regression
                'NaiveBayes' --> Naive Bayes Classifier
                'SVM' --> Support Vector Machine Classifier
                'RandomForest' --> Random Forest Classifier

        :param noise_std: float or ndarray of shape (n_samples,), default=None
                Value added to the diagonal of the kernel matrix during fitting.
                This can prevent a potential numerical issue during fitting, by
                ensuring that the calculated values form a positive definite matrix.
                It can also be interpreted as the variance of additional Gaussian.
                measurement noise on the training observations.

                if noise_std is not None, a noise value will be estimated by maximum likelihood
                on training dataset.

        :param Kriging_model (default None):
                str, Kriging_model = 'SVM', 'RF', 'AdaB', 'MLP'
                The  machine learning models will be implemented: Support Vector Machine (SVM), 
                Random Forest(RF), AdaBoost(AdaB), and Multi-Layer Perceptron (MLP).
                The estimation uncertainity will be determined by Boostsrap sampling.
            or  
                a user defined callable Kriging model, has an attribute of <fit_pre>
                if user isn't applied one, Bgolearn will call a pre-set Kriging model
                atribute <fit_pre> : 
                input -> xtrain, ytrain, xtest ; 
                output -> predicted  mean and std of xtest

                e.g. (take GaussianProcessRegressor in sklearn):
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest):
                        # instantiated model
                        kernel = RBF()
                        mdoel = GaussianProcessRegressor(kernel=kernel).fit(xtrain,ytrain)
                        # defined the attribute's outputs
                        mean,std = mdoel.predict(xtest,return_std=True)
                        return mean,std    

                e.g. (MultiModels estimations):
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest):
                        # instantiated model
                        pre_1 = SVR(C=10).fit(xtrain,ytrain).predict(xtest) # model_1
                        pre_2 = SVR(C=50).fit(xtrain,ytrain).predict(xtest) # model_2
                        pre_3 = SVR(C=80).fit(xtrain,ytrain).predict(xtest) # model_3
                        model_1 , model_2 , model_3  can be changed to any ML models you desire
                        # defined the attribute's outputs
                        stacked_array = np.vstack((pre_1,pre_2,pre_3))
                        means = np.mean(stacked_array, axis=0)
                        std = np.sqrt(np.var(stacked_array), axis=0)
                        return mean, std    

        :param opt_num: the number of recommended candidates for next iteration, default 1. 

        :param min_search: default True -> searching the global minimum ;
                                   False -> searching the global maximum.

        :param CV_test: 'LOOCV' or an int, default False (pass test) 
                if CV_test = 'LOOCV', LOOCV will be applied,
                elif CV_test = int, e.g., CV_test = 10, 10 folds cross validation will be applied.
        
        :return: 1: array; potential of each candidate. 2: array/float; recommended candidate(s).

        """
        
        # Fit and transform the input data matrix
        Xname = data_matrix.columns
        virtual_samples = preprocess_data(virtual_samples)
        data_matrix = preprocess_data(data_matrix)
        Measured_response = preprocess_data(Measured_response)
        if Dynamic_W == False :
            pass
        elif Dynamic_W == True :
            data_matrix, Measured_response = Resampling(data_matrix,Measured_response,min_search,seed )


        row_features = copy.deepcopy(virtual_samples)
        scaler = MinMaxScaler()
        virtual_samples = scaler.fit_transform(virtual_samples)
        data_matrix = scaler.transform(data_matrix)

        timename = time.localtime(time.time())
        namey, nameM, named, nameh, namem = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min

        warnings.filterwarnings('ignore')

        if Mission == 'Classification':
            if type(Classifier) == str:
                model = Classifier_selection(Classifier)
                print(model)
                BGOmodel = Boundary(model,data_matrix, Measured_response, row_features, opt_num, virtual_samples)
                return BGOmodel
            else:
                print('Type Error! Classifier should be one of the following:')
                print('GaussianProcess; LogisticRegression;NaiveBayes;SVM;RandomForest')


        elif Mission == 'Regression':
        
            if Kriging_model == None:
                kernel = 1 * RBF() 
                if noise_std == None:
                    # call the default model;
                    class Kriging_model(object):
                        def fit_pre(self,xtrain,ytrain,xtest,):
                            # estimating Noise Level of training dataset
                            noise_ker = WhiteKernel(noise_level_bounds=(0.001,0.5))
                            GPr = GaussianProcessRegressor(kernel= 1 * RBF()+noise_ker,normalize_y=True).fit(xtrain,ytrain)
                            noise_level = np.exp(GPr.kernel_.theta[1])
        
                            # ret_std is a placeholder for homogenous noise
                            # instantiated mode
                            mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha = noise_level).fit(xtrain,ytrain)
                            # defined the attribute's outputs
                            mean,std = mdoel.predict(xtest,return_std=True)
                            return mean,std 
                    print('The internal model is instantiated with optimized homogenous noise')  

                elif type(noise_std) == float:
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
            elif type(Kriging_model) == str:
                model_type = Kriging_model
                class Kriging_model(object):
                    def fit_pre(self,xtrain,ytrain,xtest,):
                        mean,std = Bgolearn_model(xtrain,ytrain,xtest,model_type)
                        return mean,std 
                print('The internal model is assigned')

            else: 
                print('The external model is instantiated')
                pass  
            
            
            # position incluse 'self'
            if len(inspect.getfullargspec(Kriging_model().fit_pre)[0]) == 5:
                ret_noise = True
            elif len(inspect.getfullargspec(Kriging_model().fit_pre)[0]) == 4:
                ret_noise = False
            else:
                print('type ERROR! -ILLEGAL form of Krigging-')

            # fitting results
            X_true = np.array(data_matrix)
            Y_true = np.array(Measured_response)
            __fea_num = len(X_true[0])
        
            # test default model
            if CV_test == False:
                pass

            else:
                if type(CV_test) != int and CV_test != 'LOOCV':
                    print('type ERROR! - CV_test should be an int or \'LOOCV\'')
                elif CV_test == 'LOOCV':
                    print('Time consuming warning')
                    print('LeaveOneOut Cross validation is applied')
                    loo = LeaveOneOut()
                    loo.get_n_splits(X_true)
                    pre = []
                    if ret_noise == False:
                        _Y_pre, _ = Kriging_model().fit_pre(X_true , Y_true, X_true.reshape(-1,__fea_num))
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true , Y_true, virtual_samples.reshape(-1,__fea_num))
                        for train_index, test_index in loo.split(X_true):
                            X_train, X_test = X_true[train_index], X_true[test_index]
                            y_train, _ = Y_true[train_index], Y_true[test_index]
                            Y_pre, _ = Kriging_model().fit_pre( X_train , y_train, X_test)
                            pre.append(Y_pre)
                            
                    else:
                        _Y_pre, _ = Kriging_model().fit_pre(X_true , Y_true, X_true.reshape(-1,__fea_num),0.0)
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true , Y_true, virtual_samples.reshape(-1,__fea_num),0.0)
                        for train_index, test_index in loo.split(X_true):
                            X_train, X_test = X_true[train_index], X_true[test_index]
                            y_train, _ = Y_true[train_index], Y_true[test_index]
                            Y_pre, _ = Kriging_model().fit_pre( X_train , y_train, X_test,0.0)
                            pre.append(Y_pre)
                            
                else:          
                    print('Time consuming warning')
                    print('{num}-folds Cross validation is applied'.format(num=CV_test))
                    kfold = Bgo_KFold(X_true, Y_true, CV_test)
                    pre_list = []
                    index_list = []
                    
                    if ret_noise == False:
                        _Y_pre, _ = Kriging_model().fit_pre(X_true , Y_true, X_true.reshape(-1,__fea_num))
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true , Y_true, virtual_samples.reshape(-1,__fea_num))
                        for train_index, test_index in kfold:
                            X_train = X_true[train_index]  
                            y_train = Y_true[train_index]  
                            X_test = X_true[test_index]   
                            # y_test = Y_true[test_index] 
                            index_list.append(list(test_index))
                            Y_pre, _ = Kriging_model().fit_pre( X_train , y_train, X_test)
                            pre_list.append(list(Y_pre))
                            

                    for train_index, test_index in kfold:
                        _Y_pre, _ = Kriging_model().fit_pre(X_true , Y_true, X_true.reshape(-1,__fea_num),0.0)
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true , Y_true, virtual_samples.reshape(-1,__fea_num),0.0)
                        for train_index, test_index in kfold:
                            X_train = X_true[train_index]  
                            y_train = Y_true[train_index]  
                            X_test = X_true[test_index]   
                            # y_test = Y_true[test_index] 
                            index_list.append(list(test_index))  
                            Y_pre, _ = Kriging_model().fit_pre( X_train , y_train, X_test,0.0)
                            pre_list.append(list(Y_pre))

                            
                    pre_mixed =  [float(x) for item in pre_list for x in item]
                    index = [float(x) for item in index_list for x in item]
                    # match order of pre with the order of original ytrues
                    order = np.array(index).argsort()
                    pre = []
                    for i in range(len(order)):
                        pre.append(pre_mixed[order[i]])
                
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

                print('Fitting goodness on training dataset: \n' + str('  RMSE = %f' % _RMSE) +' '+ str('  MAE = %f' % _MAE)
                    +' '+ str('  R2 = %f' % _R2))

                print('Fitting goodness of {}:'.format(docu_name(CV_test)))
                print(str('  RMSE = %f' % RMSE) +' '+ str('  MAE = %f' % MAE) +' '+ str('  R2 = %f' % R2))

            

                results_dataset.to_csv('./Bgolearn/predictions{name}_{year}_{month}_{day}_{hour}_{minute}.csv'.format(name=docu_name(CV_test),year=namey, month=nameM, day=named, hour=nameh,
                                                                                minute=namem),encoding='utf-8-sig')
                
                _results_dataset.to_csv('./Bgolearn/predictionsOnTrainingDataset_{year}_{month}_{day}_{hour}_{minute}.csv'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                                minute=namem),encoding='utf-8-sig')

                V_Xmatrix.to_csv('./Bgolearn/predictionsOfVirtualSampels_{year}_{month}_{day}_{hour}_{minute}.csv'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                                minute=namem),encoding='utf-8-sig')


            

            
            arv_vs = pd.DataFrame(np.array(virtual_samples))
            arv_vs.columns = Xname
            pre,_ = Kriging_model().fit_pre(data_matrix, Measured_response, virtual_samples)
            arv_vs['Y'] = np.array(pre)
            arv_vs.to_csv('./Bgolearn/PredictionsByBgolearn_{year}_{month}_{day}_{hour}_{minute}.csv'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                                minute=namem), encoding='utf-8-sig')

            # BGO
            if min_search == True:
                BGOmodel = Global_min(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num, ret_noise,row_features)
            elif min_search == False: 
                BGOmodel = Global_max(Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num, ret_noise,row_features)
            else:
                print('type ERROR! -opt_num-')
            return BGOmodel
        else:
            print('type ERROR! -MISSION-')

    def test(self,Ture_fun, Def_Domain,noise_std = 1e-5, Kriging_model = None, opt_num = 1 ,min_search = True):
        
        """
        PACKAGE: Bayesian global optimization learn .

        :param Ture_fun: the true function being evaluated. e.g.,
                def function(X):
                    X = np.array(X)
                    Y = 0.013*X**4 - 0.25*X**3 + 1.61*X**2 - 4.1*X + 8
                    return Y

        :param Def_Domain: discrete function Domain. e.g., Def_Domain = numpy.linspace(0,11,111)

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
        """

        print(' test model is developed for evaluating the regression efficiency')

        warnings.filterwarnings('ignore')

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
        
        # position incluse 'self'
        if len(inspect.getfullargspec(Kriging_model().fit_pre)[0]) == 5:
            ret_noise = True
        elif len(inspect.getfullargspec(Kriging_model().fit_pre)[0]) == 4:
            ret_noise = False
        else:
            print('type ERROR! -ILLEGAL form of Krigging-')

        print('Evaluation is executed')
        
        Eval_model = BGO_Efficient(Ture_fun,Def_Domain, Kriging_model, opt_num, ret_noise,min_search)
        return Eval_model
      
def Bgo_KFold(x_train, y_train,cv):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    kfolder = KFold(n_splits=cv, shuffle=True,random_state=0)
    kfold = kfolder.split(x_train, y_train)
    return kfold

def docu_name(CV_test):
    if CV_test == 'LOOCV':
        return 'LOOCV'
    elif type(CV_test) == int:
        return '{}-CVs'.format(CV_test)
    else:
        print('type error')

def Classifier_selection(Classifier):
    if Classifier == 'GaussianProcess':
        from sklearn.gaussian_process import GaussianProcessClassifier 
        model = GaussianProcessClassifier(kernel= 1*RBF(1.0) ,random_state=0)
    elif Classifier == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0,class_weight='balanced',multi_class='multinomial')
    elif Classifier == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif Classifier == 'SVM':
        from sklearn.svm import SVC
        model = SVC(probability=True)
    elif Classifier == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(max_depth=4,random_state=0)
    else :
        print('type ERROR! -Classifier-')
    return model


def preprocess_data(data):
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = np.reshape(data, (-1, 1))
    elif isinstance(data, list):
        data = np.array(data).reshape(-1, 1)
    data = np.array(data)
    return data


def Bgolearn_model(xtrain,ytrain,xtest,Kriging_model):
    models = {
    'SVM': SVR(),
    'RF': RandomForestRegressor(),
    'AdaB': AdaBoostRegressor(),
    'MLP': MLPRegressor()
    }   
    try:
        Bgo_model = models[Kriging_model]
        print('Bgolearn model : ', Bgo_model)
    except:
        print('Type Error: Kriging_model, please check your input of param Kriging_model')

    all_predictions = []
    for _ in range(10):
        # Perform Bootstrap sampling
        X_bootstrap, y_bootstrap = resample(xtrain, ytrain)
        predictions = Bgo_model.fit(X_bootstrap, y_bootstrap).predict(xtest)
        # Store the predictions
        all_predictions.append(predictions)

    # Convert the list of predictions to a NumPy array for easier calculations
    all_predictions = np.array(all_predictions)
    # Calculate mean and standard deviation across the samples
    mean = np.mean(all_predictions, axis=0)
    std = np.std(all_predictions, axis=0)
    return mean, std

def Resampling(data_matrix,Measured_response,min_search,seed_):
    
    np.random.seed(seed_)
    max_value = max(Measured_response)
    min_value = min(Measured_response)
    prob = (Measured_response - min_value) / (max_value - min_value)
    if min_search == True:
        prob = 1 - prob
    cdf = np.cumsum(prob)
    cdf_ = cdf / cdf[-1]
    uniform_samples = np.random.random_sample(len(Measured_response))
    bootstrap_idx = cdf_.searchsorted(uniform_samples, side='right')
    # searchsorted returns a scalar
    bootstrap_idx = np.array(bootstrap_idx, copy=False)
    print('Importance resampling is APPLIED','\n')

    
    return data_matrix[bootstrap_idx], Measured_response[bootstrap_idx]