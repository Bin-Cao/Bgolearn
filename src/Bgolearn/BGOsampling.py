# Standard library imports
import copy
import datetime
import inspect
import os
import time
import warnings
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from art import text2art
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.utils import resample

# Local imports
from .BgolearnFuns.BGO_eval import BGO_Efficient
from .BgolearnFuns.BGOclf import Boundary
from .BgolearnFuns.BGOmax import Global_max
from .BgolearnFuns.BGOmin import Global_min

class Bgolearn(object):
    """
    Bayesian Global Optimization Learning (Bgolearn) Class

    A comprehensive machine learning framework for materials design and optimization
    using Bayesian optimization principles. Supports both regression and classification
    tasks with various machine learning models and cross-validation techniques.

    Author: Bin CAO <binjacobcao@gmail.com>
    Institution: Hong Kong University of Science and Technology (Guangzhou)

    References:
        - Materials & Design: https://doi.org/10.1016/j.matdes.2024.112921
        - MGE Advances: https://onlinelibrary.wiley.com/doi/10.1002/mgea.70031
        - NPJ Computational Materials: https://doi.org/10.1038/s41524-024-01243-4
    """

    def __init__(self) -> None:
        """
        Initialize the Bgolearn instance.

        Creates the output directory for results and displays package information
        including author details, citations, and execution timestamp.
        """
        # Create output directory for storing results
        os.makedirs('Bgolearn', exist_ok=True)

        # Display package information and branding
        now = datetime.datetime.now()
        formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')
        print(text2art("Bgolearn"))
        print('Bgolearn, Bin CAO, Hong Kong University of Science and Technology (Guangzhou)' )
        print('URL : https://bgolearn.netlify.app/')
        print("Citation: ")
        print('Materials & Design : https://doi.org/10.1016/j.matdes.2024.112921')
        print('MGE Advances : https://onlinelibrary.wiley.com/doi/10.1002/mgea.70031')
        print('NPJ Com. Mat. : https://doi.org/10.1038/s41524-024-01243-4')
        print('Executed on :',formatted_date_time, ' | Have a great day.')
        print('='*80)


    def fit(self,data_matrix, Measured_response, virtual_samples, Mission ='Regression', Classifier = 'GaussianProcess',noise_std = None,
            Kriging_model = None, opt_num = 1 ,min_search = True, CV_test = False, Dynamic_W = False,seed=42,
            Normalize = True,):
        """
        Fit the Bayesian Global Optimization model and recommend optimal candidates.

        This method performs Bayesian optimization to identify the most promising candidates
        from a virtual sample space based on training data. It supports both regression and
        classification tasks with various machine learning models and evaluation strategies.

        Multi-objective Optimization Guidelines:
        ======================================
        For multi-target optimization, consider these approaches:

        1. Single-task conversion: Combine multiple objectives using domain knowledge
           Example: y_combined = w1*y1 + w2*y2 + ... (weighted linear combination)

        2. Pareto front analysis: Use the evaluation scores for each objective separately
           - First array: evaluation scores for each virtual sample
           - Second array: recommended candidates for current target
           - Plot scores for different objectives to identify Pareto-optimal solutions

        Contact Information:
        ===================
        Author: Bin CAO <binjacobcao@gmail.com>
        Institution: Hong Kong University of Science and Technology (Guangzhou)
        GitHub: https://github.com/Bin-Cao/Bgolearn
        Documentation: https://bgolearn.netlify.app/

        Parameters:
        -----------
        data_matrix : array-like or DataFrame
            Training dataset feature matrix (X). Contains the input features for model training.

        Measured_response : array-like
            Training dataset response values (y). Target values corresponding to data_matrix.

        virtual_samples : array-like
            Virtual candidate samples for optimization. These are the potential solutions
            to be evaluated and ranked by the optimization algorithm.

        Mission : str, default='Regression'
            Optimization task type. Options:
            - 'Regression': Continuous target optimization
            - 'Classification': Discrete class boundary optimization

        Classifier : str, default='GaussianProcess'
            Classifier model for classification tasks. Available options:
            - 'GaussianProcess': Gaussian Process Classifier (default)
            - 'LogisticRegression': Logistic Regression Classifier
            - 'NaiveBayes': Naive Bayes Classifier
            - 'SVM': Support Vector Machine Classifier
            - 'RandomForest': Random Forest Classifier

        noise_std : float, array-like, or None, default=None
            Noise standard deviation for Gaussian Process models.
            - float: Homogeneous noise across all observations
            - array-like: Heterogeneous noise (one value per training sample)
            - None: Automatic noise estimation via maximum likelihood

            This parameter helps prevent numerical issues and can represent
            measurement uncertainty in the training observations.

        Kriging_model : str, callable, or None, default=None
            Machine learning model for regression tasks. Options:

            Built-in models (str):
            - 'SVM': Support Vector Machine with bootstrap uncertainty estimation
            - 'RF': Random Forest with bootstrap uncertainty estimation
            - 'AdaB': AdaBoost with bootstrap uncertainty estimation
            - 'MLP': Multi-Layer Perceptron with bootstrap uncertainty estimation

            Custom model (callable):
            User-defined model class with 'fit_pre' method. The method should:
            - Input: xtrain, ytrain, xtest
            - Output: predicted mean and standard deviation for xtest

            Example - Gaussian Process model:
                class Kriging_model(object):
                    def fit_pre(self, xtrain, ytrain, xtest):
                        kernel = RBF()
                        model = GaussianProcessRegressor(kernel=kernel).fit(xtrain, ytrain)
                        mean, std = model.predict(xtest, return_std=True)
                        return mean, std

            Example - Ensemble model:
                class Kriging_model(object):
                    def fit_pre(self, xtrain, ytrain, xtest):
                        # Multiple model predictions
                        pred_1 = SVR(C=10).fit(xtrain, ytrain).predict(xtest)
                        pred_2 = SVR(C=50).fit(xtrain, ytrain).predict(xtest)
                        pred_3 = SVR(C=80).fit(xtrain, ytrain).predict(xtest)

                        # Ensemble statistics
                        stacked_array = np.vstack((pred_1, pred_2, pred_3))
                        mean = np.mean(stacked_array, axis=0)
                        std = np.std(stacked_array, axis=0)
                        return mean, std

        opt_num : int, default=1
            Number of optimal candidates to recommend for the next iteration.

        min_search : bool, default=True
            Optimization direction:
            - True: Search for global minimum (minimization)
            - False: Search for global maximum (maximization)

        CV_test : str, int, or False, default=False
            Cross-validation strategy for model evaluation:
            - False: Skip cross-validation
            - 'LOOCV': Leave-One-Out Cross-Validation
            - int: k-fold cross-validation (e.g., CV_test=10 for 10-fold CV)

        Dynamic_W : bool, default=False
            Enable dynamic importance resampling based on response values.
            When True, samples with better objective values have higher selection probability.

        seed : int, default=42
            Random seed for reproducible results in stochastic operations.

        Normalize : bool, default=True
            Whether to normalize input features using MinMaxScaler.
            Recommended to keep True for consistent model performance.

        Returns:
        --------
        tuple
            - evaluation_scores : array-like
                Potential/evaluation score for each virtual sample candidate
            - recommended_candidates : array-like
                Top recommended candidate(s) for next iteration
        """

        # Extract feature column names for later use
        Xname = data_matrix.columns

        # Preprocess and reshape input data to ensure consistent array format
        virtual_samples = preprocess_data(virtual_samples)
        data_matrix = preprocess_data(data_matrix)
        Measured_response = preprocess_data(Measured_response)

        # Apply dynamic importance resampling if enabled
        if Dynamic_W == False :
            pass  # Use original data without resampling
        elif Dynamic_W == True :
            # Resample training data based on response values to emphasize better samples
            data_matrix, Measured_response = Resampling(data_matrix,Measured_response,min_search,seed )

        # Store original virtual samples before normalization for output
        row_features = copy.deepcopy(virtual_samples)

        # Apply feature normalization if enabled
        if Normalize == True:
            # Fit scaler on virtual samples and transform both datasets
            scaler = MinMaxScaler()
            virtual_samples = scaler.fit_transform(virtual_samples)
            data_matrix = scaler.transform(data_matrix)
        else:pass

        # Generate timestamp for output file naming
        timename = time.localtime(time.time())
        namey, nameM, named, nameh, namem = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min

        # Suppress sklearn warnings for cleaner output
        warnings.filterwarnings('ignore')

        # Handle classification tasks
        if Mission == 'Classification':
            if type(Classifier) == str:
                # Use predefined classifier from string identifier
                model = Classifier_selection(Classifier)
                print(model)
                BGOmodel = Boundary(model,data_matrix, Measured_response, row_features, opt_num, virtual_samples)
                return BGOmodel
            else:
                # Display error message for invalid classifier type
                print('Type Error! Classifier should be one of the following:')
                print('GaussianProcess; LogisticRegression;NaiveBayes;SVM;RandomForest')

        # Handle regression tasks
        elif Mission == 'Regression':

            # Configure Kriging model based on user input
            if Kriging_model == None:
                # Use default Gaussian Process with RBF kernel
                kernel = 1 * RBF()

                if noise_std == None:
                    # Automatic noise estimation via maximum likelihood
                    class Kriging_model(object):
                        def fit_pre(self,xtrain,ytrain,xtest,):
                            # Estimate optimal noise level using WhiteKernel
                            noise_ker = WhiteKernel(noise_level_bounds=(0.001,0.5))
                            GPr = GaussianProcessRegressor(kernel= 1 * RBF()+noise_ker,normalize_y=True).fit(xtrain,ytrain)
                            noise_level = np.exp(GPr.kernel_.theta[1])

                            # Create final model with estimated noise level
                            mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha = noise_level).fit(xtrain,ytrain)
                            # Generate predictions with uncertainty estimates
                            mean,std = mdoel.predict(xtest,return_std=True)
                            return mean,std
                    print('The internal model is instantiated with optimized homogenous noise')

                elif type(noise_std) == float:
                    # Fixed homogeneous noise across all observations
                    class Kriging_model(object):
                        def fit_pre(self,xtrain,ytrain,xtest,):
                            # Apply fixed noise level to all training points
                            mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha = noise_std**2).fit(xtrain,ytrain)
                            # Generate predictions with uncertainty estimates
                            mean,std = mdoel.predict(xtest,return_std=True)
                            return mean,std
                    print('The internal model is instantiated with homogenous noise: %s' % noise_std)
                
                elif type(noise_std) == np.ndarray:
                    # Handle heterogeneous noise (different noise for each data point)
                    class Kriging_model(object):
                        def fit_pre(self, xtrain, ytrain, xtest, ret_std=0.0):
                            """Fit Gaussian Process with heterogeneous noise."""
                            # Check if noise array matches training data size
                            if len(xtrain) == len(noise_std):
                                # Use provided noise values
                                mdoel = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=noise_std**2).fit(xtrain, ytrain)
                            elif len(xtrain) == len(noise_std) + 1:
                                # Append noise for new data point
                                new_alpha = np.append(noise_std, ret_std)
                                mdoel = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=new_alpha**2).fit(xtrain, ytrain)
                            else:
                                print('the input data is not matched with heterogeneous noise size')
                            # Generate predictions with uncertainty
                            mean, std = mdoel.predict(xtest, return_std=True)
                            return mean, std
                    print('The internal model is instantiated with heterogeneous noise')
            elif type(Kriging_model) == str:
                # Use string-specified model type
                model_type = Kriging_model
                class Kriging_model(object):
                    def fit_pre(self, xtrain, ytrain, xtest):
                        """Fit model using string-specified type."""
                        mean, std = Bgolearn_model(xtrain, ytrain, xtest, model_type)
                        return mean, std
                print('The internal model is assigned')

            else:
                # Use externally provided model
                print('The external model is instantiated')
                pass


            # Determine noise handling capability by inspecting method signature
            # Position includes 'self' parameter
            if len(inspect.getfullargspec(Kriging_model().fit_pre)[0]) == 5:
                ret_noise = True  # Model supports noise parameter
            elif len(inspect.getfullargspec(Kriging_model().fit_pre)[0]) == 4:
                ret_noise = False  # Model does not support noise parameter
            else:
                print('type ERROR! -ILLEGAL form of Kriging-')

            # Prepare training data for model fitting
            X_true = np.array(data_matrix)
            Y_true = np.array(Measured_response)
            __fea_num = len(X_true[0])

            # Cross-validation testing (optional)
            if CV_test == False:
                # Skip cross-validation testing
                pass

            else:
                # Validate CV_test parameter
                if type(CV_test) != int and CV_test != 'LOOCV':
                    print('type ERROR! - CV_test should be an int or \'LOOCV\'')
                elif CV_test == 'LOOCV':
                    # Leave-One-Out Cross-Validation
                    print('Time consuming warning')
                    print('LeaveOneOut Cross validation is applied')
                    loo = LeaveOneOut()
                    loo.get_n_splits(X_true)
                    pre = []  # Store cross-validation predictions

                    if ret_noise == False:
                        # Model without noise parameter
                        _Y_pre, _ = Kriging_model().fit_pre(X_true, Y_true, X_true.reshape(-1, __fea_num))
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true, Y_true, virtual_samples.reshape(-1, __fea_num))
                        for train_index, test_index in loo.split(X_true):
                            X_train, X_test = X_true[train_index], X_true[test_index]
                            y_train, _ = Y_true[train_index], Y_true[test_index]
                            Y_pre, _ = Kriging_model().fit_pre(X_train, y_train, X_test)
                            pre.append(Y_pre)

                    else:
                        # Model with noise parameter
                        _Y_pre, _ = Kriging_model().fit_pre(X_true, Y_true, X_true.reshape(-1, __fea_num), 0.0)
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true, Y_true, virtual_samples.reshape(-1, __fea_num), 0.0)
                        for train_index, test_index in loo.split(X_true):
                            X_train, X_test = X_true[train_index], X_true[test_index]
                            y_train, _ = Y_true[train_index], Y_true[test_index]
                            Y_pre, _ = Kriging_model().fit_pre(X_train, y_train, X_test, 0.0)
                            pre.append(Y_pre)

                else:
                    # K-Fold Cross-Validation
                    print('Time consuming warning')
                    print('{num}-folds Cross validation is applied'.format(num=CV_test))
                    kfold = Bgo_KFold(X_true, Y_true, CV_test)
                    pre_list = []  # Store predictions for each fold
                    index_list = []  # Store test indices for each fold

                    if ret_noise == False:
                        # Model without noise parameter
                        _Y_pre, _ = Kriging_model().fit_pre(X_true, Y_true, X_true.reshape(-1, __fea_num))
                        V_Y_pre, V_Y_std = Kriging_model().fit_pre(X_true, Y_true, virtual_samples.reshape(-1, __fea_num))
                        for train_index, test_index in kfold:
                            X_train = X_true[train_index]
                            y_train = Y_true[train_index]
                            X_test = X_true[test_index]
                            # Store test indices for result reconstruction
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