"""
BGO Evaluation Module

This module provides evaluation tools for Bayesian Global Optimization (BGO) algorithms.
It includes performance metrics, statistical analysis, and visualization capabilities
for comparing different acquisition functions and optimization strategies.

Key Features:
- Trail-based performance evaluation
- Opportunity cost analysis
- Statistical comparison between acquisition functions
- Success rate counting
- Visualization of optimization paths

Author: Bin Cao
Institution: Hong Kong University of Science and Technology (Guangzhou)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
from .BGOmax import Global_max
from .BGOmin import Global_min


def cal_area(array, length):
    """
    Calculate the area under a curve using trapezoidal rule.

    Args:
        array: Array of y-values representing the curve
        length: Step size between x-values

    Returns:
        float: Area under the curve
    """
    area = 0
    for i in range(len(array)-1):
        area += (array[i] + array[i+1])/2 * length
    return area


def Cal_total_area(array, length=1):
    """
    Calculate total area for multiple curves.

    Args:
        array: List of arrays, each representing a curve [[curve1], [curve2], ...]
        length: Step size between x-values (default=1)

    Returns:
        float: Sum of areas under all curves
    """
    Total_area = 0
    for i in range(len(array)):
        Total_area += cal_area(array[i], length)
    return Total_area

class BGO_Efficient(object):
    """
    Bayesian Global Optimization Efficiency Evaluator

    This class provides comprehensive evaluation tools for assessing the performance
    of different acquisition functions in Bayesian optimization. It supports various
    metrics including convergence analysis, opportunity cost evaluation, and
    statistical comparisons.

    Attributes:
        Ture_fun: True objective function for evaluation
        Def_Domain: Definition domain (search space) for optimization
        dim: Dimensionality of the search space
        total_space: Total number of points in the search space
        Ymax: Maximum value in the true function
        Ymin: Minimum value in the true function
        ref_scale: Reference scale (Ymax - Ymin) for normalization
        Kriging_model: Surrogate model for optimization
        opt_num: Number of optimal candidates to recommend
        ret_noise: Whether to return noise estimates
        min_search: Whether to search for minimum (True) or maximum (False)
    """

    def __init__(self, Ture_fun, Def_Domain, Kriging_model, opt_num, ret_noise, min_search):
        """
        Initialize the BGO efficiency evaluator.

        Args:
            Ture_fun: True objective function to evaluate
            Def_Domain: Search space definition
            Kriging_model: Surrogate model for predictions
            opt_num: Number of optimal candidates per iteration
            ret_noise: Whether to return noise estimates
            min_search: True for minimization, False for maximization
        """
        self.Ture_fun = Ture_fun
        self.Def_Domain = np.array(Def_Domain)
        self.dim = len(pd.DataFrame(self.Def_Domain).iloc[0,:])
        self.total_space = len(self.Def_Domain)
        self.Ymax = Ture_fun(self.Def_Domain).max()
        self.Ymin = Ture_fun(self.Def_Domain).min()
        self.ref_scale = self.Ymax - self.Ymin
        self.Kriging_model = Kriging_model
        self.opt_num = opt_num
        self.ret_noise = ret_noise
        self.min_search = min_search
        # Create output directory for results
        os.makedirs('Bgolearn', exist_ok=True)
       
    def Call(self, BGO_mdoel, UTFs='EI', param_one=None, param_two=None):
        """
        Call the specified acquisition function with appropriate parameters.

        This method serves as a dispatcher for different acquisition functions,
        handling parameter passing and function selection.

        Args:
            BGO_mdoel: BGO model instance with acquisition function methods
            UTFs: Acquisition function name (default='EI')
                Available options: 'EI', 'EI_plugin', 'Augmented_EI', 'EQI',
                'Reinterpolation_EI', 'UCB', 'PoI', 'PES', 'Knowledge_G'
            param_one: First parameter for the acquisition function (optional)
            param_two: Second parameter for the acquisition function (optional)

        Returns:
            Result from the specified acquisition function
        """
        if UTFs == 'EI':
            # Expected Improvement
            return BGO_mdoel.EI()
        elif UTFs == 'EI_plugin':
            # Expected Improvement with plugin estimator
            return BGO_mdoel.EI_plugin()
        elif UTFs == 'Augmented_EI':
            # Augmented Expected Improvement
            if param_one == None and param_two == None:
                return BGO_mdoel.Augmented_EI()
            elif param_one != None and param_two != None:
                return BGO_mdoel.Augmented_EI(alpha=param_one, tao=param_two)
            elif param_one != None and param_two == None:
                return BGO_mdoel.Augmented_EI(alpha=param_one)
            else:
                return BGO_mdoel.Augmented_EI(tao=param_two)
        elif UTFs == 'EQI':
            # Expected Quantile Improvement
            if param_one == None and param_two == None:
                return BGO_mdoel.EQI()
            elif param_one != None and param_two != None:
                return BGO_mdoel.EQI(beta=param_one, tao_new=param_two)
            elif param_one != None and param_two == None:
                return BGO_mdoel.EQI(beta=param_one)
            else:
                return BGO_mdoel.EQI(tao_new=param_two)
        elif UTFs == 'Reinterpolation_EI':
            # Reinterpolation Expected Improvement
            return BGO_mdoel.Reinterpolation_EI()
        elif UTFs == 'UCB':
            # Upper Confidence Bound
            if param_one == None:
                return BGO_mdoel.UCB()
            else:
                return BGO_mdoel.UCB(alpha=param_one)
        elif UTFs == 'PoI':
            # Probability of Improvement
            if param_one == None:
                return BGO_mdoel.PoI()
            else:
                return BGO_mdoel.PoI(tao=param_one)
        elif UTFs == 'PES':
            # Predictive Entropy Search
            if param_one == None:
                return BGO_mdoel.PES()
            else:
                return BGO_mdoel.PES(sam_num=param_one)
        elif UTFs == 'Knowledge_G':
            # Knowledge Gradient
            if param_one == None:
                return BGO_mdoel.Knowledge_G()
            else:
                return BGO_mdoel.Knowledge_G(MC_num=param_one)
        else:
            print('type ERROR! -UTFs-')



    def Trail(self, trails=100, Max_inter=500, tol=0.1, ini_nb=None, UTFs='EI', param_one=None, param_two=None):
        """
        Evaluate optimization performance through multiple independent trials.

        This method runs multiple optimization trials to assess the convergence
        performance of different acquisition functions. It measures the number
        of iterations required to reach the global optimum within a specified tolerance.

        Args:
            trails (int, default=100): Total number of independent trials to run
            Max_inter (int, default=500): Maximum iterations allowed per trial
            tol (float, default=0.1): Convergence tolerance criteria:
                - For minimization: current_optimal <= (1+tol) * global_optimal
                - For maximization: current_optimal >= (1-tol) * global_optimal
            ini_nb (int, optional): Number of initial training samples
                If None, uses 1% of total search space size
            UTFs (str, default='EI'): Acquisition function to evaluate
                Options: 'EI', 'EI_plugin', 'Augmented_EI', 'EQI',
                'Reinterpolation_EI', 'UCB', 'PoI', 'PES', 'Knowledge_G'
            param_one (optional): First parameter for the acquisition function
            param_two (optional): Second parameter for the acquisition function

        Returns:
            tuple: (mean_iterations, variance_iterations)
                - mean_iterations: Average number of iterations to convergence
                - variance_iterations: Variance in iteration counts across trials

        Note:
            Results are automatically saved to './Bgolearn/Trail_{UTFs}.csv'
        """
        # Validate and set initial sample size
        if ini_nb is None:
            ini_nb = int(0.01 * self.total_space)  # Default: 1% of search space
        elif type(ini_nb) == int:
            pass  # Use provided value
        else:
            print('type ERROR! -ini_nb-')

        # Handle minimization problems
        if self.min_search == True:

            Iter_Num = []  # Store iteration counts for each trial
            for i in range(trails):
                # Initialize training data with random samples
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)

                Iter = 0  # Iteration counter for current trial
                for j in range(Max_inter):
                    # Create BGO model for minimization
                    BGO_mdoel = Global_min(self.Kriging_model, train_X, train_Y, self.Def_Domain,
                            self.opt_num, self.ret_noise, self.Def_Domain)
                    # Get next candidate using specified acquisition function
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)
                    Iter += 1

                    # Check convergence criteria for minimization
                    if np.array(new_Y).min() <= (1+tol) * self.Ymin:
                        Iter_Num.append(Iter)
                        print('\n')
                        break
                    elif Iter == Max_inter - 1:
                        # Maximum iterations reached without convergence
                        Iter_Num.append(Max_inter)
                        print('\n')
                        break
                    else:
                        # Add new sample to training data and continue
                        train_X = np.append(train_X, new_X).reshape(-1, self.dim)
                        train_Y = np.append(train_Y, new_Y)

        
        
        elif self.min_search == False:
            Iter_Num = []
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)
                
                Iter = 0
                for j in range(Max_inter):
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_max(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)
                    Iter += 1
        
                    if np.array(new_Y).max() >= (1-tol) * self.Ymax:
                        Iter_Num.append(Iter)
                        print('\n')
                        break
                    elif Iter == Max_inter - 1: 
                        Iter_Num.append(Max_inter)
                        print('\n')
                        break
                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)
        else:
            print('Type Error: min_search')

        mean = np.array(Iter_Num).sum()/trails
        var = ((np.array(Iter_Num) - mean)**2).sum() / (trails-1)
        pd.DataFrame(Iter_Num).to_csv('./Bgolearn/Trail_{name}.csv'.format(name = UTFs),encoding='utf-8-sig')
        print('Mean: {}'.format(mean))
        print('Variance: {}'.format(var))

        return mean,var
    



    def Opp_Cost(self,trails = 10, Max_inter = 500, threshold = 0.05, ini_nb = None, UTFs = 'EI',param_one = None, param_two = None):
        """
        : param trails: int, default = 10. the total number of trails.
        
        : param Max_inter: int, default = 500, the maximum number of iterations in each trail.
        
        : param threshold: float, default = 0.05, the coverage critera of normalized OC value.
        
        : param ini_nb: int, default = None,the number of initial sampled training data. 
            If ini_nb = None, ini_nb = 0.01*total number of Def_Domain is applied
        
        : param UTFs: string, default = 'EI', the evaluated acaquisition function.
            e.g., 'EI','EI_plugin','Augmented_EI','EQI','Reinterpolation_EI','UCB','PoI','PES','Knowledge_G'
        
        : param param_one, param_two : default = None, the parametres of the UTFs, more detailes see (Utility Function): https://github.com/Bin-Cao/Bgolearn 
        """
        if ini_nb is None:
            ini_nb = int(0.01 * self.total_space)
        elif type(ini_nb) == int:
            pass
        else:
            print('type ERROR! -ini_nb-')
        
        if self.min_search == True:
            
            OC_list = []
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)
                
                OC_value = []
                Iter = 0
                for j in range(Max_inter):
                    Iter += 1
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_min(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)

                    OC_inter = (train_Y.min() - self.Ymin) / self.ref_scale
                    OC_value.append(OC_inter)
        
                    if OC_inter <= threshold:
                        OC_list.append(OC_value)
                        print('\n')
                        break
                    elif Iter == Max_inter - 1: 
                        OC_list.append(OC_value)
                        print('\n')
                        break
                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)
        elif self.min_search == False:
            
            OC_list = []
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)
                
                OC_value = []
                Iter = 0
                for j in range(Max_inter):
                    Iter +=1
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_max(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)

                    OC_inter = (self.Ymax - train_Y.max()) / self.ref_scale
                    OC_value.append(OC_inter)
        
                    if OC_inter <= threshold:
                        OC_list.append(OC_value)
                        print('\n')
                        break
                    elif Iter == Max_inter - 1: 
                        OC_list.append(OC_value)
                        print('\n')
                        break
                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)
        else:
            print('Type Error: min_search')

        OC_area = round(Cal_total_area(OC_list, 1),4)

        plt.figure()
        ax = plt.subplot()
        plt.title('Average Area: {}'.format(OC_area))
        plt.ylabel('OCi',fontsize = 15)
        plt.xlabel('iter num',fontsize = 15)
        for i in range(len(OC_list)):
            ax.plot(range(len(OC_list[i])),OC_list[i])
        plt.savefig('./Bgolearn/OC_path_{name}.png'.format(name = UTFs),dpi=800)
        plt.show()

    def Pdf(self,trails = 200, Max_inter = 500, tol = 0.1, num_bins = 20, ini_nb = None, UTFs = 'EI',param_one = None, param_two = None,Ref_UTFs = 'EI',ref_param_one = None, ref_param_two = None,):
        """
        : param trails: int, default = 200. the total number of trails.

        : param Max_inter: int, default = 500, the maximum number of iterations in each trail.

        : param tol: float, default = 0.1, the tolerance of coverage critera. viz., 
            (Search minimum)if current optimal <= (1+tol) * global optimal, this trail treats as coveraged.
            (Search maximum)if current optimal >= (1-tol) * global optimal, this trail treats as coveraged.

        : param num_bins: int, default = 20, the number of bins in the histogram

        : param ini_nb: int, default = None,the number of initial sampled training data. 
            If ini_nb = None, ini_nb = 0.01*total number of Def_Domain is applied

        : param UTFs: string, default = 'EI', the evaluated acaquisition function.
            e.g., 'EI','EI_plugin','Augmented_EI','EQI','Reinterpolation_EI','UCB','PoI','PES','Knowledge_G'

        : param param_one, param_two : default = None, the parametres of the UTFs, more detailes see (Utility Function): https://github.com/Bin-Cao/Bgolearn 
        
        : param Ref_UTFs: string, default = 'EI', the reference acaquisition function.
            e.g., 'EI','EI_plugin','Augmented_EI','EQI','Reinterpolation_EI','UCB','PoI','PES','Knowledge_G'
        
        : param Ref_param_one, Ref_param_two : default = None, the parametres of the UTFs, more detailes see (Utility Function): https://github.com/Bin-Cao/Bgolearn 
        """
        if ini_nb is None:
            ini_nb = int(0.01 * self.total_space)
        elif type(ini_nb) == int:
            pass
        else:
            print('type ERROR! -ini_nb-')
        
        
        if self.min_search == True:
            
            Iter_Num = []
            ref_Iter_Num = []
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)
                
                # loop one with same initial training data
                print('Iteration in {name1} method:'.format(name1=UTFs))
                Iter = 0
                for j in range(Max_inter):
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_min(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)
                    Iter += 1
        
                    if np.array(new_Y).min() <= (1+tol) * self.Ymin:
                        Iter_Num.append(Iter)
                        print('\n')
                        break
                    elif Iter == Max_inter - 1: 
                        Iter_Num.append(Max_inter)
                        print('\n')
                        break
                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)

                # loop two with same initial training data
                print('Iteration in {name2} method:'.format(name2=Ref_UTFs))
                ref_Iter = 0
                for j in range(Max_inter):
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_min(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, Ref_UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)
                    ref_Iter += 1
        
                    if np.array(new_Y).min() <= (1+tol) * self.Ymin:
                        ref_Iter_Num.append(Iter)
                        print('\n')
                        break
                    elif ref_Iter == Max_inter - 1:
                        ref_Iter_Num.append(Max_inter)
                        print('\n')
                        break
                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)

        elif self.min_search == False:
            
            Iter_Num = []
            ref_Iter_Num = []
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)
                
                # loop one with same initial training data

                print('Iteration in {name1} method:'.format(name1 = UTFs))
                Iter = 0
                for j in range(Max_inter):
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_max(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)
                    Iter += 1
        
                    if np.array(new_Y).max() >= (1-tol) * self.Ymax:
                        Iter_Num.append(Iter)
                        print('\n')
                        break
                    elif Iter == Max_inter - 1: 
                        Iter_Num.append(Max_inter)
                        print('\n')
                        break
                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)

                # loop two with same initial training data
                print('Iteration in {name2} method:'.format(name2=Ref_UTFs))
                ref_Iter = 0
                for j in range(Max_inter):
                    # Max_inter = 500 is a enough large number
                    BGO_mdoel = Global_max(self.Kriging_model,train_X, train_Y, self.Def_Domain, 
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, Ref_UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)
                    ref_Iter += 1
        
                    if np.array(new_Y).max() >= (1-tol) * self.Ymax:
                        ref_Iter_Num.append(Iter)
                        print('\n')
                        break

                    elif ref_Iter == Max_inter - 1:
                        ref_Iter_Num.append(Max_inter)
                        print('\n')
                        break

                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)
        
        diff = np.array(Iter_Num) - np.array(ref_Iter_Num)
        pd.DataFrame(diff).to_csv('./Bgolearn/∆{name1}-{name2}.csv'.format(name1 = Ref_UTFs,name2 = UTFs),encoding='utf-8-sig')

        # plot
        diff_pd = pd.DataFrame(diff)
        print(diff_pd.describe()) 
        x = diff_pd.iloc[:,0] 
        mu =np.mean(x) 
        sigma =np.std(x) 
        mu,sigma

        prob = 1- st.norm.cdf((0 - mu)/(sigma+1e-10))

        num_bins = num_bins
        n, bins, patches = plt.hist(x, num_bins,density= True,stacked=True,facecolor='blue', edgecolor="b", alpha=0.5) 
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')  
        plt.xlabel('∆') 
        plt.ylabel('p.d.f.') 
        plt.title(r'$\mu={mu}$,$\sigma={sigma}$,$P(>0)={prob}$'.format(mu=round(mu,4), sigma=round(sigma,4),prob=round(prob,4)))
        plt.subplots_adjust(left=0.15)
        plt.tick_params(labelsize=16)
        plt.savefig('./Bgolearn/∆{name1}-{name2}.png'.format(name1 = Ref_UTFs,name2 = UTFs),dpi=800)
        plt.show()

    def Count(self,trails = 100, Max_inter = 5, tol = 0.1, ini_nb = None, UTFs = 'EI',param_one = None, param_two = None):
        """
        : param trails: int, default = 100. the total number of trails.
        
        : param Max_inter: int, default = 5, the maximum number of iterations in each trail.
        
        : param tol: float, default = 0.1, the tolerance of coverage critera. viz., 
            (Search minimum)if current optimal <= (1+tol) * global optimal, this trail treats as coveraged.
            (Search maximum)if current optimal >= (1-tol) * global optimal, this trail treats as coveraged.
        
        : param ini_nb: int, default = None,the number of initial sampled training data. 
            If ini_nb = None, ini_nb = 0.01*total number of Def_Domain is applied
        
        : param UTFs: string, default = 'EI', the evaluated acaquisition function.
            e.g., 'EI','EI_plugin','Augmented_EI','EQI','Reinterpolation_EI','UCB','PoI','PES','Knowledge_G'
        
        : param param_one, param_two : default = None, the parametres of the UTFs, more detailes see (Utility Function): https://github.com/Bin-Cao/Bgolearn 
        """
        if ini_nb is None:
            ini_nb = int(0.01 * self.total_space)
        elif type(ini_nb) == int:
            pass
        else:
            print('type ERROR! -ini_nb-')
        
        if self.min_search == True:

            Success_Cot = 0
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)

                success = False

                for j in range(Max_inter):
                    # Max_inter is the threshold
                    BGO_mdoel = Global_min(self.Kriging_model,train_X, train_Y, self.Def_Domain,
                            self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)


                    if np.array(new_Y).min() <= (1+tol) * self.Ymin:
                        success = True
                        print('\n')
                        break

                    else:
                        train_X = np.append(train_X,new_X).reshape(-1,self.dim)
                        train_Y = np.append(train_Y,new_Y)
                Success_Cot += success

        elif self.min_search == False:

            Success_Cot = 0
            for i in range(trails):
                train_X = np.array(random.sample(list(self.Def_Domain), ini_nb))
                train_Y = self.Ture_fun(train_X)
                train_X = pd.DataFrame(train_X)

                success = False

                for j in range(Max_inter):
                    # Max_inter is the threshold
                    BGO_mdoel = Global_max(self.Kriging_model, train_X, train_Y, self.Def_Domain,
                                           self.opt_num, self.ret_noise,self.Def_Domain)
                    _, return_x = self.Call(BGO_mdoel, UTFs, param_one, param_two)
                    new_X = return_x
                    new_Y = self.Ture_fun(new_X)

                    if np.array(new_Y).max() >= (1 - tol) * self.Ymax:
                        success = True
                        print('\n')
                        break

                    else:
                        train_X = np.append(train_X, new_X).reshape(-1, self.dim)
                        train_Y = np.append(train_Y, new_Y)
                Success_Cot += success
        print('\n')
        print('After %f trails :' %trails +'%f times success; '% Success_Cot + '%f times failed'% (trails-Success_Cot) )
