import copy,os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import multiprocess as mp
import multiprocessing


class Global_min(object):
    def __init__(self,Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num, ret_noise, row_features):
        self.Kriging_model = Kriging_model
        self.data_matrix = np.array(data_matrix)
        self.Measured_response = np.array(Measured_response)
        __fea_num = len(self.data_matrix[0])
        self.virtual_samples = np.array(virtual_samples).reshape(-1,__fea_num)
        if ret_noise == 0:
            self.virtual_samples_mean, self.virtual_samples_std = self.Kriging_model().fit_pre(
                self.data_matrix, self.Measured_response, self.virtual_samples)
        else:
            self.virtual_samples_mean, self.virtual_samples_std = self.Kriging_model().fit_pre(
                self.data_matrix, self.Measured_response, self.virtual_samples,0.0)
        self.opt_num = opt_num
        self.ret_noise = ret_noise
        self.row_features = row_features
        warnings.filterwarnings('ignore')
        os.environ["PYTHONWARNINGS"] = "ignore"
   
    
    def EI(self,):
        """
        Expected Improvement algorith
        """
        cur_optimal_value = self.Measured_response.min()
        print('current optimal is :', cur_optimal_value)
        EI_list = []
        for i in range(len(self.virtual_samples_mean)):
            Var_Z = (cur_optimal_value - self.virtual_samples_mean[i])/self.virtual_samples_std[i] 
            EI = (cur_optimal_value - self.virtual_samples_mean[i]) * norm.cdf(Var_Z)+ self.virtual_samples_std[i] * norm_des(Var_Z)
            EI_list.append(EI)
       
        EI_list = np.array(EI_list)

        return_x = []
        if self.opt_num == 1:
            EI_opt_index = np.random.choice(np.flatnonzero(EI_list == EI_list.max()))
            print('The next datum recomended by Expected Improvement : \n x = ', self.row_features[EI_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[EI_opt_index])
            return_x.append(self.row_features[EI_opt_index])
        elif type(self.opt_num) == int:
            EI_opt_index = np.argsort(EI_list)[-self.opt_num:][::-1]  
            for j in range(len(EI_opt_index)):
                print('The {num}-th datum recomended by Expected Improvement : \n x = '.format(num =j+1), self.row_features[EI_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[EI_opt_index[j]])
                return_x.append(self.row_features[EI_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return EI_list, np.array(return_x)

    def EI_plugin(self,):
        """
        Expected improvement with “plugin”
        """
        if self.ret_noise == 0:
            __train_ypre,_ = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix)
        else:
            __train_ypre,_ = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix,0.0)
        cur_optimal_value = __train_ypre.min()
        print('current optimal is :', cur_optimal_value)
        EIp_list = []
        for i in range(len(self.virtual_samples_mean)):
            Var_Z = (cur_optimal_value - self.virtual_samples_mean[i])/self.virtual_samples_std[i] 
            EIp = (cur_optimal_value - self.virtual_samples_mean[i]) * norm.cdf(Var_Z)+ self.virtual_samples_std[i] * norm_des(Var_Z)
          
            EIp_list.append(EIp)
       
        EIp_list = np.array(EIp_list)
        
        return_x = []
        if self.opt_num == 1:
            EIp_opt_index = np.random.choice(np.flatnonzero(EIp_list == EIp_list.max()))
            print('The next datum recomended by Expected Improvement with plugin : \n x = ', self.row_features[EIp_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[EIp_opt_index])
            return_x.append(self.row_features[EIp_opt_index])    
        elif type(self.opt_num) == int:
            EIp_opt_index = np.argsort(EIp_list)[-self.opt_num:][::-1]  
            for j in range(len(EIp_opt_index)):
                print('The {num}-th datum recomended by Expected Improvement with plugin : \n x = '.format(num =j+1), self.row_features[EIp_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[EIp_opt_index[j]])
                return_x.append(self.row_features[EIp_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return EIp_list,np.array(return_x)
    
    
    def Augmented_EI(self, alpha = 1, tao = 0):
        """
        Augmented Expected Improvement
        :param alpha: tradeoff coefficient, default 1
        :param tao: noise standard deviation, default 0
        """
        # cal current optimal 
        if self.ret_noise == 0:
            Pre_response_means, Pre_response_std = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix)
        else:
            Pre_response_means, Pre_response_std = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix,0.0)
        proposal_fun_value = np.array(Pre_response_means) + alpha * np.array(Pre_response_std)
        cur_opt_index = np.random.choice(np.flatnonzero(proposal_fun_value == proposal_fun_value.min()))
        cur_optimal_value = Pre_response_means[cur_opt_index]

        print('current optimal is :', cur_optimal_value)
        AEI_list = []
        for i in range(len(self.virtual_samples_mean)):
            Var_Z = (cur_optimal_value - self.virtual_samples_mean[i])/self.virtual_samples_std[i] 
            EI = (cur_optimal_value - self.virtual_samples_mean[i]) * norm.cdf( Var_Z ) + self.virtual_samples_std[i] * norm_des(Var_Z)
            AEI = EI * (1 - tao/np.sqrt(self.virtual_samples_std[i]**2 + tao**2))
            AEI_list.append(AEI)
        AEI_list = np.array(AEI_list)
        
        return_x = []
        if self.opt_num == 1:
            AEI_opt_index = np.random.choice(np.flatnonzero(AEI_list == AEI_list.max()))
            print('The next datum recomended by Augmented_EI : \n x = ', self.row_features[AEI_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[AEI_opt_index])
            return_x.append(self.row_features[AEI_opt_index])
        elif type(self.opt_num) == int:
            AEI_opt_index = np.argsort(AEI_list)[-self.opt_num:][::-1]  
            for j in range(len(AEI_opt_index)):
                print('The {num}-th datum recomended by Augmented_EI : \n x = '.format(num =j+1), self.row_features[AEI_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[AEI_opt_index[j]])
                return_x.append(self.row_features[AEI_opt_index[j]]) 
        else:
            print('The input para. opt_num must be an int')

        return AEI_list,np.array(return_x)

    def EQI(self, beta = 0.5,tao_new = 0):
        """
        Expected Quantile Improvement
        :param beta: beta quantile number, default 0.5
        :param tao: noise standard deviation, default 0
        """
        # cal current optimal
        if self.ret_noise == 0: 
            Pre_response_means, Pre_response_std = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix)
        else:
            Pre_response_means, Pre_response_std = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix,0.0)
        q_value = np.array(Pre_response_means) + norm.ppf(beta) * np.array(Pre_response_std)
        cur_optimal_value = q_value.min()
        print('current optimal is :', cur_optimal_value)

        __mean = self.virtual_samples_mean + norm.ppf(beta) * np.sqrt(tao_new**2 * self.virtual_samples_std**2 / (tao_new**2 + self.virtual_samples_std**2))
        __std = self.virtual_samples_std**2 / np.sqrt(self.virtual_samples_std**2 + tao_new**2) 
       
        EQI_list = []
        for i in range(len(self.virtual_samples_mean)):
            
            Var_Z = (cur_optimal_value - __mean[i]) / __std[i] 
            EQI = (cur_optimal_value - __mean[i]) * norm.cdf( Var_Z ) + __std[i] * norm_des(Var_Z)
            EQI_list.append(EQI)
        EQI_list = np.array(EQI_list)
        
        return_x = []
        if self.opt_num == 1:
            EQI_opt_index = np.random.choice(np.flatnonzero(EQI_list == EQI_list.max()))
            print('The next datum recomended by Expected Quantile Improvement : \n x = ', self.row_features[EQI_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[EQI_opt_index])
            return_x.append(self.row_features[EQI_opt_index])
        elif type(self.opt_num) == int:
            EQI_opt_index = np.argsort(EQI_list)[-self.opt_num:][::-1]   
            for j in range(len(EQI_opt_index)):
                print('The {num}-th datum recomended by Expected Quantile Improvement : \n x = '.format(num =j+1), self.row_features[EQI_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[EQI_opt_index[j]])
                return_x.append(self.row_features[EQI_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')

        return EQI_list,np.array(return_x)

    def Reinterpolation_EI(self, ):
        """
        Reinterpolation Expected Improvement
        """
        if self.ret_noise == 0: 
            __update_y,_ = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix)
        else:
            __update_y,_ = self.Kriging_model().fit_pre(self.data_matrix,self.Measured_response, self.data_matrix,0.0)

        cur_optimal_value = __update_y.min()
        print('current optimal is :', cur_optimal_value)
        if self.ret_noise == 0: 
            __y,__std = self.Kriging_model().fit_pre(self.data_matrix,__update_y, self.virtual_samples)
        else:
            __y,__std = self.Kriging_model().fit_pre(self.data_matrix,__update_y, self.virtual_samples,0.0)
        REI_list = []
        for i in range(len(__y)):
            Var_Z = (cur_optimal_value - __y[i])/__std[i] 
            REI = (cur_optimal_value - __y[i]) * norm.cdf(Var_Z)+ __std[i] * norm_des(Var_Z)
            REI_list.append(REI)
       
        REI_list = np.array(REI_list)
        
        return_x = []
        if self.opt_num == 1:
            REI_opt_index = np.random.choice(np.flatnonzero(REI_list == REI_list.max()))
            print('The next datum recomended by Reinterpolation Expected Improvement : \n x = ', self.row_features[REI_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[REI_opt_index])
            return_x.append(self.row_features[REI_opt_index])
        elif type(self.opt_num) == int:
            REI_opt_index = np.argsort(REI_list)[-self.opt_num:][::-1]  
            for j in range(len(REI_opt_index)):
                print('The {num}-th datum recomended by Reinterpolation Expected Improvement : \n x = '.format(num =j+1), self.row_features[REI_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[REI_opt_index[j]])
                return_x.append(self.row_features[REI_opt_index[j]]) 
        else:
            print('The input para. opt_num must be an int')
        return REI_list,np.array(return_x)

    def UCB(self, alpha=1):
        """
        Upper confidence bound 
        :param alpha: tradeoff coefficient, default 1
        """
        UCB_list = np.array(self.virtual_samples_mean) - alpha * np.array(self.virtual_samples_std)
        
        return_x = []
        if self.opt_num == 1:
            UCB_opt_index = np.random.choice(np.flatnonzero(UCB_list == UCB_list.min()))
            print('The next datum recomended by Upper confidence bound  : \n x = ', self.row_features[UCB_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[UCB_opt_index])
            return_x.append(self.row_features[UCB_opt_index])    
        elif type(self.opt_num) == int:
            UCB_opt_index = np.argsort(UCB_list)[:self.opt_num]
            for j in range(len(UCB_opt_index)):
                print('The {num}-th datum recomended by Upper confidence bound : \n x = '.format(num =j+1), self.row_features[UCB_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[UCB_opt_index[j]])
                return_x.append(self.row_features[UCB_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')

        return UCB_list,np.array(return_x)
    
    def PoI(self, tao = 0):
        """
        Probability of Improvement
        :param tao: improvement ratio (>=0) , default 0
        """
        if tao < 0:
            print('Type Error \'EXACT\'','please input a Non-negative value of tao')
        else:
            cur_optimal_value = (1 - tao) * self.Measured_response.min() 
            print('current optimal is :', cur_optimal_value)
            PoI_list = []
            for i in range(len(self.virtual_samples_mean)):
                PoI = norm.cdf((cur_optimal_value - self.virtual_samples_mean[i])/self.virtual_samples_std[i])
                PoI_list.append(PoI)
        
            PoI_list = np.array(PoI_list)
            
            return_x = []
            if self.opt_num == 1:
                PoI_opt_index = np.random.choice(np.flatnonzero(PoI_list == PoI_list.max()))
                print('The next datum recomended by Probability of Improvement  : \n x = ', self.row_features[PoI_opt_index])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[PoI_opt_index])
                return_x.append(self.row_features[PoI_opt_index])
            elif type(self.opt_num) == int:
                PoI_opt_index = np.argsort(PoI_list)[-self.opt_num:][::-1]   
                for j in range(len(PoI_opt_index)):
                    print('The {num}-th datum recomended by Probability of Improvement  : \n x = '.format(num =j+1), self.row_features[PoI_opt_index[j]])
                    print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[PoI_opt_index[j]])
                    return_x.append(self.row_features[PoI_opt_index[j]])
            else:
                print('The input para. opt_num must be an int')
            return PoI_list,np.array(return_x)
    
    def Thompson_sampling(self,):
        # x* is derived by searching at the vistual space 
        # sample = np.len(virtual_samples)
        optimal_value_set = []
        for i in range(len(self.virtual_samples)):
            y_value = np.random.normal(loc = self.virtual_samples_mean[i],scale = self.virtual_samples_std[i])
            optimal_value_set.append(y_value)
        index = np.where(np.array(optimal_value_set)==np.array(optimal_value_set).min())[0][0]
        return self.virtual_samples[index], optimal_value_set[index],self.virtual_samples_std[index]
    
    def PES(self, sam_num = 500):
        """
        Predictive Entropy Search
        :param sam_num: number of optimal drawn from p(x*|D), D is support set, default 500
        """
        # sam_num: number of optimal drawn from p(x*|D),D is support set
        # ref. paper: Predictive Entropy Search for Efficient Global Optimization of Black-box Functions
        Entropy_y_ori = 0.5*np.log(2*np.pi*np.e*(self.virtual_samples_std**2))
    
        # defined Monte carol
        Entropy_y_conditional = np.zeros(len(self.virtual_samples))
        for i in range(sam_num):   
            sample_x, sample_y,sample_noise = self.Thompson_sampling()
            
            archive_sample_x = copy.deepcopy(self.data_matrix)
            archive_sample_y = copy.deepcopy(self.Measured_response)
            
            archive_sample_x = np.append(archive_sample_x, sample_x)
            archive_sample_y = np.append(archive_sample_y, sample_y)
            fea_num = len(self.data_matrix[0])
            # return a callable model
            if self.ret_noise == True:
                _, post_std = self.Kriging_model().fit_pre(archive_sample_x.reshape(-1, fea_num), archive_sample_y, self.virtual_samples,sample_noise)
            else:
                _, post_std = self.Kriging_model().fit_pre(archive_sample_x.reshape(-1, fea_num), archive_sample_y, self.virtual_samples)
            
            Entropy_y_conditional += 0.5*np.log(2*np.pi*np.e*(post_std**2))
        estimated_Entropy_y_conditional = Entropy_y_conditional / sam_num            
        PES_list = Entropy_y_ori - estimated_Entropy_y_conditional
        
        return_x = []
        if self.opt_num == 1:
            PES_opt_index = np.random.choice(np.flatnonzero(PES_list == PES_list.max()))
            print('The next datum recomended by Predictive Entropy Search  : \n x = ', self.row_features[PES_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[PES_opt_index])
            return_x.append(self.row_features[PES_opt_index])
        elif type(self.opt_num) == int:
            PES_opt_index = np.argsort(PES_list)[-self.opt_num:][::-1]   
            for j in range(len(PES_opt_index)):
                print('The {num}-th datum recomended by Predictive Entropy Search  : \n x = '.format(num =j+1), self.row_features[PES_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[PES_opt_index[j]])
                return_x.append(self.row_features[PES_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return PES_list,np.array(return_x)
    
  
    def __Knowledge_G_per_sample(self, func_bytes:bytes, MC_num:int, virtual_samples, v_sample_mean, v_sample_std, archive_sample_x, archive_sample_y, x_value, fea_num, ret_noise):
        MC_batch_min = 0
        for _ in range(MC_num):
            # generate y value
            y_value = np.random.normal(loc = v_sample_mean, scale = v_sample_std)
            # update the sample x and sample y
            archive_sample_x[-len(x_value):] = x_value
            if isinstance(y_value, float):
                archive_sample_y[-1] = y_value
            elif isinstance(y_value, np.ndarray):
                archive_sample_y[-len(y_value):] = y_value
            # calculate the post mean
            if ret_noise == True:
                # return a callable model
                post_mean, _ = func_bytes.fit_pre(archive_sample_x.reshape(-1, fea_num), archive_sample_y, virtual_samples, v_sample_std)
            else:
                post_mean, _ = func_bytes.fit_pre(archive_sample_x.reshape(-1, fea_num), archive_sample_y, virtual_samples)
            MC_batch_min += post_mean.min()
        return MC_batch_min
    
    def Knowledge_G(self,MC_num = 1, Proc_num:int=None):
        """
        Calculate the Knowledge Gradient for Bayesian optimization.

        :param MC_num: Number of Monte Carlo simulations, default 1. (1-10)
        :param Proc_num: Number of processors, default None (0) for single process.
        :return: Knowledge Gradient values and recommended data points.

        for windows operating systems, please ensures that the code inside the if __name__ == '__main__': 
        e.g.,
        import multiprocessing as mp
        if __name__ == '__main__':
            # Freeze support for Windows
            mp.freeze_support()

            # Call your function
            Mymodel.Knowledge_G(MC_num=100,Proc_num=6)
        """
        current_min = self.virtual_samples_mean.min()
        KD_list = []
        fea_num = len(self.data_matrix[0])
        archive_sample_x = np.append(self.data_matrix[:], self.virtual_samples[0])
        archive_sample_y = np.append(self.Measured_response[:], self.virtual_samples_mean[0])
        K_model = self.Kriging_model()
        results = []
        i = 0
        if not Proc_num:
            print('Execution using a single process')
            for x_value, v_sample_mean, v_sample_std in zip(self.virtual_samples, self.virtual_samples_mean, self.virtual_samples_std):
                MC_batch_min= self.__Knowledge_G_per_sample(K_model, MC_num, self.virtual_samples, v_sample_mean, v_sample_std, archive_sample_x, archive_sample_y, x_value, fea_num, self.ret_noise)
                MC_result = MC_batch_min / MC_num
                KD_list.append( current_min - MC_result)
                i += 1
                print('The Monte Carlo simulation has been performed {num} times.'.format(num = i * MC_num))
        else:

            # Call this function at the beginning of your script
            setup_multiprocessing()
            print('Execution using multiple processes, processes num = {},'.format(Proc_num) )
            with mp.get_context("spawn").Pool(Proc_num) as pool:
                results=[pool.apply_async(self.__Knowledge_G_per_sample, args=(K_model, MC_num, self.virtual_samples, v_sample_mean, v_sample_std, archive_sample_x, archive_sample_y, x_value, fea_num, self.ret_noise)) for x_value, v_sample_mean, v_sample_std in zip(self.virtual_samples, self.virtual_samples_mean, self.virtual_samples_std)]
                for idx, rst in enumerate(results):
                    MC_batch_min = rst.get()
                    MC_result = MC_batch_min / MC_num
                    KD_list.append( current_min - MC_result)
                    i += 1
                    print('The Monte Carlo simulation has been performed {num} times.'.format(num = i * MC_num))
        KD_list = np.array(KD_list)

        return_x = []
        if self.opt_num == 1:
            KD_opt_index = np.random.choice(np.flatnonzero(KD_list == KD_list.max()))
            print('The next datum recomended by Knowledge Gradient : \n x = ', self.row_features[KD_opt_index])
            print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[KD_opt_index])
            return_x.append(self.row_features[KD_opt_index])
        elif type(self.opt_num) == int:
            KD_opt_index = np.argpartition(KD_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(KD_opt_index)):
                print('The {num}-th datum recomended by Knowledge Gradient : \n x = '.format(num =j+1), self.row_features[KD_opt_index[j]])
                print('The predictions of Bgolearn are : \n y = ', self.virtual_samples_mean[KD_opt_index[j]])
                return_x.append(self.row_features[KD_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return KD_list,np.array(return_x)
    

def setup_multiprocessing():
    if multiprocessing.get_start_method() != 'fork':
        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            print('\'fork\' method not available, using the default')

# cal norm prob.
def norm_des(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2)