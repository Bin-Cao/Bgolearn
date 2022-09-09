import copy
import warnings
import numpy as np
from scipy.stats import norm

# cal norm prob.
def norm_des(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2)

class Global_min(object):
    def __init__(self,Kriging_model,data_matrix, Measured_response, virtual_samples, opt_num, ret_noise):
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
        warnings.filterwarnings('ignore')
   
    
    
    def EI(self,):
        cur_optimal_value = self.Measured_response.min()
        print('current optimal is :', cur_optimal_value)
        EI_list = []
        for i in range(len(self.virtual_samples_mean)):
            Var_Z = (cur_optimal_value - self.virtual_samples_mean[i])/self.virtual_samples_std[i] 
            EI = (cur_optimal_value - self.virtual_samples_mean[i]) * norm.cdf(Var_Z)+ self.virtual_samples_std[i] * norm_des(Var_Z)
          
            EI_list.append(EI)
       
        EI_list = np.array(EI_list)
        
        if self.opt_num == 1:
            EI_opt_index = np.random.choice(np.flatnonzero(EI_list == EI_list.max()))
            print('The next datum recomended by Expected Improvement : \n x = ', self.virtual_samples[EI_opt_index])
        elif type(self.opt_num) == int:
            EI_opt_index = np.argpartition(EI_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(EI_opt_index)):
                print('The {num}-th datum recomended by Expected Improvement : \n x = '.format(num =j+1), self.virtual_samples[EI_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return EI_list

    
    
    def EI_plugin(self,):
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
        
        if self.opt_num == 1:
            EIp_opt_index = np.random.choice(np.flatnonzero(EIp_list == EIp_list.max()))
            print('The next datum recomended by Expected Improvement with plugin : \n x = ', self.virtual_samples[EIp_opt_index])
        elif type(self.opt_num) == int:
            EIp_opt_index = np.argpartition(EIp_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(EIp_opt_index)):
                print('The {num}-th datum recomended by Expected Improvement with plugin : \n x = '.format(num =j+1), self.virtual_samples[EIp_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return EIp_list
    
    
    def Augmented_EI(self, alpha = 1, tao = 0):
        """
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
        
        if self.opt_num == 1:
            AEI_opt_index = np.random.choice(np.flatnonzero(AEI_list == AEI_list.max()))
            print('The next datum recomended by Augmented_EI : \n x = ', self.virtual_samples[AEI_opt_index])
        elif type(self.opt_num) == int:
            AEI_opt_index = np.argpartition(AEI_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(AEI_opt_index)):
                print('The {num}-th datum recomended by Augmented_EI : \n x = '.format(num =j+1), self.virtual_samples[AEI_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')

        return AEI_list

    def EQI(self, beta = 0.5,tao_new = 0):
        """
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
        
        if self.opt_num == 1:
            EQI_opt_index = np.random.choice(np.flatnonzero(EQI_list == EQI_list.max()))
            print('The next datum recomended by Expected Quantile Improvement : \n x = ', self.virtual_samples[EQI_opt_index])
        elif type(self.opt_num) == int:
            EQI_opt_index = np.argpartition(EQI_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(EQI_opt_index)):
                print('The {num}-th datum recomended by Expected Quantile Improvement : \n x = '.format(num =j+1), self.virtual_samples[EQI_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')

        return EQI_list

    def Reinterpolation_EI(self, ):
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
        
        if self.opt_num == 1:
            REI_opt_index = np.random.choice(np.flatnonzero(REI_list == REI_list.max()))
            print('The next datum recomended by Reinterpolation Expected Improvement : \n x = ', self.virtual_samples[REI_opt_index])
        elif type(self.opt_num) == int:
            REI_opt_index = np.argpartition(REI_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(REI_opt_index)):
                print('The {num}-th datum recomended by Reinterpolation Expected Improvement : \n x = '.format(num =j+1), self.virtual_samples[REI_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return REI_list



    def UCB(self, alpha=1):
        """
        :param alpha: tradeoff coefficient, default 1
        """
        UCB_list = np.array(self.virtual_samples_mean) - alpha * np.array(self.virtual_samples_std)
        
        if self.opt_num == 1:
            UCB_opt_index = np.random.choice(np.flatnonzero(UCB_list == UCB_list.min()))
            print('The next datum recomended by Upper confidence bound  : \n x = ', self.virtual_samples[UCB_opt_index])
        elif type(self.opt_num) == int:
            UCB_opt_index = np.argpartition(UCB_list, self.opt_num)[:self.opt_num]
            for j in range(len(UCB_opt_index)):
                print('The {num}-th datum recomended by Upper confidence bound : \n x = '.format(num =j+1), self.virtual_samples[UCB_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')

        return UCB_list
    
    def PoI(self, tao = 0):
        """
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
            
            if self.opt_num == 1:
                PoI_opt_index = np.random.choice(np.flatnonzero(PoI_list == PoI_list.max()))
                print('The next datum recomended by Probability of Improvement  : \n x = ', self.virtual_samples[PoI_opt_index])
            elif type(self.opt_num) == int:
                PoI_opt_index = np.argpartition(PoI_list, -self.opt_num)[-self.opt_num:]
                for j in range(len(PoI_opt_index)):
                    print('The {num}-th datum recomended by Probability of Improvement  : \n x = '.format(num =j+1), self.virtual_samples[PoI_opt_index[j]])
            else:
                print('The input para. opt_num must be an int')
            return PoI_list
    
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
        
        if self.opt_num == 1:
            PES_opt_index = np.random.choice(np.flatnonzero(PES_list == PES_list.max()))
            print('The next datum recomended by Predictive Entropy Search  : \n x = ', self.virtual_samples[PES_opt_index])
        elif type(self.opt_num) == int:
            PES_opt_index = np.argpartition(PES_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(PES_opt_index)):
                print('The {num}-th datum recomended by Predictive Entropy Search  : \n x = '.format(num =j+1), self.virtual_samples[PES_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return PES_list
    

    def Knowledge_G(self,MC_num = 50):
        """
        :param MC_num: number of Monte carol,  default 50
        """
        current_min = self.virtual_samples_mean.min()
        KD_list = []
        vir_num = len(self.virtual_samples)
        for i in range(vir_num):
            x_value = self.virtual_samples[i]
            MC_batch_min = 0
            for j in range(MC_num):
                y_value = np.random.normal(loc = self.virtual_samples_mean[i],scale = self.virtual_samples_std[i])
                archive_sample_x = copy.deepcopy(self.data_matrix)
                archive_sample_y = copy.deepcopy(self.Measured_response)
                
                archive_sample_x = np.append(archive_sample_x, x_value)
                archive_sample_y = np.append(archive_sample_y, y_value)
                fea_num = len(self.data_matrix[0])
                if self.ret_noise == True:
                    # return a callable model
                    post_mean, _ = self.Kriging_model().fit_pre(archive_sample_x.reshape(-1, fea_num),archive_sample_y,self.virtual_samples,self.virtual_samples_std[i])
                else:
                    post_mean, _ = self.Kriging_model().fit_pre(archive_sample_x.reshape(-1, fea_num),archive_sample_y,self.virtual_samples)
                MC_batch_min += post_mean.min()
                MC_times = i * MC_num + j+1
                if MC_times % 2000 == 0:
                    print('The {num}-th Monte carol simulation'.format(num = MC_times))
            MC_result = MC_batch_min / MC_num
            KD_list.append( current_min - MC_result)
        KD_list = np.array(KD_list)
        if self.opt_num == 1:
            KD_opt_index = np.random.choice(np.flatnonzero(KD_list == KD_list.max()))
            print('The next datum recomended by Knowledge Gradient : \n x = ', self.virtual_samples[KD_opt_index])
        elif type(self.opt_num) == int:
            KD_opt_index = np.argpartition(KD_list, -self.opt_num)[-self.opt_num:]
            for j in range(len(KD_opt_index)):
                print('The {num}-th datum recomended by Knowledge Gradient : \n x = '.format(num =j+1), self.virtual_samples[KD_opt_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return KD_list

