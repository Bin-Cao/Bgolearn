import warnings
import numpy as np

class Boundary(object):
    def __init__(self,model,data_matrix, Measured_response, virtual_samples, opt_num):
        warnings.filterwarnings('ignore')
        self.model = model
        self.data_matrix = np.array(data_matrix)
        self.Measured_response = np.array(Measured_response)
        __fea_num = len(self.data_matrix[0])
        self.virtual_samples = np.array(virtual_samples).reshape(-1,__fea_num)
        self.probs = model.fit(data_matrix, Measured_response).predict_proba(self.virtual_samples)
        self.opt_num = opt_num
        

    def Least_cfd(self,):
        Lc = []
        for i in range(len(self.probs)):
            max_pro = np.array(self.probs[i]).max()
            Lc.append(1 - max_pro)
        LcValue =  np.array(Lc)

        return_x = []
        if self.opt_num == 1:
            LcValue_index = np.random.choice(np.flatnonzero(LcValue == LcValue.max()))
            print('The next datum recomended by Least confidence : \n x = ', self.virtual_samples[LcValue_index])
            return_x.append(self.virtual_samples[LcValue_index])
        elif type(self.opt_num) == int:
            LcValue_index = np.argpartition(LcValue, -self.opt_num)[-self.opt_num:]
            for j in range(len(LcValue_index)):
                print('The {num}-th datum recomended by Least confidence : \n x = '.format(num =j+1), self.virtual_samples[LcValue_index[j]])
                return_x.append(self.virtual_samples[LcValue_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return LcValue,np.array(return_x)

    def Margin_S(self,):
        Margin = []
        for i in range(len(self.probs)):
            targ_list = list(self.probs[i])
            max_prb = np.array(targ_list).max()
            targ_list.remove(max_prb)
            secmax_prb = np.array(targ_list).max()
            Mg_values = 1.0 + secmax_prb - max_prb
            Margin.append(Mg_values)
        MgValue =  np.array(Margin)

        return_x = []
        if self.opt_num == 1:
            Mg_index = np.random.choice(np.flatnonzero(MgValue == MgValue.max()))
            print('The next datum recomended by Margin sampling : \n x = ', self.virtual_samples[Mg_index])
            return_x.append(self.virtual_samples[Mg_index])
        elif type(self.opt_num) == int:
            Mg_index = np.argpartition(MgValue, -self.opt_num)[-self.opt_num:]
            for j in range(len(Mg_index)):
                print('The {num}-th datum recomended by Margin sampling : \n x = '.format(num =j+1), self.virtual_samples[Mg_index[j]])
                return_x.append(self.virtual_samples[Mg_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return MgValue,np.array(return_x)


    def Entropy(self,):
        Entropy_value = []
        for i in range(len(self.probs)):
            Etp = 0
            for j in range(len(self.probs[i])):
                Etp -= self.probs[i][j] * np.log(self.probs[i][j] + 1e-10)
            Entropy_value.append(Etp)
        EtpValue =  np.array(Entropy_value)

        return_x = []
        if self.opt_num == 1:
            Etp_index = np.random.choice(np.flatnonzero(EtpValue == EtpValue.max()))
            print('The next datum recomended by Entropy-based approach : \n x = ', self.virtual_samples[Etp_index])
            return_x.append(self.virtual_samples[Etp_index])
        elif type(self.opt_num) == int:
            Etp_index = np.argpartition(EtpValue, -self.opt_num)[-self.opt_num:]
            for j in range(len(Etp_index)):
                print('The {num}-th datum recomended by Entropy-based approach : \n x = '.format(num =j+1), self.virtual_samples[Etp_index[j]])
                return_x.append(self.virtual_samples[Etp_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return EtpValue,np.array(return_x)
    


    
   