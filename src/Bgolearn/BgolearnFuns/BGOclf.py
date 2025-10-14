"""
BGO Classification Module

This module implements Bayesian Global Optimization for classification problems.
It provides uncertainty-based sampling strategies for active learning in
classification tasks, focusing on decision boundary exploration.

Key Features:
- Least Confidence sampling strategy
- Margin Sampling for boundary exploration
- Entropy-based uncertainty quantification
- Support for multi-class classification problems

Author: Bin Cao
Institution: Hong Kong University of Science and Technology (Guangzhou)
"""

import warnings
import numpy as np


class Boundary(object):
    """
    Classification Boundary Exploration for Active Learning

    This class implements uncertainty-based sampling strategies for classification
    problems in Bayesian optimization. It focuses on exploring decision boundaries
    by selecting samples with high prediction uncertainty.

    Attributes:
        model: Classification model (e.g., SVM, Random Forest)
        data_matrix: Training feature matrix
        Measured_response: Training class labels
        virtual_samples: Candidate points for evaluation
        probs: Predicted class probabilities for virtual samples
        opt_num: Number of optimal candidates to recommend
    """

    def __init__(self, model, data_matrix, Measured_response, virtual_samples, opt_num, scale_virtual_samples):
        """
        Initialize the Boundary exploration classifier.

        Args:
            model: Classification model with fit and predict_proba methods
            data_matrix: Training feature matrix
            Measured_response: Training class labels
            virtual_samples: Candidate points for classification
            opt_num: Number of optimal candidates to recommend
            scale_virtual_samples: Scaled virtual samples for prediction
        """
        warnings.filterwarnings('ignore')
        self.model = model
        self.data_matrix = np.array(data_matrix)
        self.Measured_response = np.array(Measured_response)
        __fea_num = len(self.data_matrix[0])
        self.virtual_samples = np.array(virtual_samples).reshape(-1, __fea_num)
        # Generate probability predictions for all virtual samples
        self.probs = model.fit(data_matrix, Measured_response).predict_proba(scale_virtual_samples)
        self.opt_num = opt_num
        

    def Least_cfd(self):
        """
        Least Confidence sampling strategy for active learning.

        This method selects samples with the lowest confidence in their predicted
        class labels. It identifies points where the classifier is most uncertain,
        making them valuable for improving the decision boundary.

        Returns:
            tuple: (confidence_values, optimal_candidates)
                - confidence_values: Least confidence scores for all virtual samples
                - optimal_candidates: Top candidates with lowest confidence

        Mathematical Formula:
            LC(x) = 1 - max(P(y|x))
            where P(y|x) is the predicted probability for the most likely class
        """
        Lc = []  # Least confidence scores
        for i in range(len(self.probs)):
            # Find maximum probability (highest confidence class)
            max_pro = np.array(self.probs[i]).max()
            # Calculate least confidence (1 - max probability)
            Lc.append(1 - max_pro)
        LcValue = np.array(Lc)

        return_x = []
        if self.opt_num == 1:
            # Select single candidate with highest uncertainty
            LcValue_index = np.random.choice(np.flatnonzero(LcValue == LcValue.max()))
            print('The next datum recommended by Least confidence : \n x = ', self.virtual_samples[LcValue_index])
            return_x.append(self.virtual_samples[LcValue_index])
        elif type(self.opt_num) == int:
            # Select multiple candidates with highest uncertainties
            LcValue_index = np.argpartition(LcValue, -self.opt_num)[-self.opt_num:]
            for j in range(len(LcValue_index)):
                print('The {num}-th datum recommended by Least confidence : \n x = '.format(num=j+1), self.virtual_samples[LcValue_index[j]])
                return_x.append(self.virtual_samples[LcValue_index[j]])
        else:
            print('The input para. opt_num must be an int')
        return LcValue, np.array(return_x)

    def Margin_S(self):
        """
        Margin Sampling strategy for active learning.

        This method selects samples with the smallest margin between the two most
        likely class predictions. It focuses on points near decision boundaries
        where the classifier is most uncertain between competing classes.

        Returns:
            tuple: (margin_values, optimal_candidates)
                - margin_values: Margin scores for all virtual samples
                - optimal_candidates: Top candidates with smallest margins

        Mathematical Formula:
            Margin(x) = 1 + P_2nd(y|x) - P_1st(y|x)
            where P_1st and P_2nd are the highest and second-highest class probabilities
        """
        Margin = []  # Margin scores
        for i in range(len(self.probs)):
            targ_list = list(self.probs[i])
            # Find highest probability (most confident class)
            max_prb = np.array(targ_list).max()
            targ_list.remove(max_prb)
            # Find second highest probability (second most confident class)
            secmax_prb = np.array(targ_list).max()
            # Calculate margin (smaller margin = more uncertainty)
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
        """
        Entropy-based approach
        """
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
    


    
   