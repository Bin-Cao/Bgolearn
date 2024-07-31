
<h1 align="center">
  <a href=""><img src="https://user-images.githubusercontent.com/86995074/232675281-97ee5a19-b238-4d83-913c-7b0489807fa9.jpeg" alt="Bgolearn" width="150"></a>
  <br>
  Bgolearn
  <br>
</h1>


🤝🤝🤝 Please star ⭐️ it for promoting open source projects 🌍 ! Thanks ! For inquiries or assistance, please don't hesitate to contact us at bcao686@connect.hkust-gz.edu.cn (Dr. CAO Bin).

**Bgolearn** has been implemented in the machine learning platform [MLMD](http://123.60.55.8/). **Bgolearn Code** : [here](https://colab.research.google.com/drive/1OSc-phxm7QLOm8ceGJiIMGGz9riuwP6Q?usp=sharing) The **video of Bgolearn** has been uploaded to platforms : [BiliBili](https://www.bilibili.com/video/BV1Ae411J76z/?spm_id_from=333.999.0.0&vd_source=773e0c92141f498497cfafd0112fc146). [YouTube](https://www.youtube.com/watch?v=MSG6wcBol64&t=48s).


![Screenshot 2023-11-16 at 11 23 35](https://github.com/Bin-Cao/Bgolearn/assets/86995074/cd0d24e4-06db-45f7-b6d6-12750fa8b819)


# Python package - Bgolearn 

**No gradient** information is used
![plot](https://github.com/Bin-Cao/Bgolearn/assets/86995074/d4e43900-eadb-4ddf-af46-0208314de41a)



## Installing / 安装
    pip install Bgolearn 
    
## Checking / 查看
    pip show Bgolearn 
    
## Updating / 更新
    pip install --upgrade Bgolearn


## Template 
``` javascript
# import BGOsampling after installation 
# 安装后, 通过此命令调用BGOsampling类
import Bgolearn.BGOsampling as BGOS

# import your dataset (Samples have been characterized)
# 导入研究的数据集(已经表征过的样本)
data = pd.read_csv('data.csv') 
# features 
x = data.iloc[:,:-1]
# response / target 
y = data.iloc[:,-1]

# virtual samples which have same feature dimension with x
# 设计的虚拟样本, 与x具有相同的维度
vs = pd.read_csv('virtual_data.csv') 

# instantiate class
# 实例化类 Bgolearn
Bgolearn = BGOS.Bgolearn() 

# Pass parameters to the function
# 传入参数
Mymodel = Bgolearn.fit(data_matrix = x, Measured_response = y, virtual_samples = vs)

# derive the result by EI
# 通过EI导出结果
Mymodel.EI()
```

## Multi-task design
    pip install BgoKit 
    
``` javascript
from BgoKit import ToolKit
# vs is the virtual samples
# score_1,score_2 are output of Bgolearn
# score_1, _= Mymodel_1.EI() ; score_2, _= Mymodel_2.EI()

Model = ToolKit.MultiOpt(vs,[score_1,score_2])
Model.BiSearch()
Model.plot_distribution()
```
See : [Link](https://github.com/Bin-Cao/Bgolearn/blob/main/Template/%E4%B8%AD%E6%96%87%E7%A4%BA%E4%BE%8B/%E5%A4%9A%E7%9B%AE%E6%A0%87%E5%AE%9E%E7%8E%B0/%E5%A4%9A%E7%9B%AE%E6%A0%87.ipynb)
<img src="https://github.com/Bin-Cao/Bgolearn/assets/86995074/41c90c29-364c-47cc-aefe-4433f7d93e23" alt="1" width="300" height="300">

## UI
The User Interface of the Bgolearn Platform : [**BgoFace**](https://github.com/Bgolearn/BgoFace)
 
## cite
1:
    Cao B., Su T, Yu S, Li T, Zhang T, Zhang J, Dong Z, Zhang Ty. Active learning accelerates the discovery of high strength and high ductility lead-free solder alloys, Materials & Design, 2024, 112921, ISSN 0264-1275, https://doi.org/10.1016/j.matdes.2024.112921.

2:
    Ma J.∔, Cao B.∔, Dong S, Tian Y, Wang M, Xiong J, Sun S. MLMD: a programming-free AI platform to predict and design materials. npj Comput Mater 10, 59 (2024). https://doi.org/10.1038/s41524-024-01243-4


## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao686@connect.hkust-gz.edu.cn) in case of any problems/comments/suggestions in using the code. 

## Contributing / 共建
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.

## License and Usage
© 2024 Bgolearn Development Team. All rights reserved.

This software is provided for academic and research purposes only. Commercial use is strictly prohibited. Any violation of these terms will be subject to appropriate actions.

---
### for regression
- 1.Expected Improvement algorith (期望提升函数)

- 2.Expected improvement with “plugin” (有“plugin”的期望提升函数)

- 3.Augmented Expected Improvement (增广期望提升函数)

- 4.Expected Quantile Improvement (期望分位提升函数)

- 5.Reinterpolation Expected Improvement (重插值期望提升函数)

- 6.Upper confidence bound (高斯上确界函数)

- 7.Probability of Improvement (概率提升函数)

- 8.Predictive Entropy Search (预测熵搜索函数)

- 9.Knowledge Gradient (知识梯度函数)

###  for classification
- 1.Least Confidence (欠信度函数)

- 2.Margin Sampling (边界函数)

- 3.Entropy-based approach (熵索函数)


``` javascript
Signature:
Bgolearn.fit(
    data_matrix,
    Measured_response,
    virtual_samples,
    Mission='Regression',
    Classifier='GaussianProcess',
    noise_std=None,
    Kriging_model=None,
    opt_num=1,
    min_search=True,
    CV_test=False,
    Dynamic_W=False,
    seed=42,
)
Docstring:
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
File:      ~/miniconda3/lib/python3.9/site-packages/Bgolearn/BGOsampling.py
Type:      method
```
