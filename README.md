
[![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/Bgolearn/)
# Python package - Bgolearn （V1.05）



![Screen Shot 2022-07-11 at 9 13 28 AM](https://user-images.githubusercontent.com/86995074/178176016-8a79db81-fcfb-4af0-9b1c-aa4e6a113b5e.png)

## 为材料设计而生！
## （ A Bayesian global optimization package for material design ）Version 1, Jul, 2022


Reference paper : V. Picheny, T. Wagner, and D. Ginsbourger. “A Benchmark of Kriging-Based Infill Criteria for Noisy Optimization”. In: Structural and Multidisciplinary Optimization 48.3 (Sept. 2013), pp. 607–626. issn: 1615-1488. 


Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Content
Bgolearn guides subsequent material design based on existed experimental data. Which includes: 1.Expected Improvement algorithm, 2.Expected improvement with “plugin”，3.Augmented Expected Improvement，4.Expected Quantile Improvement，5.Reinterpolation Expected Improvement， 6.Upper confidence bound，7.Probability of Improvement，8.Predictive Entropy Search，9.Knowledge Gradient, a total of nine Utility Functions. Predictive Entropy Search，Knowledge Gradient are implemented based on Monte Carlo simulation.（贝叶斯优化设计，根据已有的实验数据对后续材料设计作出指导，本算法包共包括：期望最大化算法，期望最大化算法改进（考虑数据噪声），上确界方法，期望提升方法，熵搜索，知识梯度方法等在内的共计9种贝叶斯采样方法。其中熵搜索和知识梯度方法基于蒙特卡洛实现）

## Installing / 安装
    pip install Bgolearn 

## Updating / 更新
    pip install --upgrade Bgolearn

## Running / 运行
### Ref.https://github.com/Bin-Cao/Bgolearn/blob/main/Template/demo.ipynb

```javascript
import Bgolearn.BGOsampling as BGOS 

data = pd.read_csv('data.csv')
data_matrix = data.iloc[:,:-1]
Measured_response = data.iloc[:,-1]

# design virtual samples
virtual_samples = np.linspace(0,11,100)

# define a callable Kriging model
class Kriging_model(object):
    def fit_pre(self,xtrain,ytrain,xtest):
        # instantiated model
        kernel = RBF() + WhiteKernel()
        mdoel = GaussianProcessRegressor(kernel=kernel,normalize_y=True,).fit(xtrain,ytrain)
        # defined the attribute's outputs
        mean,std = mdoel.predict(xtest,return_std=True)
        return mean,std    


Bgolearn = BGOS.Bgolearn()

# min_search = False:  searching the global maximum
model = Bgolearn.fit(Kriging_model,data_matrix,Measured_response,virtual_samples,opt_num = 3,min_search = True)

# Expected Improvement 
model.EI()
```

## Utility Function 效用函数: 
+ 1:Expected Improvement 

        model.EI()
+ 2:Expected improvement with “plugin”

        model.EI_plugin()
+ 3:Augmented Expected Improvement 

        model.Augmented_EI(alpha = 1, tao = 0)
        :param alpha: tradeoff coefficient, default 1
        :param tao: noise standard deviation, default 0
+ 4:Expected Quantile Improvement 

        model.EQI(beta = 0.5,tao_new = 0)
        :param beta: beta quantile number, default 0.5
        :param tao: noise standard deviation, default 0

        
+ 5:Reinterpolation Expected Improvement

        model.Reinterpolation_EI()
+ 6:Upper confidence bound

        model.UCB(alpha=1)
        :param alpha: tradeoff coefficient, default 1
+ 7:Probability of Improvement

        model.PoI(tao = 0)
        :param tao: improvement ratio (>=0) , default 0
+ 8:Predictive Entropy Search

        model.PES(sam_num = 500)
        :param sam_num: number of optimal drawn from p(x*|D), default 500
+ 9:Knowledge Gradient

        model.KD(MC_num = 500)
        :param MC_num: number of Monte carlo, default 500

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

