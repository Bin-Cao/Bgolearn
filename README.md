
<h1 align="center">
  <a href=""><img src="https://user-images.githubusercontent.com/86995074/232675281-97ee5a19-b238-4d83-913c-7b0489807fa9.jpeg" alt="Bgolearn" width="150"></a>
  <br>
  Bgolearn
  <br>
</h1>


🤝🤝🤝 Please star ⭐️ it for promoting open source projects 🌍 ! Thanks !

If you are using this code, please cite:
    
    Cao Bin, Zhang Tong-yi, Wang Yuanhao, Tian Yuan, Sun Sheng. Bayesian global optimization package for material design [2022SR1481726], 2022, Software copyright, GitHub : github.com/Bin-Cao/Bgolearn.
    
## links
![Screenshot 2023-11-16 at 11 23 35](https://github.com/Bin-Cao/Bgolearn/assets/86995074/cd0d24e4-06db-45f7-b6d6-12750fa8b819)

- https://www.wheelodex.org/projects/bgolearn/
- https://pypi.tuna.tsinghua.edu.cn/simple/bgolearn/
- [user count](https://pypistats.org/packages/bgolearn)


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

## Download History （- Nov16，2023）
![WechatIMG4661](https://github.com/Bin-Cao/Bgolearn/assets/86995074/591e26b4-c8c3-4a17-ae8b-b3bcf9237514)


if you have any questions or need help, you are welcome to contact me

Source code: [![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/Bgolearn/)




# Python package - Bgolearn 

**No gradient** information is used
![plot](https://github.com/Bin-Cao/Bgolearn/assets/86995074/d4e43900-eadb-4ddf-af46-0208314de41a)


## Package Document / 手册
see 📒 [Bgolearn](https://bgolearn.netlify.app) (Click to view）

见 📒 [中文说明](https://mp.weixin.qq.com/s/y-i_2ixbtJOv-nEYDu9THg) (Click to view）

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

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



## Installing / 安装
    pip install Bgolearn 
    
## Checking / 查看
    pip show Bgolearn 
    
## Updating / 更新
    pip install --upgrade Bgolearn


     
## Update log / 日志
Before version 2.0, function building

Bgolearn V2.1.1 Jun 9, 2023. *para noise_std* By default, the built-in Gaussian process model estimates the noise of the input dataset by maximum likelihood, and yields in a more robust model.


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




``` javascript
Thank you for choosing Bgolearn for material design. 
Bgolearn is developed to facilitate the application of machine learning in research.
Bgolearn is designed for optimizing single-target material properties.
The BgoKit package is being developed to facilitate multi-task design.


If you need to perform multi-target optimization, here are two important reminders:

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
```

## References / 参考文献
See : [papers](https://github.com/Bin-Cao/Bgolearn/tree/main/Refs)

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao686@connect.hkust-gz.edu.cn) in case of any problems/comments/suggestions in using the code. 

## Contributing / 共建
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.
