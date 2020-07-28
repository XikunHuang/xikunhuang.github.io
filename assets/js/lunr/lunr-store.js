var store = [{
        "title": "MLAPP笔记-概率",
        "excerpt":"简介 概率两大派: 频率学派(frequentist) 贝叶斯学派(Bayesian) 频率学派将概率解释为事件在多次实验下发生的频率(long run frequencies of events). 贝叶斯学派用概率来量化我们对一些事情的不确定性, 因此概率本质上与信息有关, 而与实验次数无关. 举个例子, 对于“硬币正面朝上的概率为0.5”这句表述, 频率学派是在说如果我们掷硬币很多次, 那么大概有一半的次数硬币朝上. 而贝叶斯学派是在说我们相信下一次掷硬币出现正面和反面的可能性相同. 概率论 本小节简单介绍了一些概率论的基本概念和公式, 这里不多赘述. 只记录一些看了有些启发的. 贝叶斯公式 贝叶斯公式给出的结果往往反“直觉”, 当然这里的“直觉”是错的. 书中正文和习题中给了三个例子. 医疗诊断 &gt; 假设如果人类患有某种疾病y, 那么在检查时指标x为阳性的概率为0.8. 现在如果有个人检测到指标x为阳性, 那么请问: 这个人患有疾病y的概率是多少? 脱口而出0.8? 正确答案是不知道, 因为给的信息不足以做出判断. 我们再知道两个信息就可以做出判断: 人群中该疾病患病率是多少? 一个没患病的人检测到指标x为阳性的概率是多少? 这里我们假设 \\(p(y=1)=0.004, p(x=1\\vert y=0)=0.1\\), 那么由此可以计算出\\(p(y=1\\vert x=1)=0.031\\). 换句话说, 即使指标x被检测出阳性, 这个人也只有大约3%的概率真正患有疾病y! 公诉人与辩护人谬论 &gt; 假设有一起案件,...","categories": ["技术"],
        "tags": ["MLAPP","机器学习"],
        "url": "https://xikunhuang.github.io/%E6%8A%80%E6%9C%AF/MLAPP%E7%AC%94%E8%AE%B0-%E6%A6%82%E7%8E%87/",
        "teaser": null
      },{
        "title": "MLAPP笔记-离散数据生成模型",
        "excerpt":"简介 先来说一下生成式模型(generative models) 和 判别式模型(discriminative models). 对于分类问题, 我们的目标是基于有限的训练样本集尽可能准确地估计出后验概率\\(p(c\\vert x)\\), 而估计该后验概率有两种方式: 判别式模型: 直接建模\\(p(c \\vert x)\\). 生成式模型: 先对联合分布\\(p(x,c)\\)建模, 再由此得到\\(p(c \\vert x)\\). 本章着眼于生成式模型, 由贝叶斯定理可得 \\[p(c\\vert x)=\\frac{p(x\\vert c)p(c)}{p(x)}\\] 所以问题关键就在于求得 类条件概率分布(class-conditional density) \\(p(x\\vert c)\\) 先验分布 \\(p(c)\\) 概率分布 继续进行之前, 这里先罗列一下本章涉及的一些概率分布. 伯努利分布和二项分布 可用来建模掷硬币的结果 伯努利分布 \\[X \\sim Ber(\\theta)\\] \\[Ber(x\\vert \\theta) = \\theta^{\\mathbb{I}(x=1)}(1-\\theta)^{\\mathbb{I}(x=0)}\\] 即随机变量\\(X \\in \\{0,1\\}\\), 且 \\[p(X=1) =...","categories": ["技术"],
        "tags": ["MLAPP","机器学习"],
        "url": "https://xikunhuang.github.io/%E6%8A%80%E6%9C%AF/MLAPP%E7%AC%94%E8%AE%B0-%E7%A6%BB%E6%95%A3%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B/",
        "teaser": null
      },{
        "title": "MLAPP笔记-高斯模型",
        "excerpt":"简介 本章讨论多元高斯或者称多元正态(MVN), 数学要求稍高一些. 基础 \\(D\\)维MVN的pdf为 \\[\\mathcal{N}(x|\\mu,\\Sigma) = \\frac{1}{(2\\pi)^{D/2}|\\Sigma|^{1/2}}\\exp[-\\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu)]\\] 上述pdf式中指数里的表达式去掉\\(-\\frac{1}{2}\\)后实际上就是\\(x\\)和\\(\\mu\\)之间的Mahalanobis距离, 关于这个距离更多的知识可以参考Wikipedia. 现在考虑这样一个问题: MVN的pdf的等值线是什么样的呢? 如果协方差矩阵\\(\\Sigma\\)是对角阵, 那么Mahalanobis距离就变成了 \\[(x-\\mu)^T\\Sigma^{-1}(x-\\mu) = \\sum_{i=1}^{D}\\frac{1}{\\lambda_i}(x_i-\\mu_i)^2 = \\sum_{i=1}^{D}\\frac{1}{\\lambda_i} y_i^2\\] 其中\\(\\lambda_i\\)是协方差矩阵的对角元素, \\(y_i = x_i-\\mu_i\\) 在二维情形下这个等值线就是椭圆. \\(\\frac{1}{\\lambda_1}y_1^2 + \\frac{1}{\\lambda_2} y_2^2 = k\\) 如果协方差不是对角阵, 那么由于它是实对称矩阵, 因此可以正交分解为 \\(\\Sigma = U\\Lambda U^T\\) 此时\\(y_i = u_i^T(x-\\mu)\\) 由此也可以看出Mahalanobis距离与欧氏距离的联系. MLE for an MVN 最大似然估计是估计MVN的参数的方法之一. 但是最大似然估计有过拟合的缺点,后续会讨论MVN参数的Bayes推断, 这种方法可以消除过拟合, 并且可以为估计提供置信度....","categories": ["技术"],
        "tags": ["MLAPP","机器学习"],
        "url": "https://xikunhuang.github.io/%E6%8A%80%E6%9C%AF/MLAPP%E7%AC%94%E8%AE%B0-%E9%AB%98%E6%96%AF%E6%A8%A1%E5%9E%8B/",
        "teaser": null
      }]
