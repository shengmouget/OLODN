# 面向人员密集与遮挡的实时目标检测算法

<a href="sw20000829@163.com" title="超链接title">ShengWei</a>    <a href="sw20000829@163.com" title="超链接title">DLOU</a>

**摘 要 ：目的** 在计算机视觉领域，人员密集场景的目标检测对于实时系统尤其关键，如监控和人员行为分析系统。如果场景人员目标众多，影响算法的检测速度，会使系统出现延迟和检测精度下降的情况，为了避免这类情况发生，这就要求检测模型在硬件计算资源有限的情况下，仍然快速且准确地处理复杂的人员以及人员行为的遮挡问题，即人员的类间遮挡以及人员行为的类间遮挡和类内遮挡。本文针对以上问题，设计了一种轻量级的实时目标检测算法。**方法** 本文设计了一种基于YOLO范式的轻量级目标检测网络。该网络由特征提取部分也就是主干网络（BackBone），特征融合部分（Neck）和输出预测部分（Head）三部分组成。首先在主干网络部分，通过快速网络块对输入的数据进行特征提取，数据每经过一个特征提取部分都会在嵌入层降采样，保证足够的感受野，同时还会经过一个增强位置注意力机制模块，更加关注人员与人员之间行为之间的遮挡边界信息，为了减少信息丢失，在主干网络的最后使用特征金字塔串联汇聚模块，通过整合不同尺度的特征信息，增强模型对不同尺度人员和受遮挡人员的识别能力。然后，在主干部分提取的特征会通过增强位置注意力机制模块和特征金字塔串联汇聚模块传输到特征融合部分，通过分组洗牌卷积（GSConv，Grouped Shuffle Convolution）改善特征的信息流动和整合，有效增强特征表达能力，在不增加计算负担的情况下，保证信息在特征图中的全面分布。最后，将融合后的特征在预测部分通过任务对齐的单阶段目标检测（TOOD，Task-aligned One-stage Object Detection）思想，拉近分类和定位任务的最佳锚点，有效提高遮挡条件下的对象识别准确性。**结果** 实验结果表明，本文算法在WiderPerson数据集上，模型实现了64.8%的召回率，比YOLOv8-n模型高出0.2%，模型的参数仅1.8M，是YOLOv8-n参数量的一半，且在CPU和GPU设备上的运行效率均快于其它模型。在UpDown数据集上，分类错误率和未检测到的真实目标错误率分别为2.9%和1.8%，比YOLOv8低0.1%和0.2%。**结论** 通过实验验证和主客观评价，证明了本文算法的可行性，在计算资源有限的设备中，也能高效处理人员密集场景中的遮挡问题。

<img src="./image/jiegou.svg" alt="图片alt" title="图片title">

代码运行步骤：

~~~shell
# 使用conda创建虚拟环境
conda create -n name python=3.8
# 安装环境
pip install -r requirements.txt
~~~

请运行train.py文件，引入yaml路径，以及数据集路径。

数据集方式请联系作者。
