# RL4UC文档

本科一学位毕业设计项目，使用强化学习方法研究电力系统机组组合问题。机组组合问题是电力系统经济调度中的重要内容，根据预期的全天负荷曲线及各机组的成本价格曲线，给出各机组的开关决策变量以及出力大小。因此决策变量包含离散变量和连续变量，此问题则成为MILP问题。使用传统MILP算法进行求解在大系统中将会非常耗时，不能满足电力市场化下的市场快速出清要求。因此，使用强化学习方法来对机组组合问题求解进行研究。

## 文件夹

### 代码文件夹

-  pytorch-drl4vrp-master：参考代码

### 数据文件夹

-  data：模仿学习MATLAB结果
   -  onoff：启停结果
   -  power：功率结果
-  fake_load：旧的生成负荷数据，弃用
-  imitate_load, imitate_load_windows3, imitate_load_windows6：使用几台电脑运算得出的MILP结果，合并得到data文件夹

### 模型文件夹

-  state_dict：包含几个文件夹以及模型文件。文件夹中是强化学习训练的模型，模型文件是模仿学习训练的模型文件。其中state_dict/imitate_actor_parameter_0521_001633.pt为后续强化学习训练的基础。
-  state_dict_new0.3, state_dict_new2, state_dict_new3：比较混乱的模型存储文件夹，保存的是在几次修改代码后训练模型得到的结果，感觉用处不大

### 结果文件夹

-  figure：模仿学习过程中产生的结果，包含训练误差，测试误差以及每个epoch消耗时间
-  imitate_test：使用模仿学习训练的网络对365条真实数据计算的结果
   -  difference：imitate和result之差
   -  imitate：模仿学习给出的调度计划
   -  result：matlab给出的调度计划
-  log：训练过程输出结果，包含每个epoch内batch的训练结果以及每一个epoch的总结（特征中不包含power时的结果），txt格式
-  power_log：特征中引入power后的训练过程输出结果，txt格式
-  result：训练过程中间输出结果，csv格式
   -  imitate：模仿学习结果
   -  reinforce：强化学习结果
-  train_figure：训练过程中间输出结果图

## 文件

### csv文件

-  fake_load.csv：根据真实负荷曲线生成的负荷曲线，每条真实曲线生成50条负荷曲线。基于标幺化之后的真实数据使用dataGenerator生成
-  imitate_load_power.csv：将fake_load中每天的负荷曲线随机选择25条，得到模仿学习的负荷曲线
-  reinforce_load.csv：将fake_load中每天的负荷曲线随机选择25条，得到强化学习的负荷曲线
-  load.csv：江苏省原始负荷数据

### py文件

-  trainer.py：进行模仿学习和强化学习训练
   -  类
      -  ctiric：critic网络的定义
   -  函数
      -  validate：在不同模式下使用不同方法，使用真实数据对给定的actor网络进行测试（针对训练过程中的测试）
      -  imitate：进行多个epoch的模仿学习训练，每一轮训练中包含训练过程以及测试过程
      -  train：进行强化学习训练
      -  test：对训练完成后的模型进行测试，包含对模仿学习模型以及强化学习模型的训练（针对学习完成的模型的测试）
      -  train_init：进行训练前的准备工作，包括从生成数据集、actor网络和critic网络
-  model.py：
   -  类
      -  Encoder：对动静态数据进行嵌入的conv网络
      -  StateCritic：状态价值评判网络
      -  Resnet：策略模型，给出每一步的策略
      -  RL4UC：加入电力系统安全约束的24h机组组合问题求解
-  UC.py：
   -  类
      -  UCDataset：对机组组合问题的初始状态进行定义，并且给出mask函数以及类内动态数据更新方法
   -  函数
      -  get_data：从pandapower中读取数据，例如得到opf结果
      -  save_data：将数据存入pandapower中，以实现修改机组最大最小出力等用途
      -  reward：计算机组运行成本
-  utils.py：
   -  函数：
      -  matlab2pp：按照手工核算的顺序将matlab中使用的matpower包中的机组顺序与pandapower中的机组顺序对齐（对于rts网络）

### ipynb文件

-  dataGenerator.ipynb：本来应该是生成负荷数据的程序的，但是后来被被其他程序覆盖了，找不回来了
-  test.ipynb：模仿学习测试结果

### zip文件

-  scuc程序.zip：MILP求解机组组合问题
