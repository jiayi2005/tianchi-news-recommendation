根据以下方案进行了整合修改




零基础入门推荐系统 - 新闻推荐 Top2  

比赛地址: https://tianchi.aliyun.com/competition/entrance/531842/introduction

# 解决方案
采用多种召回方式：itemcf 召回，binetwork 召回，基于 word2vec 的 i2i 召回，usercf 召回，youtubednn 召回，并加入冷启动召回兜底。合并去重并删除没有召回到真实商品的用户数据后，利用特征工程+ LGB 二分类模型进行排序，同时支持 DIN 排序模型。

# 复现步骤
操作系统：ubuntu 16.04  
```
pip install requirements.txt
cd code
bash test.sh
```

可选开启 DIN 排序：
```
USE_DIN=1 bash test.sh
```
