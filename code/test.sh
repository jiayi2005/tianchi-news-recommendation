cd "$(dirname "$0")"

# 使用 tianchi conda 环境的 Python
PYTHON=/opt/anaconda3/envs/tianchi/bin/python

time=$(date "+%Y-%m-%d-%H:%M:%S")

# 处理数据
$PYTHON data.py --mode valid --logfile "${time}.log"

# itemcf 召回
$PYTHON recall_itemcf.py --mode valid --logfile "${time}.log"

# binetwork 召回
$PYTHON recall_binetwork.py --mode valid --logfile "${time}.log"

# w2v 召回
$PYTHON recall_w2v.py --mode valid --logfile "${time}.log"

# usercf 召回
$PYTHON recall_usercf.py --mode valid --logfile "${time}.log"

# youtubednn 召回
$PYTHON recall_youtubednn.py --mode valid --logfile "${time}.log"

# 冷启动召回
$PYTHON recall_coldstart.py --mode valid --logfile "${time}.log"

# 召回合并
$PYTHON recall.py --mode valid --logfile "${time}.log"

# 排序特征
$PYTHON rank_feature.py --mode valid --logfile "${time}.log"

# lgb 模型训练
$PYTHON rank_lgb.py --mode valid --logfile "${time}.log"

# din 模型训练（可选，依赖 torch）
if [[ "${USE_DIN}" == "1" ]]; then
  $PYTHON rank_din.py --mode valid --logfile "${time}.log"
fi
