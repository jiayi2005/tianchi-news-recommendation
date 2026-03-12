cd "$(dirname "$0")"

PYTHON=/opt/anaconda3/envs/tianchi/bin/python
time=$(date "+%Y-%m-%d-%H:%M:%S")

# lgb 模型训练（feature.pkl 已存在，从这里续跑）
$PYTHON rank_lgb.py --mode valid --logfile "${time}.log"

# din 模型训练（可选，依赖 torch）
if [[ "${USE_DIN}" == "1" ]]; then
  $PYTHON rank_din.py --mode valid --logfile "${time}.log"
fi
