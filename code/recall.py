import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('threading')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        score_range = user_score_max[user_id] - user_score_min[user_id]
        if score_range == 0:
            ans.append(10**-3)
            continue
        ans.append((sim_score - user_score_min[user_id]) / score_range +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    if cnt == 0:
        return 0
    return hit_cnt / cnt


def get_hot_items(df_click, topn=200):
    df_stat = df_click.groupby('click_article_id').agg(
        click_cnt=('click_article_id', 'count'),
        last_ts=('click_timestamp', 'max')).reset_index()
    df_stat.sort_values(['click_cnt', 'last_ts'], ascending=False, inplace=True)
    return df_stat['click_article_id'].values.tolist()[:topn]


def fill_cold_start(recall_df, df_query, df_click, min_recall=50):
    hot_items = get_hot_items(df_click, topn=min_recall * 5)

    user_item_dict = recall_df.groupby('user_id')['article_id'].agg(
        lambda x: set(x)).to_dict()

    data_list = []
    for user_id, item_id in tqdm(df_query.values):
        existed = user_item_dict.get(user_id, set())
        need = min_recall - len(existed)
        if need <= 0:
            continue

        rec_items = []
        for item in hot_items:
            if item in existed:
                continue
            rec_items.append(item)
            if len(rec_items) >= need:
                break

        if not rec_items:
            continue

        scores = [1.0 / (idx + 1) for idx in range(len(rec_items))]

        df_temp = pd.DataFrame({
            'user_id': user_id,
            'article_id': rec_items,
            'sim_score': scores
        })

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        data_list.append(df_temp)

    if not data_list:
        return recall_df

    df_extra = pd.concat(data_list, sort=False)
    df_extra['user_id'] = df_extra['user_id'].astype('int')
    df_extra['article_id'] = df_extra['article_id'].astype('int')

    recall_df = pd.concat([recall_df, df_extra], sort=False)
    recall_df = recall_df.drop_duplicates(['user_id', 'article_id'])
    return recall_df


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = [
        'itemcf', 'w2v', 'binetwork', 'usercf', 'youtubednn', 'coldstart'
    ]

    weights = {
        'itemcf': 1,
        'binetwork': 1,
        'w2v': 0.1,
        'usercf': 0.8,
        'youtubednn': 1.0,
        'coldstart': 0.05,
    }
    recall_list = []
    recall_dict = {}
    used_methods = []
    for recall_method in recall_methods:
        recall_file = f'{recall_path}/recall_{recall_method}.pkl'
        if not os.path.exists(recall_file):
            log.warning(f'缺少召回文件: {recall_file}，跳过该召回方式')
            continue
        recall_result = pd.read_pickle(recall_file)
        weight = weights.get(recall_method, 1.0)

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result
        used_methods.append(recall_method)

    # 求相似度
    for recall_method1, recall_method2 in permutations(used_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()

    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    recall_final = recall_final.merge(recall_score, how='left')

    # 冷启动兜底，保证每个用户至少有一定数量的召回
    recall_final = fill_cold_start(recall_final,
                                   df_query,
                                   df_click,
                                   min_recall=50)

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')
