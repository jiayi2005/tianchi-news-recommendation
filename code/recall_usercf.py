import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

max_threads = min(multitasking.config['CPU_CORES'], 4)
multitasking.set_max_threads(max_threads)
multitasking.set_engine('threading')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# CLI args
parser = argparse.ArgumentParser(description='usercf recall')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# Init logger
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'usercf recall, mode: {mode}')


def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(
        lambda x: list(x)).reset_index()
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))

    user_cnt = defaultdict(int)
    for user_id, items in user_item_dict.items():
        user_cnt[user_id] = len(items)

    user_sim = {}
    for item, users in tqdm(item_user_dict.items()):
        user_num = len(users)
        if user_num > 500:  # 跳过热门物品，防止内存爆炸
            continue
        for u in users:
            user_sim.setdefault(u, {})
            for v in users:
                if u == v:
                    continue
                user_sim[u].setdefault(v, 0)
                user_sim[u][v] += 1 / math.log(1 + user_num)

    for u, related_users in tqdm(user_sim.items()):
        for v, cuv in related_users.items():
            user_sim[u][v] = cuv / math.sqrt(user_cnt[u] * user_cnt[v])

    return user_sim, user_item_dict


@multitasking.task
def recall(df_query, user_sim, user_item_dict, worker_id, topk_user=20,
           topk_item=100, max_hist=50):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        if user_id not in user_item_dict:
            continue
        if user_id not in user_sim:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_set = set(interacted_items)

        rank = defaultdict(float)
        sim_users = sorted(user_sim[user_id].items(),
                           key=lambda d: d[1],
                           reverse=True)[:topk_user]

        for sim_user, sim_score in sim_users:
            items = user_item_dict.get(sim_user, [])
            for loc, item in enumerate(items[::-1][:max_hist]):
                if item in interacted_set:
                    continue
                rank[item] += sim_score * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:topk_item]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        if not item_ids:
            continue

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    if data_list:
        df_data = pd.concat(data_list, sort=False)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    os.makedirs('../user_data/tmp/usercf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/usercf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/usercf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/usercf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    user_sim, user_item_dict = cal_sim(df_click)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(user_sim, f)

    # Recall
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = max(1, total // n_split)

    # Clear temp folder
    for path, _, file_list in os.walk('../user_data/tmp/usercf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, user_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('merge tasks')

    dfs = []
    for path, _, file_list in os.walk('../user_data/tmp/usercf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            dfs.append(df_temp)
    if dfs:
        df_data = pd.concat(dfs, ignore_index=True)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    # Sort by score
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # Evaluate recall metrics
    if mode == 'valid':
        log.info('eval recall metrics')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'usercf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # Save recall result
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_usercf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_usercf.pkl')
