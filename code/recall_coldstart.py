import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

random.seed(2020)

# CLI args
parser = argparse.ArgumentParser(description='coldstart recall')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--topn', type=int, default=50)
parser.add_argument('--min_hist', type=int, default=1)

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
topn = args.topn
min_hist = args.min_hist

# Init logger
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'coldstart recall, mode: {mode}')


def build_hot_items(df_click, topn=50):
    df_stat = df_click.groupby('click_article_id').agg(
        click_cnt=('click_article_id', 'count'),
        last_ts=('click_timestamp', 'max')).reset_index()
    df_stat.sort_values(['click_cnt', 'last_ts'], ascending=False, inplace=True)
    return df_stat['click_article_id'].values.tolist()[:topn]


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    user_hist_len = df_click.groupby('user_id')['click_article_id'].count()
    user_hist_len = user_hist_len.to_dict()
    user_item_dict = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).to_dict()

    hot_items = build_hot_items(df_click, topn=topn * 5)

    data_list = []
    for user_id, item_id in tqdm(df_query.values):
        hist_len = user_hist_len.get(user_id, 0)
        if hist_len > min_hist:
            continue

        interacted = set(user_item_dict.get(user_id, []))

        rec_items = []
        for item in hot_items:
            if item in interacted:
                continue
            rec_items.append(item)
            if len(rec_items) >= topn:
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

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    if data_list:
        df_data = pd.concat(data_list, sort=False)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    # Sort
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)

    # Evaluate recall metrics
    if mode == 'valid':
        log.info('eval recall metrics')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'coldstart: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # Save recall result
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_coldstart.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_coldstart.pkl')
