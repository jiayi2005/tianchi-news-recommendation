import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('threading')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# CLI args
parser = argparse.ArgumentParser(description='youtubednn recall')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--decay', type=float, default=0.7)

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
seq_len = args.seq_len
decay = args.decay

# Init logger
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'youtubednn recall, mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         vector_size=256,
                         window=3,
                         min_count=1,
                         sg=1,
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10,
                         epochs=1)
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map


def build_user_embedding(hist_items, article_vec_map, decay):
    if not hist_items:
        return None
    hist_items = hist_items[::-1]
    weights = np.array([decay**i for i in range(len(hist_items))],
                       dtype=np.float32)
    emb_list = []
    valid_weights = []
    for item, w in zip(hist_items, weights):
        if item not in article_vec_map:
            continue
        emb_list.append(article_vec_map[item])
        valid_weights.append(w)

    if not emb_list:
        return None

    emb_mat = np.vstack(emb_list)
    w = np.array(valid_weights, dtype=np.float32)
    user_emb = np.average(emb_mat, axis=0, weights=w)
    return user_emb


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id, seq_len=50, decay=0.7):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_set = set(interacted_items)
        hist_items = interacted_items[-seq_len:]

        user_emb = build_user_embedding(hist_items, article_vec_map, decay)
        if user_emb is None:
            continue

        item_ids, distances = article_index.get_nns_by_vector(
            user_emb, 200, include_distances=True)
        sim_scores = [2 - distance for distance in distances]

        rank = defaultdict(float)
        for relate_item, wij in zip(item_ids, sim_scores):
            if relate_item in interacted_set:
                continue
            rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
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

    os.makedirs('../user_data/tmp/youtubednn', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/youtubednn/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    if os.path.exists(w2v_file):
        with open(w2v_file, 'rb') as f:
            article_vec_map = pickle.load(f)
    else:
        article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                                   model_path)
        with open(w2v_file, 'wb') as f:
            pickle.dump(article_vec_map, f)

    # Build ANN index for item embeddings
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # Recall
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = max(1, total // n_split)

    # Clear temp folder
    for path, _, file_list in os.walk('../user_data/tmp/youtubednn'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i,
               seq_len=seq_len, decay=decay)

    multitasking.wait_for_tasks()
    log.info('merge tasks')

    dfs = []
    for path, _, file_list in os.walk('../user_data/tmp/youtubednn'):
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
            f'youtubednn: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # Save recall result
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_youtubednn.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_youtubednn.pkl')
