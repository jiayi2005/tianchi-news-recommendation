import argparse
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)
np.random.seed(seed)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise ImportError('rank_din.py requires torch. Please install torch.') from e

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# CLI args
parser = argparse.ArgumentParser(description='din rank')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', default='cpu')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
epochs = args.epochs
batch_size = args.batch_size
seq_len = args.seq_len
embed_dim = args.embed_dim
lr = args.lr

# Init logger
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'din rank, mode: {mode}')


def build_mapping(values):
    uniq = pd.Series(values).dropna().unique().tolist()
    return {v: i + 1 for i, v in enumerate(uniq)}


def build_article_cate_map(article_path):
    if not os.path.exists(article_path):
        log.warning(f'missing article file: {article_path}, use default cate 0')
        return {}, {0: 0}

    df_article = pd.read_csv(article_path)
    if 'category_id' not in df_article.columns:
        log.warning('articles.csv missing category_id, use default cate 0')
        return {}, {0: 0}

    article_cate = dict(zip(df_article['article_id'], df_article['category_id']))
    cate_map = build_mapping(df_article['category_id'])
    article_cate_idx = {k: cate_map[v] for k, v in article_cate.items()}
    return article_cate_idx, cate_map


def resolve_article_path():
    """Prefer local project articles metadata; keep legacy fallback for compatibility."""
    candidates = ['../data/articles.csv', '../tcdata/articles.csv']
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def build_user_hist_cache(df_click, item_map, seq_len):
    df_click = df_click.sort_values(['user_id', 'click_timestamp'])
    user_hist = df_click.groupby('user_id')['click_article_id'].agg(list)

    cache = {}
    for user_id, items in user_hist.items():
        mapped = [item_map.get(i, 0) for i in items if item_map.get(i, 0) != 0]
        if not mapped:
            continue
        mapped = mapped[-seq_len:]
        pad_len = seq_len - len(mapped)
        cache[user_id] = ([0] * pad_len + mapped,
                          [0] * pad_len + [1] * len(mapped))
    return cache


def prepare_arrays(df_feature, user_map, item_map, article_cate_idx,
                   user_hist_cache, seq_len):
    user_ids = df_feature['user_id'].values
    item_ids = df_feature['article_id'].values

    user_idx = np.array([user_map.get(u, 0) for u in user_ids],
                        dtype=np.int64)
    item_idx = np.array([item_map.get(i, 0) for i in item_ids],
                        dtype=np.int64)
    cate_idx = np.array([article_cate_idx.get(i, 0) for i in item_ids],
                        dtype=np.int64)

    hist_seq = np.zeros((len(df_feature), seq_len), dtype=np.int64)
    hist_mask = np.zeros((len(df_feature), seq_len), dtype=np.float32)

    for i, u in enumerate(user_ids):
        cache = user_hist_cache.get(u)
        if cache is None:
            continue
        hist_seq[i] = cache[0]
        hist_mask[i] = cache[1]

    return user_idx, item_idx, cate_idx, hist_seq, hist_mask


class DIN(nn.Module):
    def __init__(self, n_users, n_items, n_cates, embed_dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(n_cates, embed_dim, padding_idx=0)

        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, user_ids, item_ids, cate_ids, hist_item_ids, hist_mask):
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(item_ids)
        cate_emb = self.cate_emb(cate_ids)
        hist_emb = self.item_emb(hist_item_ids)

        target = item_emb.unsqueeze(1).expand(-1, hist_emb.size(1), -1)
        att_input = torch.cat(
            [hist_emb, target, hist_emb * target, hist_emb - target], dim=-1)
        att_score = self.att_mlp(att_input).squeeze(-1)
        att_score = att_score.masked_fill(hist_mask < 0.5, -1e9)
        att_weight = torch.softmax(att_score, dim=1)
        hist_rep = torch.sum(att_weight.unsqueeze(-1) * hist_emb, dim=1)

        x = torch.cat([user_emb, item_emb, cate_emb, hist_rep], dim=-1)
        logits = self.out_mlp(x).squeeze(-1)
        return logits


def train_model(model, train_loader, device, epochs, lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            user_ids, item_ids, cate_ids, hist_seq, hist_mask, labels = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            cate_ids = cate_ids.to(device)
            hist_seq = hist_seq.to(device)
            hist_mask = hist_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(user_ids, item_ids, cate_ids, hist_seq, hist_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        log.debug(f'epoch {epoch + 1} loss: {total_loss / len(train_loader):.6f}')


def predict(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predict'):
            user_ids, item_ids, cate_ids, hist_seq, hist_mask = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            cate_ids = cate_ids.to(device)
            hist_seq = hist_seq.to(device)
            hist_mask = hist_mask.to(device)

            logits = model(user_ids, item_ids, cate_ids, hist_seq, hist_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    if preds:
        return np.concatenate(preds)
    return np.array([])


if __name__ == '__main__':
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        log.warning('CUDA not available, fallback to CPU')
        device = 'cpu'

    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        model_dir = '../user_data/model'
    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        model_dir = '../user_data/model'

    os.makedirs(model_dir, exist_ok=True)

    article_path = resolve_article_path()

    if mode == 'valid':
        # Build mappings and train in valid mode
        user_map = build_mapping(
            pd.concat([df_click['user_id'], df_feature['user_id']]))
        item_map = build_mapping(
            pd.concat([df_click['click_article_id'], df_feature['article_id']]))
        article_cate_idx, cate_map = build_article_cate_map(article_path)

        user_hist_cache = build_user_hist_cache(df_click, item_map, seq_len)

        user_idx, item_idx, cate_idx, hist_seq, hist_mask = prepare_arrays(
            df_feature, user_map, item_map, article_cate_idx, user_hist_cache,
            seq_len)

        train_mask = df_feature['label'].notnull().values
        test_mask = df_feature['label'].isnull().values

        train_labels = df_feature.loc[train_mask, 'label'].values.astype(
            np.float32)

        X_train = TensorDataset(
            torch.from_numpy(user_idx[train_mask]),
            torch.from_numpy(item_idx[train_mask]),
            torch.from_numpy(cate_idx[train_mask]),
            torch.from_numpy(hist_seq[train_mask]),
            torch.from_numpy(hist_mask[train_mask]),
            torch.from_numpy(train_labels),
        )

        train_loader = DataLoader(X_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

        n_users = max(user_map.values()) + 1
        n_items = max(item_map.values()) + 1
        n_cates = max(cate_map.values()) + 1

        model = DIN(n_users, n_items, n_cates, embed_dim)
        train_model(model, train_loader, device, epochs, lr)

        torch.save(model.state_dict(), os.path.join(model_dir, 'din.pt'))
        meta = {
            'user_map': user_map,
            'item_map': item_map,
            'cate_map': cate_map,
            'article_cate_idx': article_cate_idx,
            'seq_len': seq_len,
            'embed_dim': embed_dim,
        }
        with open(os.path.join(model_dir, 'din_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        # Evaluate
        train_pred_dataset = TensorDataset(
            torch.from_numpy(user_idx[train_mask]),
            torch.from_numpy(item_idx[train_mask]),
            torch.from_numpy(cate_idx[train_mask]),
            torch.from_numpy(hist_seq[train_mask]),
            torch.from_numpy(hist_mask[train_mask]),
        )
        train_pred_loader = DataLoader(train_pred_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)
        train_preds = predict(model, train_pred_loader, device)

        df_eval = df_feature.loc[train_mask, ['user_id', 'article_id', 'label']]
        df_eval = df_eval.copy()
        df_eval['pred'] = train_preds
        df_eval.sort_values(['user_id', 'pred'],
                            ascending=[True, False],
                            inplace=True)
        total = df_eval[df_eval['label'].notnull()].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_eval, total)
        log.debug(
            f'din: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

        # Build submission for unlabeled users
        test_dataset = TensorDataset(
            torch.from_numpy(user_idx[test_mask]),
            torch.from_numpy(item_idx[test_mask]),
            torch.from_numpy(cate_idx[test_mask]),
            torch.from_numpy(hist_seq[test_mask]),
            torch.from_numpy(hist_mask[test_mask]),
        )
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_preds = predict(model, test_loader, device)

        df_pred = df_feature.loc[test_mask, ['user_id', 'article_id']].copy()
        df_pred['pred'] = test_preds

        df_sub = gen_sub(df_pred)
        df_sub.sort_values(['user_id'], inplace=True)
        os.makedirs('../prediction_result', exist_ok=True)
        df_sub.to_csv('../prediction_result/result_din.csv', index=False)

    else:
        # Online mode uses pretrained model and mappings
        meta_path = os.path.join(model_dir, 'din_meta.pkl')
        model_path = os.path.join(model_dir, 'din.pt')
        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise FileNotFoundError('missing din model/meta, train in valid mode first')

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        user_map = meta['user_map']
        item_map = meta['item_map']
        cate_map = meta['cate_map']
        article_cate_idx = meta['article_cate_idx']
        seq_len = meta['seq_len']
        embed_dim = meta['embed_dim']

        user_hist_cache = build_user_hist_cache(df_click, item_map, seq_len)

        user_idx, item_idx, cate_idx, hist_seq, hist_mask = prepare_arrays(
            df_feature, user_map, item_map, article_cate_idx, user_hist_cache,
            seq_len)

        n_users = max(user_map.values()) + 1
        n_items = max(item_map.values()) + 1
        n_cates = max(cate_map.values()) + 1

        model = DIN(n_users, n_items, n_cates, embed_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        test_dataset = TensorDataset(
            torch.from_numpy(user_idx),
            torch.from_numpy(item_idx),
            torch.from_numpy(cate_idx),
            torch.from_numpy(hist_seq),
            torch.from_numpy(hist_mask),
        )
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        test_preds = predict(model, test_loader, device)

        df_pred = df_feature[['user_id', 'article_id']].copy()
        df_pred['pred'] = test_preds

        df_sub = gen_sub(df_pred)
        df_sub.sort_values(['user_id'], inplace=True)
        os.makedirs('../prediction_result', exist_ok=True)
        df_sub.to_csv('../prediction_result/result_din.csv', index=False)
