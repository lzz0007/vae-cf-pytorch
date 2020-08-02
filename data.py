import os
import pandas as pd
from scipy import sparse
import numpy as np
import pickle
import re
import spacy
from nltk.corpus import stopwords
import string
import gzip
import json
from tqdm import tqdm


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
        # tp = tp[tp['userId'].isin(usercount.index[usercount <= 100])]

    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


tok = spacy.load('en')
stop = stopwords.words('english')


def tokenize(text):
    if text.isdigit():
        res = str(text)
    else:
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        res = re.sub(r"[^A-Za-z]+", " ", nopunct)
        res = ' '.join([word for word in res.split() if word not in (stop)])
    if not res:
        res = 'none'
    return [token.text for token in tok.tokenizer(res)]


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000  # UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def prepare_dataset(sentences, items, word_to_id):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing string of words and word indexes
    """
    data = []
    for i in range(len(sentences)):
        item = items[i]
        str_words = list(set([w for w in sentences[i]]))[:100]
        words = [word_to_id[w if w in word_to_id else '<UNK>'] for w in str_words]
        data.append({'str_words': str_words, 'words': words, 'item': item})
    return data


def sparse_mat(data, item2id, word_to_id):
    # max(meta['encoded'].apply(lambda x: len(x)))
    ulist = []
    ilist = []
    for i in range(len(data)):
        iid = data[i]['item']  # item id
        titles = data[i]['words']  # item title
        # meta_titles[iid] = row
        for t in titles:
            ulist.append(iid)
            ilist.append(t)
    mat = sparse.csr_matrix((np.ones_like(ulist), (ulist, ilist)), dtype='float64',
                            shape=(len(item2id), len(word_to_id)))
    return mat


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    data = pd.DataFrame.from_dict(df, orient='index')
    data = data[['title', 'image', 'asin']]
    data = data[~data.title.fillna('').str.contains('getTime')]
    # data['category'] = category
    return data


if __name__ == '__main__':

    which_dataset_to_use = 4  # in {0, 1, 2, 3}, see below.
    dataset = {0: 'ml-latest-small', 1: 'ml-1m', 2: 'ml-20m', 3: 'netflix', 4: 'amazon'}

    n_heldout_users = (50, 500, 10000, 40000, 900)[which_dataset_to_use]
    DATA_DIR = 'data/%s/' % dataset[which_dataset_to_use]
    print("Load and Preprocess %s dataset" % dataset[which_dataset_to_use])


    # Load Data
    if 'ml-1m' in DATA_DIR:
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'),
                               header=None, delimiter='::',
                               names=['userId', 'movieId', 'rating', 'timestamp'])
    elif 'amazon' in DATA_DIR:
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'Luxury_Beauty.csv'), header=None,
                             names=['movieId', 'userId', 'rating', 'timestamp'])
    else:
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)

    raw_data = raw_data[raw_data['rating'] > 3.5]

    # Load title data
    meta = getDF(os.path.join(DATA_DIR, 'meta_Luxury_Beauty.json.gz'))
    meta = meta.drop_duplicates(subset='asin', keep='first')
    # find null title
    meta = meta[meta['title'].notnull()]
    meta_item = meta['asin'].unique()
    # remove items which dont have title
    raw_data = raw_data[raw_data['movieId'].isin(meta_item)]

    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, 5, 0)
    raw_data = raw_data.drop_duplicates(subset=['movieId', 'userId'], keep='first')

    # Shuffle User Indices
    unique_uid = user_activity.index
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    n_users = unique_uid.size
    # n_heldout_users = 10000

    # Split Train/Validation/Test User Indices
    # tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    # vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    # te_users = unique_uid[(n_users - n_heldout_users):]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(raw_data)

    # train_plays = raw_data.loc[raw_data['userId'].isin(unique_uid)]
    unique_sid = pd.unique(vad_plays_tr['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(DATA_DIR, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    # mapping = {'item2id': show2id, 'user2id': profile2id}
    # with open(os.path.join(pro_dir, 'mapping.pickle'), 'wb') as f:
    #     pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

    # vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    # vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    # vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    # test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    # test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    # test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    train_data = numerize(vad_plays_tr, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_plays_te = vad_plays_te[vad_plays_te['movieId'].isin(unique_sid)]
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    # test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    # test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    # test_data_te = numerize(test_plays_te, profile2id, show2id)
    # test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print('total no of users:', len(unique_uid))
    print('total no of items:', len(unique_sid))
    print('total no of interactions for train:', train_data.shape[0])
    print('sparsity:',  train_data.shape[0]/(len(unique_uid)*len(unique_sid)))

    # process title data
    meta = meta[meta['asin'].isin(unique_sid)]
    meta['item_id'] = meta['asin'].apply(lambda x: show2id[x])
    meta = meta.sort_values('item_id')
    meta['title_length'] = meta['title'].apply(lambda x: len(str(x).split()))
    np.mean(meta['title_length'])
    np.max(meta['title_length'])

    sentences = [tokenize(s[0]) for _, s in meta.iterrows()]
    items = [show2id[s[2]] for _, s in meta.iterrows()]
    dico_words, word_to_id, id_to_word = word_mapping(sentences)

    res = prepare_dataset(sentences, items, word_to_id)
    # max_word = max([len(i['words']) for i in res])

    meta_mat = sparse_mat(res, show2id, word_to_id)

    metaset = {'meta_mat': meta_mat, 'vocab2index': word_to_id, 'index2vocab': id_to_word, 'item2index': show2id,
               'user2index': profile2id, 'meta_titles': res, 'sentences': sentences, 'items': items}

    with open(os.path.join(DATA_DIR, 'meta_encoded.pickle'), 'wb') as f:
        pickle.dump(metaset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")

    # meta['processed_title'] = meta['title'].apply(lambda x: tokenize(x))
    # meta.columns
    # tmp = meta[['title', 'processed_title']]
    # tmp.to_csv('title_names.csv', header=False)