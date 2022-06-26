import json

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import to_array, sequence_padding
from sklearn.metrics.pairwise import cosine_similarity
import os

# bert config
config_path = 'en_1024/bert_config.json'
checkpoint_path = 'en_1024/bert_model.ckpt'
dict_path = 'en_1024/vocab.txt'
tokenizer = Tokenizer(load_vocab(dict_path), do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path)
MAX_LEN = 30
PADDING_LEN = 50

# privacy config
title_path = 'cleared_all.json'
sample_path = 'samples.json'
result_dir = 'sim_result/'


def extract_emb_feature(sample_ids, article):
    result = [list() for i in range(len(article))]
    title_idx = 0
    for title_token, title_segment in article:
        for sample_token, sample_segment in sample_ids:
            token_list = [title_token, sample_token]
            segment_list = [title_segment, sample_segment]
            token_list = sequence_padding(token_list, length=PADDING_LEN)
            segment_list = sequence_padding(segment_list, length=PADDING_LEN)
            token_list, segment_list = to_array(token_list, segment_list)
            predict = np.mean(model.predict([token_list, segment_list]), axis=1)
            # print(predict.shape)
            result[title_idx].append(cosine_similarity(predict))
        title_idx += 1
    return result


if __name__ == '__main__':
    samples = dict()
    with open(sample_path, 'r') as f:
        samples = json.loads(f.read())
    sample_ids = dict()
    for key in samples.keys():
        sample_ids[key] = [tokenizer.encode(s, maxlen=MAX_LEN) for s in samples[key]]
    articles = dict()
    article_ids = dict()
    with open(title_path, 'r') as f:
        articles = json.loads(f.read())
    for key in articles.keys():
        if os.path.exists(result_dir + key + '.json'):
            continue
        article_ids[key] = [tokenizer.encode(s, maxlen=MAX_LEN) for s in articles[key]]
    for article_key in article_ids.keys():
        print(article_key + " begin")
        result = dict()
        article = article_ids[article_key]
        # print(articles[article_key])
        for sample_key in sample_ids.keys():
            # print(samples[sample_key])
            emb_feature = extract_emb_feature(sample_ids[sample_key], article)
            result[sample_key] = []
            for i in range(len(article)):
                for j in range(len(sample_ids[sample_key])):
                    result[sample_key].append((i, j, emb_feature[i][j][0][1]))
        with open(result_dir + article_key + '.json', 'w', encoding='utf-8') as f:
            f.write(str(result))
        print(article_key + " end")
