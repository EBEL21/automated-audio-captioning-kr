from tools.csv_functions import read_csv_file
from itertools import chain, count
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordPOSTagger, pos_tag
from nltk.corpus.reader.wordnet import NOUN, ADJ, ADV, VERB
import pickle


def get_keyword_indices(meta_keys, words_list):
    keyword_indices = []
    for kws in meta_keys:
        ind_list = []
        for kw in kws:
            try:
                ind = words_list.index(kw)
            except ValueError:
                continue
            ind_list.append(ind)
        keyword_indices.append(ind_list)
    return keyword_indices


if __name__ == "__main__":

    # load metadata csv files
    dir_path = 'data/clotho_csv_files'
    meta_dev = read_csv_file('clotho_metadata_development.csv', dir_path)
    meta_eval = read_csv_file('clotho_metadata_evaluation.csv', dir_path)

    # load StanfordPOSTagger -> pos_tag is better
    # jar = "./stanford-postagger-full-2020-08-06/stanford-postagger.jar"
    # model = "./stanford-postagger-full-2020-08-06/models/english-bidirectional-distsim.tagger"
    # tagger = StanfordPOSTagger(model, jar, encoding="utf-8")


    lmt = WordNetLemmatizer()

    dict_pos_map = {
        'NN': NOUN,
        'NNS': NOUN,
        'MD': NOUN,
        'VBG': VERB,
        'VBN': VERB,
        'VBD': VERB,
        'VB': VERB,
        'VBP': VERB,
        'VBZ': VERB,
        'JJ': ADJ,
        'JJS': ADJ,
        'JJR': ADJ,
        'RB': ADV
    }

    # keywords split and update
    for csv_entry in chain(meta_dev, meta_eval):
        keywords = csv_entry.get('keywords')
        keywords = keywords.split(';')
        keywords_set = set()
        for keyword in keywords:
            tag_tuple = pos_tag([keyword])[0]
            word = tag_tuple[0]
            pos = tag_tuple[1]
            if pos in ['NNP', 'NNPS', 'IN', 'PRP$', 'WDT', 'SYM']:
                keywords_set.add(word.lower())
            elif pos in ['CD', 'DT', 'CC']:
                continue
            else:
                try:
                    tmp = lmt.lemmatize(word.lower(), pos=dict_pos_map[pos])
                except KeyError:
                    print(word, pos)
                    raise KeyError('Unexpected POS')
                keywords_set.add(tmp)
        csv_entry.update({'keywords': keywords_set})

    # make metadata keyword dict
    file_names_dev = [
        csv_field.get('file_name')
        for csv_field in meta_dev
    ]

    meta_keys_dev = [
        csv_field.get('keywords')
        for csv_field in meta_dev
    ]

    file_names_eval = [
        csv_field.get('file_name')
        for csv_field in meta_eval
    ]

    meta_keys_eval = [
        csv_field.get('keywords')
        for csv_field in meta_eval
    ]

    meta_keys_counter = Counter(chain.from_iterable(meta_keys_dev))
    # total 355 words
    words_list = [key for key in meta_keys_counter.keys()
                  if meta_keys_counter[key] >= 10]

    keyword_indices_dev = get_keyword_indices(meta_keys_dev, words_list)
    keyword_indices_eval = get_keyword_indices(meta_keys_eval, words_list)

    with open('./data/pickles/dev_keywords.p', 'wb') as f:
        pickle.dump(keyword_indices_dev, f)
        f.close()

    with open('./data/pickles/eval_keywords.p', 'wb') as f:
        pickle.dump(keyword_indices_eval, f)
        f.close()