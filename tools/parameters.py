from pathlib import Path


class parameters:
    batch_size = 16
    nhead = 4
    nhid = 192
    nlayers = 2
    ninp = 64
    ntoken = 4367 + 1
    clip_grad = 2.5
    lr = 3e-4 # learning rate
    beam_width = 3
    training_epochs = 100
    log_interval = 100
    checkpoint_save_interval = 5
    load_model = False
    is_training = True

    input_field_name = 'features'
    output_field_name = 'words_ind'
    load_into_memory = False

    data_path_development = r'./data/data_splits/development'
    data_path_evaluation = r'./data/data_splits/evaluation'
    data_path_test = r'./data/data_splits/test'

    word_dict_pickle_path = './data/pickles/words_list.p'
    word_freq_pickle_path = './data/pickles/words_frequencies.p'

    keyword_path_development = './data/pickles/dev_keywords.p'
    keyword_path_evaluation = './data/pickles/eval_keywords.p'
