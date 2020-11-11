from modules.decoder_transformer import TransformerModel
from pathlib import Path

import torch
from torch import Tensor, cuda, load as pt_load
import torch.nn.functional as F

import numpy as np
from typing import List, Dict
from data_handlers.clotho_loader import get_clotho_loader
from tools.file_io import load_yaml_file
from tools.argument_parsing import get_argument_parser
from processes.method import _load_indices_file, evaluate_metrics
from tools.word_embedding import align_word_embedding
from tools.beam import beam_search


def greedy_decode(model, src, max_len=30, start_symbol_ind=0):
    batch_size = src.size()[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)
    encoded = model.encode(src)
    for i in range(max_len - 1):
        # ys_i:(batch_size, T_pred=i+1)
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(encoded, ys, target_mask=target_mask)  # (T_out, batch_size, nhid)
        prob = model.generator(out[-1, :])  # (T_-1, batch_size, nhid)
        next_word = torch.argmax(prob, dim=1)  # (batch_size)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys

def evaluation():
    output_y_hat = []
    output_y = []
    f_names = []

    model.eval()
    with torch.no_grad():
        for ind, example in enumerate(validation_data):
            x, y, _, f_names_tmp = [i.to(device) if isinstance(i, Tensor)
                                    else i for i in example]
            y_hat = beam_search(model, x, 30, beam_size=3)
            y = y[:, 1:]
            try:
                output_y_hat.extend(y_hat)
                output_y.extend(y.cpu())
            except AttributeError:
                print("attr error")
                pass
            except TypeError:
                print("type error")
                pass

    print("evaluation end")
    print("decode captions start")

    captions_gt: List[Dict] = []
    captions_pred: List[Dict] = []

    file_names = sorted(list(data_path_evaluation.iterdir()))
    for f_name, pred, ref in zip(file_names, output_y_hat, output_y):
        gt_caption = []
        predicted_caption = []

        for i, word in enumerate(ref):
            if word.item() == 9:
                break
            gt_caption.append(indices_list[word.item()])

        for i, word in enumerate(pred[1:]):
            if word.item() == 9:
                break
            predicted_caption.append(indices_list[word.item()])

        gt_caption = ' '.join(gt_caption)
        predicted_caption = ' '.join(predicted_caption)

        print(gt_caption)
        print(predicted_caption)
        print('----------------------------------')

        f_n = f_name.stem.split('.')[0]

        if f_n not in f_names:
            f_names.append(f_n)
            captions_pred.append({
                'file_name': f_n,
                'caption_predicted': predicted_caption})
            captions_gt.append({
                'file_name': f_n,
                'caption_1': gt_caption})
        else:
            for d_i, d in enumerate(captions_gt):
                if f_n == d['file_name']:
                    len_captions = len([i_c for i_c in d.keys()
                                        if i_c.startswith('caption_')]) + 1
                    d.update({f'caption_{len_captions}': gt_caption})
                    captions_gt[d_i] = d
                    break

    print("decode captions end")
    print("calculate metrics start")

    metrics = evaluate_metrics(captions_pred, captions_gt)

    for metric, values in metrics.items():
        print(f'{metric:<7s}: {values["score"]:7.4f}')


if __name__ == "__main__":
    batch_size = 16
    nhead = 4
    nhid = 192
    nlayers = 2
    ninp = 64
    ntoken = 4367 + 1
    clip_grad = 2.5
    lr = 3e-4  # learning rate
    beam_width = 3
    training_epochs = 50
    log_interval = 100
    checkpoint_save_interval = 5

    device = torch.device('cuda:0')

    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

    print("load settings start")

    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    settings_training = settings['dnn_training_settings']['training'],
    settings_data = settings['dnn_training_settings']['data'],
    settings_io = settings['dirs_and_files']

    indices_list = _load_indices_file(
        settings['dirs_and_files'],
        settings['dnn_training_settings']['data'])

    data_path_evaluation = Path(
        settings_io['root_dirs']['data'],
        settings_io['dataset']['features_dirs']['output'],
        settings_io['dataset']['features_dirs']['evaluation'])

    validation_data = get_clotho_loader(
        settings_io['dataset']['features_dirs']['evaluation'],
        is_training=False,
        settings_data=settings['dnn_training_settings']['data'],
        settings_io=settings['dirs_and_files'])

    print("load settings end")
    print("load pre-trained models start")

    pretrain_emb = align_word_embedding(r'data/pickles/words_list.p', r'outputs/models/w2v_192.mod', ntoken,
                                        nhid)
    # pretrain_cnn = torch.load(Path('outputs/models/TagModel_60.pt'), map_location='cuda')

    model = TransformerModel(
        ntoken, ninp, nhead, nhid, nlayers, batch_size, dropout=0.2, pretrain_cnn=None,
        pretrain_emb=pretrain_emb, freeze_cnn=False
    )

    model.load_state_dict(pt_load('outputs/models/epoch_090_transformer_model.pt', map_location=device))
    # model.load_encoder(pretrain_cnn, pretrain_emb)
    model.to(device)

    print('--------------------------------------------------------')

    print("load pre-trained models end")
    print("evaluation start")
