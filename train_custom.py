from modules.decoder_transformer import TransformerModel
from pathlib import Path

import torch
from torch import Tensor, cuda, load as pt_load
import torch.nn.functional as F

from processes import method, dataset
from typing import List, Dict
from data_handlers.clotho_loader import get_clotho_loader
from tools.file_io import load_yaml_file
from tools.printing import init_loggers
from tools.argument_parsing import get_argument_parser
from processes.method import _decode_outputs, _load_indices_file, evaluate_metrics
from tools.beam import Beam


def greedy_decode(model, src, max_len, start_symbol_ind=0):
    ys = torch.ones(src.size()[0], 1).fill_(0).long().to(device)  # ys_0: (batch_size,T_pred=1)
    encoded = model.encode(x)
    for i in range(30 - 1):
        # ys_i:(batch_size, T_pred=i+1)
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(encoded, ys, target_mask=target_mask)  # (T_out, batch_size, nhid)
        prob = model.generator(out[-1, :])  # (T_-1, batch_size, nhid)
        next_word = torch.argmax(prob, dim=1)  # (batch_size)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


def beam_search(model, src, max_len=30, start_symbol_ind=0, end_symbol_ind=9, beam_size=1):
    device = src.device
    batch_size = src.size()[0]
    memory = model.encoder(src)
    ys = torch.ones(src.size()[0], 1).fill_(0).long().to(device)

    first_time = True

    beam = [Beam(beam_size, device, start_symbol_ind, end_symbol_ind)
            for _ in range(batch_size)]

    for i in range(max_len):
        if all((b.done() for b in beam)):
            break
        ys = torch.cat([b.get_current_state() for b in beam], dim=0).to(device).requires_grad_(False)

        # get input mask
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, ntoken) for first time,
        # (T_out, batch_size*beam_size, ntoken) in other times
        out = F.log_softmax(out[-1, :], dim=-1)  # (batch_size, ntoken) for first time,
        # (batch_size*beam_size, ntoken) in other times

        beam_batch = 1 if first_time else beam_size
        for j, b in enumerate(beam):
            b.advance(out[j * beam_batch:(j + 1) * beam_batch, :], first_time)  # update each beam

        if first_time:
            first_time = False  # reset the flag
            # after the first run, the beam expands, so the memory needs to expands too.
            memory = memory.repeat_interleave(beam_size, dim=1)

        output = [b.get_output() for b in beam]
        return output


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

    device = torch.device('cuda')
    model = TransformerModel(
        ntoken, ninp, nhead, nhid, nlayers, batch_size, dropout=0.5, pretrain_cnn=None,
        pretrain_emb=None, freeze_cnn=True
    )
    model.load_state_dict(pt_load(
        Path('outputs/models/best.pt'), map_location='cuda'
    ))

    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

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

    output_y_hat = []
    output_y = []
    f_names = []

    model.eval()
    with torch.no_grad():
        for i, example in enumerate(validation_data):
            device = next(model.parameters()).device
            x, y, f_names_tmp = [i.to(device) if isinstance(i, Tensor)
                                 else i for i in example]
            y_hat = beam_search(model, x, 30, beam_size=3)
            # f_names.extend(f_names_tmp)
            y = y[:, 1:]
            try:
                output_y_hat.extend(y_hat.cpu())
                output_y.extend(y.cpu())
            except AttributeError:
                pass
            except TypeError:
                pass

    captions_gt: List[Dict] = []
    captions_pred: List[Dict] = []
    file_names = sorted(list(data_path_evaluation.iterdir()))
    for f_name, pred, ref in zip(file_names, output_y_hat, output_y):
        gt_caption = []
        predicted_caption = []

        for i, word in enumerate(ref[1:]):
            if word.item() == 9:
                break
            gt_caption.append(indices_list[word.item()])

        for i, word in enumerate(pred[1:]):
            if word.item() == 9:
                break
            predicted_caption.append(indices_list[word.item()])

        gt_caption = ' '.join(gt_caption)
        predicted_caption = ' '.join(predicted_caption)

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

    print("PREDICT: ", captions_pred)

    metrics = evaluate_metrics(captions_pred, captions_gt)

    for metric, values in metrics.items():
        print(f'{metric:<7s}: {values["score"]:7.4f}')
