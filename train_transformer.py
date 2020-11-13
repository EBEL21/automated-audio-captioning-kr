from tools.beam import beam_search
from eval_metrics import evaluate_metrics
from modules.decoder_transformer import TransformerModel
from pathlib import Path

import torch
from torch import Tensor, cuda, load as pt_load
from data_handlers.clotho_loader import get_clotho_loader
from tools.file_io import load_yaml_file
from tools.argument_parsing import get_argument_parser
from processes.method import _load_indices_file
from tools.word_embedding import align_word_embedding, LabelSmoothingLoss
import time
from tools.parameters import parameters as param
from typing import List, Dict


def get_padding(tgt, tgt_len):
    # tgt: (batch_size, max_len)
    device = tgt.device
    batch_size = tgt.size()[0]
    max_len = tgt.size()[1]
    mask = torch.zeros(tgt.size()).type_as(tgt).to(device)
    for i in range(batch_size):
        d = tgt[i]
        num_pad = max_len - int(tgt_len[i].item())
        mask[i][max_len - num_pad:] = 1
        # tgt[i][max_len - num_pad:] = pad_idx

    # mask:(batch_size,max_len)
    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()
    return mask


def evaluation():
    output_y_hat = []
    output_y = []
    f_names = []

    model.eval()
    with torch.no_grad():
        for ind, example in enumerate(eval_data):
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

    captions_gt: List[Dict] = []
    captions_pred: List[Dict] = []

    file_names = sorted(list(Path(param.data_path_evaluation).iterdir()))
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


def train():
    model.train()
    total_loss = 0.
    batch = 0
    start_time = time.time()
    log_interval = param.log_interval
    for src, tgt, tgt_len, _ in train_data:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_pad_mask = get_padding(tgt, tgt_len)
        tgt_in = tgt[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_in, target_padding_mask=tgt_pad_mask)

        loss_text = criterion(output.contiguous().view(-1, param.ntoken),
                              tgt_y.transpose(0, 1).contiguous().view(-1))
        loss = loss_text
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), param.clip_grad)
        optimizer.step()
        total_loss += loss_text.item()

        batch += 1

        if batch % log_interval == 0 and batch > 0:
            mean_loss = total_loss / batch
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | '
                  'loss {:5.2f}'.format(epoch, batch, len(train_data), elapsed * 1000 / log_interval, mean_loss))
            start_time = time.time()


if __name__ == "__main__":

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

    train_data = get_clotho_loader(
        split='development',
        data_dir=param.data_path_development,
        is_training=True,
        input_field_name=param.input_field_name,
        output_field_name=param.output_field_name,
        batch_size=param.batch_size,
        augment=True)

    eval_data = get_clotho_loader(
        split='evaluation',
        data_dir=param.data_path_evaluation,
        is_training=False,
        input_field_name=param.input_field_name,
        output_field_name=param.output_field_name,
        batch_size=param.batch_size)

    print("load settings end")
    print("load pre-trained models start")

    pretrain_emb = align_word_embedding(r'data/pickles/words_list.p', r'outputs/models/w2v_192.mod', param.ntoken,
                                        param.nhid)
    pretrain_cnn = torch.load(Path('outputs/models/TagModel_20.pt'), map_location='cuda')

    model = TransformerModel(
        param.ntoken, param.ninp, param.nhead, param.nhid, param.nlayers, param.batch_size, dropout=0.2,
        pretrain_cnn=pretrain_cnn, pretrain_emb=None, freeze_cnn=False
    )

    if param.load_model:
        model.load_state_dict(pt_load('outputs/models/epoch_042_fine_tuning_model.pt', map_location=device))
    # model.load_encoder(pretrain_cnn, pretrain_emb)
    model.to(device)

    print('--------------------------------------------------------')
    criterion = LabelSmoothingLoss(param.ntoken, smoothing=0.1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=param.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    if param.is_training:
        epoch = 1
        while epoch < param.training_epochs + 1:
            scheduler.step(epoch)
            train()
            if epoch % 3 == 0:
                evaluation()
            torch.save(model.state_dict(), './outputs/models/epoch_{:03d}_transformer_model.pt'.format(epoch))
            epoch += 1
    else:
        evaluation()
