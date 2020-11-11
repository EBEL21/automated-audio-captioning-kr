import openl3
import soundfile as sf
from pathlib import Path
import numpy as np
from itertools import chain


def feature_extraction_l3(file_name):
    audio, sr = sf.read(file_name)
    emb, ts = openl3.get_audio_embedding(audio, sr, content_type='env',
                                         embedding_size=512)
    return emb


if __name__ == "__main__":

    dir_dev = Path('./data/clotho_audio_files/development')
    dir_eval = Path('./data/clotho_audio_files/evaluation')

    output_dir_dev = Path('./data/openl3/development')
    output_dir_eval = Path('./data/openl3/evaluation')

    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="env",
                                                     embedding_size=512)
    dev_files_list = []
    for data_file_name in dir_dev.iterdir():
        dev_files_list.append(str(data_file_name))
    openl3.process_audio_file(dev_files_list, model=model, output_dir=output_dir_dev, batch_size=16)

    eval_files_list = []
    for data_file_name in dir_eval.iterdir():
        eval_files_list.append(str(data_file_name))
    openl3.process_audio_file(eval_files_list, model=model, output_dir=output_dir_eval, batch_size=16)
