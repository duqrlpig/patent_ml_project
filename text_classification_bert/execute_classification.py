

import torch
import numpy as np
import pickle

# from sklearn.metrics import matthews_corrcoef, confusion_matrix

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tools import *
from multiprocessing import Pool, cpu_count
import convert_examples_to_features

import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import pandas as pd
import re
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "/content/gdrive/My Drive/Colab Notebooks/byBert/data/"

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'yelp.tar.gz'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'yelp'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'{DATA_DIR}outputs/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
#REPORTS_DIR = f'{DATA_DIR}reports/{TASK_NAME}_evaluation_reports_20041902/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = f'{DATA_DIR}cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 128
EVAL_BATCH_SIZE = 8
OUTPUT_MODE = 'classification'

def get_ref_data(text_list, dep_list):
    values = []
    
    for idx in range(len(dep_list)):
        if dep_list[idx] == 0:
            values.append(-1)
        else:
            text = get_subs_claim_text(text_list[idx])
            numList = re.findall(r'\b\d+\b', text)
            # print(f"numList: {numList}, {text_list[idx]}")
            if len(numList) > 0:
                values.append(numList[0])
            else:
                values.append(-1)
    print(values)
    return values


def get_subs_claim_text(text):
    if 'claim' in text:
        index = text.index("claim")
        text = text[index:]
    elif 'CLAIM' in text:
        index = text.index("CLAIM")
        text = text[index:]
    print(f"sub text: {text}")
    return text
        
        


def execute_classification(FILE_DIR):

    test_df = pd.read_csv(FILE_DIR+"test.csv", header=None)
    test_df.head()
    test_df[0] = (test_df[0] == 2).astype(int)
    test_df.head()

    dev_df_bert = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
    })
    
    dev_df_bert.head()
    col_text_list = dev_df_bert['text'].tolist()

    dev_df_bert.to_csv(FILE_DIR + 'dev.tsv', sep='\t', index=False, header=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)

    processor = BinaryClassificationProcessor()
    eval_examples = processor.get_dev_examples(FILE_DIR)
    label_list = processor.get_labels() # [0, 1] for binary classification
    eval_examples_len = len(eval_examples)

    label_map = {label: i for i, label in enumerate(label_list)}
    eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

    process_count = cpu_count() - 1
    print(process_count)
    # if __name__ ==  '__main__':
    print(f'Preparing to convert {eval_examples_len} examples..')
    print(f'Spawning {process_count} processes..')

    global eval_features
    with Pool(process_count) as p:
        eval_features = list(p.imap(convert_examples_to_features.convert_example_to_feature, eval_examples_for_processing))

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    # Load pre-trained model (weights)
    model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
    model.to(device)
    model.eval()

    values = []

    print('start classification')

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        
        for value in logits:
            if value[0] < 0:
                values.append(1)
            else:
                values.append(0)

    return values
