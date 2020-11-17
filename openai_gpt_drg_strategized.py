# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

"""
import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange
import time
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--max_seq_length', type=int, default=110)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Set the seed for random, numpy, PyTorch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    
    ###### NOTE: MODIFIED PARTS ######
    # special_tokens = ['<ATTR_WORDS>','<CON_START>','<START>','<END>']
    special_tokens = ['<STR>','<CONTEXT>','<START>','<END>', \
                      '<Actually>',
                      '<Adverb.Just>',
                      '<Affirmation>',
                      '<Apology>',
                      '<By.The.Way>',
                      '<Indicative>',
                      '<Conj.Start>',
                      '<Subjunctive>',
                      '<Filler>',
                      '<For.Me>',
                      '<For.You>',
                      '<Gratitude>',
                      '<Hedges>',
                      '<Greeting>',
                      '<Please>',
                      '<Please.Start>',
                      '<Reassurance>',
                      '<Swearing>']
    
    ###### END OF CHANGE ######
    
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    start_token_id = tokenizer.convert_tokens_to_ids(['<START>'])[0]
    model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
    model.to(device)
    
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load and encode dataset
    def tokenize_and_encode(file_path):
        '''
        This method tokenizes the input data and encodes it using the OpenAIGPTTokenizer
        :param file_path: Path of the input file, dtype: str
        :return: encoded dataset  dtype: list
        '''
        with open(file_path, 'r') as in_fp:
            lines = in_fp.read().splitlines()
        #lines = lines[:40000]
        tokenized_dataset = lines
        for i, line in enumerate(tqdm(lines)):
            token = tokenizer.tokenize(line)[:512]
            tokenized_dataset[i] = tokenizer.convert_tokens_to_ids(token)
        return tokenized_dataset

    logger.info("Encoding dataset...")
    train_dataset = tokenize_and_encode(args.train_dataset)
    #train_dataset = tokenize_and_encode[:100]
    eval_dataset = tokenize_and_encode(args.eval_dataset)

    train_dataset = [x for x in train_dataset if len(x) <= args.max_seq_length and start_token_id in x]
    eval_dataset = [x for x in eval_dataset if len(x) <= args.max_seq_length and start_token_id in x]
    print("Training samples = {}".format(len(train_dataset)))
    print("Validation samples = {}".format(len(eval_dataset)))
    print("Example = {}".format(train_dataset[0]))
    time.sleep(2)
    # Compute the mex input length for the Transformer
    input_length = max(max(len(t) for t in train_dataset), max(len(q) for q in eval_dataset))
    if n_gpu > 1:
        input_length = min(input_length, model.module.config.n_positions)
    else:
        input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    print("Input Length = {}".format(input_length))
    
    
    def pre_process_dataset(encoded_dataset, input_length, start_token_id):
        """
        This method is to create torch tensor of input ids and lm labels
        :param encoded_dataset: Input dataset, dtype: list
        :param input_length: Maximum length of sentence from training and eval dataset, dtype: int
        :param start_token_id: id of the '<START>' token, dtype: int
        :return: torch.tensor of size [len(encoded_dataset), 2]
        """

        n_batch = len(encoded_dataset)
        input_ids = np.zeros(shape=(n_batch, input_length), dtype=np.int64)
        lm_labels = np.full(shape=(n_batch, input_length), fill_value=-1, dtype=np.int64)

        for i, tokens in enumerate(encoded_dataset):
            try:
                start_id_index = tokens.index(start_token_id)
                input_ids[i, :len(tokens)] = tokens
                start_id_index = tokens.index(start_token_id)
                lm_labels[i, start_id_index : len(tokens)-1] = tokens[start_id_index + 1: len(tokens)]
                # LM loss calculate only for tokens after <START> token in the sentence
                #lm_labels[i, :len(tokens)-1] = tokens[1:]
            except ValueError:
                #print("Index {} doesn't have start token".format(i))
                print("Example = {}".format(tokens))
                raise ValueError("Example {} doesn't have start token".format(i))
                 

        input_ids = torch.tensor(input_ids)
        lm_labels = torch.tensor(lm_labels)
        tensor_dataset = (input_ids, lm_labels)
        #tensor_dataset.append(torch.tensor(d) for d in all_inputs)

        return tensor_dataset

    # Prepare input tensors and dataloders
    train_tensor_dataset = pre_process_dataset(train_dataset, input_length, start_token_id=start_token_id)
    eval_tensor_dataset = pre_process_dataset(eval_dataset, input_length, start_token_id=start_token_id)

    print("Training Example Input ids= {}".format(train_tensor_dataset[0][0]))
    print("Training Example Language Modeling ids = {}".format(train_tensor_dataset[1][0]))
    time.sleep(10)
    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels = batch
                loss = model(input_ids, lm_labels=lm_labels)
                if n_gpu > 1:
                    loss.mean().backward()
                else:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if n_gpu > 1:
                    tmp_loss = loss.mean().item()
                else:
                    tmp_loss = loss.item()
                exp_average_loss = tmp_loss if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * tmp_loss
                nb_tr_steps += 1
                tr_loss += tmp_loss
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "pytorch_model_zero_grad_{}.bin".format(epoch+1))
            config = model.module.config if hasattr(model, 'module') else model.config
            torch.save(model_to_save.state_dict(), output_model_file)

            model_state_dict = torch.load(output_model_file)
            model = OpenAIGPTLMHeadModel(config)
            model.load_state_dict(model_state_dict)
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

    # Save a trained model
    # if args.do_train:
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    #     output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    #     config = model.config
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #
    #     # Load a trained model that you have fine-tuned
    #     model_state_dict = torch.load(output_model_file)
    #     model = OpenAIGPTLMHeadModel(config)
    #     model.load_state_dict(model_state_dict)
    #     model.to(device)

    if args.do_eval:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, lm_labels = batch
            with torch.no_grad():
                lm_loss = model(input_ids, lm_labels=lm_labels)

            eval_loss += lm_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == '__main__':
    main()
