import torch
import datasets
import numpy as np
import logging
from typing import List, Dict, Any, Union
IGNORE_INDEX = -100
logger = logging.getLogger('data')

def load_hf_dataset(path, **kwargs):
    dataset = datasets.load_dataset(path, **kwargs)
    return dataset

def preprocess_chat_instance(tokenizer, template_config: Dict[str, Any], prompt_msgs: Union[List[str], str], response_msgs: Union[List[str], str], max_length: int, predict_with_generate: bool=False) -> Dict[str, torch.Tensor]:
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = ([prompt_msgs], [response_msgs])
    if template_config['apply_chat_template']:
        chat = []
        system_prompt = template_config.get('system_prompt', None)
        if system_prompt:
            chat += [{'role': 'system', 'content': system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{'role': 'user', 'content': prompt}]
            chat += [{'role': 'assistant', 'content': response}]
        date_str = template_config.get('date_string', None)
        date_info = {'date_string': date_str} if date_str is not None else {}
        chat_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, **date_info)
        wrapped_prompt = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True, **date_info)
        prompt_ids = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, **date_info)
    else:
        wrapped_prompt = ''
        system_prompt_with_special_tokens = template_config.get('system_prompt_with_special_tokens', None)
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        n_few_shot = len(prompt_msgs) - 1
        for i in range(n_few_shot):
            fs_prompt, fs_response = (prompt_msgs[i], response_msgs[i])
            wrapped_prompt += template_config['user_start_tag'] + fs_prompt + template_config['user_end_tag'] + template_config['asst_start_tag'] + fs_response + template_config['asst_end_tag']
        final_prompt, final_response = (prompt_msgs[-1], response_msgs[-1])
        wrapped_prompt += template_config['user_start_tag'] + final_prompt + template_config['user_end_tag'] + template_config['asst_start_tag']
        chat_ids = tokenizer(wrapped_prompt + final_response, add_special_tokens=True, max_length=max_length, truncation=True)['input_ids']
        prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=True, max_length=max_length, truncation=True)['input_ids']
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]
    len_matched = len(prompt_ids)
    item = {}
    if predict_with_generate:
        item['input_ids'] = prompt_ids
        labels = chat_ids
    else:
        item['input_ids'] = chat_ids
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
        if len(prompt_ids) == len(chat_ids):
            logger.warning('Tokenization mismatch: no valid target tokens for loss computation')
    item['labels'] = labels
    item['attention_mask'] = [1] * len(item['input_ids'])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item

def preprocess_pretraining_instance(tokenizer, prefix: str, text_content: str, max_length: int, predict_with_generate: bool=False, insert_space: bool=False) -> Dict[str, torch.Tensor]:
    full_seq_ids = tokenizer(prefix + (' ' if insert_space else '') + text_content, add_special_tokens=True)['input_ids']
    prefix_ids = tokenizer(prefix, add_special_tokens=True)['input_ids']
    prefix_len = len(prefix_ids)
    full_seq_ids = full_seq_ids[:prefix_len + max_length]
    len_matched = prefix_len
    if len_matched == 0:
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    item = {}
    if predict_with_generate:
        item['input_ids'] = prefix_ids
    else:
        item['input_ids'] = full_seq_ids
    item['labels'] = labels
    item['attention_mask'] = [1] * len(item['input_ids'])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
