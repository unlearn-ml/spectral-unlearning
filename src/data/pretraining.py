import hashlib
import logging
import os
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from data.utils import load_hf_dataset, add_dataset_index, preprocess_pretraining_instance
_logger = logging.getLogger('data.pretraining')
_CHUNK_CACHE_DIR = Path(os.environ.get('OPEN_UNLEARNING_CACHE_DIR', str(Path.home() / '.cache' / 'open-unlearning'))) / 'chunks'

def _chunk_cache_key(hf_args, tokenizer, text_key, max_length):
    data_files = hf_args.get('data_files', '')
    if isinstance(data_files, str):
        paths = [data_files]
    else:
        paths = list(data_files)
    spec_parts = []
    for p in paths:
        abs_p = os.path.abspath(p)
        try:
            st = os.stat(abs_p)
            spec_parts.append(f'{abs_p}:{st.st_mtime_ns}:{st.st_size}')
        except OSError:
            spec_parts.append(f'{abs_p}:missing')
    tok_name = getattr(tokenizer, 'name_or_path', None) or tokenizer.__class__.__name__
    spec = '|'.join(sorted(spec_parts)) + f'||{tok_name}||text_key={text_key}||maxlen={max_length}'
    return hashlib.sha256(spec.encode()).hexdigest()[:24]

class CompletionDataset(Dataset):

    def __init__(self, hf_args, template_args, tokenizer, prefix_key='prompt', text_key='text', max_length=2048, predict_with_generate=False, insert_space=False):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(self.tokenizer, prefix, text_content, self.max_length, self.predict_with_generate, self.insert_space)
        item_dct = {'input_ids': tokenized_data['input_ids'], 'labels': tokenized_data['labels'], 'attention_mask': tokenized_data['attention_mask']}
        if index != -1:
            item_dct['index'] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, '')
        text_content = self.data[idx].get(self.text_key, '')
        index = self.data[idx]['index']
        item = self._process_sample(pref, text_content, index)
        return item

class PretrainingDataset(Dataset):

    def __init__(self, hf_args, template_args, tokenizer, text_key='text', max_length=2048):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        cache_disabled = os.environ.get('DISABLE_CHUNK_CACHE', '') in ('1', 'true', 'True')
        cache_path = None
        if not cache_disabled:
            try:
                key = _chunk_cache_key(hf_args, tokenizer, text_key, max_length)
                cache_path = _CHUNK_CACHE_DIR / f'{key}.pkl'
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        self.chunks = pickle.load(f)
                    _logger.info(f'[chunk-cache] HIT {cache_path.name} ({len(self.chunks)} chunks)')
                    return
            except Exception as e:
                _logger.warning(f'[chunk-cache] load failed: {e}; rebuilding')
                cache_path = None
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])
        if cache_path is not None:
            try:
                _CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix('.pkl.tmp.' + os.urandom(4).hex())
                with open(tmp, 'wb') as f:
                    pickle.dump(self.chunks, f)
                os.replace(tmp, cache_path)
                _logger.info(f'[chunk-cache] SAVED {cache_path.name} ({len(self.chunks)} chunks)')
            except Exception as e:
                _logger.warning(f'[chunk-cache] save failed: {e}')

    def _chunk_raw_text(self, raw_text):
        raw_text = '\n\n'.join(raw_text)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)['input_ids']
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(self.tokenizer.decode(full_token_sequence[i * self.max_length:(i + 1) * self.max_length]))
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(self.tokenizer, '', self.chunks[idx], self.max_length)
