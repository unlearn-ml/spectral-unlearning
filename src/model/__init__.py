from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM
hf_home = os.getenv('HF_HOME', default=None)
logger = logging.getLogger(__name__)
MODEL_REGISTRY: Dict[str, Any] = {}

def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class

def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop('torch_dtype', None)
    attn_impl = model_args.get('attn_implementation', None)
    if attn_impl == 'flash_attention_2':
        assert torch_dtype in ['float16', 'bfloat16'], ValueError(f"Invalid torch_dtype '{torch_dtype}' for the requested attention implementation: 'flash_attention_2'. Supported types are 'float16' and 'bfloat16'.")
    if torch_dtype == 'float16':
        return torch.float16
    elif torch_dtype == 'bfloat16':
        return torch.bfloat16
    return torch.float32

def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError('Model config not found or model_args absent in configs/model.')
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    compute_dtype = get_dtype(model_args)
    model_handler = model_cfg.get('model_handler', 'AutoModelForCausalLM')
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop('pretrained_model_name_or_path', None)
    attn_impl = model_args.get('attn_implementation', None)
    if attn_impl == 'flash_attention_2':
        param_dtype = compute_dtype
    else:
        param_dtype = torch.float32
    try:
        model = model_cls.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=param_dtype, **model_args, cache_dir=hf_home)
    except Exception as e:
        logger.warning(f'Model {model_path} requested with {model_cfg.model_args}')
        raise ValueError(f'Error {e} while fetching model using {model_handler}.from_pretrained().')
    lora_cfg = model_cfg.get('lora_config', None)
    adapter_path = model_cfg.get('adapter_path', None)
    adapter_cfg_exists = False
    if adapter_path is not None:
        adapter_cfg_exists = os.path.exists(os.path.join(str(adapter_path), 'adapter_config.json'))
    if lora_cfg is not None and (not adapter_cfg_exists):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise ImportError('`peft` is required for lora_config. Install with `pip install peft`.') from e
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        peft_cfg = LoraConfig(r=int(lora_cfg.get('r', 16)), lora_alpha=int(lora_cfg.get('alpha', 32)), target_modules=list(lora_cfg.get('target_modules', [])), lora_dropout=float(lora_cfg.get('dropout', 0.05)), bias='none', task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, peft_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        logger.info(f'LoRA wrap: r={peft_cfg.r} alpha={peft_cfg.lora_alpha} target_modules={peft_cfg.target_modules}')
    if adapter_path is not None:
        if adapter_cfg_exists:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError('`peft` is required for adapter_path. Install with `pip install peft`.') from e
            model = PeftModel.from_pretrained(model, str(adapter_path))
            logger.info(f'Loaded LoRA adapter from {adapter_path} (base kept un-wrapped to avoid stacked adapters)')
        else:
            logger.warning(f'adapter_path={adapter_path} set but no adapter_config.json found — skipping.')
    tokenizer = get_tokenizer(tokenizer_args)
    return (model, tokenizer)

def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({'eos_token': eos_token})
    if is_added:
        logger.info('Add eos token: {}'.format(tokenizer.eos_token))
    else:
        logger.info('Replace eos token: {}'.format(tokenizer.eos_token))
    if num_added_tokens > 0:
        logger.info('New tokens have been added, make sure `resize_vocab` is True.')

def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = f"{'--' * 40}\nError {e} fetching tokenizer using AutoTokenizer.\nTokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\nFull tokenizer config: {tokenizer_cfg}\n{'--' * 40}"
        raise RuntimeError(error_message)
    if tokenizer.eos_token_id is None:
        logger.info('replacing eos_token with <|endoftext|>')
        _add_or_replace_eos_token(tokenizer, eos_token='<|endoftext|>')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info('Setting pad_token as eos token: {}'.format(tokenizer.pad_token))
    return tokenizer
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
