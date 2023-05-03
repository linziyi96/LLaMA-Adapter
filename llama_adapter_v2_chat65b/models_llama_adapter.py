import functools
from pathlib import Path
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import fairscale.nn.model_parallel.initialize as fs_init

from llama import ModelArgs, Tokenizer, LLaMA, Transformer


def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]), force=True)
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))
            
            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0: # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                        'attention_norm.weight',
                        'ffn_norm.weight',
                        ]
                column_parallel_names = [
                        'attention.wq.weight',
                        'attention.wk.weight',
                        'attention.wv.weight',
                        'feed_forward.w1.weight',
                        'feed_forward.w3.weight',
                        ]
                row_parallel_names = [
                        'attention.wo.weight',
                        'feed_forward.w2.weight',
                        ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else: # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params


def Llama_adapter(args, model_name, adapter_len=0, adapter_layer=0, add_bias=False, add_scale=False, train_norm=False, **kwargs):
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(args.llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(
        # caching configuration
        max_seq_len=args.max_seq_len if hasattr(args, 'max_seq_len') else 2048,

        adapter_len=adapter_len,
        adapter_layer=adapter_layer,
        add_bias=add_bias,
        add_scale=add_scale,
        train_norm=train_norm,

        # other args
        **params
    )

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_adapter = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    missing_keys, unexpected_keys = model_llama_adapter.load_state_dict(checkpoint, strict=False)

    for i in range(model_args.n_layers):
        if i < model_args.n_layers - adapter_layer:
            del model_llama_adapter.layers[i].attention.gate

    for name, param in model_llama_adapter.named_parameters():
        requires_grad = \
            name.endswith('.gate') or \
            name == 'adapter_query' or \
            (train_norm and '_norm.' in name) or \
            name.endswith('_bias') or \
            name.endswith('_scale')
            
        if requires_grad:
            param.data = param.data.float()
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model_llama_adapter


'''
for saving and loading peft-ed checkpoints. marks the parallel type of each param tensor.
key: the param name suffix
value: the dimension along which the full tensor is splitted. -1 for fully replicated tensors.
'''
param_suffix_and_parallel_type = [
        ('.gate', -1),
        ('adapter_query', -1),
        ('.attention_norm.weight', -1),
        ('.ffn_norm.weight', -1),
        ('.attention.wq.weight', 0),
        ('.attention.wk.weight', 0),
        ('.attention.wv.weight', 0),
        ('.attention.wo.weight', 1),
        ('.attention.wq_scale', -1),
        ('.attention.wk_scale', -1),
        ('.attention.wo_scale', -1),
        ('.attention.wo_scale', 0),
        ('.attention.wq_bias', 0),
        ('.attention.wk_bias', 0),
        ('.attention.wv_bias', 0),
        ('.attention.wo_bias', -1),
        ('.feed_forward.w1.weight', 0),
        ('.feed_forward.w2.weight', 1),
        ('.feed_forward.w3.weight', 0),
        ('.feed_forward.w1_scale', -1),
        ('.feed_forward.w2_scale', 0),
        ('.feed_forward.w3_scale', -1),
        ('.feed_forward.w1_bias', 0),
        ('.feed_forward.w2_bias', -1),
        ('.feed_forward.w3_bias', 0),
        ]
def _get_gather_dim(name):
    global param_suffix_and_parallel_type
    for k, v in param_suffix_and_parallel_type:
        if name.endswith(k):
            return v
    assert False, 'the gathering method of tensor "%s" is unknown.' % name

'''
this function creates the state dict of the PEFT-ed model. Specifically:
1. exclude the params that does not require grad;
2. gather model parallel weights to a single state dict.
'''
def create_gathered_state_dict(model):
    print('creating gathered state dict...')
    mp_size = fs_init.get_model_parallel_world_size()
    gathered_state_dict = OrderedDict()
    n_params = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        gather_dim = _get_gather_dim(n)
        if gather_dim >= 0:
            gathered_shape = [x for i, x in enumerate(p.size()) if i != gather_dim else x * mp_size]
            gathered_tensor = torch.empty(gathered_shape, device=p.device, dtype=p.dtype)
            dist.all_gather_into_tensor(gathered_tensor, p, group=fs_init.get_model_parallel_group())
            gathered_state_dict[n] = gathered_tensor.cpu()
        else:
            gathered_state_dict[n] = p.data.cpu()
        n_params += gathered_state_dict[n].numel()
    print('# params in the gathered state dict:', n_params, '(%.2f M)' % (n_params / 1000000))
    return gathered_state_dict


'''
this function distributes the state dict of the PEFT-ed model.
the saved checkpoint is expected to be gathered using the create_gathered_state_dict function above.
should only pass the state dict if rank=0 (pass state_dict=None if rank!=0).
'''
def distribute_peft_state_dict(state_dict):
    mp_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    if dist.get_rank() == 0:
        state_dict_meta = [(k, v.shape, v.dtype) for k, v in state_dict.items()]
        dist.broadcast_object_list([state_dict_meta], src=0)
    else:
        assert state_dict is None
        recv_list = [None]
        dist.broadcast_object_list(recv_list, src=0)
        state_dict_meta = recv_list

    out_state_dict = {}
    for k, shape, dtype in state_dict_meta:
        if dist.get_rank() == 0:
            comm_tensor = state_dict[k].cuda()
        else:
            comm_tensor = torch.empty(shape, dtype=dtype, device='cuda')
        dist.broacast(comm_tensor, src=0)
        
        gather_dim = _get_gather_dim(k)
        slice_list = []
        for dim, dim_size in enumerate(shape):
            if dim != gather_dim:
                slice_list.append(slice(None))
            else:
                assert dim_size % mp_size == 0
                shard_size = dim_size // mp_size
                slice_list.append(slice(shard_size * mp_rank, shard_size * (mp_rank + 1)))
        out_state_dict[k] = comm_tensor[slice_list].cpu()
    return out_state_dict


'''
a small helper function that only load model in rank 0.
reduces io and memory usage.
'''
def load_and_distribute_peft_state_dict(path):
    if dist.get_rank() == 0:
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['model']
    else:
        state_dict = None

    return distribute_peft_state_dict(state_dict)


'''
load state dict for peft model.
for peft model the saved state dict does not contain frozen weights.
gives warning for params:
1. do not require grad but exist in the state dict.
2. require grad but do not exist in the state dict.
'''
def peft_load_state_dict(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    missing_keys = set(missing_keys)
    for n, p in model.named_parameters():
        if p.requires_grad and n in missing_keys:
            print('[WARNING] Found parameters that require grad but not loaded from the state dict:', n)
        elif not p.requires_grad and n not in missing_keys:
            print('[WARNING] Found parameters that do not require grad but overrided by the state dict:', n)
    for n in unexpected_keys:
        print('[WARNING] Found unexpected parameters in the state dict:', n)


# set recommended archs
Llama7B_bias_scale_norm_tuning = functools.partial(Llama_adapter, model_name='7B', add_bias=True, add_scale=True, train_norm=True)
Llama13B_bias_scale_norm_tuning = functools.partial(Llama_adapter, model_name='13B', add_bias=True, add_scale=True, train_norm=True)
Llama30B_bias_scale_norm_tuning = functools.partial(Llama_adapter, model_name='30B', add_bias=True, add_scale=True, train_norm=True)
Llama65B_bias_scale_norm_tuning = functools.partial(Llama_adapter, model_name='65B', add_bias=True, add_scale=True, train_norm=True)
