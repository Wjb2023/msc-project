import math
import copy
import json
import csv
import ast
from math import sqrt
from random import choice, randint
from pathlib import Path
from shutil import rmtree
from functools import wraps, partial
from typing_extensions import Annotated
import pandas as pd

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from lion_pytorch import Lion
from accelerate import Accelerator, DistributedType

import torchaudio
from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking, MelSpectrogram, AmplitudeToDB

import torch.distributed as dist

from x_clip.tokenizer import tokenizer

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from beartype.typing import Union, List, Optional, Tuple, Callable

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    wavs = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    raw_texts = List[str],
    texts = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.long and t.ndim == 2]
    ],
    audio_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 1]
    ],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 1]
    ],
)


#functions

def exists(val):
    return val is not None

def first(it):
    return it[0]

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def default(val, d):
    return val if exists(val) else d

def default2(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def interspersed_indices(layers, total_layers):
    assert total_layers >= layers
    step = total_layers / layers
    return (torch.arange(0, layers) * step).floor().long()

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

# optimizer functions

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

#decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

#tensor functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def comcosida_sinusoid(t):
    return 0.5 * (1 + torch.sin(-torch.pi / 2 * t))

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_same_dim(t):
    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = all_gather_same_dim(size)
        sizes = torch.stack(sizes)

    if torch.unique(sizes).numel() == 1:
        gathered_tensors = all_gather_same_dim(t)
        return torch.cat(gathered_tensors, dim = dim), sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        ctx.batch_sizes = batch_sizes.tolist()
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        if ctx.all_reduce_grads:
            dist.all_reduce(grads)

        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None, None
    
class AllGather(nn.Module):
    def __init__(
        self,
        dim,
        *,
        all_reduce_grads = False
    ):
        super().__init__()
        self.dim = dim
        self.all_reduce_grads = all_reduce_grads
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def forward(
        self,
        x,
        sizes = None
    ):
        if not self.is_distributed:
            return x, None

        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)
    
#biasless layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None,
        return_all_layers = False
    ):
        layers = []

        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x
            layers.append(x)

        if not return_all_layers:
            return x

        return x, torch.stack(layers[:-1])
    
class AudioSpectrogramTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        accept_spec = False,
        accept_spec_time_first = True,
        spec_n_fft = 400,
        n_mels = 64,
        spec_power = 2,
        spec_win_length = 400,
        spec_hop_length = 100,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        patch_dropout_prob = 0.25
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        # MelSpectrogram and Log Scaling
        self.mel_spec = MelSpectrogram(
            n_fft = spec_n_fft,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode,
            n_mels = n_mels
        )
        self.amplitude_to_db = AmplitudeToDB(stype='power')

        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout
        )

        self.norm = LayerNorm(dim)

        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias

        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        x,
        force_no_patch_dropout = False,
        return_all_layers = False
    ):
        batch, device = x.shape[0], x.device
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b t f -> b f t')

        if not self.accept_spec:
            x = self.mel_spec(x)

        if self.training:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): 
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # to patches

        x = self.to_patch_tokens(x)

        # get number of patches along height and width

        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions

        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding

        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout

        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias

        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # attention, what else

        x, all_layers = self.transformer(x, rel_pos_bias = rel_pos_bias, return_all_layers = True)

        # final global average and norm (most recent papers show this is superior to CLS token)

        x = reduce(x, 'b n d -> b d', 'mean')

        out = self.norm(x)

        if not return_all_layers:
            return out

        return out, all_layers
    
class TextTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.depth = depth
        self.max_seq_len = max_seq_len

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.pad_id = pad_id
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None,
        return_all_layers = False
    ):
        assert exists(x) ^ exists(raw_texts)

        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts, truncate_text=True).to(self.device)
        
        # Ensure the input tensor is of type LongTensor
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a tensor")

        x = x.long()

        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device

        # token embedding + positional embedding

        x = self.token_emb(x)

        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        x = x + self.pos_emb(torch.arange(n, device = device))

        # cls tokens, as in bert

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # account for attending to cls token with self attention mask

        mask = F.pad(mask, (1, 0), value = True)

        # attention

        x, all_layers = self.transformer(x, mask = mask, return_all_layers = True)

        # unpack the cls tokens

        cls_tokens, _ = unpack(x, ps, 'b * d')

        out = self.norm(cls_tokens)

        if not return_all_layers:
            return out

        return out, all_layers
    
class SigmoidContrastiveLearning(nn.Module):

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

        self.all_gather = AllGather(dim = 1, all_reduce_grads = True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        device = self.device

        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        text_latents, rank_sizes = self.all_gather(text_latents)

        n = text_latents.shape[1]

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp() + self.bias

        labels = torch.eye(n, device = device)

        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim = 0)
            labels = labels_by_ranks[dist.get_rank()]

        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        return -F.logsigmoid(labels * sims).sum() / n
    
class SoftmaxContrastiveLearning(nn.Module):
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        comcosida = False,
        init_temp = 10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning
        self.comcosida = comcosida
        if self.comcosida:
            print('Using CoMCoSIDA')

        self.all_gather = AllGather(dim = 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents, audio_embeds, text_embeds):
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        batch = audio_latents.shape[1]

        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp()

        cosine_sims_exp = sims.exp()

        numerator = matrix_diag(cosine_sims_exp)

        if self.comcosida:
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            audio_sims = einsum('i d, j d -> i j', audio_embeds, audio_embeds)
            audio_sims = comcosida_sinusoid(audio_sims)
            audio_sims = audio_sims.masked_fill(eye, 1.)
            cosine_sims_exp = cosine_sims_exp * audio_sims

            text_sims = einsum('i d, j d -> i j', text_embeds, text_embeds)
            text_sims = comcosida_sinusoid(text_sims)
            text_sims = text_sims.masked_fill(eye, 1.)
            cosine_sims_exp = cosine_sims_exp * audio_sims

        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()
    
class MultiLayerContrastiveLoss(nn.Module):
    def __init__(
        self,
        *,
        audio_dim,
        text_dim,
        dim_latent,
        layers,
        decoupled_contrastive_learning = False,
        comcosida = False,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        self.layers = layers

        self.audio_norm = LayerNorm(audio_dim, scale = False)
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        self.text_norm = LayerNorm(text_dim, scale = False)
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning, comcosida=comcosida)
        self.contrast = klass(layers = layers)

    def forward(self, *, audio_layers, text_layers, audio_pre_embeds, text_pre_embeds):
        device, batch = audio_layers.device, audio_layers.shape[1]

        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        audio_latents = l2norm(audio_latents)

        text_cls_tokens = text_layers[:, :, 0]
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        text_latents = l2norm(text_latents)

        return self.contrast(audio_latents, text_latents, audio_pre_embeds, text_pre_embeds)
    
class MuLaN(nn.Module):
    @beartype
    def __init__(
        self,
        audio_transformer: AudioSpectrogramTransformer,
        text_transformer: TextTransformer,
        dim_latent = 256,                     
        decoupled_contrastive_learning = False, 
        comcosida = False,
        hierarchical_contrastive_loss = False,
        hierarchical_contrastive_loss_layers = None,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        self.dim_latent = dim_latent

        self.audio = audio_transformer
        self.text = text_transformer


        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning, comcosida=comcosida)
        self.contrast = klass()

        self.multi_layer_contrastive_learning = None

        if hierarchical_contrastive_loss:
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1)
            assert num_layers > 0

            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim = self.audio.dim,
                text_dim = self.text.dim,
                dim_latent = dim_latent,
                layers = num_layers,
                decoupled_contrastive_learning = decoupled_contrastive_learning,
                comcosida=comcosida,
                sigmoid_contrastive_loss = sigmoid_contrastive_loss
            )

    def get_audio_latents(
        self,
        wavs,
        return_all_layers = False
    ):
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers = True)
        audio_latents = self.audio_to_latents(audio_embeds)
        out = l2norm(audio_latents)

        if not return_all_layers:
            return out

        return out, audio_layers

    @beartype
    def get_text_latents(
        self,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_all_layers = False
    ):
        text_embeds, text_layers = self.text(texts, raw_texts = raw_texts, return_all_layers = True)
        text_latents = self.text_to_latents(text_embeds)
        out = l2norm(text_latents)

        if not return_all_layers:
            return out

        return out, text_layers

    @beartype
    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        audio_embeds = None,
        text_embeds = None,
        return_latents = False,
        return_similarities = False,
        return_pairwise_similarities = False
    ):
        batch, device = wavs.shape[0], wavs.device

        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers = True)
        text_latents, text_layers = self.get_text_latents(texts, raw_texts = raw_texts, return_all_layers = True)

        if return_latents:
            return audio_latents, text_latents

        if return_similarities:
            return einsum('i d, i d -> i', audio_latents, text_latents)

        if return_pairwise_similarities:
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim

        cl_loss = self.contrast(audio_latents, text_latents, audio_embeds, text_embeds)

        if not exists(self.multi_layer_contrastive_learning):
            return cl_loss

        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]

        # whether to do cl loss across all layers, from ViCHA paper

        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers = audio_layers,
            text_layers = text_layers,
            audio_pre_embeds = audio_embeds,
            text_pre_embeds = text_embeds
        )

        return cl_loss + hierarchical_cl_loss
    
@beartype
class MuLaNTrainer(nn.Module):
    def __init__(
        self,
        mulan: MuLaN,
        dataset: Dataset,
        *,
        num_train_epochs = 1,
        batch_size,
        data_max_length = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        betas = (0.9, 0.99),
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 45,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        use_lion = False,
        force_clear_prev_results = None,  
        patience = 20
    ):
        super().__init__()
        assert batch_size > 1, 'batch size must be greater than 1 for contrastive learning (but ideally as large as possible)'

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.mulan = mulan

        self.register_buffer('steps', torch.Tensor([0]))

        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.steps_per_epoch = len(dataset) // batch_size

        if num_train_epochs is not None:
            self.num_train_steps = self.steps_per_epoch * num_train_epochs
        else:
            self.num_train_steps = None  

        self.save_every = self.steps_per_epoch

        # optimizers

        optim_klass = Lion if use_lion else Adam
        self.optim = optim_klass(mulan.parameters(), lr = lr, betas = betas)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        self.data_max_length = data_max_length

        # create dataset

        self.ds = dataset
        self.ds_fields = None

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, pad_to_longest = False, drop_last = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, pad_to_longest = False, drop_last = True)

        # prepare with accelerator

        (
            self.mulan,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.mulan,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        hps = dict(
            num_train_steps = self.num_train_steps,
            data_max_length = data_max_length,
            learning_rate = lr
        )

        self.accelerator.init_trackers("mulan", config = hps)

        # results folder

        self.results_folder = Path(results_folder)

        if force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        self.log_file = self.results_folder / 'training_log.csv'
        if not self.log_file.exists():
            with open(self.log_file, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

        # to device

        self.mulan.to(self.device)

        # early stopping variables
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def save(self, path):
        # Save model, optimizer states, and training progress
        pkg = dict(
            model=self.accelerator.get_state_dict(self.mulan),
            optim=self.optim.state_dict(),
            steps=self.steps.item(),
            best_val_loss=self.best_val_loss,
            epochs_no_improve=self.epochs_no_improve
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location='cpu')

        mulan = self.accelerator.unwrap_model(self.mulan)
        mulan.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

        # Restore training progress
        self.steps = torch.tensor(pkg['steps'], device=self.device)
        self.best_val_loss = pkg.get('best_val_loss', float('inf'))
        self.epochs_no_improve = pkg.get('epochs_no_improve', 0)

    def _find_latest_checkpoint(self):
        checkpoints = list(self.results_folder.glob('mulan.epoch_*.pt'))
        if not checkpoints:
            return None

        # Sort checkpoints by epoch number
        def extract_epoch_number(path):
            filename = path.stem
            epoch_str = filename.split('epoch_')[-1]
            return int(epoch_str)

        checkpoints = sorted(checkpoints, key=extract_epoch_number)
        return str(checkpoints[-1])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            #self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            #print(self.ds_fields)
            self.ds_fields = ('wavs', 'raw_texts', 'audio_embeds', 'text_embeds')
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        data_kwargs =  dict(zip(self.ds_fields, data))

        wavs = data_kwargs['wavs']
        data_kwargs.update(wavs = wavs[..., :self.data_max_length])

        return data_kwargs

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.mulan.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            loss = self.mulan(**data_kwargs)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.mulan.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step = steps)

        self.steps += 1
        return logs
    
    def validate(self):
        self.mulan.eval()
        val_loss = 0
        count = 0

        with torch.no_grad():
            for data in self.valid_dl:
                data_kwargs = self.data_tuple_to_kwargs(data)
                loss = self.mulan(**data_kwargs)
                val_loss += loss.item()
                count += 1

        val_loss /= count
        self.print(f'Validation Loss: {val_loss}')
        return val_loss

    def train(self, log_fn: Callable = noop, resume_path: Optional[str] = None):
        # If a resume path is provided, load the checkpoint
        if resume_path:
            self.load(resume_path)
            self.print(f'Resumed training from checkpoint: {resume_path}')
        else:
            # Automatically look for the latest checkpoint in the results folder
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                self.load(latest_checkpoint)
                self.print(f'Resumed training from the latest checkpoint: {latest_checkpoint}')

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

            # Check if we have completed an epoch
            if int(self.steps.item()) % self.steps_per_epoch == 0:
                # Run validation at the end of each epoch
                val_loss = self.validate()

                # Save model and log results
                epoch_number = int(self.steps.item() // self.steps_per_epoch)
                if self.is_main:
                    model_path = str(self.results_folder / f'mulan.epoch_{epoch_number}.pt')
                    self.save(model_path)

                    # Log to CSV
                    with open(self.log_file, mode='a') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch_number, logs['loss'], val_loss])

                    self.print(f'Epoch {epoch_number}: Model saved and logs updated.')

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1

                if self.epochs_no_improve >= self.patience:
                    self.print('Early stopping triggered!')
                    break

        self.print('training complete')

class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, text_dir, csv_path, transform=None, segment_duration=10):
        """
        Args:
            audio_dir (str or Path): Path to the directory containing .wav files.
            text_dir (str or Path): Path to the directory containing text JSON files.
            csv_path (str or Path): Path to the CSV file mapping numeric IDs to alphanumeric IDs.
            transform (callable, optional): Optional transform to be applied on the audio data.
        """
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.transform = transform
        self.segment_duration = segment_duration

        # Load the CSV file containing the ID mappings
        self.id_mapping = pd.read_csv(csv_path)

        # Filter out rows where either the .wav file or the JSON file does not exist
        self.id_mapping = self.id_mapping[
            self.id_mapping.apply(lambda row: (self.audio_dir / f"{row['track_7digitalid']}.clip.wav").exists() and 
                                               (self.text_dir / f"{row['track_id']}_comments.json").exists(), axis=1)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.id_mapping)

    def __getitem__(self, idx):
        # Get the corresponding row from the CSV
        row = self.id_mapping.iloc[idx]
        numeric_id = row['track_7digitalid']
        alphanumeric_id = row['track_id']
        audio_embedding = row['audio_embedding']
        text_embedding = row['text_embedding']

        audio_embedding = torch.tensor(ast.literal_eval(audio_embedding))
        text_embedding = torch.tensor(ast.literal_eval(text_embedding))

        # Load the .wav file
        audio_path = self.audio_dir / f"{numeric_id}.clip.wav"
        waveform, sample_rate = torchaudio.load(audio_path)

        segment_length = self.segment_duration * sample_rate
        if waveform.shape[-1] > segment_length:
            max_start = waveform.shape[-1] - segment_length
            start = randint(0, max_start)
            end = start + segment_length
            waveform = waveform[..., start:end]
        
        if waveform.ndim == 2:
            waveform = torch.mean(waveform, dim=0)

        text_path = self.text_dir / f"{alphanumeric_id}_comments.json"
        with open(text_path, 'r') as f:
            comments = json.load(f)

        assert comments, f"No comments found for {alphanumeric_id}"

        # Randomly sample one text label from the comments list
        text_label = choice(comments)

        return waveform, text_label, audio_embedding, text_embedding
    
def check_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available. Using CPU for training.")
    
def main(audio_dir, text_dir, csv_path):
    print('Starting training')
    check_available_gpus()

    audio_transformer = AudioSpectrogramTransformer(dim=256, depth=12, heads=10, attn_dropout=0.5, ff_dropout=0.5) 
    text_transformer = TextTransformer(dim=256, depth=12, heads=10, attn_dropout=0.5, ff_dropout=0.5)
    mulan_model = MuLaN(audio_transformer, text_transformer, hierarchical_contrastive_loss=True, comcosida=True)

    dataset = AudioTextDataset(audio_dir, text_dir, csv_path, segment_duration=5)

    # Instantiate the trainer
    trainer = MuLaNTrainer(
        mulan=mulan_model,
        dataset=dataset,
        batch_size=16,  
        num_train_epochs=500,  
        lr=1e-4,  
        grad_accum_every=1,
        patience=500
    )

    # Start training
    print('Model ready. Starting now...')
    trainer.train()
    

audio_dir = "../Data/wavs"
text_dir = "../Data/comments"
csv_path = "./track_ids.csv"

if __name__ == '__main__':
    main(audio_dir, text_dir, csv_path)