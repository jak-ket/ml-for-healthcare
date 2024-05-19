# Inspiration:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
# https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_10_3_transformer_timeseries.ipynb


import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch import optim, Tensor
from typing import Optional
import torch.nn.functional as F


def get_fastpath_enabled() -> bool:
    _is_fastpath_enabled: bool = True
    """Returns whether fast path for TransformerEncoder and MultiHeadAttention
    is enabled, or ``True`` if jit is scripting.

    ..note:
        The fastpath might not be run even if ``get_fastpath_enabled`` returns
        ``True`` unless all conditions on inputs are met.
    """
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    return True



class TransformerEncoderLayerCustom(nn.TransformerEncoderLayer):
  def __init__(self, *args, **kwargs):
    super(TransformerEncoderLayerCustom, self).__init__(*args, **kwargs)

  def _sa_block(self, x: Tensor,attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
      """
      self attention block for TransformerEncoderLayer
      """
      x, attn_output_weights = self.self_attn(x, x, x,
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask,
                          need_weights=True,
                          average_attn_weights=True,
                          is_causal=is_causal
                          )
      return self.dropout1(x), attn_output_weights


  def forward(
              self,
              src: Tensor,
              src_mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None,
              is_causal: bool = False) -> Tensor:
          """
          TransformerEncoderLayer
          """
          src_key_padding_mask = F._canonical_mask(
              mask=src_key_padding_mask,
              mask_name="src_key_padding_mask",
              other_type=F._none_or_dtype(src_mask),
              other_name="src_mask",
              target_type=src.dtype
          )

          src_mask = F._canonical_mask(
              mask=src_mask,
              mask_name="src_mask",
              other_type=None,
              other_name="",
              target_type=src.dtype,
              check_other=False,
          )

          is_fastpath_enabled = get_fastpath_enabled()

          
          why_not_sparsity_fast_path = ''
          if not is_fastpath_enabled:
              why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
          elif not src.dim() == 3:
              why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
          elif self.training:
              why_not_sparsity_fast_path = "training is enabled"
          elif not self.self_attn.batch_first:
              why_not_sparsity_fast_path = "self_attn.batch_first was not True"
          elif self.self_attn.in_proj_bias is None:
              why_not_sparsity_fast_path = "self_attn was passed bias=False"
          elif not self.self_attn._qkv_same_embed_dim:
              why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
          elif not self.activation_relu_or_gelu:
              why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
          elif not (self.norm1.eps == self.norm2.eps):
              why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
          elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
              why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
          elif self.self_attn.num_heads % 2 == 1:
              why_not_sparsity_fast_path = "num_head is odd"
          elif torch.is_autocast_enabled():
              why_not_sparsity_fast_path = "autocast is enabled"
          if not why_not_sparsity_fast_path:
              tensor_args = (
                  src,
                  self.self_attn.in_proj_weight,
                  self.self_attn.in_proj_bias,
                  self.self_attn.out_proj.weight,
                  self.self_attn.out_proj.bias,
                  self.norm1.weight,
                  self.norm1.bias,
                  self.norm2.weight,
                  self.norm2.bias,
                  self.linear1.weight,
                  self.linear1.bias,
                  self.linear2.weight,
                  self.linear2.bias,
              )

              
              
              _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
              if torch.overrides.has_torch_function(tensor_args):
                  why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
              elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                  why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                                f"{_supported_device_type}")
              elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                  why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                                "input/output projection weights or biases requires_grad")

              if not why_not_sparsity_fast_path:
                  merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                  return torch._transformer_encoder_layer_fwd(
                      src,
                      self.self_attn.embed_dim,
                      self.self_attn.num_heads,
                      self.self_attn.in_proj_weight,
                      self.self_attn.in_proj_bias,
                      self.self_attn.out_proj.weight,
                      self.self_attn.out_proj.bias,
                      self.activation_relu_or_gelu == 2,
                      self.norm_first,
                      self.norm1.eps,
                      self.norm1.weight,
                      self.norm1.bias,
                      self.norm2.weight,
                      self.norm2.bias,
                      self.linear1.weight,
                      self.linear1.bias,
                      self.linear2.weight,
                      self.linear2.bias,
                      merged_mask,
                      mask_type,
                  )

          x = src
          if self.norm_first:
              y, attn_output_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
              x = x + y
              x = x + self._ff_block(self.norm2(x))
          else:
              
              y, attn_output_weights = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
              x = self.norm1(x + y)
              x = self.norm2(x + self._ff_block(x))

          return x, attn_output_weights


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            
            return src_size[0]
        else:
            
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        
        
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


class TransformerEncoderCustom(nn.TransformerEncoder):
  
  def __init__(self, *args, **kwargs):
    super(TransformerEncoderCustom, self).__init__(*args, **kwargs)

  def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        attn_weights = []
        for mod in self.layers:
            output, weights = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
            attn_weights.append(weights)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesClassifierTransformer(nn.Module):
    def __init__(self, seq_len, d_model, num_layers, num_heads, dim_feedforward, dropout, activation):
        super(TimeSeriesClassifierTransformer, self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        self.embed = nn.Linear(1, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder_layer = TransformerEncoderLayerCustom(
            d_model=d_model, 
            nhead=num_heads, 
            activation=activation,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )

        self.transformer_encoder = TransformerEncoderCustom(
          self.encoder_layer, 
          num_layers
        )

        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
                

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.embed(x)
        x = self.pos_encoder(x)
        x, attn_output_weights = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        logits = self.linear(x)
        probas = self.sigmoid(logits)
        return probas, attn_output_weights


def train_transformer(
  model, epochs, batch_size, train_loader, lr, weight_decay, 
  val_loader, val_score=None
):
    criterion = torch.nn.BCELoss() 
    optimizer = optim.Adam(
      model.parameters(), 
      lr=lr,
      weight_decay=weight_decay
    )

    
    best_score = - np.inf   
    best_weights = None

    for epoch in range(epochs):
        model.train()
        loss_total = 0
        for nr_batch, (X_batch, y_batch) in enumerate(train_loader):

            if X_batch.shape[0] != batch_size:
                continue
            
            optimizer.zero_grad()
            
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        if val_loader != None:
          
          model.eval()
          
          y_pred = []
          y_true = []
          for nr_batch, (X_batch, y_batch) in enumerate(train_loader):
            y_pred_batch, _ = model(X_batch)
            y_pred.append(y_pred_batch.squeeze().cpu().detach().numpy().round())
            y_true.append(y_batch.cpu().detach().numpy())

          score = val_score(list(itertools.chain(*y_pred)), list(itertools.chain(*y_true))) 

          score = float(score)
          if score > best_score:
              best_score = score
              best_weights = copy.deepcopy(model.state_dict())
        else:
          score = "not evaluated"
    
        print(f"epoch: {epoch}, train loss: {loss_total / nr_batch}, validation score: {score}")

    
    if val_loader != None:
      model.load_state_dict(best_weights)
