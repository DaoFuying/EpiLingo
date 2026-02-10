# dna_lm_models.py
import torch
import torch.nn as nn

from transformers import (
    GPT2Config,
    GPT2Model,
    LlamaConfig,
    LlamaModel,
    T5Config,
    T5EncoderModel,
)


class DNAGPTForSequenceClassification(nn.Module):
    """
    用 GPT2-style decoder-only Transformer 作为 backbone，
    再加一个简单分类头。
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 2,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_position_embeddings,
            n_ctx=max_position_embeddings,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # GPT2 使用 causal mask，因此 attention_mask 只用于 padding
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, L, D)

        # 使用 masked mean pooling（和 Mamba 保持一致）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = (hidden_states * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1e-6)
        else:
            hidden_states = hidden_states.mean(dim=1)

        x = self.dropout(hidden_states)
        logits = self.classifier(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss

        return outputs


class DNALLAMAForSequenceClassification(nn.Module):
    """
    用 LLaMA-style encoder 作为 backbone。
    注意：这里是随机初始化的小 LLaMA 模型，用于“框架对比”，不是加载真实大 LLaMA 权重。
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 2,
        d_model: int = 256,
        n_layer: int = 8,
        n_head: int = 8,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            intermediate_size=intermediate_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-5,
        )
        self.model = LlamaModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, L, D)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = (hidden_states * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1e-6)
        else:
            hidden_states = hidden_states.mean(dim=1)

        x = self.dropout(hidden_states)
        logits = self.classifier(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss

        return outputs


class DNAT5ForSequenceClassification(nn.Module):
    """
    仅使用 T5 encoder 作为 backbone（encoder-only），
    再加分类头。
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 2,
        d_model: int = 256,
        d_ff: int = 1024,
        n_layer: int = 6,
        n_head: int = 8,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=n_layer,
            num_heads=n_head,
            dropout_rate=dropout,
            feed_forward_proj="relu",
        )
        self.encoder = T5EncoderModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, L, D)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = (hidden_states * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1e-6)
        else:
            hidden_states = hidden_states.mean(dim=1)

        x = self.dropout(hidden_states)
        logits = self.classifier(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss

        return outputs
