import logging
import torch
import torch.nn as nn
import featurization as ft
from packaging import version
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.modeling_bert import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertPooler
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import logger
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union

class my_BertEmbeddings(BertEmbeddings):
        """Construct the embeddings from word, position and token_type embeddings."""

        def __init__(self, config):
            super().__init__(config)
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.vgene_embeddings = nn.Embedding(config.vgene_vocab_size, config.hidden_size) # add vgene_embedding
            
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            if version.parse(torch.__version__) > version.parse("1.6.0"):
                self.register_buffer(
                    "token_type_ids",
                    torch.zeros(self.position_ids.size(), dtype=torch.long),
                    persistent=False,
                )

        def forward(
            self, input_ids=None, vgene_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

            # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
            # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
            # issue #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            vgene_embeddings = self.vgene_embeddings(vgene_ids) # add vgene_embedding
            embeddings = inputs_embeds + token_type_embeddings + vgene_embeddings # add vgene_embedding
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

class my_BertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = my_BertEmbeddings(config=config) # add vgene_embedding
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        vgene_ids=None, # add vgene_embedding
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            vgene_ids = vgene_ids, # add vgene_embedding
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class my_BertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = my_BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        vgene_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            vgene_ids=vgene_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class my_BertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = my_BertModel(config)
        classifier_dropout = (0.2)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier_0 = nn.Linear(config.hidden_size, 128)
        self.classifier = nn.Linear(128, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        vgene_ids=None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            vgene_ids=vgene_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_0(pooled_output)
        logits = self.classifier(logits)
        # logits = self.dropout(self.classifier_0(pooled_output))
        # logits = self.classifier(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def get_bert_model(**kwargs) -> nn.Module:
    """
    Load the bert model
    """
    logging.info("Loading vanilla BERT model") 
    config = BertConfig(
        **kwargs,
        vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL), # 20+5
        pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD], # PAD
    )
    model = my_BertForMaskedLM(config)
    return model