from torch import nn
import torch
from models.modeling_llama import LlamaForCausalLM
from models.Point_BERT import Point_BERT
from transformers import BitsAndBytesConfig
import torch.nn.functional as nnf


class AdaptiveLLM(nn.Module):
    def __init__(self, llm_config, encoder_config):
        super(AdaptiveLLM, self).__init__()
        self._llm_config = llm_config
        self._encoder_config = encoder_config
        self.dtype = torch.float16
        
        # For debug in 3090
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype
        )
        self.llm = LlamaForCausalLM.from_pretrained('ckpts/Llama-2-7b-hf', 
                                                    config=self._llm_config, 
                                                    torch_dtype=self.dtype, 
                                                    low_cpu_mem_usage=True,
                                                    quantization_config=bnb_config)
            
        self.encoder = Point_BERT(self._encoder_config)
        
        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(self._encoder_config.transformer_config.trans_dim, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
        )
        
    def _loss_caption(self, logits, target, mask):
        loss_per_word = nnf.cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            target,
            reduction='none',
        )
        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)

        return final_loss
        
    def forward(self, data_dict):
        with torch.no_grad():
            points = data_dict['points'].cuda()
            num_group = data_dict['num_groups'][0].item()
            group_size = data_dict['group_size'][0].item()
            cls_tokens, encoder_tokens = self.encoder(points, num_group=num_group, group_size=group_size, forward_encoder=True)
            # Concat cls_tokens and encoder_tokens
            cls_tokens = cls_tokens.unsqueeze(1)
            vision_embed = torch.cat((cls_tokens, encoder_tokens), dim=1)
        
        vision_embed.requires_grad_(True)
        vision_embed = self.encoder_to_llm_projection(vision_embed)
        
        # Debug
        embedding_layer = self.llm.get_input_embeddings()
        
        input_mask = data_dict['attention_mask'].cuda()
        input_ids = data_dict['input_ids'].cuda()
        vision_mask = torch.ones_like(vision_embed[..., 0])
        # ---- batch x (ntoken + nword) x n_embd
        # Debug
        inputs_embeds = torch.cat((vision_embed, embedding_layer(input_ids)), dim=1)
        attention_mask = torch.cat((vision_mask, input_mask), dim=1)
        
        # Calculate llm loss
        outputs = self.llm(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask.to(self.dtype),
            output_attentions=False,
        )
        
        gradient_mask = data_dict['gradient_mask'].cuda()
        loss = self._loss_caption(
            logits = outputs.logits[:, vision_embed.shape[1] - 1: -1],
            target = input_ids,
            mask = gradient_mask.to(self.dtype),
        )
        
        return loss