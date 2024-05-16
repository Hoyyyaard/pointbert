from torch import nn
import torch
from models.modeling_llama import LlamaForCausalLM
from models.Point_BERT import Point_BERT
from transformers import BitsAndBytesConfig
import torch.nn.functional as nnf
from transformers import AutoTokenizer
from tools.generation_utils import generation


class AdaptiveLLM(nn.Module):
    def __init__(self, llm_config, encoder_config):
        super(AdaptiveLLM, self).__init__()
        self._llm_config = llm_config
        self._encoder_config = encoder_config
        self.dtype = torch.float16
        
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf')
        
        # For debug in 3090
        if encoder_config.quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,                     
                bnb_8bit_use_double_quant=True,        
                bnb_8bit_quant_type="nf8",            
                bnb_8bit_compute_dtype=self.dtype      
            )
            self.llm = LlamaForCausalLM.from_pretrained('ckpts/Llama-2-7b-hf', 
                                                        config=self._llm_config, 
                                                        torch_dtype=self.dtype, 
                                                        low_cpu_mem_usage=True,
                                                        quantization_config=bnb_config)
        else:
            self.llm = LlamaForCausalLM.from_pretrained('ckpts/Llama-2-7b-hf', 
                                            config=self._llm_config, 
                                            torch_dtype=self.dtype, 
                                            low_cpu_mem_usage=True)
            
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
        
    def forward(self, data_dict, eval=False):
        with torch.no_grad():
                    
            points = data_dict['points']
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
        
        if not eval:
            
            input_mask = data_dict['attention_mask']
            input_ids = data_dict['input_ids']
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
            
            gradient_mask = data_dict['gradient_mask']
            loss = self._loss_caption(
                logits = outputs.logits[:, vision_embed.shape[1] - 1: -1],
                target = input_ids,
                mask = gradient_mask.to(self.dtype),
            )
            
            return loss
        
        else:
            
            caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 4 
            }
            
            # attentions = [None] * vision_embed.shape[0]
            instruction = data_dict['instruction']
            instruction_mask = data_dict['instruction_mask']
            
            output_ids = []
            
            for batch_id in range(vision_embed.shape[0]):
                sample_instruction = instruction[batch_id]     
                sample_mask = instruction_mask[batch_id]     # ntoken
                
                output = generation(
                    self.llm, 
                    inputs_embeds=torch.cat(
                        [
                            vision_embed[batch_id].unsqueeze(0).to(self.dtype),   # 1 x nprefix x n_embd
                            embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0).to(self.dtype)
                        ],
                        dim=1
                    ),
                    max_length=128,
                    **caption_config,
                )
                output_ids.append(output['output_ids'])
            
            output_ids = torch.cat(output_ids, dim=0)
            
            return output_ids