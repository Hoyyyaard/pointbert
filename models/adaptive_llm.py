from torch import nn
import torch
from models.modeling_llama import LlamaForCausalLM
from models.Point_BERT import PointTransformer
from transformers import BitsAndBytesConfig
import torch.nn.functional as nnf
from transformers import AutoTokenizer
from tools.generation_utils import generation
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from models.dvae import Group
import numpy as np
import os 
from pointnet2_ops import pointnet2_utils


class OpensceneEncoder(nn.Module):
    def __init__(self, encoder_config):
        super(OpensceneEncoder, self).__init__()
        self.encoder_config = encoder_config
        self.openscene_features_base_dir = encoder_config.openscene_features_base_dir
        # As different levels have different number of groups, we temporally set num_group and group_size to -1
        self.pointcloud_tokenizer = Group(num_group=-1, group_size=-1)
        self.level_group_map = {
            'scene': (self.encoder_config.NUM_GROUP, self.encoder_config.GROUP_SIZE),
            'instance': (self.encoder_config.INSTANCE_NUM_GROUP, self.encoder_config.INSTANCE_GROUP_SIZE),
            'region': (self.encoder_config.REGION_NUM_GROUP, self.encoder_config.REGION_GROUP_SIZE),
        }
        self.scene_npoints = self.encoder_config.N_POINTS
        self.region_npoints = self.encoder_config.REGION_N_POINTS
        self.instance_npoints = self.encoder_config.INSTANCE_N_POINTS
        
    def forward(self, xyzs, pointcloud_features, level, valid_labels=None):
        B, _, dim = pointcloud_features.shape
        max_token_num = self.level_group_map['scene'][0]
        region_token_num = self.level_group_map['region'][0]
        instance_token_num = self.level_group_map['instance'][0]
        all_fts = torch.zeros((B, max_token_num, dim), device=pointcloud_features.device, dtype=pointcloud_features.dtype)
        all_fts_mask = torch.zeros((B, max_token_num), device=pointcloud_features.device, dtype=pointcloud_features.dtype)
        
        # Different level data has different number of groups
        # We process them seperately
        instance_idx = [li for li,l in enumerate(level) if l=='instance']
        region_idx = [li for li,l in enumerate(level) if l=='region']
        scene_idx = [li for li,l in enumerate(level) if l=='scene']
        
        # if len(instance_idx) > 0:
        #     # For instance level we only need to avg pooling the features
        #     instance_fts = pointcloud_features[instance_idx][:, :self.instance_npoints, :]
        #     instance_fts = instance_fts.mean(1).unsqueeze(1)
        #     ## Padding
        #     padding_instance_embed = torch.zeros((instance_fts.shape[0], max_token_num-instance_token_num, instance_fts.shape[-1]), device=instance_fts.device)
        #     instance_fts = torch.cat([padding_instance_embed, instance_fts], dim=1)
        #     all_fts[instance_idx] = instance_fts
        #     for id in instance_idx:
        #         all_fts_mask[id, -instance_token_num:] = 1
        
        # For region and scene level we need to group and gather the features
        if len(scene_idx) > 0:
            self.pointcloud_tokenizer.num_group, self.pointcloud_tokenizer.group_size = self.level_group_map['scene']
            scene_fts = pointcloud_features[scene_idx][:, :self.scene_npoints, :]
            xyz = xyzs[scene_idx][:, :self.scene_npoints, :]
            scene_pointclouds = torch.cat([xyz, scene_fts, valid_labels.unsqueeze(-1)], dim=-1).contiguous()
            ## batch_size, num_group, group_size, 768
            scene_neighborhood, scene_center = self.pointcloud_tokenizer(scene_pointclouds)
            ## Dop xyz
            valid_neighborhood = scene_neighborhood[... , -1]
            scene_fts = scene_neighborhood[... , 3:-1].mean(-2)
            all_fts[scene_idx] = scene_fts
            all_fts_mask[scene_idx] = (valid_neighborhood.sum(-1) > 0).float()
        
        # if len(region_idx) > 0:
        #     self.pointcloud_tokenizer.num_group, self.pointcloud_tokenizer.group_size = self.level_group_map['region']
        #     region_fts = pointcloud_features[region_idx][:, :self.region_npoints, :]
        #     xyz = xyzs[region_idx][:, :self.region_npoints, :]
        #     region_pointclouds = torch.cat([xyz, region_fts], dim=-1)
        #     ## batch_size, num_group, group_size, 768
        #     region_neighborhood, region_center = self.pointcloud_tokenizer(region_pointclouds)
        #     ## Dop xyz
        #     region_fts = region_neighborhood[... , 3:].mean(-2)
        #     ## Padding
        #     padding_region_embed = torch.zeros((region_fts.shape[0], max_token_num-region_token_num, region_fts.shape[-1]), device=region_fts.device)
        #     region_fts = torch.cat([padding_region_embed, region_fts], dim=1)
        #     all_fts[region_idx] = region_fts
        #     for id in region_idx:
        #         all_fts_mask[id, -region_token_num:] = 1
        
        return all_fts, all_fts_mask, scene_center
        
        

class AdaptiveLLM(nn.Module):
    def __init__(self, llm_config, encoder_config, finetune=False):
        super(AdaptiveLLM, self).__init__()
        self._llm_config = llm_config
        self._encoder_config = encoder_config
        self.dtype = torch.float16
        self.OPENSCENE = self._encoder_config.NAME == 'Openscene'
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf')
        
        # For debug in 3090
        if encoder_config.quantization:
            print_log("Quantization is enabled")
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
                                                        quantization_config=bnb_config,
                                                        )
        else:
            self.llm = LlamaForCausalLM.from_pretrained('ckpts/Llama-2-7b-hf', 
                                            config=self._llm_config, 
                                            torch_dtype=self.dtype, 
                                            low_cpu_mem_usage=True,)
            if finetune:
                self.llm.model.gradient_checkpointing_enable()
                self.llm.model.gradient_checkpointing = True
                print_log("Gradient checkpointing is enabled")
        
        # FIXME
        # Expand LLM vocalubary when finetune as there are grouding data
        if finetune:
            special_tokens = ['<obj>', '</obj>']
            xyz_prompt = '<loc{}>'
            for i in range(255):
                special_tokens.append(xyz_prompt.format(i))
            whl_prompt = '<whl{}>'
            for i in range(255):
                special_tokens.append(whl_prompt.format(i))
            self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
            self.llm.resize_token_embeddings(len(self.tokenizer))
        
        if not self.OPENSCENE:
            self.encoder = PointTransformer(self._encoder_config)
        else:
            self.encoder = OpensceneEncoder(self._encoder_config)
    
        self.xyz_projection = nn.Sequential(
            nn.Linear(3 , 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim + 128 , self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
        )
        # self.encoder_to_llm_projection = self.encoder_to_llm_projection.to(self.dtype)
        
    def wrap_fsdp(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import (
            enable_wrap,
            transformer_auto_wrap_policy,
            wrap,
        )
        import functools
        from models.modeling_llama import LlamaDecoderLayer
        llama_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )
        self.llm = FSDP(self.llm, device_id=torch.cuda.current_device(), auto_wrap_policy=llama_auto_wrap_policy)
        # tmp_module_list = []
        # for li,layer in enumerate(self.llm.model.layers):
        #     with enable_wrap(wrapper_cls=FSDP, device_id=torch.cuda.current_device()):
        #         tmp_module_list.append(wrap(layer))    
        # tmp_module_list = nn.ModuleList(tmp_module_list)
        # setattr(self.llm.model, 'layers', tmp_module_list)
        # with enable_wrap(wrapper_cls=FSDP, device_id=device_id):
        #     self.llm.model.layers = wrap(self.llm.model.layers)
    
    def _find_all_linear_names(self, model):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    def wrap_lora(self):
        from peft import LoraConfig, get_peft_model
        lora_r: int = 64
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_bias = "none"
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self._find_all_linear_names(self.llm),
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        print("Lora is enabled")
    
    def load_model_from_ckpt(self, bert_ckpt_path, args, finetune=False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        ckpt = torch.load(bert_ckpt_path, map_location=map_location)
        if not finetune:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                    base_ckpt[k.replace("transformer_q.", "encoder.")] = base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k.replace("base_model.", "encoder.")] = base_ckpt[k]
                del base_ckpt[k]
        else:
            base_ckpt = ckpt['base_model']
            
        # uclip2
        # base_ckpt = {k.replace("module.point_encoder.", "encoder."): v for k, v in ckpt['state_dict'].items() if k.find('point_encoder') != -1}
            
        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys and torch.cuda.current_device() == 0:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys and torch.cuda.current_device() == 0:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )
        if torch.cuda.current_device() == 0:
            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')
    
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
            # num_group = data_dict['num_groups'][0].item()
            # group_size = data_dict['group_size'][0].item()
            
            # TODO
            num_group = data_dict['num_groups']
            group_size = data_dict['group_size']
            level = data_dict['level']
            
            if not self.OPENSCENE:
                cls_tokens, encoder_tokens, center = self.encoder(points, num_group=num_group, group_size=group_size, level=level, forward_llm=True)
                # Concat cls_tokens and encoder_tokens
                cls_tokens = cls_tokens.unsqueeze(1)
                vision_embed = torch.cat((cls_tokens, encoder_tokens), dim=1)
            
                # Only use cls token
                # vision_embed = torch.cat([cls_tokens, encoder_tokens.max(1)[0]], dim = -1).unsqueeze(1)
                
                # Find "instance" in level and use avg feature to represent instance
                # instance_indices = [i for i, x in enumerate(level) if x == 'instance' or x == 'region']
                # if len(instance_indices) > 0:
                #     vision_embed.requires_grad_(False)
                #     instance_embed = vision_embed[instance_indices]
                #     avg_instance_embed = (instance_embed[:, 0, :] + instance_embed[:, 1:, :].max(1)[0]).unsqueeze(1).repeat(1, vision_embed.shape[1], 1)
                #     # padding_instance_embed = torch.zeros((avg_instance_embed.shape[0], vision_embed.shape[1]-1, vision_embed.shape[-1]), device=vision_embed.device)
                #     # avg_instance_embed = torch.cat([avg_instance_embed, padding_instance_embed], dim=1)
                #     vision_embed[instance_indices] = avg_instance_embed
                    
                #     vision_mask = torch.ones_like(vision_embed[..., 0])
                #     vision_mask[instance_indices, 1:] = 0
                # else:
                vision_mask = torch.ones_like(vision_embed[..., 0])
            else:
                vision_embed, vision_mask, center = self.encoder(points, data_dict['features'], level, data_dict['valid_label'])
        
        vision_embed.requires_grad_(True)
        center.requires_grad_(True)
        vision_embed = torch.cat([vision_embed, self.xyz_projection(center)], dim=-1)
        vision_embed = vision_embed
        vision_embed = self.encoder_to_llm_projection(vision_embed)

        if not eval:
            
            input_mask = data_dict['attention_mask']
            input_ids = data_dict['input_ids']
            
            # ---- batch x (ntoken + nword) x n_embd
            # inputs_embeds = torch.cat((vision_embed, embedding_layer(input_ids)), dim=1)
            # attention_mask = torch.cat((vision_mask, input_mask), dim=1)
            
            # Calculate llm loss
            # outputs = self.llm(
            #     inputs_embeds=inputs_embeds.to(self.dtype),
            #     attention_mask=attention_mask.to(self.dtype),
            #     output_attentions=False,
            # )
            outputs = self.llm(
                vision_embeds=vision_embed.to(self.dtype),
                vision_mask=vision_mask.to(self.dtype),
                input_ids=input_ids,
                attention_mask=input_mask.to(self.dtype),
                output_attentions=True,
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
            'num_beams': None
            }
            
            # attentions = [None] * vision_embed.shape[0]
            instruction = data_dict['instruction']
            instruction_mask = data_dict['instruction_mask']
            
            output_ids = []
            
            for batch_id in range(vision_embed.shape[0]):
                sample_instruction = instruction[batch_id]     
                sample_mask = instruction_mask[batch_id]     # ntoken
                
                # output = generation(
                #     self.llm, 
                #     inputs_embeds=torch.cat(
                #         [
                #             vision_embed[batch_id][vision_mask[batch_id] == 1].unsqueeze(0).to(self.dtype),   # 1 x nprefix x n_embd
                #             embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0).to(self.dtype)
                #         ],
                #         dim=1
                #     ),
                #     max_length=128,
                #     **caption_config,
                # )
                # vision_embed_per_batch = vision_embed[batch_id][vision_mask[batch_id] == 1].unsqueeze(0).to(self.dtype)
                output = generation(
                    self.llm, 
                    vision_embeds=vision_embed[batch_id].unsqueeze(0).to(self.dtype),
                    vision_mask=vision_mask[batch_id].unsqueeze(0).to(self.dtype),
                    input_ids=sample_instruction[sample_mask == 1].unsqueeze(0),
                    attention_mask=torch.ones_like(sample_instruction[sample_mask == 1].unsqueeze(0), dtype=self.dtype),
                    max_length=128,   
                    **caption_config,
                )
                output_ids.append(output['output_ids'])
            
            output_ids = torch.cat(output_ids, dim=0)
            
            return output_ids