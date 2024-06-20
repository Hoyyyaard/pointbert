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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    transformer_auto_wrap_policy,
    wrap,
)
import functools
from models.modeling_llama import LlamaDecoderLayer
import numpy as np
import os 
from utils.logger import *
from pointnet2_ops import pointnet2_utils
from tools.position_embedding import PositionEmbeddingCoordsSine


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
        
    def forward(self, xyzs, pointcloud_features, level):
        B, _, dim = pointcloud_features.shape
        max_token_num = self.level_group_map['scene'][0]
        all_fts = torch.zeros((B, max_token_num, dim), device=pointcloud_features.device, dtype=pointcloud_features.dtype)
        all_fts_mask = torch.ones((B, max_token_num), device=pointcloud_features.device, dtype=pointcloud_features.dtype)
        scene_idx = [li for li,l in enumerate(level) if l=='scene']
        if len(scene_idx) > 0:
            self.pointcloud_tokenizer.num_group, self.pointcloud_tokenizer.group_size = self.level_group_map['scene']
            scene_fts = pointcloud_features[scene_idx][:, :self.scene_npoints, :]
            xyz = xyzs[scene_idx][:, :self.scene_npoints, :]
            scene_pointclouds = torch.cat([xyz, scene_fts], dim=-1).contiguous()
            ## batch_size, num_group, group_size, 768
            scene_neighborhood, scene_center = self.pointcloud_tokenizer(scene_pointclouds)
            ## Drop xyz
            neighborhood_xyz = scene_neighborhood[... , :3]
            scene_fts = scene_neighborhood[... , 3:].mean(-2)
            all_fts[scene_idx] = scene_fts
        else:
            assert False
        
        return all_fts, all_fts_mask, scene_center, neighborhood_xyz
        
        
class AdaptiveLLM(nn.Module):
    def __init__(self, llm_config, encoder_config, finetune=False, logger=None, args=None):
        super(AdaptiveLLM, self).__init__()
        self._llm_config = llm_config
        self._encoder_config = encoder_config
        self.dtype = torch.float32 if self._encoder_config.DTYPE == 'FP32' else torch.float16
        self.OPENSCENE = self._encoder_config.NAME == 'Openscene'
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/Llama-2-7b-hf')
        self.logger = get_logger(args.log_name)
        self.args = args
        
        self.FLEX = hasattr(self._encoder_config, 'FLEX')
        self._llm_config.FLEX = self.FLEX
        self._llm_config.DENSE_TOKEN_NUM = self._encoder_config.DENSE_TOKEN_NUM if hasattr(self._encoder_config, 'DENSE_TOKEN_NUM') else None
        self._llm_config.DENSE_TOKEN_SELECT_THRESHOLD = self._encoder_config.DENSE_TOKEN_SELECT_THRESHOLD if hasattr(self._encoder_config, 'DENSE_TOKEN_SELECT_THRESHOLD') else None
        self._llm_config.DENSE_TOKEN_SELECT_TOPK = self._encoder_config.DENSE_TOKEN_SELECT_TOPK if hasattr(self._encoder_config, 'DENSE_TOKEN_SELECT_TOPK') else None
        
        model_path = 'ckpts/Llama-2-7b-hf'
        if hasattr(self._encoder_config,"LLAVA"):
            model_path = 'ckpts/llava-v1.5-7b'
            print_log("Using LLAVA Finetune")
            
        # For debug in 3090
        if encoder_config.quantization:
            print_log("Quantization is enabled")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                     
                bnb_4bit_use_double_quant=True,        
                bnb_4bit_quant_type="nf4",            
                bnb_4bit_compute_dtype=self.dtype      
            )
            self.llm = LlamaForCausalLM.from_pretrained(model_path, 
                                                        config=self._llm_config, 
                                                        torch_dtype=self.dtype, 
                                                        low_cpu_mem_usage=True,
                                                        quantization_config=bnb_config,
                                                        )
        else:
            self.llm = LlamaForCausalLM.from_pretrained(model_path, 
                                            config=self._llm_config, 
                                            torch_dtype=self.dtype, 
                                            )
            if not self.FLEX:
                self.llm.model.gradient_checkpointing_enable()
                self.llm.model.gradient_checkpointing = True
                print_log("Gradient checkpointing is enabled", logger=self.logger)
        
        if not self.OPENSCENE:
            self.encoder = PointTransformer(self._encoder_config)
        else:
            self.encoder = OpensceneEncoder(self._encoder_config)
        
        # Expand LLM vocalubary when finetune as there are grouding data
        special_tokens = ['<vp_placehold>', '<scene>', '<scene_placehold>', '</scene>', '<obj>', '</obj>']
        # xyz_prompt = '<loc{}>'
        # for i in range(255):
        #     special_tokens.append(xyz_prompt.format(i))
        # whl_prompt = '<whl{}>'
        # for i in range(255):
        #     special_tokens.append(whl_prompt.format(i))
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Visual Prompt version
        self.visual_nquery = 8
        self.box_prompt_projector = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self.visual_nquery * self._llm_config.hidden_size),
        )
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self.visual_nquery * self._llm_config.hidden_size),
        )
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=self._encoder_config.trans_dim, 
            pos_type='fourier', 
            normalize=True
        )
        
        # Given xyz and token mask version
        self.xyz_projection = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim , self._encoder_config.trans_dim),
            nn.ReLU(),
            nn.Linear(self._encoder_config.trans_dim, self._encoder_config.trans_dim),
        )
        
        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(self._encoder_config.trans_dim , self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
        )

    def _fps(self, points, number):
        '''
            data B N 3
            number int
        '''
        fps_idx = pointnet2_utils.furthest_point_sample(points.contiguous(), number) 
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return points
    
    def _get_hd_corpus(self, data_dict, center, neighborhood_xyz):
        '''
            neighborhood_xyz: [bs, 128, 384, 3]
        '''
        # Extract {TOKEN_NUM} dense tokens per token
        TOKEN_NUM = self._llm_config.DENSE_TOKEN_NUM
        bs = neighborhood_xyz.shape[0]
        # Get dense token center in neighborhood [bs, 128, 4, 3]
        dense_token_center = self._fps(neighborhood_xyz.view(-1, neighborhood_xyz.shape[-2], neighborhood_xyz.shape[-1]), TOKEN_NUM).view(bs, -1, TOKEN_NUM, 3)

        tokenizer = Group(num_group=dense_token_center.shape[-3]*dense_token_center.shape[-2], group_size=neighborhood_xyz.shape[-2])
        
        dense_points = data_dict['hd_points']
        dense_features = data_dict['hd_features']
        
        dense_scene_pointclouds = torch.cat([dense_points, dense_features], dim=-1).contiguous()
        ## batch_size, num_group, group_size, 768
        dense_neighborhood, dense_center = tokenizer(dense_scene_pointclouds)

        # [bs, 128*TOKEN_NUM, 384, 768]
        dense_fts = dense_neighborhood[... , 3:].mean(-2)
        
        point_cloud_dims_min, _ = dense_points[..., :3].min(dim=1)
        point_cloud_dims_max, _ = dense_points[..., :3].max(dim=1)
        point_cloud_dims = [
            point_cloud_dims_min,
            point_cloud_dims_max,
        ]
        
        return {
            # 'dense_points':dense_points, 
            'dense_center':dense_center,
            'point_cloud_dims': point_cloud_dims,
            'dense_features':dense_fts.view(bs, -1, TOKEN_NUM, self._encoder_config.trans_dim).contiguous()
        }
        
    def wrap_model(self):
        if self._encoder_config.distributed == 'DDP':
            return self.wrap_ddp()
        elif self._encoder_config.distributed == 'FSDP':
            return self.wrap_fsdp()
        else:
            assert False
    
    def wrap_fsdp(self):
        llama_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )
        self.llm = FSDP(self.llm, device_id=torch.cuda.current_device(), auto_wrap_policy=llama_auto_wrap_policy)
        self.encoder = self.encoder.to(torch.cuda.current_device())
        self.encoder_to_llm_projection = self.encoder_to_llm_projection.to(torch.cuda.current_device())
        self.xyz_projection = self.xyz_projection.to(torch.cuda.current_device())
        self.pos_emb3d = self.pos_emb3d.to(torch.cuda.current_device())
        self.box_prompt_projector = self.box_prompt_projector.to(torch.cuda.current_device())
        self.click_prompt_projector = self.click_prompt_projector.to(torch.cuda.current_device())
        print_log('Using FSDP')
        return self
    
    def wrap_ddp(self):
        self = self.to(torch.cuda.current_device())
        self = nn.parallel.DistributedDataParallel(self, 
                                                    device_ids=[self.args.local_rank % torch.cuda.device_count()]
                                                    )
        print_log('Using Distributed Data parallel' )
        return self
    
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
        
        # tmp_ckpt = {k:v for k,v in ckpt['base_model'].items() if not k.find('llm.') != -1}
        # ckpt = {'base_model': tmp_ckpt}
        
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
        print_log(incompatible, logger = 'Transformer')
        

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

        return final_loss + 0.
    
    def expand_prompt_representation(self, prompt_feature, prompt_mask=None):
        # input:
        #   prompt_feature: batch x nprompt x (ntkn x channel)
        #   prompt_mask: batch x nprompt
        # output:
        #   prompt_feature: batch x (nprompt x ntkn) x channel
        #   prompt_mask: batch x (nprompt x ntkn)
        batch_size, nprompt = prompt_feature.shape[:2]
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_feature[..., 0])
        prompt_mask = prompt_mask.unsqueeze(-1).repeat(1, 1, self.visual_nquery)
        prompt_mask = prompt_mask.reshape(batch_size, nprompt * self.visual_nquery)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt, self.visual_nquery, self._llm_config.hidden_size)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt * self.visual_nquery, self._llm_config.hidden_size)
        return prompt_feature, prompt_mask
    
    def forward(self, data_dict, eval=False):
        with torch.no_grad():
                    
            points = data_dict['points']
            level = data_dict['level']
            features = data_dict['features']
            vision_embed, vision_mask, center, neighborhood_xyz = self.encoder(points, features, level)
                
        # Visual prompt code
        batch_size = vision_embed.shape[0]
        # # Generate box and click prompt
        box_mask = data_dict['box_mask']
        click_prompt = data_dict['click_query']
        click_mask = data_dict['click_mask']
        box_prompt = data_dict['box_query'].unsqueeze(1)
        visual_prompt = [torch.zeros(batch_size, 0, self._llm_config.hidden_size).to(vision_embed.device)]
        visual_mask = [torch.zeros(batch_size, 0).to(vision_embed.device)]
        box_prompt = self.box_prompt_projector(box_prompt)
        box_prompt, box_mask = self.expand_prompt_representation(box_prompt, box_mask)
        visual_prompt.append(box_prompt)
        visual_mask.append(box_mask) 
        
        point_cloud_dims_min, _ = data_dict['points'][..., :3].min(dim=1)
        point_cloud_dims_max, _ = data_dict['points'][..., :3].max(dim=1)
        point_cloud_dims = [
            point_cloud_dims_min,
            point_cloud_dims_max,
        ]
        click_xyz = click_prompt     # batch x nquery x 3
        click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)
        click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))
        click_prompt, click_mask = self.expand_prompt_representation(click_prompt, click_mask)
        visual_prompt.append(click_prompt)
        visual_mask.append(click_mask)

        ## concat box and click prompts as well as prompt masks
        prompt_feature = torch.cat(visual_prompt, dim=1)   # batch x (2 x ntoken) x channel
        prompt_mask = torch.cat(visual_mask, dim=1)        # batch x (2 x ntoken)
            
        if self.FLEX:
            hd_corpus = self._get_hd_corpus(data_dict, center, neighborhood_xyz)
            hd_center = hd_corpus['dense_center']
            hd_point_cloud_dims = hd_corpus['point_cloud_dims']
            hd_vision_embed = hd_corpus['dense_features']
            bsz, _, TN, dim = hd_vision_embed.shape
            hd_vision_embed = hd_vision_embed.view(bsz, -1, dim).contiguous()
            
            hd_pos_embed = self.pos_emb3d(hd_center, input_range=hd_point_cloud_dims)
            pos_embed = self.pos_emb3d(center, input_range=point_cloud_dims)
            all_pos_embed = torch.cat((pos_embed, hd_pos_embed), dim=-1)
            all_pos_embed = self.xyz_projection(all_pos_embed.permute(0, 2, 1))
            pos, hd_pos = all_pos_embed.split([vision_embed.shape[1], hd_vision_embed.shape[1]], dim=1)
            all_vision_embed = torch.cat((vision_embed+pos, hd_vision_embed+hd_pos), dim=1)
            all_vision_embed = self.encoder_to_llm_projection(all_vision_embed)
            vision_embed, hd_vision_embed = all_vision_embed.split([vision_embed.shape[1], hd_vision_embed.shape[1]], dim=1)
            hd_vision_embed = hd_vision_embed.to(self.dtype)
            # Set hd corpus for self attention layer
            for i in range(len(self.llm.model.layers)):
                self.llm.model.layers[i].self_attn.hd_features = hd_vision_embed.view(bsz, -1, TN, self._llm_config.hidden_size).contiguous()
                self.llm.model.layers[i].self_attn.step_ratio = data_dict['step_ratio']
        else:
            pos_embed = self.pos_emb3d(center, input_range=point_cloud_dims)
            pos_embed = self.xyz_projection(pos_embed.permute(0, 2, 1))
            vision_embed = self.encoder_to_llm_projection(vision_embed + pos_embed) 
        
        if not eval:
            input_mask = data_dict['attention_mask']
            input_ids = data_dict['input_ids']
            
            outputs = self.llm(
                vision_embeds=vision_embed.to(self.dtype),
                vision_mask=vision_mask.to(self.dtype),
                visual_prompt_embeds=prompt_feature.to(self.dtype),
                visual_prompt_mask=prompt_mask.to(self.dtype),
                input_ids=input_ids,
                attention_mask=input_mask.to(self.dtype),
                tokenizer = self.tokenizer,
                output_attentions=False,
            )
            
            gradient_mask = data_dict['gradient_mask']
            loss = self._loss_caption(
                logits = outputs.logits[:, -(gradient_mask.shape[1]+1): -1],
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
            attentions = []
            
            for batch_id in range(vision_embed.shape[0]):
                sample_instruction = instruction[batch_id]     
                sample_mask = instruction_mask[batch_id]     # ntoken
                
                output = generation(
                    self.llm, 
                    vision_embeds=vision_embed[batch_id].unsqueeze(0).to(self.dtype),
                    vision_mask=vision_mask[batch_id].unsqueeze(0).to(self.dtype),
                    input_ids=sample_instruction[sample_mask == 1].unsqueeze(0),
                    attention_mask=torch.ones_like(sample_instruction[sample_mask == 1].unsqueeze(0)).to(self.dtype),
                    visual_prompt_embeds=prompt_feature[batch_id].unsqueeze(0).to(self.dtype),
                    visual_prompt_mask=prompt_mask[batch_id].unsqueeze(0).to(self.dtype),
                    tokenizer = self.tokenizer,
                    max_length=128,   
                    **caption_config,
                )
                output_ids.append(output['output_ids'])
                if not output['attentions'] is None:
                    attn = torch.cat(output['attentions'], dim=0)
                    attentions.append(attn)
            
            output_ids = torch.cat(output_ids, dim=0)
            if len(attentions) > 0:
                attentions = torch.stack(attentions, dim=0)
            
            return output_ids, attentions, center