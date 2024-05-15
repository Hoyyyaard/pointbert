from torch import nn
import torch
from models.modeling_llama import LlamaForCausalLM
from models.Point_BERT import Point_BERT

class AdaptiveLLM(nn.Module):
    def __init__(self, llm_config, encoder_config):
        super(AdaptiveLLM, self).__init__()
        self._llm_config = llm_config
        self._encoder_config = encoder_config
        
        # self.llm = LlamaForCausalLM.from_pretrained('ckpts/Llama-2-7b-hf', config=self._llm_config, torch_dtype=torch.float16, )#low_cpu_mem_usage=True)
        self.encoder = Point_BERT(self._encoder_config)
        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(self._encoder_config.transformer_config.trans_dim, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
            nn.Linear(self._llm_config.hidden_size, self._llm_config.hidden_size),
            nn.ReLU(),
        )
        
    def forward(self):
        pass