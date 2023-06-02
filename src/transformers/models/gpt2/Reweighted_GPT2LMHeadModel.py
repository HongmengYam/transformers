import torch
import torch.nn as nn
from torch.nn import functional as F
from tensordict import TensorDict
from tqdm import tqdm
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.onnx import config
from transformers import GPT2TokenizerFast
import wandb
import numpy as np
from transformers import OPTForCausalLM, LlamaForCausalLM, GPT2LMHeadModel, BertForMaskedLM
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles

plt.rcParams.update(bundles.iclr2023())



class Reweighted_GPT2LMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(PretrainedConfig())
        self.config = config
        self.P = GPT2LMHeadModel.from_pretrained("gpt2")
        # self.P = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        # self.P = LlamaForCausalLM.from_pretrained("../llama_hf_7B")

        self.W = GPT2LMHeadModel.from_pretrained("gpt2")
        # self.W = OPTForCausalLM.from_pretrained("facebook/opt-350m")

        # adapt vocab size of W for P
        self.adaptor = nn.Linear(self.W.config.vocab_size, self.P.config.vocab_size)

        self.reward = nn.Linear(self.P.config.vocab_size, 1)

        for p in self.P.parameters():
            p.requires_grad = False

        model_parameters = filter(lambda p: p.requires_grad, self.W.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f"N trainable params W: {params}")

    def parameters(self):
        return (
            list(self.W.parameters())
            + list(self.adaptor.parameters())
            + list(self.reward.parameters())
        )

    def forward(self, idx, attn_mask):
        # print(torch.cuda.max_memory_allocated(device="cuda:0"))
        P = self.P(input_ids=idx, attention_mask=attn_mask)
        P = P.logits.detach()

        W = self.W(input_ids=idx, attention_mask=attn_mask)
        W = self.adaptor(W.logits)

        prob_P = F.softmax(P, dim=-1)
        prob_W = F.softmax(W, dim=-1)

        prob = prob_P * prob_W
        prob = prob / prob.sum(-1, keepdim=True)

        return prob

def test():
    print("start of test")
    # Instantiate the model
    config = GPT2Config.from_pretrained('gpt2')
    model = Reweighted_GPT2LMHeadModel(config)

    # Load GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Encode a text input
    text = "Hello, how are you?"
    print(text)
    inputs = tokenizer(text, return_tensors='pt')

    # Extract inputs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Run the model
    output = model(input_ids, attention_mask)

    print(output)

    # encode context the generation is conditioned on
    input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')

    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

class Reweighted_GPT2LMHeadModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

test()