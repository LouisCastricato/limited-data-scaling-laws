import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from critic_models import GPTSentimentELOCritic, T5SentimentELOCritic
from tqdm import tqdm 
import pandas as pd

def load_elo_model():
    # load critic model
    elo_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").cuda()
    elo_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    prompt_dir = "datasets/prompts_reprocessed.csv"
    suffix = "positive"
    return T5SentimentELOCritic(model, tokenizer, prompt_dir, suffix=suffix)