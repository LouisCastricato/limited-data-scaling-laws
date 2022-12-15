import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import pandas as pd
from tqdm import tqdm

# Defines the reward model that sits on top of an autoregressive language model
class GPTRewardModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(name)
        self.config = model.config
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        input_ids = input_ids.to(next(self.parameters()).device)
        attention_mask = attention_mask.to(next(self.parameters()).device)
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask
        )

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        return rewards

class ReviewRewardDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        # datadir is a csv with columns ["name", "review", "normalized elo"]
        self.data = pd.read_csv(data_dir)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # get the product name
        name = self.data["name"].iloc[idx]
        # get the review
        review = self.data["review"].iloc[idx]
        # tokenize
        string = "Product name: " + name + " Product review: " + review
        out = self.tokenizer(string, return_tensors="pt", padding=False, truncation=False)
        # remove the last token
        out["input_ids"] = out["input_ids"][:,:-1].squeeze()
        out["attention_mask"] = out["attention_mask"][:,:-1].squeeze()
        # add the reward
        out["reward"] = torch.tensor(self.data["normalized elo"].iloc[idx])
        return out

def train_rm():
    # load the reward model
    reward_model = GPTRewardModel("EleutherAI/gpt-neo-125M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # load the dataset
    dataset = ReviewRewardDataset("rewards.csv", tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    grad_accum = 1
    # train the reward model
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)
    for idx, batch in (pbar := tqdm(enumerate(dataloader), total=len(dataloader))):
        # forward pass
        rewards = reward_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # calculate loss
        loss = torch.mean((rewards - batch["reward"])**2)
        # backward pass
        loss.backward()
        if (idx + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        pbar.set_description(f"Loss: {loss.item():.4f}")
    # save the model    
    reward_model.save_pretrained("reward_model")
    tokenizer.save_pretrained("reward_model")

if __name__ == "__main__":
    train_rm()
