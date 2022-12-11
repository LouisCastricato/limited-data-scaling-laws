from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, DefaultDataCollator
import torch

from torch.utils.data import default_collate, IterableDataset, Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

class AutoRegressiveDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len, device="cuda"):
        """
        dataset: list of strings
        tokenizer: tokenizer object
        seq_len: sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device

        self.dataset = "".join(dataset)
        self.tokenized_dataset = self.tokenizer(self.dataset, padding=False, truncation=False)['input_ids']
    def __iter__(self):
        return self
    def __next__(self):
        # take a random sample from the dataset of length seq_len
        start = torch.randint(0, len(self.tokenized_dataset) - self.seq_len, (1,)).item()
        end = start + self.seq_len
        out_tensor = torch.tensor(self.tokenized_dataset[start:end], device=self.device)
    
        return {
            "input_ids": out_tensor,
            "attention_mask": torch.ones_like(out_tensor, device=self.device),
        }

class SingleSampleDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len, device="cuda"):
        """
        dataset: list of strings
        tokenizer: tokenizer object
        seq_len: sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device
        self.untokenized_dataset = self.dataset

        self.tokenized_dataset = self.tokenizer(self.dataset, padding=False, truncation=False)

    def __len__(self):
        return len(self.untokenized_dataset)

    def __getitem__(self, idx):
        # retrieve tokenized_dataset at idx
        tokenized_sample = self.tokenizer(self.untokenized_dataset[idx], padding=False, truncation=False)

        return {
            "input_ids": torch.tensor(tokenized_sample['input_ids']).to(self.device),
            "attention_mask": torch.tensor(tokenized_sample['attention_mask']).to(self.device),
        }


# download only the tokenizer for now. prepare the dataset, and then download the model
model_name = "EleutherAI/pythia-1.3b-deduped"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.3b")

# set eos and pad
tokenizer.eos_token = "<|endoftext|>"
tokenizer.eos_token_id = 0

# add special pad
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# parameters
epochs = 1
seq_len = 1024
bs = 15
grad_acc_steps = 1
validate_every = 20

# load finetuning_examples.csv, which is comma delimited
df = pd.read_csv("datasets/finetuning_examples.csv", sep=",")


def collate(string):
    split_string = string.split('\n')
    name = split_string[0]
    review = '\n'.join(split_string[1:])

    # append EOT after
    output_string = "Product name: " + name + "\nProduct review: " + review + "\n"
    return  tokenizer.decode(tokenizer(output_string)['input_ids'] + [0])

# collate over the entire dataset
dataset = []
for i in range(len(df)):
    dataset.append(collate(df.iloc[i]['text']))

# take the first 95% of the dataset for training
train_dataset = dataset[:int(len(dataset)*0.95)]
# take the last 5% of the dataset for validation
validation_dataset = dataset[int(len(dataset)*0.95):]

# download the model
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# instatiate the optimizer with warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: float(epochs) / (float(epochs) + 0.01 * x))

train = SingleSampleDataset(train_dataset, tokenizer, seq_len)
validation = SingleSampleDataset(validation_dataset, tokenizer, seq_len)

# instantiate the data collator, which will pad the input
collator = DataCollatorWithPadding(return_tensors="pt", padding=True, tokenizer=tokenizer)
train_dataloader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collator)
val_dataloader = DataLoader(validation, batch_size=bs, shuffle=True, collate_fn=collator)

count = 0
last_val_los = 0
for _ in (pbar := tqdm(range(epochs))):
    # train
    for train_elems in train_dataloader:
        for k, v in train_elems.items():
            train_elems[k] = v.to("cuda")

        # forward pass
        outputs = model(input_ids=train_elems['input_ids'], attention_mask=train_elems['attention_mask'], labels=train_elems['input_ids'])
        loss = outputs.loss

        # backward pass
        loss.backward()

        # gradient accumulation
        if (count+1) % grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # update the learning rate
        if (count+1) % validate_every == 0:
            # validation
            avg_val_loss = 0
            with torch.no_grad():
                for val_elems in val_dataloader:
                    for k, v in val_elems.items():
                        val_elems[k] = v.to("cuda")

                    # forward pass
                    outputs = model(input_ids=val_elems['input_ids'], labels=val_elems['input_ids'])
                    avg_val_loss += outputs.loss.item()
                # log the loss
                avg_val_loss /= len(val_dataloader)
                last_val_los = avg_val_loss
            
        count += 1

        # log the loss
        pbar.set_description(f"train loss: {loss.item()}, val loss: {last_val_los}")

# save the model and the tokenizer
model.save_pretrained("finetuned_student_model")
tokenizer.save_pretrained("finetuned_student_model")
