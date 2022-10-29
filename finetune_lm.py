from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from torch.utils.data import default_collate
import pandas as pd
from tqdm import tqdm

class AutoRegressiveDataset(torch.utils.data.IterableDataset):
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

        self.tokenized_dataset = self.tokenizer("".join(self.dataset), padding=False, truncation=False)['input_ids']
    def __iter__(self):
        return self
    def __next__(self):
        # take a random sample from the dataset of length seq_len
        start = torch.randint(0, len(self.tokenized_dataset) - self.seq_len, (1,)).item()
        end = start + self.seq_len
        return torch.tensor(self.tokenized_dataset[start:end]).to(self.device)

# download only the tokenizer for now. prepare the dataset, and then download the model
model_name = "EleutherAI/pythia-1.3b-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# parameters
epochs = 5
seq_len = 1024
samples_per_epoch = 64
bs = 2
grad_acc_steps = 8

# load finetuning_examples.csv, which is comma delimited
df = pd.read_csv("finetuning_examples.csv", sep=",")


def collate(string):
    split_string = string.split('\n')
    name = split_string[0]
    review = '\n'.join(split_string[1:])

    # append EOT after
    output_string = "Product name: " + name + "\nProduct review: " + review
    return tokenizer.decode(tokenizer(output_string)['input_ids'] +[0])

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0 / (1.0 + 0.01 * x))

train_iter = iter(AutoRegressiveDataset(train_dataset, tokenizer, seq_len))
validation_iter = iter(AutoRegressiveDataset(validation_dataset, tokenizer, seq_len))

count = 0
last_val_los = 0
for _ in (pbar := tqdm(range(epochs))):
    # train
    for sample in range(samples_per_epoch//bs):

        # get the next batch
        train_elems = default_collate([next(train_iter) for _ in range(bs)])

        # forward pass
        outputs = model(train_elems, labels=train_elems)
        loss = outputs.loss

        # backward pass
        loss.backward()

        # gradient accumulation
        if (count+1) % grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            count = 0
        count += 1

        # log the loss
        pbar.set_description(f"train loss: {loss.item()}, val loss: {last_val_los}")
    
    # validation
    avg_val_loss = 0
    with torch.no_grad():
        for sample in range((samples_per_epoch//2)//bs):
            # get the next batch
            validation_elems = default_collate([next(validation_iter) for _ in range(bs)])
            # forward pass
            outputs = model(validation_elems, labels=validation_elems)
            avg_val_loss += outputs.loss.item()
        # log the loss
        avg_val_loss /= (samples_per_epoch//2)//bs
        last_val_los = avg_val_loss

# save the model
model.save_pretrained("finetuned_student_model")
