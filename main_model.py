from preprocessing import d_norm, delete_hot_pixels, read_folder
import pandas as pd
from model import Discriminator
import torch

data = pd.DataFrame(read_folder("data_for_prediction")).T
for col in data.columns:
    data[col] = delete_hot_pixels(data[col].to_list())

data = d_norm(data)

model = Discriminator(in_dim=len(data.columns), hidden_dims=[128, 32], out_dim=1)

result = model(torch.tensor(data.to_numpy(), dtype=torch.float))
with open("result.txt", "w", encoding="utf-16-le") as f:
    for name, value in zip(data.index, result):
        f.write(f"{name}: {value.item()} \n")