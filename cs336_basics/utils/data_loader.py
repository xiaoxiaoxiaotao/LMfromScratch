import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ):
    data_size = len(dataset)
    batch_indices = np.random.randint(
        low=0,
        high=data_size - context_length,
        size=batch_size
    )
    x_batch = []
    y_batch = []
    for idx in batch_indices:
        x = dataset[idx:idx+context_length]
        y = dataset[idx+1:idx+context_length+1]
        x_batch.append(x)
        y_batch.append(y)
    
    x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)

    return x_tensor, y_tensor
    