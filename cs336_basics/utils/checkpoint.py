import torch

def save_checkpoint(model, optimizer, iteration, out):
    obj = {}
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    obj["iteration"] = iteration
    torch.save(obj,out)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    iteration = obj["iteration"]
    return iteration