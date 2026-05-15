import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../eomt')))

import torch
import yaml
import importlib
from erfnet import ERFNet


def load_erfnet(weights_path, num_classes=20):
    """Initializes and loads weights for ERFNet."""
    model = ERFNet(num_classes)
    
    state_dict = torch.load(weights_path, map_location='cpu')
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
        else:
            own_state[name].copy_(param)
            
    return model

def load_eomt(config_path, weights_path, device):
    """Initializes and loads weights for EoMT using Lightning yaml config files."""
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module = getattr(importlib.import_module(data_module_name), class_name)
    data_kwargs = config["data"].get("init_args", {})
    data_meta = data_module(path="", batch_size=1, num_workers=0, check_empty_targets=False, **data_kwargs)

    # Extract encoder and network from config
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_cls = getattr(importlib.import_module(encoder_cfg["class_path"].rsplit(".", 1)[0]), encoder_cfg["class_path"].rsplit(".", 1)[1])
    encoder = encoder_cls(img_size=data_meta.img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_cls = getattr(importlib.import_module(network_cfg["class_path"].rsplit(".", 1)[0]), network_cfg["class_path"].rsplit(".", 1)[1])
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(masked_attn_enabled=False, num_classes=data_meta.num_classes, encoder=encoder, **network_kwargs)

    # Instantiate final Lightning module
    lit_cls = getattr(importlib.import_module(config["model"]["class_path"].rsplit(".", 1)[0]), config["model"]["class_path"].rsplit(".", 1)[1])
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    data_init_args = config.get("data", {}).get("init_args", {})
    if "stuff_classes" in data_init_args:
        model_kwargs['stuff_classes'] = data_init_args["stuff_classes"]
    elif hasattr(data_meta, 'stuff_classes'):
        model_kwargs['stuff_classes'] = data_meta.stuff_classes
    model = lit_cls(img_size=data_meta.img_size, num_classes=data_meta.num_classes, network=network, **model_kwargs).to(device)

    # Load checkpoint weights
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    return model