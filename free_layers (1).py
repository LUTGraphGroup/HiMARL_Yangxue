# freeze_layers.py

import torch


def freeze_layers(model, strategy="all"):
    """
    Adjusts the trainability of the transformer model layers based on the given strategy.

    Parameters:
    - model (torch.nn.Module): The model whose layers will be frozen/unfrozen.
    - strategy (str): The finetuning strategy to apply. Possible values are:
        * 'no' - Freeze all layers, no fine-tuning.
        * 'last' - Only unfreeze the last transformer layer and the pooler.
        * 'last_three' - Unfreeze the last three transformer layers and the pooler.
        * 'first' - Only unfreeze the first transformer layer.
        * 'middle' - Only unfreeze middle layers (e.g., layer 6 and layer 7).
        * 'custom' - Unfreeze specific layers, customize based on layer names.
        * 'all' - Unfreeze all layers, full fine-tuning.

    Raises:
    - ValueError: If an unsupported finetuning strategy is provided.
    """

    if strategy == "no":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        print("All layers frozen. No fine-tuning will be done.")

    elif strategy == "last":
        # Only unfreeze the last transformer layer and the pooler
        for name, param in model.named_parameters():
            if "layer.11" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Only the last layer and pooler are unfrozen for fine-tuning.")

    elif strategy == "last_three":
        # Unfreeze the last three transformer layers and the pooler
        for name, param in model.named_parameters():
            if "layer.9" in name or "layer.10" in name or "layer.11" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("The last three layers and pooler are unfrozen for fine-tuning.")

    elif strategy == "first":
        # Only unfreeze the first transformer layer
        for name, param in model.named_parameters():
            if "layer.0" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Only the first layer is unfrozen for fine-tuning.")

    elif strategy == "middle":
        # Only unfreeze middle layers (e.g., layer 6 and layer 7)
        for name, param in model.named_parameters():
            if "layer.6" in name or "layer.7" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Only middle layers (6 and 7) are unfrozen for fine-tuning.")

    elif strategy == "custom":
        # Custom strategy: manually specify which layers to unfreeze
        for name, param in model.named_parameters():
            if "layer.10" in name or "layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Custom layers (layer 10 and layer 11) are unfrozen for fine-tuning.")

    elif strategy == "all":
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        print("All layers are unfrozen for fine-tuning.")

    else:
        raise ValueError(f"Unsupported finetuning strategy: {strategy}")
