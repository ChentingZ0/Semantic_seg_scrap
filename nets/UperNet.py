import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import UperNetForSemanticSegmentation


class UperNetSwin(nn.Module):
    def __init__(self, num_classes=4, backbone="swintransformer_tiny", pretrained=True):
        super(UperNetSwin, self).__init__()
        # Load the pretrained UperNet-Swin model
        if backbone == "swintransformer_tiny":
            backbone_model = "openmmlab/upernet-swin-tiny"
        elif backbone == "swintransformer_large":
            backbone_model = "openmmlab/upernet-swin-large"
        else:
            print("wrong backbone")
        self.model = UperNetForSemanticSegmentation.from_pretrained(backbone_model)

        # Update the main classifier layer in `decode_head`
        self.model.decode_head.classifier = torch.nn.Conv2d(
            in_channels=self.model.decode_head.classifier.in_channels,  # Keep the same input channels
            out_channels=num_classes,  # Change the output channels to 4 classes
            kernel_size=self.model.decode_head.classifier.kernel_size,
            stride=self.model.decode_head.classifier.stride,
            padding=self.model.decode_head.classifier.padding
        )

        self.model.auxiliary_head.classifier = torch.nn.Conv2d(
            in_channels=self.model.auxiliary_head.classifier.in_channels,  # Keep the input channels unchanged
            out_channels=num_classes,  # Set the number of output classes to 4
            kernel_size=self.model.auxiliary_head.classifier.kernel_size,  # Keep other parameters unchanged
            stride=self.model.auxiliary_head.classifier.stride,
            padding=self.model.auxiliary_head.classifier.padding
        )
        # ------------------------------------------------------#
        #   把classifier head的weights全部消除掉，只保留backbone
        for name, module in self.model.decode_head.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():  # Ensure gradients are not tracked
                    module.weight.zero_()  # Zero out weights
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.zero_()  # Zero out biases

            # Handle BatchNorm running statistics (if applicable)
            if isinstance(module, torch.nn.BatchNorm2d):
                with torch.no_grad():
                    module.running_mean.zero_()
                    module.running_var.zero_()

        # Zero out all parameters in the auxiliary_head (if it exists)
        if hasattr(self.model, 'auxiliary_head'):
            for name, module in self.model.auxiliary_head.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    with torch.no_grad():
                        module.weight.zero_()  # Zero out weights
                if hasattr(module, 'bias') and module.bias is not None:
                    with torch.no_grad():
                        module.bias.zero_()  # Zero out biases

                # Handle BatchNorm running statistics (if applicable)
                if isinstance(module, torch.nn.BatchNorm2d):
                    with torch.no_grad():
                        module.running_mean.zero_()
                        module.running_var.zero_()
    def forward(self, x):
        # Initial x should be the batch input tensor -> pixel_values: torch.Size([1, 3, 512, 512])
        H, W = x.size(2), x.size(3)

        # Forward pass through the backbone and feature extraction stages
        outputs = self.model(x)  # This will pass through the backbone, ASPP-like modules, etc.

        # Extract the logits (similar to your DeepLab example)
        logits = outputs.logits  # Assuming logits are the output of the decode_head

        # Optional: You can further process logits if needed (e.g., upsampling or merging with auxiliary outputs)
        # Resize the logits to match the input size (similar to DeepLab)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)

        return logits
