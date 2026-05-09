import timm
import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.params.model_params import ViTHFParams


class ViTHF(BaseModel):
    def __init__(self, model_params: ViTHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        kwargs = {"num_classes": 0, "pretrained": load_pretrained}
        # Override the timm model's native input resolution if requested. timm
        # interpolates pretrained position embeddings to the new size at load.
        if getattr(model_params, "img_size", None) is not None:
            kwargs["img_size"] = model_params.img_size
        self.model = timm.create_model(self.model_name, **kwargs)

    def forward(self, image):
        image_embeddings = self.model.forward_intermediates(image)[0]
        return image_embeddings

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
