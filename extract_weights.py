from byol.models import BYOL
import torch

model = BYOL.load_from_checkpoint("byol.ckpt")

encoder = model.encoder

torch.save(encoder.state_dict(), "encoder.pt")
