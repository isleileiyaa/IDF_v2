from chronos import BaseChronosPipeline
import torch

print("Loading Chronos-Bolt from AutoGluon/Chronos...")

pipeline = BaseChronosPipeline.from_pretrained(
    "autogluon/chronos-bolt-base",
    device_map="cpu"
)

model = pipeline.model

save_path = "./checkpoints/base/autogluon_model.pth"

torch.save(model.state_dict(), save_path)

print("Saved to:", save_path)