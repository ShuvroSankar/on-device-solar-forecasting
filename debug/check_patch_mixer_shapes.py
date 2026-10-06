import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

model = TinyTimeMixerForPrediction.from_pretrained("ttm_full_finetuned_lr1e-4/ttm_finetuned")
model.eval()

# Find the exact module and its weight shape
for name, module in model.named_modules():
    if "mixers.0.patch_mixer.mlp.fc1" in name:
        print(f"{name}: weight shape = {tuple(module.weight.shape)}")

# Hook it to print the REAL input shape during a real forward pass
def hook(mod, inp, out):
    print(f"Runtime input to fc1: {tuple(inp[0].shape)}")
    print(f"fc1 weight expects in_features = {mod.weight.shape[1]}")

handles = []
for name, module in model.named_modules():
    if "mixers.0.patch_mixer.mlp.fc1" in name:
        handles.append(module.register_forward_hook(hook))

past_values = torch.randn(1, 52, 1)
freq_token = torch.tensor([3], dtype=torch.long)
with torch.no_grad():
    model(past_values=past_values, freq_token=freq_token)

for h in handles:
    h.remove()
