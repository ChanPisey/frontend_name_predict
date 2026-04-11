"""
One-time script: re-save optimized_gender_model.pt as a clean state_dict
so torch.load(..., weights_only=True) works in Docker.

Run this LOCALLY (not in Docker):
    python convert_model.py
"""
import torch

src = "optimized_gender_model.pt"
dst = "optimized_gender_model.pt"  # overwrites in place

print(f"Loading {src} with weights_only=False (trusted local env)...")
checkpoint = torch.load(src, map_location="cpu", weights_only=False)

# If already a state_dict, save as-is
if isinstance(checkpoint, dict):
    state_dict = checkpoint
    print("Checkpoint is already a dict (state_dict). Saving as-is.")
# If it's a full model object, extract state_dict
elif hasattr(checkpoint, "state_dict"):
    state_dict = checkpoint.state_dict()
    print("Checkpoint is a full model object. Extracting state_dict.")
else:
    raise TypeError(f"Unknown checkpoint type: {type(checkpoint)}")

torch.save(state_dict, dst)
print(f"Saved clean state_dict to {dst}")
print("Done. You can now rebuild Docker.")
