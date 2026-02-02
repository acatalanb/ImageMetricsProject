import os
import torch
import shutil

# 1. Locate the Torch cache directory
cache_dir = torch.hub.get_dir()
checkpoints_dir = os.path.join(cache_dir, 'checkpoints')

print(f"Checking for corrupted weights in: {checkpoints_dir}")

# 2. Delete ResNet files if they exist
if os.path.exists(checkpoints_dir):
    files = os.listdir(checkpoints_dir)
    deleted = False
    for f in files:
        if "resnet" in f.lower():
            file_path = os.path.join(checkpoints_dir, f)
            try:
                os.remove(file_path)
                print(f"✅ Deleted corrupted file: {f}")
                deleted = True
            except Exception as e:
                print(f"❌ Could not delete {f}: {e}")

    if not deleted:
        print("No ResNet files found. The cache might already be clean.")
    else:
        print("🎉 Cleanup complete! Run your training script again.")
else:
    print("No checkpoints directory found. You are good to go.")