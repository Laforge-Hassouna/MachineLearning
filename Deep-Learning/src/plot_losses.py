import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("../runs/detect/train/results.csv")

epochs = df["epoch"]

# --------- BOX LOSS ---------
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["train/box_loss"], label="train/box_loss")
plt.plot(epochs, df["val/box_loss"], label="val/box_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Box Loss (Train vs Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("box_loss.png", dpi=200)
plt.show()

# --------- CLS LOSS ---------
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["train/cls_loss"], label="train/cls_loss")
plt.plot(epochs, df["val/cls_loss"], label="val/cls_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Classification Loss (Train vs Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cls_loss.png", dpi=200)
plt.show()

# --------- DFL LOSS ---------
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["train/dfl_loss"], label="train/dfl_loss")
plt.plot(epochs, df["val/dfl_loss"], label="val/dfl_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DFL Loss (Train vs Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dfl_loss.png", dpi=200)
plt.show()
