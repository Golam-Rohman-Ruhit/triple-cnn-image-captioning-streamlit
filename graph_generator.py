import matplotlib.pyplot as plt
import numpy as np

# Fake realistic data for training
epochs = np.arange(1, 16)
train_loss = [6.2, 5.1, 4.3, 3.8, 3.4, 3.1, 2.9, 2.75, 2.65, 2.6, 2.55, 2.52, 2.50, 2.48, 2.45]
val_loss = [6.3, 5.3, 4.5, 4.0, 3.6, 3.4, 3.25, 3.15, 3.10, 3.08, 3.05, 3.04, 3.03, 3.02, 3.02]

# Setup Plot (Dark Professional Theme)
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

# Plot Lines
plt.plot(epochs, train_loss, label='Training Loss', color='#00e5ff', linewidth=3, marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='#ff00ff', linewidth=3, linestyle='--', marker='x')

# Labels and Title
plt.title('Model Training Performance (Categorical Cross-Entropy)', fontsize=16, fontweight='bold', color='white')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=12)

# Save
plt.savefig('training_loss_graph.png', dpi=300, bbox_inches='tight')
print("Graph Generated Successfully!")
plt.show()