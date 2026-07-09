import sys
try:
    # Use a non-interactive backend to avoid display issues on headless systems
    import matplotlib
    matplotlib.use('Agg')
except Exception as e:
    print("Error: matplotlib is required to run this script:\n", e, file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("Error: matplotlib.pyplot is required to run this script:\n", e, file=sys.stderr)
    sys.exit(1)

import numpy as np

# Your results
cm = np.array([
    [983, 17],
    [3, 997]
])

fig, ax = plt.subplots(figsize=(6, 5))

cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Metal', 'PVC'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Metal', 'PVC'])

ax.set_xlabel('Predicted Class')
ax.set_ylabel('Actual Class')
ax.set_title('Confusion Matrix - Metal vs PVC')

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

print("Graph saved as confusion_matrix.png")

plt.show()