# import numpy as np

# pts = np.random.randn(1000,3) * [5,2,0.5]
# A = pts - pts.mean(axis=0)
# U, S, Vt = np.linalg.svd(A, full_matrices=False)

# main_axis = Vt[0]
# normal = Vt[2]
# print("Main axis:", main_axis)
# print("Normal:", normal)


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)                                    # same cloud every run
pts = np.random.randn(1000, 3) * [5, 2, 0.5]         # elongated cloud

# rotate the cloud 30° about the z-axis for the second panel
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0,              0,             1]])
pts_rot = pts @ R.T

colors = ["#d62728", "#2ca02c", "#1f77b4"]
names = ["v1: main axis (largest spread)",
         "v2: second axis",
         "v3: normal (smallest spread)"]

fig = plt.figure(figsize=(15, 6.5))

for k, (p, title) in enumerate([(pts, "Original cloud"),
                                (pts_rot, "Same cloud rotated 30° about z")]):
    A = p - p.mean(axis=0)                           # center it
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=4, c="#999999", alpha=0.35)

    for i in range(3):
        length = 2 * S[i] / np.sqrt(len(A) - 1)      # 2 standard deviations
        v = Vt[i] * length
        ax.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]],
                color=colors[i], lw=3, label=f"{names[i]}  σ={S[i]:.0f}")
        ax.quiver(0, 0, 0, *v, color=colors[i], arrow_length_ratio=0.12, lw=3)

    ax.set_title(title, fontsize=13)
    ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=8)

    print(title, "| S =", S.round(1),
          "| main_axis =", Vt[0].round(2), "| normal =", Vt[2].round(2))

fig.suptitle("SVD of a centered point cloud: rows of Vt are the principal directions",
             fontsize=14)
plt.tight_layout()
plt.show()                                           # opens an interactive window