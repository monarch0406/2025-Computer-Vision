import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 畫相機外觀 (像小金字塔)
def draw_camera(ax, R, T, color, label):
    camera_size = 1.5
    center = T.reshape(3)

    # 定義四個角落
    corners = np.array([
        [-0.5, -0.5, 1],
        [ 0.5, -0.5, 1],
        [ 0.5,  0.5, 1],
        [-0.5,  0.5, 1]
    ]) * camera_size

    # 旋轉+平移
    corners = (R @ corners.T).T + center

    # 畫四個面
    for i in range(4):
        verts = [ [center, corners[i], corners[(i+1)%4]] ]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.7))

    # 顯示文字
    ax.text(center[0], center[1], center[2]+0.5, label, color=color, fontsize=10)

def visualize(pts, R1, T1, R2, T2, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-5, 15)

    # 畫黑白棋盤格
    for r in range(1, 9):
        for c in range(1, 4):
            idx = [r*4+c-5, r*4+c-4, r*4+c, r*4+c-1]
            fourCorner = pts[idx]
            color = 'black' if (r+c)%2 == 0 else 'white'
            ax.add_collection3d(Poly3DCollection([fourCorner], facecolors=color, linewidths=0.2, edgecolors='k', alpha=0.9))

    # 畫相機
    cameraPos1 = (-R1.T @ T1).reshape(3)
    cameraPos2 = (-R2.T @ T2).reshape(3)

    draw_camera(ax, R1.T, cameraPos1, 'red', 'View 1')
    draw_camera(ax, R2.T, cameraPos2, 'blue', 'View 2')

    # 算相機夾角
    v1 = R1.T @ np.array([0, 0, 1])
    v2 = R2.T @ np.array([0, 0, 1])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    angle = math.degrees(math.acos(np.clip(np.dot(v1, v2), -1, 1)))
    print('Angle between two cameras: ', angle)

    plt.savefig('./output/' + save_name + '.png')
    plt.show()
    plt.close(fig)

