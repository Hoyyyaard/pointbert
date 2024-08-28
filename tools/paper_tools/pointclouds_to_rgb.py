import numpy as np
import open3d as o3d
import pyrender
import matplotlib.pyplot as plt
import torch
import trimesh

# 将点云转换为mesh
def point_cloud_to_mesh(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255)

    # 法向量估计
    point_cloud.estimate_normals()

    # 使用Poisson重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    
    # 去除低密度的三角形
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh

# 渲染mesh并捕捉图像
def render_mesh(mesh, capture_point):
    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加mesh到可视化窗口
    vis.add_geometry(mesh)

    # 设置相机位置
    ctr = vis.get_view_control()
    parameters = ctr.convert_to_pinhole_camera_parameters()
    parameters.extrinsic = np.array([
        [0, 0, 1, capture_point[0]],
        [0, 1, 0, capture_point[1]],
        [1, 0, 0, capture_point[2]],
        [0, 0, 0, 1]
    ])
    ctr.convert_from_pinhole_camera_parameters(parameters)

    # 渲染
    vis.poll_events()
    vis.update_renderer()

    # 捕捉图像
    image = vis.capture_screen_float_buffer(do_render=True)

    # # 关闭窗口
    vis.destroy_window()

    # # 显示图像
    plt.imshow(np.asarray(image))
    plt.show()

# 主函数
def main():
    pcd_data = torch.load("data/SceneVerse/3RScan/scan_data/pcd_with_global_alignment/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca.pth")
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    mesh = point_cloud_to_mesh(points, colors)
    print(mesh)
    capture_point = np.array([0., 0., 5.])  # 设置捕捉点，这里可以根据需要调整
    render_mesh(mesh, capture_point)

if __name__ == "__main__":
    main()
