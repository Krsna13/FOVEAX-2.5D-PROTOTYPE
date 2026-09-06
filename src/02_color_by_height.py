import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def main():
    sample_data = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud(sample_data.path)
    points = np.asarray(point_cloud.points)

    z = points[:, 2]
    z_normalized = (z - z.min()) / max(z.max() - z.min(), 1e-8)

    colormap = plt.colormaps["turbo"]
    colors = colormap(z_normalized)[:, :3]

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    )

    o3d.visualization.draw_geometries(
        [point_cloud, axis],
        window_name="FOVEAX - Point Cloud Colored by Height",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
