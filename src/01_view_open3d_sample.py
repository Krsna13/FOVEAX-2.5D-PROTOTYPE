import numpy as np
import open3d as o3d


def main():
    sample_data = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud(sample_data.path)
    points = np.asarray(point_cloud.points)

    print("Sample file:", sample_data.path)
    print("Point cloud:", point_cloud)
    print("Point array shape:", points.shape)
    print("Number of points:", len(points))

    print("\nFirst 5 points [x, y, z]:")
    print(points[:5])

    print("\nCoordinate ranges:")
    print(f"x: {points[:, 0].min():.3f} to {points[:, 0].max():.3f}")
    print(f"y: {points[:, 1].min():.3f} to {points[:, 1].max():.3f}")
    print(f"z: {points[:, 2].min():.3f} to {points[:, 2].max():.3f}")

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    )

    o3d.visualization.draw_geometries(
        [point_cloud, axis],
        window_name="FOVEAX Phase 1 - Open3D Sample Point Cloud",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
