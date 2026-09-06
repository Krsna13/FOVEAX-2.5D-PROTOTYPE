import numpy as np
import open3d as o3d


def main():
    sample_data = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud(sample_data.path)
    print("Original points:", len(point_cloud.points))

    downsampled = point_cloud.voxel_down_sample(voxel_size=0.03)
    print("After voxel downsampling:", len(downsampled.points))

    clean_cloud, inlier_indices = downsampled.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    print("After outlier removal:", len(clean_cloud.points))

    clean_cloud.paint_uniform_color([0.2, 0.8, 1.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    )

    o3d.visualization.draw_geometries(
        [clean_cloud, axis],
        window_name="FOVEAX - Filtered Point Cloud",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
