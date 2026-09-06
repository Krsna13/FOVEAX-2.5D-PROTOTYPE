from pathlib import Path
import open3d as o3d


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    sample_data = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud(sample_data.path)

    downsampled = point_cloud.voxel_down_sample(voxel_size=0.03)
    filtered_cloud, _ = downsampled.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    output_path = output_dir / "filtered_sample_cloud.ply"
    success = o3d.io.write_point_cloud(str(output_path), filtered_cloud)

    orig_count = len(point_cloud.points)
    filt_count = len(filtered_cloud.points)
    reduction = (1.0 - (filt_count / orig_count)) * 100.0

    print("Saved successfully:", success)
    print("Output file:", output_path)
    print("Original point count:", orig_count)
    print("Filtered point count:", filt_count)
    print(f"Point reduction: {reduction:.2f}%")


if __name__ == "__main__":
    main()
