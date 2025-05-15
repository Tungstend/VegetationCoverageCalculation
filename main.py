import rasterio
import numpy as np
from skimage.filters import threshold_otsu


def vegetation_segmentation(input_tif, output_tif, total_samples=10000, precision=4, random_seed=None):
    """
    基于频率倒数加权的无监督植被分类

    参数：
    input_tif: 输入NDVI TIFF文件路径
    output_tif: 输出分类结果TIFF路径
    total_samples: 总采样数量（默认10000）
    precision: NDVI值离散化精度（小数点位数，默认4）
    random_seed: 随机种子（默认无）
    """
    print(f"total_samples: {total_samples}")

    # 读取NDVI数据
    with rasterio.open(input_tif) as src:
        profile = src.profile
        ndvi = src.read(1)
        ndvi = np.where(ndvi == profile['nodata'], np.nan, ndvi)

    # 提取有效NDVI值并展平
    valid_ndvi = ndvi[~np.isnan(ndvi)].ravel()
    if valid_ndvi.size == 0:
        raise ValueError("Input contains no valid NDVI values")

    # 离散化处理（保留指定位数小数）
    scaled_ndvi = np.round(valid_ndvi, decimals=precision)

    # 计算唯一值及其频次
    unique_values, counts = np.unique(scaled_ndvi, return_counts=True)
    freq_dict = dict(zip(unique_values, counts))

    # 计算反频率权重（避免除零错误）
    eps = 1e-6
    weights = 1.0 / (np.vectorize(freq_dict.get)(scaled_ndvi) + eps)

    # 标准化权重
    weights /= weights.sum()

    # 随机抽样（带权重）
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(
        len(scaled_ndvi),
        size=min(total_samples, len(scaled_ndvi)),
        replace=False,
        p=weights
    )
    balanced_ndvi = valid_ndvi[sample_indices]

    # 计算Otsu阈值
    threshold = threshold_otsu(balanced_ndvi)
    print(f"Optimized threshold: {threshold:.4f}")

    # 生成分类结果
    vegetation_mask = np.where(ndvi < threshold, 1, 2).astype(np.uint8)
    valid_mask = ~np.isnan(ndvi)
    vegetation_valid = vegetation_mask[valid_mask]
    vegetation_pixels = np.count_nonzero(vegetation_valid == 2)
    print(f"植被覆盖率:      {vegetation_pixels / vegetation_valid.size:.2%}")

    # 保存结果
    profile.update({
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0,
        'count': 1
    })

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(vegetation_mask, 1)

    print("\n")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', required=True, help='Input NDVI TIFF file')
    # parser.add_argument('-o', '--output', required=True, help='Output classification TIFF')
    # parser.add_argument('-n', '--samples', type=int, default=10000,
    #                     help='Total samples for balancing (default: 10000)')
    # parser.add_argument('-p', '--precision', type=int, default=4,
    #                     help='NDVI discretization precision (default: 4)')
    # parser.add_argument('-r', '--seed', type=int,
    #                     help='Random seed for reproducibility')
    # args = parser.parse_args()

    input_tif = "D:\\Hydrogeology\\QGIS\\VegetationCoverage\\raw_ndvi.tif"  # 输入文件路径
    output_tif = "D:\\Hydrogeology\\QGIS\\VegetationCoverage\\otsu_ndvi.tif"  # 输出文件路径

    #for i in range(1, 21):
    #    total_samples = i * 1000
    vegetation_segmentation(
        input_tif,
        output_tif,
        total_samples=20000,
        precision=4,
        random_seed=1
    )