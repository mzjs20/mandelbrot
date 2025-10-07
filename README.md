# 曼德勃罗集生成器

一个用Rust编写的高性能曼德勃罗集分形图像生成器，支持并行计算和超高分辨率输出。

## 特点

- 利用多线程并行计算加速图像生成
- 实时显示生成进度
- 可配置的图像参数（分辨率、迭代次数、坐标范围等）
- 优化的PNG图像保存，支持压缩级别调整
- 自动检查宽高比以避免图像变形
- 内存高效的图像处理

## 安装

需要Rust开发环境（cargo），克隆仓库后执行：

```bash
cargo build --release
```

生成的可执行文件位于 `target/release/` 目录下。

## 使用方法

直接运行编译好的可执行文件：

```bash
./target/release/mandelbrot_renderer
```

程序会生成默认配置的曼德勃罗集图像，并保存为 `mandelbrot.png`（或配置中指定的文件名）。

## 配置选项

可以通过修改 `main()` 函数中的 `Config` 结构体来调整生成参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| width | 图像宽度（像素） | 1920 |
| height | 图像高度（像素） | 1080 |
| x_min, x_max | 计算区域的X轴范围 | -2.0, 1.0 |
| y_min, y_max | 计算区域的Y轴范围 | -1.0, 1.0 |
| max_iter | 最大迭代次数（影响细节和计算时间） | 1000 |
| output_filename | 输出文件名 | "mandelbrot.png" |
| png_compression_level | PNG压缩级别 (0-9) | 2 |
| png_filter_type | PNG过滤类型 | NoFilter |

### 示例配置

生成超高分辨率图像：
```rust
let config = Config {
    width: 38400,
    height: 21600,
    x_min: -2.0,
    x_max: 1.0,
    y_min: -0.84375,
    y_max: 0.84375,
    max_iter: 500,
    output_filename: "mandelbrot_ultra_high_res.png",
    png_compression_level: 1,
    png_filter_type: FilterType::NoFilter,
};
```

放大特定区域：
```rust
let config = Config {
    width: 1920,
    height: 1080,
    x_min: -0.74887,
    x_max: -0.74882,
    y_min: 0.06515,
    y_max: 0.06515 + (0.74887-0.74882)*(1080.0/1920.0),
    max_iter: 2000,
    output_filename: "mandelbrot_zoom.png",
    ..Config::default()
};
```

## 依赖库

- `image` - 图像处理和PNG编码
- `rayon` - 并行计算支持
- `indicatif` - 进度条显示
- `num_cpus` - 获取CPU核心数

