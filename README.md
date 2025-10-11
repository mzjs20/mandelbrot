# 曼德勃罗集生成器

一个用Rust编写的曼德勃罗集分形图像生成器，支持并行计算、JSON配置和实时进度展示。

## 特点

- 利用多线程并行计算加速图像生成（基于`rayon`库，自动适配CPU核心数）
- 精细化进度展示：多线程实时进度条+图像保存进度条，直观了解生成状态
- 支持JSON配置文件：通过外部配置文件灵活调整参数，无需修改代码
- 可配置的图像参数（分辨率、迭代次数、坐标范围等）
- 优化的PNG图像保存，支持压缩级别（0-9）和过滤类型调整
- 自动检查宽高比：提示图像与计算区域比例不匹配的潜在变形问题
- 内存高效的图像处理：分块并行处理，降低大尺寸图像内存占用

## 安装

### 前置要求
- Rust开发环境（推荐1.60.0及以上版本），可通过[rustup](https://rustup.rs/)安装
- 部分系统可能需要安装libpng开发库（如Ubuntu需安装`libpng-dev`，Fedora需安装`libpng-devel`）

### 编译步骤
克隆仓库后执行：
```bash
cargo build --release
```
生成的可执行文件位于 `target/release/` 目录下（Windows系统为`mandelbrot_renderer.exe`）。

## 使用方法

1. 准备配置文件（默认读取`config.json`，可通过命令行参数指定其他路径）
2. 运行编译好的可执行文件：
   ```bash
   # 使用默认配置文件 config.json
   ./target/release/mandelbrot_renderer
   
   # 指定自定义配置文件
   ./target/release/mandelbrot_renderer my_config.json
   
   # Windows系统
   .\target\release\mandelbrot_renderer.exe
   ```

程序会根据配置文件生成曼德勃罗集图像，并保存为指定文件名。

## 配置选项

配置通过JSON文件定义，支持以下参数：

| 参数 | 说明 | 默认值（示例） |
|------|------|--------|
| width | 图像宽度（像素） | 1920 |
| height | 图像高度（像素） | 1080 |
| x_min, x_max | 计算区域的X轴范围（复数平面实部） | -2.0, 1.0 |
| y_min, y_max | 计算区域的Y轴范围（复数平面虚部） | -1.0, 1.0 |
| max_iter | 最大迭代次数（值越大细节越丰富，计算时间越长） | 1000 |
| output_filename | 输出文件名（需包含.png扩展名） | "mandelbrot.png" |
| png_compression_level | PNG压缩级别 (0-9，0=最快，9=最小体积) | 2 |
| png_filter_type | PNG过滤类型（优化压缩效率） | "nofilter" |

### PNG过滤类型说明
`png_filter_type` 可接受的值：
- `"nofilter"` 或 `"none"`：无过滤（默认）
- `"sub"`：基于前一像素的差异过滤
- `"up"`：基于上一行像素的差异过滤
- `"avg"`：基于前一像素和上一行对应像素的平均值过滤
- `"paeth"`：基于Paeth预测器的过滤（适合复杂纹理）

### 示例配置文件

**超高分辨率图像（config_ultra_high_res.json）**：
```json
{
  "width": 38400,
  "height": 21600,
  "x_min": -2.0,
  "x_max": 1.0,
  "y_min": -0.84375,
  "y_max": 0.84375,
  "max_iter": 500,
  "output_filename": "mandelbrot_ultra_high_res.png",
  "png_compression_level": 1,
  "png_filter_type": "nofilter"
}
```

**放大特定细节区域（config_zoom.json）**：
```json
{
  "width": 1920,
  "height": 1080,
  "x_min": -0.74887,
  "x_max": -0.74882,
  "y_min": 0.06515,
  "y_max": 0.06515 + (0.74887-0.74882)*(1080.0/1920.0),
  "max_iter": 2000,
  "output_filename": "mandelbrot_zoom.png",
  "png_compression_level": 5,
  "png_filter_type": "paeth"
}
```

## 性能说明

- 计算速度取决于：图像分辨率（像素数量）、最大迭代次数、CPU核心数
- 4K分辨率（3840x2160）+1000次迭代：8核CPU约10-30秒
- 超高分辨率（38400x21600）：建议降低压缩级别（1-2），确保系统内存≥16GB

## 依赖库

- `image` - 图像处理和PNG编码
- `rayon` - 并行计算支持
- `indicatif` - 多线程进度条和保存进度展示
- `num_cpus` - 自动检测CPU核心数
- `serde` 和 `serde_json` - JSON配置文件解析

## 常见问题

- **配置文件错误**：检查JSON格式是否正确（可使用在线JSON验证工具）
- **编译失败**：确保安装libpng开发库，或更新Rust工具链：`rustup update`
- **内存不足**：降低图像分辨率或分多次生成局部区域
- **图像变形**：程序会提示宽高比不匹配，可调整`y_min`/`y_max`使 `(x_max-x_min)/(y_max-y_min) ≈ width/height`