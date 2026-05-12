use image::{
    codecs::png::{PngEncoder, CompressionType, FilterType},
    ImageBuffer, Rgb, ExtendedColorType, ImageEncoder, RgbImage,
};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};

/// 配置参数结构体
#[derive(Debug, Deserialize)]
struct Config {
    width: u32,
    height: u32,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    max_iter: u32,
    output_filename: String,
    png_compression_level: u32,
    png_filter_type: String,
}

impl Config {
    /// 从文件加载配置
    fn from_file(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("无法打开配置文件: {}", e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("无法读取配置文件: {}", e))?;
        serde_json::from_str(&contents).map_err(|e| format!("配置文件格式错误: {}", e))
    }

    /// 将字符串转换为FilterType
    fn get_filter_type(&self) -> FilterType {
        match self.png_filter_type.to_lowercase().as_str() {
            "none" | "nofilter" => FilterType::NoFilter,
            "sub" => FilterType::Sub,
            "up" => FilterType::Up,
            "avg" => FilterType::Avg,
            "paeth" => FilterType::Paeth,
            _ => FilterType::NoFilter,
        }
    }
}

/// 优化后的曼德勃罗集计算（主心形/周期2检测 + 周期性检测）
#[inline(always)]
fn mandelbrot(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    // 1. 主心形检测 - 直接判定是否在主心形内，跳过大量无用迭代
    let q = (c_re - 0.25).powi(2) + c_im.powi(2);
    if q * (q + (c_re - 0.25)) <= 0.25 * c_im.powi(2) {
        return max_iter;
    }

    // 2. 周期2圆盘检测 - 直接判定是否在左侧圆盘内
    if (c_re + 1.0).powi(2) + c_im.powi(2) <= 0.0625 {
        return max_iter;
    }

    let mut z_re = 0.0;
    let mut z_im = 0.0;
    let mut z_re_old = 0.0;
    let mut z_im_old = 0.0;
    let mut period = 0;

    for i in 0..max_iter {
        let z_re_squared = z_re * z_re;
        let z_im_squared = z_im * z_im;

        if z_re_squared + z_im_squared > 4.0 {
            return i;
        }

        let z_im_new = 2.0 * z_re * z_im + c_im;
        z_re = z_re_squared - z_im_squared + c_re;
        z_im = z_im_new;

        // 3. 周期性检测 - 如果陷入循环，提前退出
        if z_re == z_re_old && z_im == z_im_old {
            return max_iter;
        }

        period += 1;
        if period > 20 {
            period = 0;
            z_re_old = z_re;
            z_im_old = z_im;
        }
    }

    max_iter
}

/// 将迭代次数转换为RGB颜色
#[inline(always)]
fn iter_to_color(iter: u32, max_iter: u32) -> Rgb<u8> {
    if iter == max_iter {
        Rgb([0, 0, 0])
    } else {
        let t = iter as f64 / max_iter as f64;
        let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u8;
        let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8;
        let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8;
        Rgb([r, g, b])
    }
}

/// 极致优化：查找表 (LUT) + 粗粒度无锁并行 + 批量进度刷新
fn generate_mandelbrot(config: &Config) -> RgbImage {
    let (width, height) = (config.width, config.height);
    let x_range = config.x_max - config.x_min;
    let y_range = config.y_max - config.y_min;
    let pixel_width = x_range / width as f64;
    let pixel_height = y_range / height as f64;
    let max_iter = config.max_iter;

    println!("开始绘制曼德勃罗集...");
    println!("分辨率: {}x{}", width, height);
    println!("迭代次数: {}", max_iter);

    let start_time = Instant::now();

    // 1. 预计算颜色查找表 (LUT)
    // 将数百万次浮点计算提前完成，热循环中直接查表，省去海量浮点运算
    let lut: Vec<[u8; 3]> = (0..=max_iter)
        .map(|i| iter_to_color(i, max_iter).0)
        .collect();

    let pb = ProgressBar::new(height as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} 行 ({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );

    // 预先分配完整的像素内存（一次性分配，绝不中途再申请）
    let mut raw_pixels = vec![0u8; (width * height * 3) as usize];
    let processed_rows = AtomicUsize::new(0);

    // 2. 粗粒度切块：每 10 行作为一个任务块
    // 减少 Rayon 任务调度开销，模拟手动分块的优点，同时保持负载均衡
    let chunk_rows: usize = 10;
    let chunk_size = width as usize * 3 * chunk_rows;

    raw_pixels
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk_slice)| {
            let start_y = chunk_idx * chunk_rows;
            // 计算当前块的实际行数（最后一块可能不足 10 行）
            let actual_rows = chunk_slice.len() / (width as usize * 3);

            for y_offset in 0..actual_rows {
                let y = start_y + y_offset;
                let c_im = config.y_min + y as f64 * pixel_height;
                let row_start = y_offset * (width as usize * 3);

                for x in 0..width as usize {
                    let c_re = config.x_min + x as f64 * pixel_width;

                    // 核心计算
                    let iter = mandelbrot(c_re, c_im, max_iter) as usize;
                    let color = &lut[iter]; // 直接查表，零浮点开销

                    let idx = row_start + x * 3;
                    chunk_slice[idx] = color[0];
                    chunk_slice[idx + 1] = color[1];
                    chunk_slice[idx + 2] = color[2];
                }
            }

            // 3. 批量刷新进度条：每处理完一个块才更新，减少终端 IO 阻塞
            let prev = processed_rows.fetch_add(actual_rows, Ordering::Relaxed);
            // 每累积 50 行才真正刷新一次终端显示
            if (prev / 50) < ((prev + actual_rows) / 50) {
                pb.set_position((prev + actual_rows) as u64);
            }
        });

    pb.finish_with_message("绘制完成!");

    let image = ImageBuffer::from_raw(width, height, raw_pixels).unwrap();

    let duration = start_time.elapsed();
    println!("所有线程完成! 绘制耗时: {:?}", duration);

    image
}

/// 图像保存函数
fn save_image_with_progress(
    image: &RgbImage,
    filename: &str,
    compression: CompressionType,
    filter: FilterType,
) -> Result<(), String> {
    let (width, height) = (image.width(), image.height());
    println!("\n开始保存图像 ({}x{} pixels)...", width, height);

    let start_time = Instant::now();
    let file = File::create(filename).map_err(|e| format!("无法创建文件: {}", e))?;
    let encoder = PngEncoder::new_with_quality(file, compression, filter);
    let pixels = image.as_raw();

    encoder
        .write_image(pixels, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("编码失败: {}", e))?;

    let duration = start_time.elapsed();
    println!("图像保存完成! 保存耗时: {:?}", duration);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "config.json"
    };

    let config = Config::from_file(config_path).expect("加载配置文件失败");

    let aspect_ratio = config.width as f64 / config.height as f64;
    let region_ratio = (config.x_max - config.x_min) / (config.y_max - config.y_min);
    if (aspect_ratio - region_ratio).abs() > 0.001 {
        eprintln!("警告: 图像宽高比与区域宽高比不匹配，可能导致图像变形");
    }

    let image = generate_mandelbrot(&config);

    let compression = match config.png_compression_level {
        0 => CompressionType::Fast,
        1..=8 => CompressionType::Default,
        9 => CompressionType::Best,
        _ => CompressionType::Default,
    };
    let filter_type = config.get_filter_type();

    if let Err(e) = save_image_with_progress(&image, &config.output_filename, compression, filter_type) {
        eprintln!("保存图像失败: {}", e);
    } else {
        println!("图像已保存至: {}", config.output_filename);
    }
}
