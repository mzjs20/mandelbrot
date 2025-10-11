use image::{
    codecs::png::{PngEncoder, CompressionType, FilterType},
    ImageBuffer, RgbImage, Rgb, ExtendedColorType, ImageEncoder,
};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant, Duration};
use num_cpus;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// 配置参数结构体，支持从JSON反序列化
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
        // 读取文件内容
        let mut file = File::open(path)
            .map_err(|e| format!("无法打开配置文件: {}", e))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("无法读取配置文件: {}", e))?;

        // 解析JSON
        serde_json::from_str(&contents)
            .map_err(|e| format!("配置文件格式错误: {}", e))
    }

    /// 将字符串转换为FilterType
    fn get_filter_type(&self) -> FilterType {
        match self.png_filter_type.to_lowercase().as_str() {
            "none" | "nofilter" => FilterType::NoFilter,
            "sub" => FilterType::Sub,
            "up" => FilterType::Up,
            "avg" => FilterType::Avg,
            "paeth" => FilterType::Paeth,
            _ => {
                eprintln!("未知的过滤类型 '{}'，使用默认值 NoFilter", self.png_filter_type);
                FilterType::NoFilter
            }
        }
    }
}

/// 计算曼德勃罗集的迭代次数
fn mandelbrot(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let mut z_re = 0.0;
    let mut z_im = 0.0;

    for i in 0..max_iter {
        let z_re_squared = z_re * z_re;
        let z_im_squared = z_im * z_im;

        if z_re_squared + z_im_squared > 4.0 {
            return i;
        }

        let z_im_new = 2.0 * z_re * z_im + c_im;
        z_re = z_re_squared - z_im_squared + c_re;
        z_im = z_im_new;
    }

    max_iter
}

/// 将迭代次数转换为RGB颜色
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

/// 生成曼德勃罗集图像
fn generate_mandelbrot(config: &Config) -> RgbImage {
    let (width, height) = (config.width, config.height);
    let image = Arc::new(Mutex::new(ImageBuffer::new(width, height)));

    let x_range = config.x_max - config.x_min;
    let y_range = config.y_max - config.y_min;
    let pixel_width = x_range / width as f64;
    let pixel_height = y_range / height as f64;

    println!("开始绘制曼德勃罗集...");
    println!("分辨率: {}x{}", width, height);
    println!("迭代次数: {}", config.max_iter);
    println!("绘制区域: x[{}, {}], y[{}, {}]",
             config.x_min, config.x_max,
             config.y_min, config.y_max);

    let start_time = Instant::now();
    let num_cores = num_cpus::get() as u32;
    let chunk_height = height / num_cores;
    println!("使用 {} 个线程并行处理，每个线程处理 {} 行\n", num_cores, chunk_height);

    // 创建多进度条管理器
    let multi_progress = MultiProgress::new();
    let pbs = Arc::new(Mutex::new(Vec::with_capacity(num_cores as usize)));

    // 先创建所有进度条
    for chunk in 0..num_cores {
        let total_lines = if chunk == num_cores - 1 {
            height - chunk * chunk_height
        } else {
            chunk_height
        };

        let pb = multi_progress.add(ProgressBar::new(total_lines as u64));
        let style = ProgressStyle::default_bar()
            .template(&format!("线程 {}: [{{bar:40}}] {{pos}}/{{len}} ({{percent}}%)", chunk + 1))
            .unwrap()
            .progress_chars("#>-");
        pb.set_style(style);
        pb.set_position(0);

        pbs.lock().unwrap().push(pb);
    }

    // 并行处理图像
    (0..num_cores).into_par_iter().for_each(|chunk| {
        let start_y = chunk * chunk_height;
        let end_y = if chunk == num_cores - 1 {
            height
        } else {
            start_y + chunk_height
        };

        // 获取当前线程对应的进度条
        let pb = {
            let pbs = pbs.lock().unwrap();
            pbs[chunk as usize].clone()
        };

        // 处理当前块的每一行
        for y in start_y..end_y {
            let c_im = config.y_min + y as f64 * pixel_height;

            for x in 0..width {
                let c_re = config.x_min + x as f64 * pixel_width;
                let iter = mandelbrot(c_re, c_im, config.max_iter);
                let color = iter_to_color(iter, config.max_iter);

                let mut img = image.lock().unwrap();
                img.put_pixel(x, y, color);
            }

            // 每处理10行更新一次进度条
            if (y - start_y) % 10 == 0 || y == end_y - 1 {
                pb.set_position((y - start_y + 1) as u64);
            }
        }

        pb.finish();
    });

    // 等待所有进度条完成
    let _ = multi_progress;

    let duration = start_time.elapsed();
    println!("\n所有线程完成! 绘制耗时: {:?}", duration);

    Arc::try_unwrap(image).unwrap().into_inner().unwrap()
}

/// 带实时进度的图像保存函数
fn save_image_with_progress(image: &RgbImage, filename: &str, compression: CompressionType, filter: FilterType) -> Result<(), String> {
    let (width, height) = (image.width(), image.height());
    let total_steps = 50; // 进度条总步数，用于平滑显示

    println!("\n开始保存图像 ({}x{} pixels)...", width, height);

    // 创建保存进度条
    let pb = ProgressBar::new(total_steps);
    pb.set_style(ProgressStyle::default_bar()
        .template("保存进度: [{bar:40}] {pos}/{len} ({percent}%)")
        .unwrap()
        .progress_chars("#>-"));

    // 打开文件
    let file = File::create(filename)
        .map_err(|e| format!("无法创建文件: {}", e))?;

    // 配置PNG编码器
    let encoder = PngEncoder::new_with_quality(file, compression, filter);

    // 获取原始像素数据
    let pixels = image.as_raw();

    // 启动进度更新线程（通过计时估算进度）
    let pb_clone = pb.clone();
    let handle = thread::spawn(move || {
        let start_time = Instant::now();
        // 预计最长保存时间（根据经验值设置）
        let max_expected_time = Duration::from_secs(30);
        let interval = Duration::from_millis(200);

        while pb_clone.position() < total_steps {
            thread::sleep(interval);
            let elapsed = start_time.elapsed();
            if elapsed >= max_expected_time {
                break; // 超过预期时间则停止更新
            }

            // 计算进度百分比
            let progress = (elapsed.as_secs_f64() / max_expected_time.as_secs_f64())
                .min(1.0) * total_steps as f64;
            pb_clone.set_position(progress as u64);
        }
    });

    // 执行编码
    let start_time = Instant::now();
    encoder.write_image(
        pixels,
        width,
        height,
        ExtendedColorType::Rgb8,
    ).map_err(|e| format!("编码失败: {}", e))?;

    // 确保进度条显示100%
    pb.set_position(total_steps);
    handle.join().unwrap_or(()); // 忽略线程 join 错误
    pb.finish();

    let duration = start_time.elapsed();
    println!("图像保存完成! 保存耗时: {:?}", duration);

    Ok(())
}

fn main() {
    // 从命令行参数获取配置文件路径，默认为"config.json"
    let args: Vec<String> = std::env::args().collect();
    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "config.json"
    };

    // 加载配置文件
    let config = Config::from_file(config_path)
        .expect("加载配置文件失败");

    let aspect_ratio = config.width as f64 / config.height as f64;
    let region_ratio = (config.x_max - config.x_min) / (config.y_max - config.y_min);
    println!("图像宽高比: {:.4}, 区域宽高比: {:.4}", aspect_ratio, region_ratio);
    if (aspect_ratio - region_ratio).abs() > 0.001 {
        eprintln!("警告: 图像宽高比与区域宽高比不匹配，可能导致图像变形");
    }

    let image = generate_mandelbrot(&config);

    // 转换压缩级别为CompressionType
    let compression = match config.png_compression_level {
        0 => CompressionType::Fast,
        1..=8 => CompressionType::Default,
        9 => CompressionType::Best,
        _ => CompressionType::Default,
    };

    // 获取过滤类型
    let filter_type = config.get_filter_type();

    // 使用带进度的保存函数
    if let Err(e) = save_image_with_progress(&image, &config.output_filename, compression, filter_type) {
        eprintln!("保存图像失败: {}", e);
    } else {
        println!("图像已保存至: {}", config.output_filename);
    }
}
