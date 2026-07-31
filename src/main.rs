#![feature(portable_simd)]
#![allow(clippy::too_many_arguments)]

mod color;
mod config;
mod perturbation;
mod render;

use config::Config;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::ImageEncoder;
use std::fs::File;
use std::time::Instant;

fn map_compression(level: u32) -> CompressionType {
    match level {
        0 => CompressionType::Fast,
        1..=8 => CompressionType::Default,
        9 => CompressionType::Best,
        _ => CompressionType::Default,
    }
}

fn parse_filter_type(s: &str) -> FilterType {
    match s.to_lowercase().as_str() {
        "none" | "nofilter" => FilterType::NoFilter,
        "sub" => FilterType::Sub,
        "up" => FilterType::Up,
        "avg" => FilterType::Avg,
        "paeth" => FilterType::Paeth,
        _ => FilterType::NoFilter,
    }
}

fn save_image(image: &image::RgbImage, filename: &str, compression: CompressionType, filter: FilterType) -> Result<(), String> {
    let (width, height) = (image.width(), image.height());
    println!("\n开始保存图像 ({}x{} pixels)...", width, height);

    let start = Instant::now();
    let file = File::create(filename).map_err(|e| format!("无法创建文件: {}", e))?;
    let encoder = PngEncoder::new_with_quality(file, compression, filter);
    encoder
        .write_image(image.as_raw(), width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("编码失败: {}", e))?;

    println!("图像保存完成! 保存耗时: {:?}", start.elapsed());
    Ok(())
}

/// 解析命令行参数中的配置文件路径。
///
/// 支持多种写法（后到先得）：
/// - `prog config.json`（位置参数）
/// - `prog -c config.json` / `prog --config config.json`
/// - `prog --config=config.json`
/// - `prog --config_deep.json`（`--` 前缀 + `.json` 结尾，去掉前缀即文件名）
/// - `prog -- config.json`
///
/// 缺省为 `config.json`。
fn config_path_from_args() -> String {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--" || a == "-c" || a == "--config" {
            if let Some(v) = args.get(i + 1) {
                return v.clone();
            }
        } else if let Some(v) = a.strip_prefix("--config=") {
            if !v.is_empty() {
                return v.to_string();
            }
        } else if a.starts_with('-') && a.ends_with(".json") {
            // 兼容 `--config_deep.json` 这类写法：去掉前导 `-` 即得文件名
            return a.trim_start_matches('-').to_string();
        } else if a.starts_with('-') {
            // 未知 flag，忽略
        } else {
            positional = Some(a.clone());
        }
        i += 1;
    }
    positional.unwrap_or_else(|| "config.json".to_string())
}

fn main() {
    let config_path = config_path_from_args();

    let config = Config::from_file(&config_path).expect("加载配置文件失败");
    let view = config.resolve_view().expect("解析视图失败");

    let start = Instant::now();
    let image = render::render(&config, &view).expect("渲染失败");
    println!("渲染总耗时: {:?}", start.elapsed());

    let compression = map_compression(config.png_compression_level);
    let filter = parse_filter_type(&config.png_filter_type);

    if let Err(e) = save_image(&image, &config.output_filename, compression, filter) {
        eprintln!("保存图像失败: {}", e);
    } else {
        println!("图像已保存至: {}", config.output_filename);
    }
}
