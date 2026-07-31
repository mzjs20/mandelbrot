use rug::Float;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

/// 渲染配置。
///
/// 视图有两种指定方式：
/// 1. 深缩放（高精度）：给出 `center_re` / `center_im`（任意精度十进制字符串）+ `zoom`。
///    坐标用 MPFR 解析，可放大到 1e18 以上。
/// 2. 浅缩放（向后兼容）：给出 `x_min` / `x_max` / `y_min` / `y_max`（f64）。
///    受 f64 精度限制，实际缩放上限约 1e13。
#[derive(Debug, Deserialize)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub max_iter: u32,
    pub output_filename: String,
    pub png_compression_level: u32,
    pub png_filter_type: String,

    // 浅缩放边界（f64，可选）
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,

    // 深缩放高精度中心点 + 缩放倍率（字符串，可选）
    pub center_re: Option<String>,
    pub center_im: Option<String>,
    pub zoom: Option<String>,

    // 着色 / 性能选项
    /// smooth 着色密度（每迭代推进的调色板步数），默认 1.0
    pub color_density: Option<f64>,
    /// 是否启用级数近似加速深缩放（默认 true）
    pub series_approx: Option<bool>,
    /// 多参考点 rebase 的最大参考点数（默认 12）
    pub max_reference_points: Option<usize>,
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("无法打开配置文件: {}", e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("无法读取配置文件: {}", e))?;
        serde_json::from_str(&contents).map_err(|e| format!("配置文件格式错误: {}", e))
    }

    pub fn color_density(&self) -> f64 {
        self.color_density.unwrap_or(1.0)
    }

    pub fn series_approx(&self) -> bool {
        self.series_approx.unwrap_or(true)
    }

    pub fn max_reference_points(&self) -> usize {
        self.max_reference_points.unwrap_or(12)
    }

    /// 将配置解析为统一的视图描述。
    pub fn resolve_view(&self) -> Result<View, String> {
        let max_dim = self.width.max(self.height).max(1);

        if let (Some(re_s), Some(im_s)) = (&self.center_re, &self.center_im) {
            let zoom_f: f64 = self
                .zoom
                .as_deref()
                .unwrap_or("1")
                .parse()
                .map_err(|e| format!("zoom 解析失败: {}", e))?;
            if zoom_f <= 0.0 {
                return Err("zoom 必须为正数".to_string());
            }

            let precision = suggested_precision(zoom_f, max_dim);
            let center_re = parse_hp(re_s, precision)?;
            let center_im = parse_hp(im_s, precision)?;

            let view_width = 4.0 / zoom_f;
            let view_height = view_width * self.height as f64 / self.width as f64;

            let view = View {
                center_re,
                center_im,
                width: view_width,
                height: view_height,
                zoom: zoom_f,
                deep: true,
                precision,
            };
            view.warn_aspect(self.width, self.height);
            return Ok(view);
        }

        if let (Some(x_min), Some(x_max), Some(y_min), Some(y_max)) =
            (self.x_min, self.x_max, self.y_min, self.y_max)
        {
            let view_width = x_max - x_min;
            let view_height = y_max - y_min;
            if view_width <= 0.0 || view_height <= 0.0 {
                return Err("x_max 必须大于 x_min，y_max 必须大于 y_min".to_string());
            }
            let zoom_f = 4.0 / view_width;
            let precision = 64;
            let center_re = Float::with_val(precision, (x_min + x_max) * 0.5);
            let center_im = Float::with_val(precision, (y_min + y_max) * 0.5);

            let view = View {
                center_re,
                center_im,
                width: view_width,
                height: view_height,
                zoom: zoom_f,
                deep: false,
                precision,
            };
            view.warn_aspect(self.width, self.height);
            return Ok(view);
        }

        Err("配置必须提供 center_re/center_im/zoom（深缩放）或 x_min/x_max/y_min/y_max（浅缩放）".to_string())
    }
}

/// 统一视图描述。
pub struct View {
    /// 高精度中心实部
    pub center_re: Float,
    /// 高精度中心虚部
    pub center_im: Float,
    /// 复平面视图宽度
    pub width: f64,
    /// 复平面视图高度
    pub height: f64,
    /// 缩放倍率（zoom=1 时视图宽约 4）
    pub zoom: f64,
    /// 是否走扰动理论深缩放路径
    pub deep: bool,
    /// 参考轨道所需精度（比特）
    pub precision: u32,
}

impl View {
    pub fn center_re_f64(&self) -> f64 {
        self.center_re.to_f64()
    }
    pub fn center_im_f64(&self) -> f64 {
        self.center_im.to_f64()
    }

    fn warn_aspect(&self, width_px: u32, height_px: u32) {
        let aspect = width_px as f64 / height_px as f64;
        let region = self.width / self.height;
        if (aspect - region).abs() > 0.001 {
            eprintln!("警告: 图像宽高比与区域宽高比不匹配，可能导致图像变形");
        }
    }
}

/// 计算参考轨道所需精度。
///
/// `bits = ⌈log2(zoom)⌉ + 53(f64 尾数) + 16(保护位) + ⌈log2(max_dim)⌉`。
/// 保护位用于抵消 ε 迭代中的累积舍入误差；max_dim 项保证能分辨单个像素。
pub fn suggested_precision(zoom: f64, max_dim: u32) -> u32 {
    if zoom <= 1.0 {
        return 64;
    }
    let zoom_bits = zoom.log2().ceil().max(0.0) as u32;
    let dim_bits = (max_dim as f64).log2().ceil().max(0.0) as u32;
    (zoom_bits + 53 + 16 + dim_bits).max(64)
}

/// 将任意精度十进制字符串解析为指定精度的 Float。
fn parse_hp(s: &str, precision: u32) -> Result<Float, String> {
    let parsed = Float::parse_radix(s.trim(), 10).map_err(|e| format!("高精度坐标解析失败 {:?}: {}", s, e))?;
    Ok(Float::with_val(precision, parsed))
}
