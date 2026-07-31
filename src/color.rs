use crate::perturbation::{IterField, BAILOUT, GLITCH_MARKER};

/// 生成 cyclic 调色板（余弦调色板，首尾相接无跳变）。
pub fn build_palette(size: usize) -> Vec<[u8; 3]> {
    let a = [0.5, 0.5, 0.5];
    let b = [0.5, 0.5, 0.5];
    let c = [1.0, 1.0, 1.0];
    let d = [0.00, 0.15, 0.30];
    let tau = 2.0 * std::f64::consts::PI;

    (0..size)
        .map(|i| {
            let t = i as f64 / size as f64;
            let mut out = [0u8; 3];
            for k in 0..3 {
                let v = a[k] + b[k] * (tau * (c[k] * t + d[k])).cos();
                out[k] = (v.clamp(0.0, 1.0) * 255.0) as u8;
            }
            out
        })
        .collect()
}

/// smooth（连续）着色的归一化迭代数。
///
/// ν = n - log2( ln|z_n|² / ln(BAILOUT) )，消除整数迭代带来的色带。
#[inline(always)]
fn smooth_nu(iter: u32, mag_sq: f64) -> f64 {
    iter as f64 - (mag_sq.ln() / BAILOUT.ln()).log2()
}

#[inline(always)]
fn sample_palette(palette: &[[u8; 3]], pos: f64) -> [u8; 3] {
    let plen = palette.len() as f64;
    let mut p = pos % plen;
    if p < 0.0 {
        p += plen;
    }
    let i0 = (p as usize) % palette.len();
    let i1 = (i0 + 1) % palette.len();
    let f = p - p.floor();

    let c0 = &palette[i0];
    let c1 = &palette[i1];
    [
        (c0[0] as f64 + (c1[0] as f64 - c0[0] as f64) * f) as u8,
        (c0[1] as f64 + (c1[1] as f64 - c0[1] as f64) * f) as u8,
        (c0[2] as f64 + (c1[2] as f64 - c0[2] as f64) * f) as u8,
    ]
}

/// 将迭代场着色为 RGB 字节序列。
///
/// 集合内部（iter == max_iter）与未修正的 glitch 像素着黑色。
pub fn color_field(field: &IterField, max_iter: u32, density: f64, palette: &[[u8; 3]]) -> Vec<u8> {
    let n = field.iters.len();
    let mut rgb = vec![0u8; n * 3];

    for i in 0..n {
        let it = field.iters[i];
        let color = if it == GLITCH_MARKER || it >= max_iter || field.mag_sq[i] <= BAILOUT {
            [0u8, 0, 0]
        } else {
            let nu = smooth_nu(it, field.mag_sq[i]);
            sample_palette(palette, nu * density)
        };
        rgb[i * 3] = color[0];
        rgb[i * 3 + 1] = color[1];
        rgb[i * 3 + 2] = color[2];
    }

    rgb
}
