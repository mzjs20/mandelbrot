//! 并行渲染驱动。
//!
//! - 浅缩放：朴素 SIMD（f64 足够精确）。
//! - 深缩放：扰动理论 + 级数近似 + Pauldelbrot glitch 检测 + 多参考点 rebase
//!   + 逐像素 MPFR 精确兜底。

use crate::color::{build_palette, color_field};
use crate::config::{Config, View};
use crate::perturbation::{
    compute_deltas_hp, naive_hp, naive_scalar, naive_simd, perturb_iterate, perturb_iterate_simd,
    IterField, ReferenceOrbit, SeriesApproximation, GLITCH_MARKER,
};
use image::{ImageBuffer, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rug::{Complex, Float};
use std::simd::prelude::*;

const PALETTE_SIZE: usize = 2048;
/// 级数近似项数
const SERIES_TERMS: usize = 8;

fn make_progress(total: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "[{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos}}/{{len}} {} ({{eta}})",
                label
            ))
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb
}

/// 渲染入口：返回 RGB 图像。
pub fn render(config: &Config, view: &View) -> Result<RgbImage, String> {
    println!("分辨率: {}x{}", config.width, config.height);
    println!("迭代次数: {}", config.max_iter);
    println!("缩放倍率: {:.3e}", view.zoom);

    let field = if view.deep {
        println!("模式: 扰动理论深缩放 (精度 {} bits)", view.precision);
        render_deep(config, view)
    } else {
        println!("模式: 朴素 SIMD");
        render_naive(config, view)
    };

    let palette = build_palette(PALETTE_SIZE);
    let rgb = color_field(&field, config.max_iter, config.color_density(), &palette);

    ImageBuffer::from_raw(config.width, config.height, rgb)
        .ok_or_else(|| "无法构建图像缓冲区".to_string())
}

// ── 浅缩放：朴素 SIMD ───────────────────────────────────────────────────────

fn render_naive(config: &Config, view: &View) -> IterField {
    let w = config.width as usize;
    let h = config.height as usize;
    let max_iter = config.max_iter;
    let mut field = IterField::new(w * h);

    let x_min = view.center_re_f64() - view.width / 2.0;
    let y_min = view.center_im_f64() - view.height / 2.0;
    let pw = view.width / w as f64;
    let ph = view.height / h as f64;

    let pb = make_progress(h as u64, "行");

    field
        .iters
        .par_chunks_mut(w)
        .zip(field.mag_sq.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (it_row, mag_row))| {
            fill_naive_row(y, w, x_min, y_min, pw, ph, max_iter, it_row, mag_row);
            pb.inc(1);
        });
    pb.finish_and_clear();

    field
}

fn fill_naive_row(
    y: usize,
    w: usize,
    x_min: f64,
    y_min: f64,
    pw: f64,
    ph: f64,
    max_iter: u32,
    it_row: &mut [u32],
    mag_row: &mut [f64],
) {
    let c_im = y_min + y as f64 * ph;
    let simd_w = w / 4;
    for sx in 0..simd_w {
        let xb = sx * 4;
        let c_re = f64x4::from_array([
            x_min + xb as f64 * pw,
            x_min + (xb + 1) as f64 * pw,
            x_min + (xb + 2) as f64 * pw,
            x_min + (xb + 3) as f64 * pw,
        ]);
        let (iters, mags) = naive_simd(c_re, f64x4::splat(c_im), max_iter);
        it_row[xb..xb + 4].copy_from_slice(&iters);
        mag_row[xb..xb + 4].copy_from_slice(&mags);
    }
    for x in simd_w * 4..w {
        let c_re = x_min + x as f64 * pw;
        let (i, m) = naive_scalar(c_re, c_im, max_iter);
        it_row[x] = i;
        mag_row[x] = m;
    }
}

// ── 深缩放：扰动理论 ────────────────────────────────────────────────────────

/// 判断整个视野是否位于集合内部（主心形 cardioid 或周期-2 圆盘）。
///
/// 主心形与周期-2 圆盘都是凸集，故只需高精度检查四角：
/// 四角全在 cardioid 内 ⇒ 整帧在 cardioid 内；四角全在 bulb 内 ⇒ 整帧在 bulb 内。
/// 用 MPFR 计算避免 f64 在边界附近的舍入误判。
fn frame_inside_cardioid_or_bulb(view: &View) -> bool {
    let prec = view.precision;
    let hw = Float::with_val(prec, view.width * 0.5);
    let hh = Float::with_val(prec, view.height * 0.5);
    let cxs = [
        view.center_re.clone() - &hw,
        view.center_re.clone() + &hw,
    ];
    let cys = [
        view.center_im.clone() - &hh,
        view.center_im.clone() + &hh,
    ];

    let qtr = Float::with_val(prec, 0.25);
    let one16 = Float::with_val(prec, 0.0625);
    let one = Float::with_val(prec, 1.0);

    let mut all_cardioid = true;
    let mut all_bulb = true;
    for cx in &cxs {
        for cy in &cys {
            // 主心形：q = (c_re - 1/4)² + c_im²，q·(q + c_re - 1/4) ≤ c_im²/4
            let c_re_minus = Float::with_val(prec, cx - &qtr);
            let c_im_sq = Float::with_val(prec, cy.clone().square());
            let q = Float::with_val(prec, c_re_minus.square() + &c_im_sq);
            let q_plus = Float::with_val(prec, Float::with_val(prec, &q + cx) - &qtr);
            let lhs = Float::with_val(prec, &q * &q_plus);
            let rhs = Float::with_val(prec, &qtr * &c_im_sq);
            if lhs > rhs {
                all_cardioid = false;
            }
            // 周期-2 圆盘：(c_re + 1)² + c_im² ≤ 1/16
            let c_re_plus = Float::with_val(prec, cx + &one);
            let d = Float::with_val(prec, c_re_plus.square() + &c_im_sq);
            if d > one16 {
                all_bulb = false;
            }
            if !all_cardioid && !all_bulb {
                return false;
            }
        }
    }
    all_cardioid || all_bulb
}

fn render_deep(config: &Config, view: &View) -> IterField {
    let w = config.width as usize;
    let h = config.height as usize;
    let max_iter = config.max_iter;
    let prec = view.precision;

    // 0. 整帧在集合内部（主心形 / 周期-2 圆盘）时直接填 max_iter，
    //    跳过整个扰动 pass——避免"内部只有散点/白算几百万像素"。
    //    主心形与周期-2 圆盘都是凸集：四角全在同一个区域内 ⇒ 整帧在内部。
    if frame_inside_cardioid_or_bulb(view) {
        println!("视野全部位于集合内部（cardioid/bulb），跳过扰动迭代");
        let mut field = IterField::new(w * h);
        field.iters.fill(max_iter);
        return field;
    }

    // 1. 高精度参考轨道（视图中心）
    let c = Complex::with_val(prec, (view.center_re.clone(), view.center_im.clone()));
    let orbit = ReferenceOrbit::compute_hp(&c, max_iter);
    println!("参考轨道: 长度 {}，逃逸于 {:?}", orbit.len(), orbit.escaped_at);

    // 2. 高精度 δ（相对视图中心）
    let (dre, dim) = compute_deltas_hp(
        &view.center_re,
        &view.center_im,
        &view.center_re,
        &view.center_im,
        view.width,
        view.height,
        w as u32,
        h as u32,
    );

    // 3. 级数近似（可选）
    let delta_max = 0.5 * (view.width * view.width + view.height * view.height).sqrt();
    let series = if config.series_approx() {
        let s = SeriesApproximation::compute(&orbit, delta_max, SERIES_TERMS, max_iter);
        println!("级数近似: 跳过 {} / {} 迭代", s.skip_iterations(), max_iter);
        Some(s)
    } else {
        None
    };
    let skip = series.as_ref().map(|s| s.skip_iterations()).unwrap_or(0);

    // 4. 主渲染 pass
    let mut field = perturb_pass(
        &orbit,
        series.as_ref(),
        skip,
        &dre,
        &dim,
        w,
        h,
        max_iter,
    );

    let initial_glitches = field.glitch_count();
    println!("初次 glitch 像素: {} / {}", initial_glitches, w * h);

    // 4.5 原参考点标量 rebase 回填：SIMD 主 pass 标记的 GLITCH_MARKER 里，
    //     相当一部分只是"需要 rebase"（|z_pixel|<|ε|），标量 perturb_iterate
    //     内联 rebase 后即可消化，无需换新参考点。
    if initial_glitches > 0 {
        let glitch_idx: Vec<usize> = field
            .iters
            .iter()
            .enumerate()
            .filter(|(_, i)| **i == GLITCH_MARKER)
            .map(|(idx, _)| idx)
            .collect();
        rerender_glitches(&orbit, series.as_ref(), skip, &dre, &dim, &glitch_idx, &mut field, w, max_iter);
        let after = field.glitch_count();
        if after != initial_glitches {
            println!("标量 rebase 回填: glitch {} -> {}", initial_glitches, after);
        }
    }

    // 5. 多参考点 rebase 修正
    if field.glitch_count() > 0 {
        rebase_loop(config, view, &dre, &dim, &mut field, w, h, max_iter);
        println!("rebase 后残余 glitch 像素: {}", field.glitch_count());
    }

    // 6. 逐像素 MPFR 精确兜底（保证零伪影）
    let residual = field.glitch_count();
    if residual > 0 {
        mpfr_fallback(view, &dre, &dim, &mut field, w, max_iter);
        println!("MPFR 兜底修正 {} 个像素", residual);
    }

    field
}

#[allow(clippy::too_many_arguments)]
fn perturb_pass(
    orbit: &ReferenceOrbit,
    series: Option<&SeriesApproximation>,
    skip: u32,
    dre: &[f64],
    dim: &[f64],
    w: usize,
    h: usize,
    max_iter: u32,
) -> IterField {
    let mut field = IterField::new(w * h);
    let pb = make_progress(h as u64, "行");

    field
        .iters
        .par_chunks_mut(w)
        .zip(field.mag_sq.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (it_row, mag_row))| {
            let d_im = dim[y];
            let simd_w = w / 4;
            for sx in 0..simd_w {
                let xb = sx * 4;
                let dr = f64x4::from_array([dre[xb], dre[xb + 1], dre[xb + 2], dre[xb + 3]]);
                let di = f64x4::splat(d_im);
                let (er, ei) = match series {
                    Some(s) if skip > 0 => s.evaluate_simd(dr, di),
                    _ => (f64x4::splat(0.0), f64x4::splat(0.0)),
                };
                let (iters, mags) = perturb_iterate_simd(orbit, dr, di, max_iter, er, ei, skip);
                it_row[xb..xb + 4].copy_from_slice(&iters);
                mag_row[xb..xb + 4].copy_from_slice(&mags);
            }
            for x in simd_w * 4..w {
                let dr = dre[x];
                let (er, ei) = match series {
                    Some(s) if skip > 0 => s.evaluate(dr, d_im),
                    _ => (0.0, 0.0),
                };
                let (i, m) = perturb_iterate(orbit, dr, d_im, max_iter, er, ei, skip);
                it_row[x] = i;
                mag_row[x] = m;
            }
            pb.inc(1);
        });
    pb.finish_and_clear();

    field
}

fn rebase_loop(
    config: &Config,
    view: &View,
    base_dre: &[f64],
    base_dim: &[f64],
    field: &mut IterField,
    w: usize,
    h: usize,
    max_iter: u32,
) {
    let max_refs = config.max_reference_points();
    let total = w * h;
    let tol_count = (total as f64 * 0.0005).ceil() as usize; // 0.05%
    let prec = view.precision;
    let delta_max = 0.5 * (view.width * view.width + view.height * view.height).sqrt();

    let mut refs_used = 1usize;
    while field.glitch_count() > tol_count && refs_used < max_refs {
        let glitch_idx: Vec<usize> = field
            .iters
            .iter()
            .enumerate()
            .filter(|(_, i)| **i == GLITCH_MARKER)
            .map(|(idx, _)| idx)
            .collect();
        if glitch_idx.is_empty() {
            break;
        }

        // 选取一个 glitch 像素作为新参考点（取中位，落在 glitch 密集区）
        let pick = glitch_idx[glitch_idx.len() / 2];
        let px = pick % w;
        let py = pick / w;
        let mut new_ref_re = view.center_re.clone();
        new_ref_re += base_dre[px];
        let mut new_ref_im = view.center_im.clone();
        new_ref_im += base_dim[py];

        let new_c = Complex::with_val(prec, (new_ref_re.clone(), new_ref_im.clone()));
        let new_orbit = ReferenceOrbit::compute_hp(&new_c, max_iter);
        let (dre, dim) = compute_deltas_hp(
            &new_ref_re,
            &new_ref_im,
            &view.center_re,
            &view.center_im,
            view.width,
            view.height,
            w as u32,
            h as u32,
        );
        let new_series = if config.series_approx() {
            Some(SeriesApproximation::compute(&new_orbit, delta_max, SERIES_TERMS, max_iter))
        } else {
            None
        };
        let new_skip = new_series.as_ref().map(|s| s.skip_iterations()).unwrap_or(0);

        let before = field.glitch_count();
        rerender_glitches(
            &new_orbit,
            new_series.as_ref(),
            new_skip,
            &dre,
            &dim,
            &glitch_idx,
            field,
            w,
            max_iter,
        );
        refs_used += 1;
        println!(
            "参考点 #{} @ 像素({},{}): glitch {} -> {}",
            refs_used,
            px,
            py,
            before,
            field.glitch_count()
        );
    }
}

fn rerender_glitches(
    orbit: &ReferenceOrbit,
    series: Option<&SeriesApproximation>,
    skip: u32,
    dre: &[f64],
    dim: &[f64],
    glitch_idx: &[usize],
    field: &mut IterField,
    w: usize,
    max_iter: u32,
) {
    for &g in glitch_idx {
        let dr = dre[g % w];
        let di = dim[g / w];
        let (er, ei) = match series {
            Some(s) if skip > 0 => s.evaluate(dr, di),
            _ => (0.0, 0.0),
        };
        // 标量版支持内联 Zhuoran rebase，能消化 SIMD 标记的"需 rebase"像素
        let (iters, mags) = perturb_iterate(orbit, dr, di, max_iter, er, ei, skip);
        if iters != GLITCH_MARKER {
            field.iters[g] = iters;
            field.mag_sq[g] = mags;
        }
    }
}

fn mpfr_fallback(
    view: &View,
    base_dre: &[f64],
    base_dim: &[f64],
    field: &mut IterField,
    w: usize,
    max_iter: u32,
) {
    let prec = view.precision;
    for i in 0..field.iters.len() {
        if field.iters[i] == GLITCH_MARKER {
            let x = i % w;
            let y = i / w;
            let mut c_re = view.center_re.clone();
            c_re += base_dre[x];
            let mut c_im = view.center_im.clone();
            c_im += base_dim[y];
            let c_re = Float::with_val(prec, c_re);
            let c_im = Float::with_val(prec, c_im);
            let (it, mag) = naive_hp(&c_re, &c_im, max_iter);
            field.iters[i] = it;
            field.mag_sq[i] = mag;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perturbation::{compute_deltas_hp, naive_hp};
    use rug::Float;

    /// 在 1e14 深缩放下，对比完整渲染管线与逐像素 MPFR 真值。
    fn check_deep(series: bool) {
        let json = format!(
            r#"{{
                "width": 24, "height": 24, "max_iter": 600,
                "output_filename": "test_deep.png",
                "png_compression_level": 0, "png_filter_type": "none",
                "center_re": "-0.744047327885618596691631635069778759229",
                "center_im": "0.1098916491492589420042574693894010907545",
                "zoom": "1e14",
                "series_approx": {}
            }}"#,
            series
        );
        let config: Config = serde_json::from_str(&json).unwrap();
        let view = config.resolve_view().unwrap();
        assert!(view.deep, "应走深缩放路径");

        let field = render_deep(&config, &view);
        assert_eq!(field.glitch_count(), 0, "渲染后不应残留 glitch 像素");

        let w = config.width as usize;
        let h = config.height as usize;
        let (dre, dim) = compute_deltas_hp(
            &view.center_re,
            &view.center_im,
            &view.center_re,
            &view.center_im,
            view.width,
            view.height,
            config.width,
            config.height,
        );
        let prec = view.precision;
        let mut max_diff = 0i32;
        for y in 0..h {
            for x in 0..w {
                let mut c_re = view.center_re.clone();
                c_re += dre[x];
                let mut c_im = view.center_im.clone();
                c_im += dim[y];
                let c_re = Float::with_val(prec, c_re);
                let c_im = Float::with_val(prec, c_im);
                let (gt, _) = naive_hp(&c_re, &c_im, config.max_iter);
                let got = field.iters[y * w + x];
                let diff = (got as i32 - gt as i32).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(
            max_diff <= 2,
            "series={}: 扰动法与 MPFR 真值最大迭代差 {} 超过容差",
            series,
            max_diff
        );
    }

    #[test]
    fn deep_zoom_matches_mpfr_no_series() {
        check_deep(false);
    }

    #[test]
    fn deep_zoom_matches_mpfr_with_series() {
        check_deep(true);
    }

    /// 回归测试：1e18 深缩放图像必须有丰富细节（大量不同迭代值），
    /// 防止参考轨道在 f64 中错误逃逸导致整图退化成纯色。
    #[test]
    fn deep_zoom_image_has_detail() {
        let json = r#"{
            "width": 64, "height": 64, "max_iter": 1500,
            "output_filename": "test_detail.png",
            "png_compression_level": 0, "png_filter_type": "none",
            "center_re": "-0.744047327885618596691631635069778759229",
            "center_im": "0.1098916491492589420042574693894010907545",
            "zoom": "1e18",
            "series_approx": false
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let view = config.resolve_view().unwrap();
        let field = render_deep(&config, &view);

        let distinct: std::collections::HashSet<u32> = field.iters.iter().copied().collect();
        assert_eq!(field.glitch_count(), 0, "不应残留 glitch");
        assert!(
            distinct.len() >= 50,
            "1e18 图像细节不足：仅 {} 种迭代值（疑似退化成纯色）",
            distinct.len()
        );
    }

    /// cardioid/bulb 整帧跳过：深缩放的视野完全落在主心形内部时，
    /// 应直接填 max_iter 返回（跳过扰动 pass），且结果与 MPFR 真值一致。
    #[test]
    fn deep_zoom_inside_cardioid_skips_pass() {
        let json = r#"{
            "width": 32, "height": 32, "max_iter": 800,
            "output_filename": "test_inside.png",
            "png_compression_level": 0, "png_filter_type": "none",
            "center_re": "-0.1", "center_im": "0.0",
            "zoom": "1e12",
            "series_approx": false
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let view = config.resolve_view().unwrap();
        assert!(view.deep, "应走深缩放路径");

        let field = render_deep(&config, &view);
        // 整帧都应判为集合内部：全部像素迭代满 max_iter
        assert!(field.iters.iter().all(|&i| i == config.max_iter), "cardioid 内部应全为 max_iter");
        assert_eq!(field.glitch_count(), 0);

        // 与逐像素 MPFR 真值对比（-0.1+0i 深在主心形内部，必不逃逸）
        let prec = view.precision;
        let half = Float::with_val(prec, 0.5);
        let fw = Float::with_val(prec, 32.0);
        let fh = Float::with_val(prec, 32.0);
        let vw = Float::with_val(prec, view.width);
        let vh = Float::with_val(prec, view.height);
        let mut max_diff = 0i32;
        for y in 0..32u32 {
            for x in 0..32u32 {
                // δ = (x/32 - 0.5)·view_w，全程 MPFR 避免 f64 下溢
                let mut dx = Float::with_val(prec, x);
                dx /= &fw;
                dx -= &half;
                dx *= &vw;
                let mut dy = Float::with_val(prec, y);
                dy /= &fh;
                dy -= &half;
                dy *= &vh;
                let c_re = Float::with_val(prec, &view.center_re + &dx);
                let c_im = Float::with_val(prec, &view.center_im + &dy);
                let (gt, _) = naive_hp(&c_re, &c_im, config.max_iter);
                let got = field.iters[(y * 32 + x) as usize];
                let diff = (got as i32 - gt as i32).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(max_diff <= 2, "cardioid 内部与 MPFR 最大迭代差 {} 超容差", max_diff);
    }
}
