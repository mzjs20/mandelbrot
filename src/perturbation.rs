//! 扰动理论深缩放引擎。
//!
//! 核心思想（K.I. Martin, "Superfractalthing Maths"）：
//! 用一个高精度参考轨道 `Z_n`（MPFR 计算一次），把每个像素写成 `c = C + δ`、
//! `z_n = Z_n + ε_n`，代入 `z_{n+1} = z_n² + c` 得
//!
//! ```text
//! ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ,   ε_0 = 0
//! ```
//!
//! ε 始终保持很小，可用 f64（甚至 SIMD）逐像素迭代；只有参考轨道需要高精度。
//! 这样既能放大到 1e18 以上，又能用 SIMD 加速。
//!
//! 正确性保障：
//! - 高精度 δ（MPFR）避免 `c_pixel - C` 的灾难性抵消；
//! - Pauldelbrot glitch 检测 + 多参考点 rebase（见 render.rs）；
//! - 残余 glitch 像素用逐像素 MPFR 精确兜底（[`naive_hp`]）。

use rug::{Complex, Float};
use std::simd::cmp::SimdPartialOrd;
use std::simd::prelude::*;

/// 逃逸半径平方（|z|² > BAILOUT 即逃逸）。取 16 以提高 smooth 着色精度。
pub const BAILOUT: f64 = 16.0;

/// glitch 像素标记值。
pub const GLITCH_MARKER: u32 = u32::MAX;

/// Pauldelbrot glitch 检测容忍度：当 |z_pixel|² < |Z_ref|²·TOL² 时判定 glitch。
const GLITCH_TOLERANCE: f64 = 1e-3;
const GLITCH_TOL_SQ: f64 = GLITCH_TOLERANCE * GLITCH_TOLERANCE;

/// 逐像素结果场（Structure-of-Arrays，便于 SIMD 写入）。
pub struct IterField {
    /// 逃逸迭代数；`max_iter` 表示集合内部；`GLITCH_MARKER` 表示需修正。
    pub iters: Vec<u32>,
    /// 逃逸时的 |z|²（用于 smooth 着色）；内部 / glitch 为 0。
    pub mag_sq: Vec<f64>,
}

impl IterField {
    pub fn new(n: usize) -> Self {
        IterField {
            iters: vec![0; n],
            mag_sq: vec![0.0; n],
        }
    }

    pub fn glitch_count(&self) -> usize {
        self.iters.iter().filter(|&&i| i == GLITCH_MARKER).count()
    }
}

/// 高精度参考轨道。
#[derive(Clone)]
pub struct ReferenceOrbit {
    z_re: Vec<f64>,
    z_im: Vec<f64>,
    /// 预计算的 glitch 阈值：|Z_n|²·TOL²
    tolerance_check: Vec<f64>,
    /// 参考点 c（f64 部分）
    c_re: f64,
    c_im: f64,
    /// 参考轨道逃逸迭代（若在 max_iter 内逃逸）
    pub escaped_at: Option<u32>,
}

impl ReferenceOrbit {
    /// 用高精度算术计算参考轨道。
    pub fn compute_hp(c: &Complex, max_iter: u32) -> Self {
        let prec = c.real().prec();
        let mut z = Complex::with_val(prec, (0, 0));
        let mut z_re = Vec::with_capacity(max_iter as usize + 1);
        let mut z_im = Vec::with_capacity(max_iter as usize + 1);
        let mut tolerance_check = Vec::with_capacity(max_iter as usize + 1);
        let mut escaped_at = None;

        let c_re_f64 = c.real().to_f64();
        let c_im_f64 = c.imag().to_f64();

        for i in 0..max_iter {
            let re = z.real().to_f64();
            let im = z.imag().to_f64();
            z_re.push(re);
            z_im.push(im);
            tolerance_check.push((re * re + im * im) * GLITCH_TOL_SQ);

            z *= z.clone();
            z += c.clone();

            let re2 = z.real().to_f64();
            let im2 = z.imag().to_f64();
            if re2 * re2 + im2 * im2 > BAILOUT {
                escaped_at = Some(i);
                z_re.push(re2);
                z_im.push(im2);
                tolerance_check.push((re2 * re2 + im2 * im2) * GLITCH_TOL_SQ);
                break;
            }
        }

        if escaped_at.is_none() {
            let re = z.real().to_f64();
            let im = z.imag().to_f64();
            z_re.push(re);
            z_im.push(im);
            tolerance_check.push((re * re + im * im) * GLITCH_TOL_SQ);
        }

        ReferenceOrbit {
            z_re,
            z_im,
            tolerance_check,
            c_re: c_re_f64,
            c_im: c_im_f64,
            escaped_at,
        }
    }

    pub fn len(&self) -> u32 {
        self.z_re.len() as u32
    }

    #[inline(always)]
    pub fn z(&self, n: u32) -> (f64, f64) {
        (self.z_re[n as usize], self.z_im[n as usize])
    }

    /// ε 迭代的有效上界（参考逃逸前 / max_iter 前）。
    #[inline(always)]
    fn len_bound(&self, max_iter: u32) -> u32 {
        if let Some(esc) = self.escaped_at {
            esc.min(max_iter)
        } else {
            max_iter.min(self.len() - 1)
        }
    }
}

/// 用高精度算术计算每列 / 每行像素相对参考点的偏移 δ = c_pixel - c_ref。
///
/// ```text
/// δ_re[x] = (center_re - ref_re) + (x/width  - 0.5)·view_w
/// δ_im[y] = (center_im - ref_im) + (y/height - 0.5)·view_h
/// ```
///
/// 全程 MPFR，最后舍入到 f64（δ 本身很小，f64 足以表示）。
/// 主渲染pass参考点即视图中心（shift=0）；rebase 时参考点为别的点。
pub fn compute_deltas_hp(
    ref_re: &Float,
    ref_im: &Float,
    center_re: &Float,
    center_im: &Float,
    view_w: f64,
    view_h: f64,
    img_w: u32,
    img_h: u32,
) -> (Vec<f64>, Vec<f64>) {
    let prec = center_re.prec();
    let half = Float::with_val(prec, 0.5);
    let fw = Float::with_val(prec, img_w);
    let fh = Float::with_val(prec, img_h);
    let vw = Float::with_val(prec, view_w);
    let vh = Float::with_val(prec, view_h);

    let mut shift_re = center_re.clone();
    shift_re -= ref_re;
    let mut shift_im = center_im.clone();
    shift_im -= ref_im;

    let mut delta_re = Vec::with_capacity(img_w as usize);
    for x in 0..img_w {
        let mut t = Float::with_val(prec, x);
        t /= &fw;
        t -= &half;
        t *= &vw;
        t += &shift_re;
        delta_re.push(t.to_f64());
    }

    let mut delta_im = Vec::with_capacity(img_h as usize);
    for y in 0..img_h {
        let mut t = Float::with_val(prec, y);
        t /= &fh;
        t -= &half;
        t *= &vh;
        t += &shift_im;
        delta_im.push(t.to_f64());
    }

    (delta_re, delta_im)
}

// ── 标量扰动迭代 ────────────────────────────────────────────────────────────

/// 标量扰动迭代（通用入口）。
///
/// `eps_re/eps_im` 与 `start_n` 由级数近似提供（无级数近似时为 0 / 0）。
/// 返回 (逃逸迭代, 逃逸时 |z|²)；glitch 返回 (GLITCH_MARKER, 0)；内部返回 (max_iter, 0)。
pub fn perturb_iterate(
    orbit: &ReferenceOrbit,
    delta_re: f64,
    delta_im: f64,
    max_iter: u32,
    mut eps_re: f64,
    mut eps_im: f64,
    start_n: u32,
) -> (u32, f64) {
    let orbit_len = orbit.len_bound(max_iter);

    // 主循环：直接用高精度参考轨道 Z_n（绝不在 f64 里重新积分 z，
    // 否则 f64 中心误差会随迭代指数放大，深缩放下参考轨道会错误逃逸）。
    for n in start_n..orbit_len {
        let (z_re, z_im) = orbit.z(n);
        let sum_re = z_re + eps_re;
        let sum_im = z_im + eps_im;
        let mag_sq = sum_re * sum_re + sum_im * sum_im;

        if mag_sq > BAILOUT {
            return (n, mag_sq);
        }
        if mag_sq < orbit.tolerance_check[n as usize] {
            return (GLITCH_MARKER, 0.0);
        }

        let two_z_eps_re = 2.0 * (z_re * eps_re - z_im * eps_im);
        let two_z_eps_im = 2.0 * (z_re * eps_im + z_im * eps_re);
        let eps_sq_re = eps_re * eps_re - eps_im * eps_im;
        let eps_sq_im = 2.0 * eps_re * eps_im;
        eps_re = two_z_eps_re + eps_sq_re + delta_re;
        eps_im = two_z_eps_im + eps_sq_im + delta_im;
    }

    // 参考轨道已逃逸但像素未逃逸：从逃逸点起在 f64 内联 z（逃逸轨道不敏感，f64 足够）。
    if let Some(esc) = orbit.escaped_at {
        let from = esc.max(start_n);
        let (mut z_re, mut z_im) = orbit.z(from.min(orbit.len() - 1));
        for n in from..max_iter {
            let sum_re = z_re + eps_re;
            let sum_im = z_im + eps_im;
            let mag_sq = sum_re * sum_re + sum_im * sum_im;
            if mag_sq > BAILOUT {
                return (n, mag_sq);
            }
            let two_z_eps_re = 2.0 * (z_re * eps_re - z_im * eps_im);
            let two_z_eps_im = 2.0 * (z_re * eps_im + z_im * eps_re);
            let eps_sq_re = eps_re * eps_re - eps_im * eps_im;
            let eps_sq_im = 2.0 * eps_re * eps_im;
            eps_re = two_z_eps_re + eps_sq_re + delta_re;
            eps_im = two_z_eps_im + eps_sq_im + delta_im;

            let zn_re = z_re * z_re - z_im * z_im + orbit.c_re;
            let zn_im = 2.0 * z_re * z_im + orbit.c_im;
            z_re = zn_re;
            z_im = zn_im;
        }
    }

    (max_iter, 0.0)
}

// ── SIMD 扰动迭代 ───────────────────────────────────────────────────────────

/// SIMD 扰动迭代（4 像素并行，通用入口）。
pub fn perturb_iterate_simd(
    orbit: &ReferenceOrbit,
    delta_re: f64x4,
    delta_im: f64x4,
    max_iter: u32,
    mut eps_re: f64x4,
    mut eps_im: f64x4,
    start_n: u32,
) -> ([u32; 4], [f64; 4]) {
    let two = f64x4::splat(2.0);
    let bailout = f64x4::splat(BAILOUT);

    let mut iter_count = u32x4::splat(max_iter);
    let mut escape_mag = f64x4::splat(0.0);
    let mut active: Mask<i64, 4> = Mask::splat(true);

    let orbit_len = orbit.len_bound(max_iter);

    // 主循环：直接用高精度参考轨道 Z_n（不在 f64 重新积分，避免深缩放发散）。
    for n in start_n..orbit_len {
        let (zr, zi) = orbit.z(n);
        let zrv = f64x4::splat(zr);
        let ziv = f64x4::splat(zi);

        let sum_re = zrv + eps_re;
        let sum_im = ziv + eps_im;
        let mag_sq = sum_re * sum_re + sum_im * sum_im;

        let escaped = mag_sq.simd_gt(bailout);
        let newly_esc = escaped & active;
        if newly_esc.any() {
            iter_count = newly_esc.select(u32x4::splat(n), iter_count);
            escape_mag = newly_esc.select(mag_sq, escape_mag);
        }

        let tol = f64x4::splat(orbit.tolerance_check[n as usize]);
        let glitched = mag_sq.simd_lt(tol);
        let newly_gl = glitched & active;
        if newly_gl.any() {
            iter_count = newly_gl.select(u32x4::splat(GLITCH_MARKER), iter_count);
        }

        active = active & !escaped & !glitched;
        if !active.any() {
            break;
        }

        let two_z_eps_re = two * (zrv * eps_re - ziv * eps_im);
        let two_z_eps_im = two * (zrv * eps_im + ziv * eps_re);
        let eps_sq_re = eps_re * eps_re - eps_im * eps_im;
        let eps_sq_im = two * eps_re * eps_im;
        let new_eps_re = two_z_eps_re + eps_sq_re + delta_re;
        let new_eps_im = two_z_eps_im + eps_sq_im + delta_im;
        eps_re = active.select(new_eps_re, eps_re);
        eps_im = active.select(new_eps_im, eps_im);
    }

    let mut iter_arr = iter_count.to_array();
    let mut mag_arr = escape_mag.to_array();

    // 参考轨道已逃逸：从逃逸点起逐 lane 内联 z（逃逸轨道不敏感）。
    if let Some(esc) = orbit.escaped_at {
        let from = esc.max(start_n);
        let (z_start_re, z_start_im) = orbit.z(from.min(orbit.len() - 1));
        let active_arr = active.to_array();
        let er_arr = eps_re.to_array();
        let ei_arr = eps_im.to_array();
        let dr_arr = delta_re.to_array();
        let di_arr = delta_im.to_array();

        for lane in 0..4 {
            if active_arr[lane] && iter_arr[lane] == max_iter {
                let mut er = er_arr[lane];
                let mut ei = ei_arr[lane];
                let mut zr = z_start_re;
                let mut zi = z_start_im;
                let dr = dr_arr[lane];
                let di = di_arr[lane];

                for n in from..max_iter {
                    let sr = zr + er;
                    let si = zi + ei;
                    let mag = sr * sr + si * si;
                    if mag > BAILOUT {
                        iter_arr[lane] = n;
                        mag_arr[lane] = mag;
                        break;
                    }
                    let tze_re = 2.0 * (zr * er - zi * ei);
                    let tze_im = 2.0 * (zr * ei + zi * er);
                    let esq_re = er * er - ei * ei;
                    let esq_im = 2.0 * er * ei;
                    er = tze_re + esq_re + dr;
                    ei = tze_im + esq_im + di;

                    let zn_r = zr * zr - zi * zi + orbit.c_re;
                    let zn_i = 2.0 * zr * zi + orbit.c_im;
                    zr = zn_r;
                    zi = zn_i;
                }
            }
        }
    }

    (iter_arr, mag_arr)
}

// ── 朴素迭代（浅缩放 / 对照） ───────────────────────────────────────────────

/// 朴素标量迭代（含心形 / 周期-2 圆盘快速跳过）。
pub fn naive_scalar(c_re: f64, c_im: f64, max_iter: u32) -> (u32, f64) {
    let q = (c_re - 0.25).powi(2) + c_im.powi(2);
    if q * (q + (c_re - 0.25)) <= 0.25 * c_im.powi(2) {
        return (max_iter, 0.0);
    }
    if (c_re + 1.0).powi(2) + c_im.powi(2) <= 0.0625 {
        return (max_iter, 0.0);
    }

    let mut z_re = 0.0;
    let mut z_im = 0.0;
    for i in 0..max_iter {
        let zrs = z_re * z_re;
        let zis = z_im * z_im;
        let mag = zrs + zis;
        if mag > BAILOUT {
            return (i, mag);
        }
        let zn_im = 2.0 * z_re * z_im + c_im;
        z_re = zrs - zis + c_re;
        z_im = zn_im;
    }
    (max_iter, 0.0)
}

/// 朴素 SIMD 迭代（浅缩放路径）。
pub fn naive_simd(c_re: f64x4, c_im: f64x4, max_iter: u32) -> ([u32; 4], [f64; 4]) {
    let quarter = f64x4::splat(0.25);
    let cs = c_re - quarter;
    let q = cs * cs + c_im * c_im;
    let in_card = q.simd_le(q * cs + quarter * c_im * c_im);

    let one = f64x4::splat(1.0);
    let p0625 = f64x4::splat(0.0625);
    let cp1 = c_re + one;
    let in_bulb = (cp1 * cp1 + c_im * c_im).simd_le(p0625);

    let skip = in_card | in_bulb;
    let bailout = f64x4::splat(BAILOUT);
    let two = f64x4::splat(2.0);

    let mut z_re = f64x4::splat(0.0);
    let mut z_im = f64x4::splat(0.0);
    let mut iter = u32x4::splat(max_iter);
    let mut mag = f64x4::splat(0.0);
    let mut active: Mask<i64, 4> = !skip;

    if !active.any() {
        return (iter.to_array(), mag.to_array());
    }

    for i in 0..max_iter {
        let zrs = z_re * z_re;
        let zis = z_im * z_im;
        let m = zrs + zis;

        let escaped = m.simd_gt(bailout);
        let ne = escaped & active;
        if ne.any() {
            iter = ne.select(u32x4::splat(i), iter);
            mag = ne.select(m, mag);
        }
        active &= !escaped;
        if !active.any() {
            break;
        }

        let zn_im = two * z_re * z_im + c_im;
        let zn_re = zrs - zis + c_re;
        z_re = active.select(zn_re, z_re);
        z_im = active.select(zn_im, z_im);
    }

    (iter.to_array(), mag.to_array())
}

/// 逐像素高精度精确迭代（glitch 像素兜底，结果绝对正确）。
pub fn naive_hp(c_re: &Float, c_im: &Float, max_iter: u32) -> (u32, f64) {
    let prec = c_re.prec();
    let c = Complex::with_val(prec, (c_re.clone(), c_im.clone()));
    let mut z = Complex::with_val(prec, (0, 0));
    for i in 0..max_iter {
        let re = z.real().to_f64();
        let im = z.imag().to_f64();
        let mag = re * re + im * im;
        if mag > BAILOUT {
            return (i, mag);
        }
        z *= z.clone();
        z += c.clone();
    }
    (max_iter, 0.0)
}

// ── 级数近似（可选加速） ────────────────────────────────────────────────────
//
// ε_n 可展开为 δ 的幂级数：ε_n = A_n·δ + B_n·δ² + C_n·δ³ + ...
// 系数只依赖参考轨道：A_{n+1}=2Z_nA_n+1, B_{n+1}=2Z_nB_n+A_n², C_{n+1}=2Z_nC_n+2A_nB_n ...
// 深缩放时系数会超出 f64 范围，故用 (尾数, 指数) 形式存储避免溢出。

#[inline(always)]
fn frexp(val: f64, exp: &mut i32) -> f64 {
    let bits = val.to_bits();
    if bits == 0 {
        *exp = 0;
        return 0.0;
    }
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    if raw_exp == 0 {
        let normalized = val * (1u64 << 52) as f64;
        let nbits = normalized.to_bits();
        let nexp = ((nbits >> 52) & 0x7ff) as i32 - 52;
        let mant = f64::from_bits((nbits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000);
        *exp = nexp - 1022;
        mant
    } else if raw_exp == 0x7ff {
        *exp = 0;
        val
    } else {
        let mant = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000);
        *exp = raw_exp - 1022;
        mant
    }
}

#[inline(always)]
fn ldexp(mantissa: f64, exp: i32) -> f64 {
    if mantissa == 0.0 {
        return 0.0;
    }
    let half = exp / 2;
    let rest = exp - half;
    mantissa * 2f64.powi(half) * 2f64.powi(rest)
}

/// 复数 (尾数, 共享指数) 表示：value = (mantissa_re + i·mantissa_im)·2^exp
#[derive(Clone, Copy)]
struct Mcx {
    re: f64,
    im: f64,
    exp: i32,
}

impl Mcx {
    const ZERO: Self = Mcx { re: 0.0, im: 0.0, exp: 0 };

    fn from_complex(re: f64, im: f64) -> Self {
        if re == 0.0 && im == 0.0 {
            return Self::ZERO;
        }
        let mut re_exp = 0i32;
        let mut im_exp = 0i32;
        let re_m = frexp(re, &mut re_exp);
        let im_m = frexp(im, &mut im_exp);
        let exp = re_exp.max(im_exp);
        Mcx {
            re: ldexp(re_m, re_exp - exp),
            im: ldexp(im_m, im_exp - exp),
            exp,
        }
    }

    fn to_complex(self) -> (f64, f64) {
        (ldexp(self.re, self.exp), ldexp(self.im, self.exp))
    }

    fn reduce(&mut self) {
        if self.re == 0.0 && self.im == 0.0 {
            return;
        }
        let need = (self.re.abs() >= 2.0 || self.im.abs() >= 2.0)
            || (self.re != 0.0 && self.re.abs() < 0.25)
            || (self.im != 0.0 && self.im.abs() < 0.25);
        if !need {
            return;
        }
        let mut re_exp = 0i32;
        let mut im_exp = 0i32;
        let re_m = if self.re != 0.0 { frexp(self.re, &mut re_exp) } else { 0.0 };
        let im_m = if self.im != 0.0 { frexp(self.im, &mut im_exp) } else { 0.0 };
        let re_tot = if self.re != 0.0 { re_exp + self.exp } else { i32::MIN };
        let im_tot = if self.im != 0.0 { im_exp + self.exp } else { i32::MIN };
        let new_exp = re_tot.max(im_tot);
        self.re = if self.re != 0.0 { ldexp(re_m, re_tot - new_exp) } else { 0.0 };
        self.im = if self.im != 0.0 { ldexp(im_m, im_tot - new_exp) } else { 0.0 };
        self.exp = new_exp;
    }

    /// 乘以普通复数 (z_re, z_im)
    fn mul_z(&self, z_re: f64, z_im: f64) -> Self {
        let f = Mcx::from_complex(z_re, z_im);
        let nr = self.re * f.re - self.im * f.im;
        let ni = self.re * f.im + self.im * f.re;
        let mut r = Mcx { re: nr, im: ni, exp: self.exp + f.exp };
        r.reduce();
        r
    }

    fn mul_mcx(&self, o: &Mcx) -> Self {
        let nr = self.re * o.re - self.im * o.im;
        let ni = self.re * o.im + self.im * o.re;
        let mut r = Mcx { re: nr, im: ni, exp: self.exp + o.exp };
        r.reduce();
        r
    }

    fn mul_f64(&self, f: f64) -> Self {
        let mut r = *self;
        r.re *= f;
        r.im *= f;
        r.reduce();
        r
    }

    fn add_mcx(&self, o: &Mcx) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return *o;
        }
        if o.re == 0.0 && o.im == 0.0 {
            return *self;
        }
        let mut r = *self;
        if r.exp > o.exp {
            let shift = r.exp - o.exp;
            r.re = ldexp(r.re, shift);
            r.im = ldexp(r.im, shift);
            r.exp = o.exp;
            r.re += o.re;
            r.im += o.im;
        } else if r.exp < o.exp {
            let shift = o.exp - r.exp;
            r.re += ldexp(o.re, shift);
            r.im += ldexp(o.im, shift);
            r.exp = o.exp;
        } else {
            r.re += o.re;
            r.im += o.im;
        }
        r.reduce();
        r
    }

    fn add_real(&self, v: f64) -> Self {
        self.add_mcx(&Mcx::from_complex(v, 0.0))
    }

    fn square(&self) -> Self {
        let nr = self.re * self.re - self.im * self.im;
        let ni = 2.0 * self.re * self.im;
        let mut r = Mcx { re: nr, im: ni, exp: self.exp * 2 };
        r.reduce();
        r
    }

    /// |value|² 的 (尾数, 指数) 形式
    fn norm(&self) -> (f64, i32) {
        (self.re * self.re + self.im * self.im, self.exp * 2)
    }
}

/// 级数近似：预计算幂级数系数并据此跳过前期迭代。
pub struct SeriesApproximation {
    /// coeff[k][n] = a_{k+1}(n)
    coeff: Vec<Vec<Mcx>>,
    num_terms: usize,
    skip: u32,
}

impl SeriesApproximation {
    pub fn compute(orbit: &ReferenceOrbit, delta_max: f64, num_terms: usize, max_iter: u32) -> Self {
        let k = num_terms.clamp(2, 16);
        let orbit_len = orbit.len() as usize;

        let mut coeff: Vec<Vec<Mcx>> = (0..k).map(|_| vec![Mcx::ZERO; orbit_len]).collect();
        coeff[0][0] = Mcx::from_complex(1.0, 0.0); // A_1 = 1

        // δ^p 的 (尾数, 指数)，用于收敛判定
        let mut dmp: Vec<(f64, i32)> = Vec::with_capacity(k + 1);
        dmp.push((1.0, 0));
        let d1 = Mcx::from_complex(delta_max, 0.0);
        dmp.push((d1.re, d1.exp));
        for i in 2..=k {
            let (m0, e0) = dmp[i - 1];
            let (m1, e1) = dmp[1];
            dmp.push((m0 * m1, e0 + e1));
        }
        // 收敛容差 2^-64
        let tol_m = 0.5f64;
        let tol_e = -63i32;

        let mut skip = 0u32;
        let max_skip = (max_iter as usize).saturating_sub(10);

        for n in 0..orbit_len.saturating_sub(1) {
            let (zr, zi) = orbit.z(n as u32);

            // A_{n+1} = 2Z_nA_n + 1
            coeff[0][n + 1] = coeff[0][n].mul_z(zr, zi).mul_f64(2.0).add_real(1.0);
            if k >= 2 {
                // B_{n+1} = 2Z_nB_n + A_n²
                let t = coeff[1][n].mul_z(zr, zi).mul_f64(2.0);
                coeff[1][n + 1] = t.add_mcx(&coeff[0][n].square());
            }
            if k >= 3 {
                // C_{n+1} = 2Z_nC_n + 2A_nB_n
                let t = coeff[2][n].mul_z(zr, zi).mul_f64(2.0);
                let ab = coeff[0][n].mul_mcx(&coeff[1][n]).mul_f64(2.0);
                coeff[2][n + 1] = t.add_mcx(&ab);
            }
            if k >= 4 {
                // D_{n+1} = 2Z_nD_n + 2A_nC_n + B_n²
                let t = coeff[3][n].mul_z(zr, zi).mul_f64(2.0);
                let ac = coeff[0][n].mul_mcx(&coeff[2][n]).mul_f64(2.0);
                coeff[3][n + 1] = t.add_mcx(&ac).add_mcx(&coeff[1][n].square());
            }
            for term in 4..k {
                // 通用递推：a_{term+1}(n+1) = 2Z_na_{term+1}(n) + Σ_{i+j=term+1} a_i a_j
                let t = coeff[term][n].mul_z(zr, zi).mul_f64(2.0);
                let mut cross = Mcx::ZERO;
                for p in 0..term {
                    let qq = term - 1 - p;
                    let prod = coeff[p][n].mul_mcx(&coeff[qq][n]);
                    cross = cross.add_mcx(&prod);
                }
                coeff[term][n + 1] = t.add_mcx(&cross);
            }
            for col in coeff.iter_mut() {
                col[n + 1].reduce();
            }

            // 收敛判定：末项相对前一项可忽略时才可跳过
            if (n + 1) <= max_skip {
                let last = &coeff[k - 1][n + 1];
                let prev = &coeff[k - 2][n + 1];
                let last_zero = last.re == 0.0 && last.im == 0.0;
                let prev_zero = prev.re == 0.0 && prev.im == 0.0;
                if !last_zero && !prev_zero {
                    let (lm, le) = last.norm();
                    let (pm, pe) = prev.norm();
                    // last_term = lm·2^le · δ^k ; prev_term = pm·2^pe · δ^{k-1}
                    let last_exp = le + dmp[k].1;
                    let prev_exp = pe + dmp[k - 1].1;
                    let last_m = lm * dmp[k].0;
                    let prev_m = pm * dmp[k - 1].0;
                    // 判定 prev·tol < last → 未收敛
                    let lhs_m = prev_m * tol_m;
                    let lhs_e = prev_exp + tol_e;
                    let not_converged = if lhs_e != last_exp {
                        lhs_e < last_exp
                    } else {
                        lhs_m < last_m
                    };
                    if not_converged {
                        skip = (n + 1).saturating_sub(3) as u32;
                        let valid = if skip > 0 { skip as usize + 1 } else { 0 };
                        for col in coeff.iter_mut() {
                            col.truncate(valid);
                        }
                        break;
                    } else {
                        skip = (n + 1) as u32;
                    }
                }
            }
        }

        SeriesApproximation { coeff, num_terms: k, skip }
    }

    pub fn skip_iterations(&self) -> u32 {
        self.skip
    }

    /// 用级数近似求 ε_skip（标量）。
    pub fn evaluate(&self, delta_re: f64, delta_im: f64) -> (f64, f64) {
        let n = self.skip as usize;
        if n == 0 || n >= self.coeff[0].len() {
            return (0.0, 0.0);
        }
        let k = self.num_terms;
        let mut a_re = vec![0.0f64; k];
        let mut a_im = vec![0.0f64; k];
        for term in 0..k {
            let (re, im) = self.coeff[term][n].to_complex();
            a_re[term] = re;
            a_im[term] = im;
        }
        // Horner: ε = δ·(a_1 + δ·(a_2 + ... ))
        let mut rr = a_re[k - 1];
        let mut ri = a_im[k - 1];
        for term in (0..k - 1).rev() {
            let nr = a_re[term] + (delta_re * rr - delta_im * ri);
            let ni = a_im[term] + (delta_re * ri + delta_im * rr);
            rr = nr;
            ri = ni;
        }
        (delta_re * rr - delta_im * ri, delta_re * ri + delta_im * rr)
    }

    /// 用级数近似求 ε_skip（SIMD，4 个 δ）。
    pub fn evaluate_simd(&self, delta_re: f64x4, delta_im: f64x4) -> (f64x4, f64x4) {
        let n = self.skip as usize;
        if n == 0 || n >= self.coeff[0].len() {
            return (f64x4::splat(0.0), f64x4::splat(0.0));
        }
        let k = self.num_terms;
        let mut a_re = vec![0.0f64; k];
        let mut a_im = vec![0.0f64; k];
        for term in 0..k {
            let (re, im) = self.coeff[term][n].to_complex();
            a_re[term] = re;
            a_im[term] = im;
        }
        let mut rr = f64x4::splat(a_re[k - 1]);
        let mut ri = f64x4::splat(a_im[k - 1]);
        for term in (0..k - 1).rev() {
            let ar = f64x4::splat(a_re[term]);
            let ai = f64x4::splat(a_im[term]);
            let nr = ar + (delta_re * rr - delta_im * ri);
            let ni = ai + (delta_re * ri + delta_im * rr);
            rr = nr;
            ri = ni;
        }
        (delta_re * rr - delta_im * ri, delta_re * ri + delta_im * rr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbit_at(c_re: f64, c_im: f64, max_iter: u32, prec: u32) -> ReferenceOrbit {
        let c = Complex::with_val(prec, (c_re, c_im));
        ReferenceOrbit::compute_hp(&c, max_iter)
    }

    #[test]
    fn reference_orbit_inside() {
        let o = orbit_at(-1.0, 0.0, 100, 64);
        assert!(o.escaped_at.is_none());
    }

    #[test]
    fn reference_orbit_escape() {
        let o = orbit_at(2.0, 0.0, 100, 64);
        assert!(o.escaped_at.is_some());
        assert!(o.escaped_at.unwrap() < 5);
    }

    #[test]
    fn perturbation_matches_naive() {
        let o = orbit_at(-0.75, 0.0, 256, 96);
        let pts = [(0.3, 0.0), (1.0, 0.5), (-0.5, 0.0), (-0.75, 0.1), (-0.1, 0.8)];
        for (cr, ci) in pts {
            let (pi, _) = perturb_iterate(&o, cr - (-0.75), ci - 0.0, 256, 0.0, 0.0, 0);
            let (ni, _) = naive_scalar(cr, ci, 256);
            assert!(
                (pi as i32 - ni as i32).abs() <= 2,
                "c=({},{}): perturb={} naive={}",
                cr,
                ci,
                pi,
                ni
            );
        }
    }

    #[test]
    fn simd_matches_scalar() {
        let o = orbit_at(-0.743643887037151, 0.131825904205330, 512, 128);
        let base_re = -0.743643887037151;
        let base_im = 0.131825904205330;
        let d_re = [1e-9, 2e-9, -1e-9, 3e-9];
        let d_im = [1e-9, -2e-9, 2e-9, 0.5e-9];
        let (iters, mags) = perturb_iterate_simd(
            &o,
            f64x4::from_array(d_re),
            f64x4::from_array(d_im),
            512,
            f64x4::splat(0.0),
            f64x4::splat(0.0),
            0,
        );
        for lane in 0..4 {
            let (si, sm) = perturb_iterate(&o, d_re[lane], d_im[lane], 512, 0.0, 0.0, 0);
            assert_eq!(iters[lane], si, "lane {} iter", lane);
            if si != GLITCH_MARKER && si < 512 {
                assert!((mags[lane] - sm).abs() < 1e-6, "lane {} mag", lane);
            }
        }
        let _ = (base_re, base_im);
    }

    #[test]
    fn suggested_precision_grows_with_zoom() {
        assert_eq!(super::super::config::suggested_precision(1.0, 1920), 64);
        let p18 = super::super::config::suggested_precision(1e18, 1920);
        assert!(p18 >= 120, "p18={}", p18);
    }

    #[test]
    fn naive_hp_agrees_with_scalar_shallow() {
        // 浅缩放处 f64 与 MPFR 应高度一致
        let pts = [(0.3, 0.1), (-0.5, 0.5), (-1.2, 0.3), (0.0, 0.0)];
        for (cr, ci) in pts {
            let crf = Float::with_val(128, cr);
            let cif = Float::with_val(128, ci);
            let (hi, _) = naive_hp(&crf, &cif, 300);
            let (fi, _) = naive_scalar(cr, ci, 300);
            assert!((hi as i32 - fi as i32).abs() <= 1, "c=({},{}): hp={} f64={}", cr, ci, hi, fi);
        }
    }
}
