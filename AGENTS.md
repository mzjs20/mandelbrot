# AGENTS.md

## Build & Run

```bash
cargo build --release
./target/release/mandelbrot_renderer config.json        # 浅缩放（f64 边界）
./target/release/mandelbrot_renderer config_deep.json   # 深缩放（高精度字符串，1e18+）
```

- CLI 解析 `config_path_from_args`（`src/main.rs`）：位置参数、`-c`/`--config`、`--config=...`、`--config_deep.json`（剥前缀）、`--` 后参数均可；缺省 `config.json`。

- Requires **nightly** Rust (`#![feature(portable_simd)]` in `src/main.rs`).
- Needs a C toolchain for the `rug` crate (vendored GMP/MPFR); Linux may also need `libpng-dev` (Ubuntu) / `libpng-devel` (Fedora) for the `image` crate.

## Verification

```bash
cargo build --release && cargo clippy && cargo test
```

- `cargo test` runs 9 real tests, including full-pipeline deep-zoom comparison against per-pixel MPFR ground truth (`render::tests`) and a 1e18 image-detail regression test.
- Quick manual check: render a small deep config (low resolution, low `max_iter`) and confirm the PNG has many distinct colors (a flat/4-color image means the reference orbit bug regressed — see below).
- Note: `ld` on this machine prints harmless `GNU_PROPERTY_TYPE` warnings on link; ignore them.

## Architecture (modules)

- `src/main.rs` — CLI, config loading, PNG saving, compression/filter mapping.
- `src/config.rs` — `Config` (serde JSON) + `View` resolution. Deep zoom uses `center_re`/`center_im` (arbitrary-precision **strings**) + `zoom` (e.g. `"1e18"`), parsed by MPFR; legacy f64 `x_min/x_max/y_min/y_max` routes to shallow mode.
- `src/perturbation.rs` — deep-zoom engine: `ReferenceOrbit` (MPFR), high-precision deltas (`compute_deltas_hp`), perturbation iteration (scalar + `std::simd` `f64x4`), Pauldelbrot glitch detection, series approximation (`Mcx` mantissa+exponent to avoid f64 overflow), exact per-pixel MPFR fallback (`naive_hp`).
- `src/render.rs` — rayon parallel driver: shallow → naive SIMD; deep → perturbation pass → multi-reference rebase loop → MPFR fallback for residual glitches.
- `src/color.rs` — smooth (normalized iteration count) coloring + cyclic cosine palette.

## Non-obvious Behaviors

- **Critical pitfall**: the perturbation ε-iteration MUST use the high-precision reference orbit values (`orbit.z(n)`), never re-integrate `z` in f64 using the f64 center. At deep zoom the f64 center error (~1e-16) exceeds the view size, so re-integration diverges and every pixel escapes at the same iteration → flat/wrong image. This was the defect in the sibling project `mandelbro_video`.
- `png_compression_level` mapping is **not** linear: 0 → Fast, 1–8 → Default, 9 → Best.
- `png_filter_type` accepts both `"none"` and `"nofilter"` (case-insensitive).
- Reference precision = `⌈log2(zoom)⌉ + 53 + 16(guard) + ⌈log2(max_dim)⌉` (`config::suggested_precision`).
- Series approximation can conservatively skip **0** iterations (falls back to plain perturbation — still correct and fast).
- Residual glitches after rebase are finished with exact per-pixel MPFR (`naive_hp`), guaranteeing zero artifacts.
- Comments and stdout are in Chinese.

## See Also

- `CLAUDE.md` — older architecture overview (predates this rewrite; perturbation details now live in `src/perturbation.rs`).
- Sibling project `~/RustroverProjects/mandelbro_video` — source of the perturbation theory design; note its f64 reference re-integration bug at deep zoom (fixed here).

## Notes

- ⚠️ 仓库里有个带前导空格的残留文件 `" config_ultra_high_res.json"`（README 引用的是无空格的 `config_ultra_high_res.json`，并不存在）。内容是浅缩放 76800×43200 超清配置。运行前注意这个坑，必要时重命名或删除。
- `config_deep.json` 当前 `zoom: "1e22"`；`config.json` 是浅缩放小区域高迭代测试配置（输出 `test.png`）。
