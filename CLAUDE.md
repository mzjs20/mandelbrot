# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Build release version
cargo build --release

# Run with default config.json
./target/release/mandelbrot_renderer

# Run with custom config file
./target/release/mandelbrot_renderer path/to/config.json
```

## Architecture

This is a Mandelbrot set fractal image generator written in Rust. The codebase is a single-file application (`src/main.rs`) with the following structure:

- **Config struct**: Loads rendering parameters from JSON (resolution, coordinate bounds, iteration count, PNG settings). Supports command-line argument for config file path.

- **mandelbrot()**: Core iteration function that computes escape time for a complex point.

- **iter_to_color()**: Maps iteration count to RGB using a polynomial color gradient.

- **generate_mandelbrot()**: Parallel image generation using rayon. Divides the image into horizontal chunks (one per CPU core), each processed with its own progress bar via indicatif's MultiProgress.

- **save_image_with_progress()**: PNG encoding with configurable compression/filter settings and a time-estimated progress bar.

The program reads `config.json` by default, validates aspect ratio alignment, generates the fractal in parallel, and saves to the specified PNG file.

## Configuration

All rendering parameters are externalized to JSON config files. Key fields:
- `width`, `height`: Image resolution
- `x_min`, `x_max`, `y_min`, `y_max`: Complex plane bounds
- `max_iter`: Maximum iterations (higher = more detail, slower)
- `output_filename`: Output PNG path
- `png_compression_level`: 0 (fast) to 9 (smallest)
- `png_filter_type`: "none", "sub", "up", "avg", or "paeth"
