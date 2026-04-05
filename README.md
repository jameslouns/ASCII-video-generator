# ASCII Video Generator

Converts video files into colored ASCII art videos using GPU-accelerated frame processing.

## How It Works

Each frame is split into a grid of tiles. The average color of each tile is computed on the GPU (CUDA), and the brightness is mapped to an ASCII character. The character is then colored with the original tile color and composited back into a full-resolution frame. Audio from the original video is merged into the final output.

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- Dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** `torch`, `torchvision`, and `torchaudio` are pinned to CUDA 12.4 builds. If your CUDA version differs, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Web Interface (recommended)

```bash
python server.py
```

Then open `http://localhost:5000` in your browser. Paste a YouTube URL, configure the text scale and batch size, and start conversion. Progress streams live to the page.

### Command Line

```bash
python converter.py --video_path Videos/myvideo.mp4 --textscale 80 --framesper 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_path` | required | Path to the input `.mp4` file |
| `--textscale` | `80` | Controls ASCII grid density — higher = more characters, finer detail |
| `--framesper` | `50` | Frames processed per GPU batch — lower if you run out of VRAM |

Output is saved to the `output/` directory.

### Interactive Script

`main.py` downloads a video from YouTube and runs conversion interactively. Edit the `url` and `file_name` variables at the bottom of the file before running.

## Project Structure

```
ASCII_video_generator/
├── main.py          # Interactive entry point (YouTube download + convert)
├── converter.py     # CLI converter, used by the web server
├── server.py        # Flask web interface
├── templates/       # HTML templates for the web UI
├── Fonts/           # SpaceMono font used for character rendering
├── Videos/          # Input videos (not tracked by git)
└── output/          # Converted output videos (not tracked by git)
```
