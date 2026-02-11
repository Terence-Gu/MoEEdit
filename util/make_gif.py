#!/usr/bin/env python3
"""
Make a GIF slide from multiple PDF files.
-------------------------------------------------
Dependencies:
    pip install pdf2image pillow imageio
    # Linux: sudo apt-get install poppler-utils
"""

import os
from pdf2image import convert_from_path
from PIL import Image
import imageio.v2 as imageio    # imageio>=2

# ---------- Configuration ----------
pdf_dir   = "/home/MoEdit/results/for_gif"
gif_path  = "/home/MoEdit/results/for_gif/expert_slide.gif"
duration  = 500        # Duration of each page (seconds)
dpi       = 200        # Image resolution; can be adjusted as needed
# -------------------------------------

def collect_pdfs(directory):
    """Collect all trace_*_experts.pdf files and sort them by name"""
    files = [f for f in os.listdir(directory) if f.endswith("_experts.pdf")]
    files.sort()                # "Catalonia" "Italy" "Steve ..." ...
    return [os.path.join(directory, f) for f in files]

def pdfs_to_images(pdf_files, dpi=200):
    """Render the first page of each PDF as a PIL.Image"""
    images = []
    for pdf in pdf_files:
        # Only the first page is needed; each PDF should have only one page
        pil_pages = convert_from_path(pdf, dpi=dpi, first_page=1, last_page=1)
        images.append(pil_pages[0].convert("RGB"))
    return images

def save_gif(images, out_path, duration=1.0):
    """Save multiple PIL.Image as a looping GIF"""
    frames = [img.copy() for img in images]
    imageio.mimsave(out_path, frames, format="GIF", duration=duration, loop=0)
    print(f"GIF saved to: {out_path}")

if __name__ == "__main__":
    pdf_list = collect_pdfs(pdf_dir)
    if not pdf_list:
        raise RuntimeError(f"Directory {pdf_dir} does not contain *_experts.pdf files")
    print("Merging the following files:\n  " + "\n  ".join(pdf_list))

    imgs = pdfs_to_images(pdf_list, dpi=dpi)
    save_gif(imgs, gif_path, duration=duration)