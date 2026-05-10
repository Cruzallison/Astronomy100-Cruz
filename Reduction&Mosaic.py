#!/usr/bin/env python3
"""
KeplerCam 4-CCD science reduction — same use case and pipeline as ReduceScience.ipynb.

Overscan → trim → master bias (per CCD) → filter-matched master flat → reduced science
for CCDs 1–4. Writes:

* ``<stem>Reduced.fits`` — full multi-extension reduction (4 CCDs in separate HDUs).
* ``<stem>Reduced_mosaic.fits`` by default — single 2D image, 2×2 layout of the four
  CCDs (suitable for quick viewing, Astrometry.net, etc.).  Use ``--no-mosaic`` to
  skip this file.
* Optional ``--plot-mosaic`` — also writes ``<stem>Reduced_mosaic.png`` (matplotlib).
* Optional: ``--astrometry-ccd N`` adds a one-CCD, primary-only FITS.

**Dependencies** (install in a terminal): ``pip install astropy numpy``; optional ``matplotlib``
for ``--plot-mosaic`` (PNG preview of the 2×2 mosaic).
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional

import astropy.io.fits as fits
import numpy as np
from astropy.stats import sigma_clipped_stats

# Match ReduceScience notebook / Reduction.py: trim/overscan and per-CCD masters
LEFT_TRIM = 70
RIGHT_TRIM = 8
OVERSCAN_WIDTH = 8
OVERSCAN_SIGMA = 3.0
OVERSCAN_MAXITERS = 5


def overscan_level_per_row(data: np.ndarray) -> np.ndarray:
    overscan_region = data[:, -OVERSCAN_WIDTH:]
    levels = np.empty(overscan_region.shape[0], dtype=float)
    for i in range(overscan_region.shape[0]):
        row = overscan_region[i].astype(float, copy=False)
        row = row[np.isfinite(row)]
        if row.size == 0:
            levels[i] = np.nan
            continue
        _, med, _ = sigma_clipped_stats(row, sigma=OVERSCAN_SIGMA, maxiters=OVERSCAN_MAXITERS)
        levels[i] = med
    return levels


def apply_overscan_correction(data: np.ndarray) -> np.ndarray:
    levels = overscan_level_per_row(data)
    return data - levels[:, None]


def make_master_calibration_frame(file_list, ccd_number, left_trim, right_trim):
    stack = []
    for fn in file_list:
        data = fits.getdata(fn, ccd_number)
        data_corr = apply_overscan_correction(data)
        data_trim = data_corr[:, left_trim:-right_trim]
        stack.append(data_trim)
    stack = np.array(stack)
    return np.median(stack, axis=0)


def make_master_flat_frame(file_list, ccd_number, left_trim, right_trim, master_bias):
    stack = []
    for fn in file_list:
        data = fits.getdata(fn, ccd_number)
        data_corr = apply_overscan_correction(data)
        data_trim = data_corr[:, left_trim:-right_trim]
        flat_biascorr = data_trim - master_bias
        finite_nonzero = flat_biascorr[np.isfinite(flat_biascorr) & (flat_biascorr != 0)]
        if finite_nonzero.size == 0:
            continue
        flat_norm = flat_biascorr / np.median(finite_nonzero)
        stack.append(flat_norm)
    if not stack:
        raise RuntimeError(f"No valid flat frames for CCD {ccd_number} after bias correction.")
    return np.median(np.array(stack), axis=0)


def reduce_science_ccds(
    science_path: str,
    bias_files: list,
    all_flat_file_paths: list,
    left_trim: int,
    right_trim: int,
):
    with fits.open(science_path) as hdul:
        science_filter_local = hdul[0].header.get("FILTER", "UNKNOWN").strip().lower()

    filtered_flat_files_local: list = []
    for my_file in all_flat_file_paths:
        with fits.open(my_file) as hdul:
            primary_header = hdul[0].header
            if "FILTER" in primary_header:
                flat_filter_value = primary_header["FILTER"].strip().lower()
                if flat_filter_value == science_filter_local:
                    filtered_flat_files_local.append(my_file)

    if not filtered_flat_files_local:
        raise ValueError(
            f"No flat files found for filter '{science_filter_local}' "
            f"(science {os.path.basename(science_path)})"
        )

    master_biases_local = {}
    master_flats_local = {}
    for ccd_num in range(1, 5):
        master_biases_local[ccd_num] = make_master_calibration_frame(
            bias_files, ccd_num, left_trim, right_trim
        )
        master_flats_local[ccd_num] = make_master_flat_frame(
            filtered_flat_files_local, ccd_num, left_trim, right_trim, master_biases_local[ccd_num]
        )

    reduced_ccds_local = {}
    for ccd_num in range(1, 5):
        science_data = fits.getdata(science_path, ccd_num)
        science_corr = apply_overscan_correction(science_data)
        science_trim = science_corr[:, left_trim:-right_trim]
        science_biascorr = science_trim - master_biases_local[ccd_num]
        norm_flat = master_flats_local[ccd_num]
        flat_valid = np.isfinite(norm_flat) & (norm_flat != 0)
        science_reduced_ccd = np.full_like(science_biascorr, np.nan, dtype=float)
        science_reduced_ccd[flat_valid] = science_biascorr[flat_valid] / norm_flat[flat_valid]
        reduced_ccds_local[ccd_num] = science_reduced_ccd

    return reduced_ccds_local, science_filter_local


def write_reduced_multichip_fits(science_path: str, reduced_by_ccd: dict, out_path: str) -> None:
    out_abs = os.path.abspath(out_path)
    d = os.path.dirname(out_abs)
    if d:
        os.makedirs(d, exist_ok=True)
    with fits.open(science_path) as hdul_in:
        primary = fits.PrimaryHDU(header=hdul_in[0].header.copy())
        primary.header.add_history("Reduced: overscan, trim, bias, flat (reduce_science.py)")
        out_hdul = fits.HDUList([primary])
        for ccd_num in range(1, 5):
            hdr = hdul_in[ccd_num].header.copy()
            data = np.asarray(reduced_by_ccd[ccd_num], dtype=np.float32)
            out_hdul.append(fits.ImageHDU(data=data, header=hdr))
        # Input cameras sometimes use non–FITS-standard 'Comment' cards; strict verify fails on writeto.
        out_hdul.writeto(out_abs, overwrite=True, output_verify="ignore")


def _header_keys_remove_for_primary(hdr: fits.Header) -> None:
    """Remove extension-style cards so the header is valid for PrimaryHDU."""
    for k in ("XTENSION", "EXTNAME", "EXTVER", "EXTLEVEL"):
        if k in hdr:
            del hdr[k]


def write_astrometry_net_fits(
    science_path: str, reduced_2d: np.ndarray, ccd_num: int, out_path: str
) -> None:
    """
    Write one FITS with a single 2D array in the primary HDU.
    Astrometry.net and similar services often reject MEFs whose HDU 0 has no image.
    """
    if ccd_num not in (1, 2, 3, 4):
        raise ValueError("ccd_num must be 1–4")
    image = np.asarray(reduced_2d, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    out_abs = os.path.abspath(out_path)
    d = os.path.dirname(out_abs)
    if d:
        os.makedirs(d, exist_ok=True)
    with fits.open(science_path) as hdul_in:
        hdr = hdul_in[ccd_num].header.copy()
    _header_keys_remove_for_primary(hdr)
    hdr.add_history("Single-CCD 2D image in primary (reduce_science.py, for Astrometry.net).")
    pri = fits.PrimaryHDU(data=image, header=hdr)
    pri.header["BITPIX"] = -32
    fits.HDUList([pri]).writeto(out_abs, overwrite=True, output_verify="ignore")


def combine_reduced_ccds_mosaic(reduced_by_ccd: dict) -> np.ndarray:
    """
    Pack four reduced CCDs into one 2×2 image (same layout as ``Reduction.py``).

    Bottom row: CCD2 (left), CCD1 (right). Top row: CCD4 (left), CCD3 (right).
    """
    h, w = reduced_by_ccd[1].shape
    for ccd in (1, 2, 3, 4):
        if reduced_by_ccd[ccd].shape != (h, w):
            raise ValueError(
                f"All CCDs must have shape {(h, w)}; CCD {ccd} has {reduced_by_ccd[ccd].shape}"
            )
    combined = np.zeros((2 * h, 2 * w), dtype=np.float32)
    combined[h : 2 * h, 0:w] = np.asarray(reduced_by_ccd[2], dtype=np.float32)
    combined[h : 2 * h, w : 2 * w] = np.asarray(reduced_by_ccd[1], dtype=np.float32)
    combined[0:h, 0:w] = np.asarray(reduced_by_ccd[4], dtype=np.float32)
    combined[0:h, w : 2 * w] = np.asarray(reduced_by_ccd[3], dtype=np.float32)
    return combined


def write_astrometry_net_combined_fits(
    science_path: str, combined_2d: np.ndarray, out_path: str
) -> None:
    """Write the 4-CCD mosaic as a single primary-HDU FITS (for Astrometry.net)."""
    image = np.asarray(combined_2d, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    out_abs = os.path.abspath(out_path)
    d = os.path.dirname(out_abs)
    if d:
        os.makedirs(d, exist_ok=True)
    with fits.open(science_path) as hdul_in:
        hdr = hdul_in[0].header.copy()
    _header_keys_remove_for_primary(hdr)
    hdr.add_history(
        "4-CCD mosaic 2D in primary, KeplerCam 2x2 layout (reduce_science.py, Astrometry.net)."
    )
    pri = fits.PrimaryHDU(data=image, header=hdr)
    pri.header["BITPIX"] = -32
    fits.HDUList([pri]).writeto(out_abs, overwrite=True, output_verify="ignore")


def plot_mosaic_preview(
    mosaic_2d: np.ndarray, out_png: str, *, title: str = "Reduced mosaic (2×2 CCDs)"
) -> None:
    """Save a quick grayscale PNG of the mosaic (1–99% stretch). Requires matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("plot_mosaic_preview needs matplotlib: pip install matplotlib") from e

    z = np.asarray(mosaic_2d, dtype=float)
    zf = z[np.isfinite(z)]
    if zf.size == 0:
        raise RuntimeError("Mosaic has no finite pixels; cannot plot.")
    lo, hi = np.nanpercentile(zf, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(zf)), float(np.nanmax(zf))
        if hi <= lo:
            hi = lo + 1.0
    out_abs = os.path.abspath(out_png)
    d = os.path.dirname(out_abs)
    if d:
        os.makedirs(d, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    ax.imshow(z, origin="lower", vmin=lo, vmax=hi, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    fig.tight_layout()
    fig.savefig(out_abs, bbox_inches="tight", dpi=150)
    plt.close(fig)


def collect_calibration_paths(data_dir: str) -> tuple[list[str], list[str]]:
    """Same discovery as ReduceScience notebook / ``Reduction.py`` ``main()``."""
    bias_files = glob.glob(os.path.join(data_dir, "*BIAS*.fits")) + glob.glob(
        os.path.join(data_dir, "*bias*.fits")
    )
    all_flat_file_paths: list[str] = []
    for fn in os.listdir(data_dir):
        if "FLAT" in fn.upper() and fn.lower().endswith(".fits"):
            fp = os.path.join(data_dir, fn)
            if os.path.isfile(fp):
                all_flat_file_paths.append(fp)
    if not bias_files:
        raise FileNotFoundError(
            f"No bias FITS in {data_dir!r}. Expected files matching *BIAS*.fits or *bias*.fits"
        )
    if not all_flat_file_paths:
        raise FileNotFoundError(
            f"No flat FITS in {data_dir!r}. Expected filenames containing 'FLAT' and ending in .fits"
        )
    return bias_files, all_flat_file_paths


def run_reduce(
    science_fits: Optional[str],
    data_dir: Optional[str],
    output_dir: Optional[str],
    astrometry_ccd: Optional[int] = None,
    write_mosaic: bool = True,
    plot_mosaic: bool = True,
) -> str:
    """
    Resolve paths, reduce, and write ``stemReduced.fits`` plus, by default,
    ``stemReduced_mosaic.fits`` (4-CCD 2×2 single image in the primary HDU).
    If ``astrometry_ccd`` is 1–4, also writes ``stemReduced_ccdN_astrometry.fits``.
    Set ``write_mosaic`` to False to skip the mosaic file.
    If ``plot_mosaic`` is True, also writes ``stemReduced_mosaic.png`` (needs matplotlib).
    Returns the absolute path to the main multi-CCD product.
    """
    science = science_fits
    if science is None or str(science).strip() == "":
        science = input("Full path to science FITS: ").strip().strip('"').strip("'")

    science_path = os.path.abspath(os.path.expanduser(science))
    if not os.path.isfile(science_path):
        raise FileNotFoundError(f"Science file not found: {science_path}")

    ddir = os.path.abspath(os.path.expanduser(data_dir)) if data_dir else os.path.dirname(science_path)
    odir = os.path.abspath(os.path.expanduser(output_dir)) if output_dir else os.path.dirname(science_path)

    print(f"Science:  {science_path}")
    print(f"Cal dir:  {ddir}")
    print(f"Out dir:  {odir}")

    bias_files, all_flat_file_paths = collect_calibration_paths(ddir)
    print(f"Found {len(bias_files)} bias and {len(all_flat_file_paths)} flat candidate file(s).")

    reduced, used_filter = reduce_science_ccds(
        science_path, bias_files, all_flat_file_paths, LEFT_TRIM, RIGHT_TRIM
    )

    orig_stem = Path(science_path).stem
    # Always use lowercase .fits for the product (input may be .FITS / .fits).
    out_name = f"{orig_stem}Reduced.fits"
    out_path = os.path.join(odir, out_name)

    write_reduced_multichip_fits(science_path, reduced, out_path)
    out_abs = os.path.abspath(out_path)
    print(f"Filter: {used_filter!r}")
    print(f"Wrote: {out_abs}")

    if astrometry_ccd is not None:
        if astrometry_ccd not in (1, 2, 3, 4):
            raise ValueError("astrometry_ccd must be 1, 2, 3, or 4")
        ast_name = f"{orig_stem}Reduced_ccd{astrometry_ccd}_astrometry.fits"
        ast_path = os.path.join(odir, ast_name)
        write_astrometry_net_fits(
            science_path, reduced[astrometry_ccd], astrometry_ccd, ast_path
        )
        print(f"Astrometry.net (primary HDU = CCD {astrometry_ccd}): {os.path.abspath(ast_path)}")

    if write_mosaic or plot_mosaic:
        combo = combine_reduced_ccds_mosaic(reduced)
        if write_mosaic:
            comb_name = f"{orig_stem}Reduced_mosaic.fits"
            comb_path = os.path.join(odir, comb_name)
            write_astrometry_net_combined_fits(science_path, combo, comb_path)
            print(f"Mosaic (2×2 CCDs, single HDU): {os.path.abspath(comb_path)}")
        if plot_mosaic:
            png_name = f"{orig_stem}Reduced_mosaic.png"
            png_path = os.path.join(odir, png_name)
            plot_mosaic_preview(
                combo,
                png_path,
                title=f"{orig_stem} — reduced mosaic (2×2)",
            )
            print(f"Mosaic preview PNG: {os.path.abspath(png_path)}")

    return out_abs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reduce KeplerCam 4-CCD science FITS (same pipeline as ReduceScience.ipynb)."
    )
    p.add_argument(
        "science",
        nargs="?",
        default=None,
        help="Path to science FITS; if omitted, you are prompted (same as the notebook when SCIENCE_FITS is None).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Directory with bias/flat FITS (default: same directory as the science file).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Where to write <name>Reduced.fits (default: same directory as the science file).",
    )
    p.add_argument(
        "--astrometry-ccd",
        type=int,
        choices=(1, 2, 3, 4),
        default=None,
        metavar="N",
        help=(
            "Also write <name>Reduced_ccdN_astrometry.fits: one CCD in the primary HDU only "
            "(for Astrometry.net and similar)."
        ),
    )
    p.add_argument(
        "--no-mosaic",
        action="store_true",
        help="Do not write the 4-CCD 2x2 combined FITS (*Reduced_mosaic.fits).",
    )
    p.add_argument(
        "--plot-mosaic",
        action="store_true",
        help="Save a quick grayscale PNG (*Reduced_mosaic.png) next to the products (needs matplotlib).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_reduce(
            science_fits=args.science,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            astrometry_ccd=args.astrometry_ccd,
            write_mosaic=not args.no_mosaic,
            plot_mosaic=bool(args.plot_mosaic),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
