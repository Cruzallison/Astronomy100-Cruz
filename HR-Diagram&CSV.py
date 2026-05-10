#!/usr/bin/env python3
"""
This script is used to pair photometry from two images of the same field.
It is used to create a catalog of stars in the field and then pair the stars in the two images.
Then, the script will export the paired stars to a CSV file.
This script will also optionally plot the apertures on top of the images in an exported file.
Finally, it will also plot the isochrones on top of the HR diagram.

**Dependencies** (install in a terminal): ``pip install astropy matplotlib numpy photutils``
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils import aperture
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars

# -----------------------------------------------------------------------------
# Enter here the paths to the G-band and R-band images and the output CSV file
# -----------------------------------------------------------------------------
FITS_IMAGE_G = r"/Path/Directory/For/G-band.fits"
FITS_IMAGE_R = r"/Path/Directory/For/R-band.fits"
IDE_OUTPUT_CSV = r"/Path/Directory/For/OutputData.csv (for use in Glue or inspection)"

# ------------------------------------------------------------
# Enter here the parameters for the photometry
# ------------------------------------------------------------
IDE_HDU_G = 0
IDE_HDU_R = 0
IDE_MAX_SEP = 5.0
IDE_ZP_G = 0
IDE_ZP_R = 0
# If True (IDE run only), run SDSS step (see IDE_SDSS_MODE: zp = median offsets for all stars).
IDE_USE_SDSS = True
IDE_SDSS_RADIUS_ARCSEC = 3.0
IDE_SDSS_BATCH = 80
IDE_SDSS_DR = 17
# ``zp`` (default) or ``replace`` — passed as ``--sdss-mode`` when IDE_USE_SDSS is True.
IDE_SDSS_MODE = "zp"
# If > 0, append ABS_MAG_g / ABS_MAG_r (set 0 to skip). Same μ convention as ``hr_from_astrometry_fits``.
IDE_DISTANCE_PC = 850.0
# If non-empty, save aperture/annulus overlay PNGs (see --plot-apertures). Example: r"/path/plot_base"
IDE_PLOT_APERTURES_PREFIX = r"/users/cruzallison/astrodir/data/plot_base"
# Max sources drawn per band (first N rows); large fields stay readable.
IDE_PLOT_APERTURES_MAX = 200

# ------------------------------------------------------------
# Enter here the path to the ``*.dat`` isochrone. This should be correct but may have to change.
# ------------------------------------------------------------

IDE_PLOT_CMD_ISOCHRONE = "/data/4Gyr-isochrone.dat"
# Optional: also save CMD figure to this PNG path (empty = display only).
IDE_PLOT_CMD_OUT = ""


_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from hr_from_astrometry_fits import one_to_one_match_sky  # noqa: E402

# Match Reduction.py (single-CCD / generic field; no per-CCD right-edge map).
STAR_FWHM = 1.2
STAR_SIGMA_THRESHOLD = 4.0
STAR_EDGE_MASK_PIXELS = 25
STAR_EDGE_MASK_RIGHT_EXTRA = 60
STAR_EDGE_MASK_RIGHT_EXTRA_BY_CCD: dict[int, int] = {}
STAR_ROUNDNESS_LIMIT = 1.0
STAR_SHARP_LO = 0.2
STAR_SHARP_HI = 1.0
APERTURE_RADIUS = 6.0
ANNULUS_R_IN = 8.0
ANNULUS_R_OUT = 12.0


def _resolve_aperture_plot_paths(prefix: str) -> tuple[str, str]:
    """Return (path_g, path_r) for aperture diagnostic PNGs."""
    p = os.path.abspath(os.path.expanduser(prefix.strip()))
    if os.path.isdir(p):
        return (
            os.path.join(p, "dao_apertures_g.png"),
            os.path.join(p, "dao_apertures_r.png"),
        )
    if p.lower().endswith(".png"):
        base, _ = os.path.splitext(p)
    else:
        base = p
    return f"{base}_g.png", f"{base}_r.png"


def save_dao_aperture_diagnostic_png(
    image: np.ndarray,
    tab: Table,
    out_path: str,
    *,
    band_label: str,
    max_sources: int,
) -> None:
    """
    Grayscale image with Photutils circular apertures (lime) and sky annuli (cyan) at DAO centroids.
    """
    if "x_pix" not in tab.colnames or "y_pix" not in tab.colnames or len(tab) == 0:
        print(f"Skipping aperture plot ({band_label}): no positions in catalog.", file=sys.stderr)
        return

    import matplotlib.pyplot as plt

    x = np.asarray(tab["x_pix"], dtype=float)
    y = np.asarray(tab["y_pix"], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        print(f"Skipping aperture plot ({band_label}): no finite x/y.", file=sys.stderr)
        return

    n = len(x)
    n_draw = min(int(max_sources), n)
    if n_draw < n:
        x, y = x[:n_draw], y[:n_draw]

    positions = np.transpose((x, y))
    src_ap = aperture.CircularAperture(positions, r=APERTURE_RADIUS)
    bkg_ann = aperture.CircularAnnulus(positions, r_in=ANNULUS_R_IN, r_out=ANNULUS_R_OUT)

    img = np.asarray(image, dtype=float)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        print(f"Skipping aperture plot ({band_label}): image has no finite pixels.", file=sys.stderr)
        return
    vmin, vmax = np.percentile(finite, [2.0, 99.5])

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    src_ap.plot(ax=ax, color="lime", lw=0.9, alpha=0.85)
    bkg_ann.plot(ax=ax, color="deepskyblue", lw=0.65, alpha=0.8)
    ax.set_title(
        f"DAOStarFinder + apertures ({band_label})  r={APERTURE_RADIUS}px  "
        f"annulus {ANNULUS_R_IN}–{ANNULUS_R_OUT}px  (N={n_draw}/{n})"
    )
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.tight_layout()
    out_abs = os.path.abspath(os.path.expanduser(out_path))
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    fig.savefig(out_abs, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote aperture diagnostic ({band_label}): {out_abs}")


def mask_edges_for_detection(image_2d: np.ndarray, ccd_num: int | None = None) -> np.ndarray:
    """Same idea as ``Reduction.py``: NaN borders before DAOStarFinder."""
    arr = np.array(image_2d, dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    if STAR_EDGE_MASK_PIXELS > 0:
        arr[:STAR_EDGE_MASK_PIXELS, :] = np.nan
        arr[-STAR_EDGE_MASK_PIXELS:, :] = np.nan
        arr[:, :STAR_EDGE_MASK_PIXELS] = np.nan
        arr[:, -STAR_EDGE_MASK_PIXELS:] = np.nan
    right_extra = STAR_EDGE_MASK_RIGHT_EXTRA
    if ccd_num is not None:
        right_extra += int(STAR_EDGE_MASK_RIGHT_EXTRA_BY_CCD.get(ccd_num, 0))
    if right_extra > 0:
        arr[:, -right_extra:] = np.nan
    return arr


def _pick_first_image_hdu(path: str, hdu_prefer: int | None) -> tuple[int, np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        if hdu_prefer is not None and hdu_prefer < len(hdul):
            hdu = hdul[hdu_prefer]
            d = hdu.data
            if d is not None and getattr(d, "ndim", 0) == 2:
                return hdu_prefer, np.asarray(d, dtype=float), hdu.header.copy()
        for i, hdu in enumerate(hdul):
            d = hdu.data
            if d is None or getattr(d, "ndim", 0) != 2:
                continue
            return i, np.asarray(d, dtype=float), hdu.header.copy()
    raise ValueError(f"{path}: no 2D image HDU found (try --hdu-g / --hdu-r).")


def _flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=float)
    out = np.full_like(f, np.nan, dtype=float)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def dao_aperture_catalog(
    image: np.ndarray,
    wcs: WCS,
    *,
    ccd_num: int | None = None,
    run_extract_stars: bool = True,
) -> Table:
    """
    DAOStarFinder on background-subtracted detection frame, annulus sky, aperture sum,
    instrumental magnitude, RA/Dec from WCS.
    """
    science_for_finder = mask_edges_for_detection(image, ccd_num=ccd_num)
    valid_pixels = np.isfinite(science_for_finder)
    if np.count_nonzero(valid_pixels) == 0:
        raise RuntimeError("No valid pixels after edge masking.")
    _, median_val, std_val = sigma_clipped_stats(
        science_for_finder[valid_pixels], sigma=3.0, maxiters=5
    )
    threshold = STAR_SIGMA_THRESHOLD * std_val
    finder = DAOStarFinder(
        threshold=threshold,
        fwhm=STAR_FWHM,
        sharplo=STAR_SHARP_LO,
        sharphi=STAR_SHARP_HI,
        roundlo=-STAR_ROUNDNESS_LIMIT,
        roundhi=STAR_ROUNDNESS_LIMIT,
        exclude_border=True,
    )
    detections = finder(np.nan_to_num(science_for_finder - median_val, nan=0.0))
    if detections is None or len(detections) == 0:
        return Table(names=["ra", "dec", "mag_inst"], dtype=[float, float, float])

    x = np.asarray(detections["xcentroid"], dtype=float)
    y = np.asarray(detections["ycentroid"], dtype=float)
    hh, ww = image.shape
    edge = ANNULUS_R_OUT + 1
    right_extra = edge + STAR_EDGE_MASK_RIGHT_EXTRA
    if ccd_num is not None:
        right_extra += int(STAR_EDGE_MASK_RIGHT_EXTRA_BY_CCD.get(ccd_num, 0))
    keep = (x > edge) & (x < (ww - right_extra)) & (y > edge) & (y < (hh - edge))
    det_f = detections[keep]
    x = np.asarray(det_f["xcentroid"], dtype=float)
    y = np.asarray(det_f["ycentroid"], dtype=float)
    if len(x) == 0:
        return Table(names=["ra", "dec", "mag_inst"], dtype=[float, float, float])

    if run_extract_stars:
        try:
            nd = NDData(
                np.nan_to_num(image - median_val, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
            )
            n_try = min(40, len(det_f))
            if n_try > 0:
                extract_stars(nd, det_f[:n_try], size=25)
        except Exception:
            pass

    positions = np.transpose((x, y))
    src_ap = aperture.CircularAperture(positions, r=APERTURE_RADIUS)
    bkg_ann = aperture.CircularAnnulus(positions, r_in=ANNULUS_R_IN, r_out=ANNULUS_R_OUT)
    src_phot = aperture.aperture_photometry(image, src_ap)
    bkg_median = []
    for ann_mask in bkg_ann.to_mask(method="center"):
        annulus_data = ann_mask.multiply(image)
        ann_vals = annulus_data[ann_mask.data > 0]
        ann_vals = ann_vals[np.isfinite(ann_vals)]
        bkg_median.append(float(np.median(ann_vals)) if ann_vals.size > 0 else np.nan)
    bkg_median = np.array(bkg_median, dtype=float)
    bkg_total = bkg_median * src_ap.area
    flux_bkgsub = np.asarray(src_phot["aperture_sum"], dtype=float) - bkg_total

    ra, dec = wcs.all_pix2world(x, y, 0)
    ra = np.asarray(ra, dtype=float).ravel()
    dec = np.asarray(dec, dtype=float).ravel()
    mag_inst = _flux_to_inst_mag(flux_bkgsub)

    out = Table()
    out["x_pix"] = x
    out["y_pix"] = y
    out["ra"] = ra
    out["dec"] = dec
    out["mag_inst"] = mag_inst
    out["flux_bkgsub"] = flux_bkgsub
    return out


def build_matched_export(
    tab_g: Table,
    tab_r: Table,
    zp_g: float,
    zp_r: float,
    max_sep_arcsec: float,
) -> Table:
    mag_g = np.asarray(tab_g["mag_inst"], dtype=float) + float(zp_g)
    mag_r = np.asarray(tab_r["mag_inst"], dtype=float) + float(zp_r)
    ok_g = np.isfinite(tab_g["ra"]) & np.isfinite(tab_g["dec"]) & np.isfinite(mag_g)
    ok_r = np.isfinite(tab_r["ra"]) & np.isfinite(tab_r["dec"]) & np.isfinite(mag_r)
    tab_g = tab_g[ok_g].copy()
    tab_r = tab_r[ok_r].copy()
    mag_g = mag_g[ok_g]
    mag_r = mag_r[ok_r]
    if len(tab_g) == 0 or len(tab_r) == 0:
        raise SystemExit("No stars with finite sky position and magnitude in one or both bands.")

    i_g, i_r, seps = one_to_one_match_sky(
        tab_g["ra"], tab_g["dec"], tab_r["ra"], tab_r["dec"], float(max_sep_arcsec)
    )
    if len(i_g) == 0:
        raise SystemExit("No sky matches; increase --max-sep or check WCS / field overlap.")

    ra = 0.5 * (np.asarray(tab_g["ra"][i_g]) + np.asarray(tab_r["ra"][i_r]))
    dec = 0.5 * (np.asarray(tab_g["dec"][i_g]) + np.asarray(tab_r["dec"][i_r]))
    mg = mag_g[i_g]
    mr = mag_r[i_r]
    paired = Table()
    paired["RA"] = ra
    paired["DEC"] = dec
    paired["MAG_g"] = mg
    paired["MAG_r"] = mr
    paired["MAG_g_minus_MAG_r"] = mg - mr
    paired["match_sep_arcsec"] = seps
    return paired


def distance_modulus_parsecs(distance_pc: float) -> float:
    """Distance modulus μ = 5 log10(d) − 5 for *d* in parsecs."""
    return 5.0 * np.log10(float(distance_pc)) - 5.0


def append_absolute_magnitudes(tab: Table, distance_pc: float) -> Table:
    """If ``distance_pc`` > 0, add ``ABS_MAG_g`` and ``ABS_MAG_r`` (apparent mags minus μ)."""
    d = float(distance_pc)
    if d <= 0:
        return tab
    mu = distance_modulus_parsecs(d)
    out = tab.copy()
    out["ABS_MAG_g"] = np.asarray(out["MAG_g"], dtype=float) - mu
    out["ABS_MAG_r"] = np.asarray(out["MAG_r"], dtype=float) - mu
    return out


def load_cmd_dat_isochrone_gr_g(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PARSEC/CMD 3.x ``*.dat`` isochrone: ``#`` comment lines, whitespace-separated data.
    Parses the header line that lists column names including ``gmag`` and ``rmag`` (SDSS).
    Returns ``(g_minus_r, g_mag)`` sorted by ``Mini`` or ``Mass`` when present, else by ``gmag``.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise SystemExit(f"CMD isochrone file not found: {path}")

    colnames: list[str] | None = None
    rows: list[list[float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s[1:].strip()
                parts = body.split()
                if "gmag" in parts and "rmag" in parts:
                    colnames = parts
                continue
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue

    if colnames is None:
        raise SystemExit(
            f"{path}: no comment header with column names including 'gmag' and 'rmag' "
            "(expected PARSEC/CMD SDSS isochrone output)."
        )
    try:
        ig = colnames.index("gmag")
        ir = colnames.index("rmag")
    except ValueError as e:
        raise SystemExit(f"{path}: column header missing gmag or rmag.") from e

    if not rows:
        raise SystemExit(f"{path}: no numeric data rows found.")

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= max(ig, ir):
        raise SystemExit(f"{path}: data rows are too short for gmag/rmag column indices.")

    g_iso = arr[:, ig]
    r_iso = arr[:, ir]
    ok = np.isfinite(g_iso) & np.isfinite(r_iso)
    arr = arr[ok]
    g_iso, r_iso = g_iso[ok], r_iso[ok]
    if len(g_iso) == 0:
        raise SystemExit(f"{path}: no finite gmag/rmag values.")

    gr = g_iso - r_iso
    sort_key: np.ndarray | None = None
    for key in ("Mini", "Mass"):
        if key in colnames:
            j = colnames.index(key)
            sort_key = arr[:, j]
            break
    if sort_key is not None and np.all(np.isfinite(sort_key)):
        order = np.argsort(sort_key)
        gr, g_iso = gr[order], g_iso[order]
    else:
        order = np.argsort(g_iso)
        gr, g_iso = gr[order], g_iso[order]

    return gr, g_iso


def show_cmd_isochrone_overlay(
    stars: Table,
    isochrone_dat: str,
    *,
    save_path: str | None = None,
) -> None:
    """
    CMD / HR diagram: x = g−r, y = g. Uses ``ABS_MAG_g`` when present, else ``MAG_g``.
    Isochrone: ``gmag``, ``rmag`` from CMD ``*.dat``. Displays with ``plt.show()``; optional PNG save.
    """
    import matplotlib.pyplot as plt

    if "MAG_g_minus_MAG_r" not in stars.colnames or "MAG_g" not in stars.colnames:
        raise SystemExit("CMD plot: stars table needs MAG_g and MAG_g_minus_MAG_r.")

    gr_obs = np.asarray(stars["MAG_g_minus_MAG_r"], dtype=float)
    if "ABS_MAG_g" in stars.colnames:
        g_obs = np.asarray(stars["ABS_MAG_g"], dtype=float)
        y_label = r"$M_g$"
        data_label = "Data (ABS_MAG_g)"
        note = "Isochrone: PARSEC/CMD SDSS absolute mags; data: ABS_MAG_g (use --distance-pc)."
    else:
        g_obs = np.asarray(stars["MAG_g"], dtype=float)
        y_label = r"$m_g$ (apparent)"
        data_label = "Data (MAG_g)"
        note = "Isochrone: absolute M_g; data: apparent — overlap only if modulus matches model."

    ok = np.isfinite(gr_obs) & np.isfinite(g_obs)
    gr_obs, g_obs = gr_obs[ok], g_obs[ok]

    gr_iso, g_iso = load_cmd_dat_isochrone_gr_g(isochrone_dat)

    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    ax.scatter(
        gr_obs,
        g_obs,
        s=8,
        alpha=0.45,
        c="0.25",
        edgecolors="none",
        label=data_label,
        rasterized=True,
    )
    ax.plot(gr_iso, g_iso, color="royalblue", lw=1.6, alpha=0.9, label="Isochrone (CMD .dat)")
    ax.set_xlabel(r"$(g - r)$")
    ax.set_ylabel(y_label)
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(1.5, 10.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.set_title("Color–magnitude (SDSS)\n" + note, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if save_path and save_path.strip():
        out_abs = os.path.abspath(os.path.expanduser(save_path.strip()))
        os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
        fig.savefig(out_abs, dpi=160, bbox_inches="tight")
        print(f"Wrote CMD overlay copy: {out_abs}")
    print("Showing CMD / HR diagram (close the figure window to continue).")
    plt.show()
    plt.close(fig)


def _sdss_mag_range_ok(gmag: float, rmag: float) -> bool:
    return (
        np.isfinite(gmag)
        and np.isfinite(rmag)
        and -90.0 < gmag < 50.0
        and -90.0 < rmag < 50.0
    )


def _robust_median_1d(x: np.ndarray, *, n_sigma: float = 3.5, max_iters: int = 3) -> float:
    """Sigma-clipped median for photometric offset samples."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if x.size < 5:
        return float(np.median(x))
    ok = np.ones(x.size, dtype=bool)
    for _ in range(max_iters):
        if np.count_nonzero(ok) < 3:
            break
        med = float(np.median(x[ok]))
        mad = float(np.median(np.abs(x[ok] - med)))
        sig = 1.4826 * mad
        if sig < 1e-9:
            break
        ok = ok & (np.abs(x - med) < n_sigma * sig)
    return float(np.median(x[ok]))


def fetch_sdss_psfmags_for_positions(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    radius_arcsec: float,
    batch_size: int,
    data_release: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SDSS ``query_crossid`` batched; returns parallel ``psfMag_g``, ``psfMag_r`` (NaN if no valid row).
    """
    try:
        from astroquery.sdss import SDSS  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            "SDSS catalog step needs astroquery. Install with: pip install astroquery"
        ) from e

    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    n = len(ra_deg)
    sdss_g = np.full(n, np.nan, dtype=float)
    sdss_r = np.full(n, np.nan, dtype=float)

    r_arcsec = float(radius_arcsec)
    if r_arcsec <= 0 or r_arcsec > 179.0:
        raise SystemExit("SDSS cross-ID radius must be in (0, 180] arcseconds (server limit 3 arcmin).")

    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        coords = SkyCoord(ra=ra_deg[start:stop] * u.deg, dec=dec_deg[start:stop] * u.deg, frame="icrs")
        try:
            result = SDSS.query_crossid(
                coords,
                radius=r_arcsec * u.arcsec,
                spectro=False,
                photoobj_fields=["ra", "dec", "psfMag_g", "psfMag_r"],
                data_release=int(data_release),
            )
        except Exception as e:
            print(f"SDSS query_crossid batch [{start}:{stop}]: {e}", file=sys.stderr)
            continue

        if result is None or len(result) == 0:
            continue

        for row in result:
            try:
                name = row["name"]
            except KeyError:
                continue
            if name is None:
                continue
            sname = str(name).strip()
            if not sname.startswith("obj_"):
                continue
            try:
                i_local = int(sname.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            gi = start + i_local
            if gi < 0 or gi >= n:
                continue

            gmag = float(row["psfMag_g"])
            rmag = float(row["psfMag_r"])
            if not _sdss_mag_range_ok(gmag, rmag):
                continue
            sdss_g[gi] = gmag
            sdss_r[gi] = rmag

    return sdss_g, sdss_r


def apply_sdss_zp_offsets_to_paired(
    paired: Table,
    *,
    radius_arcsec: float = 3.0,
    batch_size: int = 80,
    data_release: int = 17,
    min_calibrators: int = 3,
) -> Table:
    """
    Cross-ID SDSS; robust median ``Δg = psfMag_g − MAG_g``, ``Δr = psfMag_r − MAG_r`` on matched
    stars; add those offsets to **every** row's ``MAG_g`` / ``MAG_r`` (and recompute color).
    """
    ra = np.asarray(paired["RA"], dtype=float)
    dec = np.asarray(paired["DEC"], dtype=float)
    n = len(ra)
    if n == 0:
        return paired

    mg0 = np.asarray(paired["MAG_g"], dtype=float).copy()
    mr0 = np.asarray(paired["MAG_r"], dtype=float).copy()

    sdss_g, sdss_r = fetch_sdss_psfmags_for_positions(
        ra,
        dec,
        radius_arcsec=radius_arcsec,
        batch_size=batch_size,
        data_release=data_release,
    )

    cal = np.isfinite(sdss_g) & np.isfinite(sdss_r) & np.isfinite(mg0) & np.isfinite(mr0)
    n_cal = int(np.count_nonzero(cal))
    if n_cal < int(min_calibrators):
        raise SystemExit(
            f"SDSS ZP: need at least {min_calibrators} stars with valid SDSS g,r and finite "
            f"instrumental mags; got {n_cal}. Try a larger --sdss-radius or check footprint."
        )

    dg = sdss_g[cal] - mg0[cal]
    dr = sdss_r[cal] - mr0[cal]
    zp_g = _robust_median_1d(dg)
    zp_r = _robust_median_1d(dr)
    if not np.isfinite(zp_g) or not np.isfinite(zp_r):
        raise SystemExit("SDSS ZP: could not compute finite robust median offsets.")

    print(
        f"SDSS ZP offsets (robust median of m_SDSS − MAG): Δg = {zp_g:.5f} mag, Δr = {zp_r:.5f} mag "
        f"(from {n_cal} cross-ID matches)."
    )

    mg = mg0 + zp_g
    mr = mr0 + zp_r
    out = paired.copy()
    out["MAG_g"] = mg
    out["MAG_r"] = mr
    out["MAG_g_minus_MAG_r"] = mg - mr
    out["sdss_zp_offset_g"] = np.full(n, zp_g, dtype=float)
    out["sdss_zp_offset_r"] = np.full(n, zp_r, dtype=float)
    out["sdss_calibrator"] = cal
    return out


def apply_sdss_replace_to_paired(
    paired: Table,
    *,
    radius_arcsec: float = 3.0,
    batch_size: int = 80,
    data_release: int = 17,
) -> Table:
    """
    Replace ``MAG_g`` / ``MAG_r`` with SDSS ``psfMag_*`` where cross-ID returns valid magnitudes;
    leave other rows unchanged.
    """
    ra = np.asarray(paired["RA"], dtype=float)
    dec = np.asarray(paired["DEC"], dtype=float)
    n = len(ra)
    if n == 0:
        return paired

    sdss_g, sdss_r = fetch_sdss_psfmags_for_positions(
        ra,
        dec,
        radius_arcsec=radius_arcsec,
        batch_size=batch_size,
        data_release=data_release,
    )

    mg = np.asarray(paired["MAG_g"], dtype=float).copy()
    mr = np.asarray(paired["MAG_r"], dtype=float).copy()
    sdss_used = np.isfinite(sdss_g) & np.isfinite(sdss_r)

    mg[sdss_used] = sdss_g[sdss_used]
    mr[sdss_used] = sdss_r[sdss_used]

    out = paired.copy()
    out["MAG_g"] = mg
    out["MAG_r"] = mr
    out["MAG_g_minus_MAG_r"] = mg - mr
    out["sdss_used"] = sdss_used
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DAO photometry on two FITS images, ZP, sky-match, export g/r paired CSV."
    )
    p.add_argument("fits_g", nargs="?", help="Image FITS (g / bluer band).")
    p.add_argument("fits_r", nargs="?", help="Image FITS (r / redder band).")
    p.add_argument("-o", "--output-csv", default="", help="Output CSV path.")
    p.add_argument("--hdu-g", type=int, default=None, help="HDU index for g image (default: auto).")
    p.add_argument("--hdu-r", type=int, default=None, help="HDU index for r image (default: auto).")
    p.add_argument("--zp-g", type=float, default=27.75689308166504, help="Additive ZP on g (same as hr --zp-a).")
    p.add_argument("--zp-r", type=float, default=22.601222633361814, help="Additive ZP on r (same as hr --zp-b).")
    p.add_argument("--max-sep", type=float, default=30.0, help="Max match radius in arcsec (default: 30).")
    p.add_argument("--ccd-num", type=int, default=None, help="Optional CCD index for Reduction right-edge mask dict.")
    p.add_argument("--no-extract-stars", action="store_true", help="Skip photutils extract_stars sanity call.")
    p.add_argument(
        "--sdss",
        action="store_true",
        help="Use SDSS cross-ID for Sloan calibration (needs astroquery + network); see --sdss-mode.",
    )
    p.add_argument(
        "--sdss-mode",
        choices=("zp", "replace"),
        default="zp",
        help="zp: robust median (m_SDSS−MAG) per band applied to all stars (default). replace: use SDSS mags only where matched.",
    )
    p.add_argument(
        "--sdss-min-calibrators",
        type=int,
        default=3,
        metavar="N",
        help="Minimum SDSS cross-ID stars required for zp mode (default 3).",
    )
    p.add_argument(
        "--sdss-radius",
        type=float,
        default=3.0,
        metavar="ARCSEC",
        help="SDSS cross-ID search radius in arcseconds (default 3; max ~180).",
    )
    p.add_argument(
        "--sdss-batch",
        type=int,
        default=80,
        metavar="N",
        help="Stars per SDSS query_crossid batch (default 80).",
    )
    p.add_argument("--sdss-dr", type=int, default=17, metavar="DR", help="SDSS data release (default 17).")
    p.add_argument(
        "--distance-pc",
        type=float,
        default=0.0,
        metavar="PC",
        help="If > 0, add ABS_MAG_g and ABS_MAG_r (M = m − μ, μ = 5·log10(d)−5 in mag). 0 = omit (default).",
    )
    p.add_argument(
        "--plot-apertures",
        default="",
        metavar="PREFIX",
        help="Save PNG overlays of apertures/annuli: PREFIX_g.png and PREFIX_r.png (or PREFIX/dao_apertures_*.png if PREFIX is a directory). Empty = skip.",
    )
    p.add_argument(
        "--plot-apertures-max",
        type=int,
        default=400,
        metavar="N",
        help="Max sources drawn per band in aperture PNGs (default 400; first N detections).",
    )
    p.add_argument(
        "--plot-cmd-isochrone",
        default="",
        metavar="FILE.dat",
        help="PARSEC/CMD .dat (gmag, rmag); show interactive CMD (x=g−r, y=M_g if ABS_MAG_g else m_g).",
    )
    p.add_argument(
        "--plot-cmd-out",
        default="",
        metavar="FILE.png",
        help="If set, also save the CMD figure to this PNG after displaying. Default: display only.",
    )
    args = p.parse_args(argv)

    fg = args.fits_g
    fr = args.fits_r
    if not fg or not fr:
        raise SystemExit("Provide two FITS paths (g then r) or use IDE constants with no argv.")

    path_g = os.path.abspath(os.path.expanduser(fg))
    path_r = os.path.abspath(os.path.expanduser(fr))
    for path in (path_g, path_r):
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")

    ig, data_g, hdr_g = _pick_first_image_hdu(path_g, args.hdu_g)
    ir, data_r, hdr_r = _pick_first_image_hdu(path_r, args.hdu_r)
    print(f"g FITS: {path_g}  (HDU {ig}, shape {data_g.shape})")
    print(f"r FITS: {path_r}  (HDU {ir}, shape {data_r.shape})")

    try:
        wcs_g = WCS(hdr_g, naxis=2)
        wcs_r = WCS(hdr_r, naxis=2)
    except Exception as e:
        raise SystemExit(f"WCS not usable (need astrometric 2D WCS in image HDU): {e}") from e

    ccd = int(args.ccd_num) if args.ccd_num is not None else None
    ext = not args.no_extract_stars
    tab_g = dao_aperture_catalog(data_g, wcs_g, ccd_num=ccd, run_extract_stars=ext)
    tab_r = dao_aperture_catalog(data_r, wcs_r, ccd_num=ccd, run_extract_stars=ext)
    print(f"Detected stars (finite aperture mags): g={len(tab_g)}, r={len(tab_r)}")

    plot_pre = args.plot_apertures.strip()
    if plot_pre:
        pg, pr = _resolve_aperture_plot_paths(plot_pre)
        save_dao_aperture_diagnostic_png(
            data_g,
            tab_g,
            pg,
            band_label="g",
            max_sources=max(1, int(args.plot_apertures_max)),
        )
        save_dao_aperture_diagnostic_png(
            data_r,
            tab_r,
            pr,
            band_label="r",
            max_sources=max(1, int(args.plot_apertures_max)),
        )

    paired = build_matched_export(tab_g, tab_r, args.zp_g, args.zp_r, args.max_sep)
    if args.sdss:
        if args.sdss_mode == "zp":
            paired = apply_sdss_zp_offsets_to_paired(
                paired,
                radius_arcsec=args.sdss_radius,
                batch_size=args.sdss_batch,
                data_release=args.sdss_dr,
                min_calibrators=args.sdss_min_calibrators,
            )
            out_only = paired[
                "RA",
                "DEC",
                "MAG_r",
                "MAG_g",
                "MAG_g_minus_MAG_r",
                "sdss_zp_offset_g",
                "sdss_zp_offset_r",
                "sdss_calibrator",
            ].copy()
        else:
            paired = apply_sdss_replace_to_paired(
                paired,
                radius_arcsec=args.sdss_radius,
                batch_size=args.sdss_batch,
                data_release=args.sdss_dr,
            )
            n_sdss = int(np.count_nonzero(np.asarray(paired["sdss_used"], dtype=bool)))
            print(f"SDSS psfMag replaced for {n_sdss} / {len(paired)} paired stars (replace mode).")
            out_only = paired["RA", "DEC", "MAG_r", "MAG_g", "MAG_g_minus_MAG_r", "sdss_used"].copy()
    else:
        out_only = paired["RA", "DEC", "MAG_r", "MAG_g", "MAG_g_minus_MAG_r"].copy()

    out_only = append_absolute_magnitudes(out_only, args.distance_pc)
    if float(args.distance_pc) > 0:
        dpc = float(args.distance_pc)
        print(f"Absolute magnitudes: d = {dpc:.1f} pc, μ = {distance_modulus_parsecs(dpc):.4f} mag.")

    print(f"Paired stars: {len(out_only)}")

    out_csv = args.output_csv.strip()
    if not out_csv:
        raise SystemExit("Set -o / --output-csv (or IDE_OUTPUT_CSV when using Run).")
    out_path = os.path.abspath(os.path.expanduser(out_csv))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cmd_iso = args.plot_cmd_isochrone.strip()
    if cmd_iso:
        save_png = args.plot_cmd_out.strip() or None
        show_cmd_isochrone_overlay(out_only, cmd_iso, save_path=save_png)

    out_only.write(out_path, format="csv", overwrite=True)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1 and FITS_IMAGE_G.strip() and FITS_IMAGE_R.strip():
        ide = [
            FITS_IMAGE_G.strip(),
            FITS_IMAGE_R.strip(),
            "-o",
            os.path.expanduser(IDE_OUTPUT_CSV.strip() or "dao_paired_g_r.csv"),
            "--max-sep",
            str(float(IDE_MAX_SEP)),
            "--zp-g",
            str(float(IDE_ZP_G)),
            "--zp-r",
            str(float(IDE_ZP_R)),
        ]
        if IDE_HDU_G is not None:
            ide.extend(["--hdu-g", str(int(IDE_HDU_G))])
        if IDE_HDU_R is not None:
            ide.extend(["--hdu-r", str(int(IDE_HDU_R))])
        if IDE_USE_SDSS:
            ide.append("--sdss")
            ide.extend(["--sdss-mode", str(IDE_SDSS_MODE)])
            ide.extend(["--sdss-radius", str(float(IDE_SDSS_RADIUS_ARCSEC))])
            ide.extend(["--sdss-batch", str(int(IDE_SDSS_BATCH))])
            ide.extend(["--sdss-dr", str(int(IDE_SDSS_DR))])
        if float(IDE_DISTANCE_PC) > 0:
            ide.extend(["--distance-pc", str(float(IDE_DISTANCE_PC))])
        if IDE_PLOT_APERTURES_PREFIX.strip():
            ide.extend(["--plot-apertures", IDE_PLOT_APERTURES_PREFIX.strip()])
            ide.extend(["--plot-apertures-max", str(int(IDE_PLOT_APERTURES_MAX))])
        if IDE_PLOT_CMD_ISOCHRONE.strip():
            ide.extend(["--plot-cmd-isochrone", IDE_PLOT_CMD_ISOCHRONE.strip()])
            if IDE_PLOT_CMD_OUT.strip():
                ide.extend(["--plot-cmd-out", os.path.expanduser(IDE_PLOT_CMD_OUT.strip())])
        raise SystemExit(main(ide))
    raise SystemExit(main())
