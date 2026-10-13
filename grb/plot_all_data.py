"""Quick-look plot of every photometric dataset returned by read_data().

Combines the optical sources (i_data, circular, sdt) into one flux-density
panel (mJy, left axis) and overlays XRT (erg/cm^2/s, right axis) on a shared
time-since-trigger x-axis. Non-detections are drawn as downward arrows.

Run from the `fitting` directory:
    python -m grb.plot_all_data
"""
import argparse
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .const import BASE_DIR
from .io import read_data
from .utils import seconds_from_trigger

SECONDS_PER_HOUR = 3600.0

# Cycled through when assigning a marker shape to each facility; reused if a
# dataset has more facilities than markers.
FACILITY_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "8"]

# Concatenation of matplotlib's three 20-color qualitative palettes, giving 60
# maximally-distinct discrete colors. Indexed directly (not sampled
# continuously) so adjacent filters never blend into similar shades the way
# `cm.tab20(np.linspace(...))` does.
FILTER_COLORS = (
    list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
)


def _load_optical(keyword, filter_col, facility_col, upper_limit_col=None):
    """Load one optical dataset and return standardized columns."""
    df = read_data(keyword, correct_galactic_extinction=True, add_converted_flux=True)

    if "time" not in df.columns:
        df = df.copy()
        df["time"] = df["date_obs"].map(seconds_from_trigger)

    out = df[["time", "flux_mJy", "flux_mJy_error"]].copy()
    out.columns = ["time_hr", "flux", "flux_err"]
    out["time_hr"] = out["time_hr"] / SECONDS_PER_HOUR
    out["filter"] = df[filter_col].astype(str)
    out["facility"] = df[facility_col].astype(str)
    out["upper_limit"] = df[upper_limit_col].astype(bool) if upper_limit_col else False
    out["source"] = keyword
    return out


def load_all_data():
    import pandas as pd

    circular = _load_optical("circular", "filter", "facility", "upper_limit")
    # NUTTelA-TAO's circular-reported photometry disagrees with (and is superseded
    # by) the dedicated i_data.csv reduction for the same facility/filter; drop it
    # here so the two don't get plotted on top of each other.
    circular = circular[circular["facility"] != "NUTTelA-TAO"]

    optical = pd.concat(
        [
            _load_optical("i_data", "filter", "facility"),
            circular,
            # sdt has no generic "facility" column; "units" is the 7DT unit-telescope ID.
            _load_optical("sdt", "filter_name", "units", "is_upper_limit"),
        ],
        ignore_index=True,
    )
    xrt = read_data("xrt")
    return optical, xrt


def plot_all_data(
    outfile, filters=None, exclude_filters=None, exclude_facilities=None, by_facility=False,
):
    optical, xrt = load_all_data()

    if filters is not None:
        missing = sorted(set(filters) - set(optical["filter"].unique()))
        if missing:
            raise ValueError(f"No data for filter(s): {missing}")
        optical = optical[optical["filter"].isin(filters)]

    if exclude_filters is not None:
        missing = sorted(set(exclude_filters) - set(optical["filter"].unique()))
        if missing:
            raise ValueError(f"No data for filter(s): {missing}")
        optical = optical[~optical["filter"].isin(exclude_filters)]

    if exclude_facilities is not None:
        missing = sorted(set(exclude_facilities) - set(optical["facility"].unique()))
        if missing:
            raise ValueError(f"No data for facility(s): {missing}")
        optical = optical[~optical["facility"].isin(exclude_facilities)]

    fig, ax_opt = plt.subplots(figsize=(9, 6))
    ax_xrt = ax_opt.twinx()

    filter_names = sorted(optical["filter"].unique())
    color_by_filter = {
        filt: FILTER_COLORS[i % len(FILTER_COLORS)] for i, filt in enumerate(filter_names)
    }

    facility_names = sorted(optical["facility"].unique())
    marker_by_facility = dict(zip(facility_names, cycle(FACILITY_MARKERS)))

    for filt in filter_names:
        filt_rows = optical[optical["filter"] == filt]
        color = color_by_filter[filt]
        facilities_here = sorted(filt_rows["facility"].unique()) if by_facility else [None]

        for fac in facilities_here:
            rows = filt_rows[filt_rows["facility"] == fac] if fac is not None else filt_rows
            marker = marker_by_facility[fac] if by_facility else "o"

            det = rows[~rows["upper_limit"]]
            if len(det):
                ax_opt.errorbar(
                    det["time_hr"], det["flux"], yerr=det["flux_err"],
                    fmt=marker, ms=4, color=color, ecolor=color, alpha=0.8,
                    label=filt if fac is None else None, zorder=3,
                )

            ul = rows[rows["upper_limit"]]
            if len(ul):
                ax_opt.errorbar(
                    ul["time_hr"], ul["flux"], yerr=0.3 * ul["flux"],
                    uplims=True, fmt=marker, ms=4, color=color, alpha=0.5, zorder=2,
                )

    xrt_yerr = [np.abs(xrt["flux_low"]), xrt["flux_high"]]
    ax_xrt.errorbar(
        xrt["time"] / SECONDS_PER_HOUR, xrt["flux"], yerr=xrt_yerr,
        fmt=".", ms=3, color="black", ecolor="gray", alpha=0.6,
        label="XRT (0.3-10 keV)", zorder=1,
    )

    ax_opt.set_xscale("log")
    ax_opt.set_yscale("log")
    ax_xrt.set_yscale("log")
    ax_opt.set_xlabel("Time since trigger [hr]")
    ax_opt.set_ylabel("Optical flux density [mJy]")
    ax_xrt.set_ylabel(r"X-ray flux [erg cm$^{-2}$ s$^{-1}$]")
    ax_opt.set_title("GRB 251013C - all photometry")

    handles_xrt, labels_xrt = ax_xrt.get_legend_handles_labels()
    if by_facility:
        # Filter labels weren't attached to any artist above (each artist is
        # split per facility instead), so build the color legend from proxies.
        filter_handles = [
            Line2D([], [], color=color_by_filter[filt], marker="o", ls="", label=filt)
            for filt in filter_names
        ]
        filter_legend = ax_opt.legend(
            filter_handles + handles_xrt, filter_names + labels_xrt,
            fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.15, 1.0),
            title="Filter",
        )
        ax_opt.add_artist(filter_legend)

        facility_handles = [
            Line2D([], [], color="gray", marker=marker_by_facility[fac], ls="", label=fac)
            for fac in facility_names
        ]
        ax_opt.legend(
            handles=facility_handles, fontsize=8, ncol=1, loc="upper left",
            bbox_to_anchor=(1.15, 1.0 - 0.045 * (len(filter_handles) // 2 + 2)),
            title="Facility",
        )
    else:
        handles_opt, labels_opt = ax_opt.get_legend_handles_labels()
        ax_opt.legend(
            handles_opt + handles_xrt, labels_opt + labels_xrt,
            fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.15, 1.0),
        )
    ax_opt.grid(which="major", linestyle="-", color="gray", alpha=0.3, lw=0.5)

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outfile", default=str(Path(BASE_DIR) / "plots" / "all_data_overview.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--filters", nargs="+", default=None,
        help="Restrict to these optical filter names (default: all available)",
    )
    parser.add_argument(
        "--exclude-filters", nargs="+", default=None,
        help="Drop these optical filter names from the plot",
    )
    parser.add_argument(
        "--exclude-facilities", nargs="+", default=None,
        help="Drop these facility names from the plot",
    )
    parser.add_argument(
        "--by-facility", action="store_true",
        help="Distinguish facilities within the same filter band by marker shape",
    )
    args = parser.parse_args()
    plot_all_data(
        args.outfile,
        filters=args.filters,
        exclude_filters=args.exclude_filters,
        exclude_facilities=args.exclude_facilities,
        by_facility=args.by_facility,
    )


if __name__ == "__main__":
    main()
