"""Common figure style (serif fonts, Okabe-Ito colours, consistent algorithm encoding)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
ALG_STYLE = {
    "GG-MOEA": dict(color="#D55E00", marker="o", ls="-"),
    "NSGA-II": dict(color="#0072B2", marker="s", ls="--"),
    "NSGA-III": dict(color="#009E73", marker="^", ls="-."),
    "MOEA/D": dict(color="#E69F00", marker="v", ls=":"),
    "RVEA": dict(color="#56B4E9", marker="D", ls="--"),
    "AGE-MOEA-II": dict(color="#CC79A7", marker="P", ls="-."),
    "CCMO": dict(color="#000000", marker="X", ls=":"),
    "CMOPSO": dict(color="#999999", marker="*", ls="--"),
}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#d9d9d9",
        "lines.linewidth": 1.2,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=600)
    plt.close(fig)
