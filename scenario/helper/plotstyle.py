# Shared figure style. Every panel is generated at the exact width it is shown
# at in the paper, so LaTeX never rescales it and the text in a figure matches
# the text around it. Scaling a 252 pt figure into a 121 pt slot is what makes
# figure labels come out half-size.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEEtran conference: one column is 252 pt, the full text block is 516 pt
COLUMN_PT = 252.0
TEXT_PT = 516.0
# caption/footnote size in a 10 pt IEEEtran document
FONT_PT = 8


def apply():
    import seaborn as sns

    sns.set_theme(style='whitegrid', font='serif')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': FONT_PT,
        'axes.labelsize': FONT_PT,
        'xtick.labelsize': FONT_PT,
        'ytick.labelsize': FONT_PT,
        'legend.fontsize': FONT_PT,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def figsize(width_pt, aspect):
    # width_pt is the on-page width, not a design width
    w = width_pt / 72.27
    return (w, w * aspect)
