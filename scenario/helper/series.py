# Plot series shared by the rq1 and rq2 figures: both vision-language backbones
# are drawn on the same axes, so (model, method) has to collapse to one label.
# Missing combinations are skipped by the plot scripts, which is how the
# CLIP-only baseline coexists with a MobileCLIP semantic run.

SERIES = {
    ('clip', 'semcom'): 'SemCom (CLIP)',
    ('mobileclip', 'semcom'): 'SemCom (MobileCLIP)',
    ('clip', 'baseline'): 'JPEG + 16-QAM',
    ('mobileclip', 'baseline'): 'JPEG + 16-QAM (MobileCLIP)',
}

SERIES_ORDER = list(SERIES.values())


def save_legend(handles, labels, out, width_pt, ncol=3):
    # Standalone legend: three series sweep across both corners of every panel,
    # so an inline legend always covers a curve. The paper places this once
    # under a subfigure pair instead.
    import matplotlib.pyplot as plt

    from . import plotstyle

    fig = plt.figure(figsize=(width_pt / 72.27, 0.3))
    fig.legend(handles, labels, loc='center', ncol=ncol, frameon=False,
               fontsize=plotstyle.FONT_PT)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'[save] {out}')
