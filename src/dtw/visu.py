#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

"""
Plotly methods to use in jupyter notebooks with DTW objects.
"""

import dtw
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sequences_table(dtw: dtw.DTW):
    """Plotly table presenting reference and test sequences found in DTW object."""
    titles = ['test', 'reference']
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=titles,
                        horizontal_spacing=0.03, specs=[[{"type": "table"}] * 2])
    for i, (seq_name, seq) in enumerate(zip(titles, [dtw.seq_test, dtw.seq_ref])):
        table = go.Table(header={'values': dtw.names, 'font': {'size': 10}, 'align': "center"},
                         cells={'values': seq.T, 'align': "left"})
        fig.add_trace(table, row=1, col=i + 1)

    fig.update_layout(title_text="DTW - Original sequences")
    return fig


def _sequence_plot(fig, fig_idx, ref_idx, ref_val, test_idx, test_val, name, pred_type=None):
    # Scatter plot of the reference sequence:
    ref_sc = go.Scatter(x=ref_idx, y=ref_val, name='reference', legendgroup=fig_idx,
                        mode='lines+markers', line=dict(color='firebrick', width=2, dash='dash'))
    # Scatter plot of the test sequence:
    test_sc = go.Scatter(x=test_idx, y=test_val, name='test', legendgroup=fig_idx,
                         mode='lines+markers', line=dict(color='royalblue', width=2, dash='dash'))
    if pred_type is not None:
        for idx, pred in enumerate(pred_type):
            if pred != '':
                fig.add_annotation(x=test_idx[idx], y=test_val[idx], text=pred, bgcolor='rgba(0,0,0,0.1)',
                                   font={"family": "Courier New", "size": 14, 'color': 'black'}, row=fig_idx + 1, col=1)
    # Add the scatter plots to the figure:
    fig.add_trace(ref_sc, row=fig_idx + 1, col=1)
    fig.add_trace(test_sc, row=fig_idx + 1, col=1)
    # Add the name of the sequence as Y-axis label:
    fig.update_yaxes(title_text=name, row=fig_idx + 1, col=1)
    fig.update_traces(textposition='top center')

    return fig


def plot_sequences(dtw: dtw.DTW):
    """Plotly scatter plot presenting reference and test sequences found in DTW object."""
    n_figs = dtw.n_dim
    ref_indexes = np.array(range(dtw.n_ref))
    test_indexes = np.array(range(dtw.n_test))

    fig = make_subplots(rows=n_figs, cols=1, vertical_spacing=0.1, subplot_titles=dtw.names)
    for i in range(n_figs):
        if dtw.n_dim == 1:
            fig = _sequence_plot(fig, i, ref_indexes + 1, dtw.seq_ref, test_indexes + 1, dtw.seq_test, name=dtw.names)
        else:
            fig = _sequence_plot(fig, i, ref_indexes + 1, dtw.seq_ref[:, i], test_indexes + 1, dtw.seq_test[:, i], name=dtw.names[i])

    fig.update_xaxes(title_text="index", row=n_figs, col=1)
    fig.update_layout(height=800, title_text="DTW - Original sequences",
                      hovermode="x unified", legend_title="Legend")
    return fig


def plot_aligned_sequences(dtw: dtw.DTW):
    """Plotly scatter plot presenting aligned reference and test sequences found in DTW object."""
    n_figs = dtw.n_dim

    ref_indexes = np.array(range(dtw.n_ref))
    results = dtw.get_better_results(start_index=0)
    seq_test = results['test']
    seq_ref = results['reference']
    pred_types = results['type']
    pred_types[np.where(pred_types == '=')] = ''
    pred_types = [pt.upper() for pt in pred_types]

    fig = make_subplots(rows=n_figs, cols=1, vertical_spacing=0.1, subplot_titles=dtw.names)
    for i in range(n_figs):
        if dtw.n_dim == 1:
            ref_val = dtw.seq_ref
            test_val = [dtw.seq_test[e] for e in seq_test]
            names = dtw.names
        else:
            ref_val = dtw.seq_ref[:, i]
            test_val = [dtw.seq_test[e, i] for e in seq_test]
            names = dtw.names[i]
        fig = _sequence_plot(fig, i, ref_indexes+1, ref_val, seq_ref+1, test_val, names, pred_types)

    fig.update_xaxes(title_text="index of reference sequence", row=n_figs, col=1)
    fig.update_layout(height=800, title_text="DTW - Aligned sequences",
                      hovermode="x unified", legend_title="Legend")
    return fig
