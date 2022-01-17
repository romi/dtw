#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       File contributor(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

"""
Dynamic time warping algorithm(s)
"""
# {# pkglts, src
# FYEO
# #}
# {# pkglts, version, after src
from . import version

__version__ = version.__version__
# #}

from dtw.dtw import DTW
