#!/bin/bash

set -x -e
"${PYTHON}" setup.py install
scons install --prefix="${PREFIX}" --jobs="${CPU_COUNT}"
rm -rf "${PREFIX}"/include