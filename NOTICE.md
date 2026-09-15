# Third-party components

## three.js

`tello_sim/viewer/vendor/three.min.js` is three.js r128, vendored so the 3D
view works with no internet connection. It is unmodified, and it carries its
own licence header.

Copyright 2010-2021 Three.js Authors, MIT licence.
https://github.com/mrdoob/three.js

## djitellopy

Imported, not vendored. Installed from PyPI by `requirements.txt`, MIT licence.
https://github.com/damiafuentes/DJITelloPy

## Not included

The tkinter controller that `run_gui_sim.py` and
`scripts/demo_drive_controller.py` can launch is not part of this repository.
No such controller is published here. Both scripts take a `--controller` path,
also settable as `TELLO_CONTROLLER`, and explain what to do when they cannot
find one.
