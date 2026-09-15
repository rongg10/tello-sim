#!/bin/bash
# Install the simulator's dependencies into the existing DJI environment.
# Double-click from Finder, or run it from a terminal.
cd "$(dirname "$0")"

DJI="$HOME/.pyenv/versions/DJI"

if [ ! -x "$DJI/bin/python" ]; then
  echo "Could not find the DJI environment at:"
  echo "    $DJI"
  echo
  echo "Either create it, or install into whatever Python you prefer with:"
  echo "    pip install -r requirements.txt"
  exit 1
fi

echo "Installing into $("$DJI/bin/python" -V) at $DJI"
echo
"$DJI/bin/pip" install -r requirements.txt || exit 1

echo
echo "Ready. To use it:"
echo "    source $DJI/bin/activate"
echo
echo "  Live 3D view you can fly with buttons:"
echo "      python run_sim3d.py --wind 1.5 --gusty"
echo
echo "  Start a simulated drone and point anything at 127.0.0.1:"
echo "      python run_sim.py"
echo
echo "  Run every scenario and make the videos:"
echo "      python run_scenario.py --all --video"
echo

# The GUI launcher needs tkinter. Say plainly whether this environment has it.
if "$DJI/bin/python" -c "import tkinter" >/dev/null 2>&1; then
  echo "  Fly the team's own tello_controller.py against it:"
  echo "      python run_gui_sim.py --wind 1.5"
  echo
else
  echo "Note: this environment has no tkinter, so run_gui_sim.py cannot start"
  echo "here (nor can the team's tello_controller.py). Everything else works."
  echo "Rebuild the environment on a Python built with Tk to get the GUI back."
  echo
fi
