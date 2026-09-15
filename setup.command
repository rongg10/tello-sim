#!/bin/bash
# Install the simulator's dependencies. Double-click from Finder, or run it
# from a terminal. Creates .venv alongside this script unless one already
# exists, or unless a virtual environment is already active.
cd "$(dirname "$0")"

if [ -n "$VIRTUAL_ENV" ]; then
  VENV="$VIRTUAL_ENV"
  echo "Using the active environment at $VENV"
else
  VENV="$PWD/.venv"
  if [ ! -x "$VENV/bin/python" ]; then
    # Prefer a Python with tkinter so run_gui_sim.py works. On macOS the
    # system Python has it; many pyenv and Homebrew builds do not.
    BASE=python3
    for candidate in /usr/bin/python3 python3; do
      if "$candidate" -c "import tkinter" >/dev/null 2>&1; then BASE="$candidate"; break; fi
    done
    echo "Creating $VENV with $("$BASE" -V)"
    "$BASE" -m venv "$VENV" || exit 1
  fi
fi

echo "Installing into $("$VENV/bin/python" -V)"
echo
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -r requirements.txt || exit 1

echo
echo "Ready. To use it:"
echo "    source $VENV/bin/activate"
echo
echo "  Live 3D view you can fly with buttons:"
echo "      python run_sim3d.py --wind 1.5 --gusty"
echo
echo "  A prompt where you type the djitellopy API itself:"
echo "      python run_cli.py"
echo
echo "  Start a simulated drone and point anything at 127.0.0.1:"
echo "      python run_sim.py"
echo
echo "  Run every scenario and make the videos:"
echo "      python run_scenario.py --all --video"
echo

# The GUI launcher needs tkinter. Say plainly whether this environment has it.
if "$VENV/bin/python" -c "import tkinter" >/dev/null 2>&1; then
  echo "  Fly your own tkinter controller against it:"
  echo "      python run_gui_sim.py --controller /path/to/controller.py"
  echo
else
  echo "Note: this environment has no tkinter, so run_gui_sim.py cannot start"
  echo "here. Everything else works. Rebuild the environment on a Python built"
  echo "with Tk to get the GUI launcher back."
  echo
fi
