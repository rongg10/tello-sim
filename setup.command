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
    # The physics engine sets the floor. MuJoCo publishes wheels for Python
    # 3.10 and newer only; on 3.9 pip falls back to building from source and
    # fails with "MUJOCO_PATH environment variable is not set", which does not
    # remotely suggest "your Python is too old".
    #
    # tkinter sets the preference. run_gui_sim.py needs it, and on macOS it is
    # easy to lack: the system Python has it but is 3.9, and Homebrew and pyenv
    # builds only have it if Tcl/Tk was installed when they were built. The one
    # that does have it is often not the one on the PATH -- a pyenv version that
    # is not the global default, say -- so pyenv's installs are searched too.
    CANDIDATES="python3.13 python3.12 python3.11 python3.10 python3"
    if command -v pyenv >/dev/null 2>&1; then
      for dir in $(ls -1dr "$(pyenv root)"/versions/[0-9]* 2>/dev/null); do
        CANDIDATES="$CANDIDATES $dir/bin/python3"
      done
    fi
    new_enough() {
      "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
    }
    has_tk() { "$1" -c 'import tkinter' >/dev/null 2>&1; }

    # First choice: new enough AND has tkinter. Otherwise settle for new enough,
    # because without physics nothing runs, whereas without tkinter only the
    # GUI launcher is missing.
    BASE=""
    for candidate in $CANDIDATES; do
      command -v "$candidate" >/dev/null 2>&1 || continue
      if new_enough "$candidate" && has_tk "$candidate"; then BASE="$candidate"; break; fi
    done
    if [ -z "$BASE" ]; then
      for candidate in $CANDIDATES; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if new_enough "$candidate"; then BASE="$candidate"; break; fi
      done
    fi
    if [ -z "$BASE" ]; then
      echo "No Python 3.10 or newer found, and MuJoCo needs one."
      echo "Install one, for example:"
      echo "    brew install python@3.13"
      echo "then run this script again."
      exit 1
    fi
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
echo "  Score the benchmark scenarios (SR, OSR, SPL, energy):"
echo "      python run_benchmark.py --all"
echo

# The GUI launcher needs tkinter. Say plainly whether this environment has it.
if "$VENV/bin/python" -c "import tkinter" >/dev/null 2>&1; then
  echo "  Fly your own tkinter controller against it:"
  echo "      python run_gui_sim.py --controller /path/to/controller.py"
  echo
else
  echo "Note: this environment has no tkinter, so run_gui_sim.py cannot start"
  echo "here. Everything else works, including the 3D view, which is a web page"
  echo "and needs no Tk at all."
  echo
  echo "The GUI launcher needs a Python that is 3.10 or newer AND has tkinter."
  echo "If you already have an environment like that, activate it and run this"
  echo "script again: it installs into the active environment. Otherwise:"
  echo "      brew install python@3.13 python-tk@3.13"
  echo "      rm -rf .venv && ./setup.command"
  echo
fi
