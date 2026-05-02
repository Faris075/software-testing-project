import pathlib
import sys

# Ensure project root is first in sys.path so local packages take priority
# over any similarly-named installed packages (e.g. site-packages/analysis).
ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
