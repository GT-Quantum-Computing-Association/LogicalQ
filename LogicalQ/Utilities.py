import os
import time
from pathlib import Path
from datetime import datetime

def sanitize_save_parameters(filename, save_dir, default_filename="file", default_save_dir="./", default_extension=".png"):
    if filename is None or filename.strip() == "":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{default_filename}_{timestamp}{default_extension}"
    else:
        base = os.path.basename(filename)
        name, ext = os.path.splitext(base)
        if ext == "":
            filename = base + default_extension
        else:
            filename = base
    
    # Determine save directory
    if Path(filename).parent != Path('.'):
        # User included directory in filename
        file_dir = str(Path(filename).parent)
        filename = Path(filename).name
        if file_dir != "":
          save_dir = file_dir
        else:
          save_dir = default_save_dir
    else:
        # No directory in filename
        if save_dir is None or save_dir.strip() == "":
            save_dir = default_save_dir

    # Ensure save_dir ends with separator
    if not save_dir.endswith(os.sep):
        save_dir += os.sep
