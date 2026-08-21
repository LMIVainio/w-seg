try:
    from ._version import version as __version__
except ImportError:
    __version__ = '0.0.1'  # Keeps your default 0.0.1 if git is missing

import os
import platform

os.environ['KERAS_BACKEND'] = 'torch'
from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._writer import write_multiple, write_single_image
from ._wsegutils import InstanceNormalization

# Fix for Windows PyTorch DLL loading issues
if platform.system() == 'Windows':
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
    except (AttributeError, OSError):
        pass

__all__ = (
    'napari_get_reader',
    'write_single_image',
    'write_multiple',
    'make_sample_data',
    'w_net_processor_widget',
    'make_widget',
    'InstanceNormalization',
)
