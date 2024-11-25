from enum import Enum

from Problems.lens_3d import setup as setup_lens_3D
from Problems.lens_3d import problem as problem_lens_3D
from Problems.lens_3d import config as config_lens_3D
from Problems.lens_3d import plot as plot_lens_3D
from Problems.cross import setup as set_cross
from Problems.cross import problem as problem_cross
from Problems.cross import config as config_cross
from Problems.cross import plot as plot_cross




class Dispenser(Enum):
    """
    Add the respective problems  here and their setup, problem and config classes as the call. Plot and data saving
    are also passed here.
    """
    LENS3D = (setup_lens_3D.LensSetup3D, problem_lens_3D.LensProblem3D, config_lens_3D.LensConfig3D,
              plot_lens_3D.plot_maker(), plot_lens_3D.data_maker())
    CROSS = (set_cross.DemultiplexerSetup, problem_cross.DemultiplexerProblem,
                     config_cross.DemultiplexerConfig,
                     plot_cross.plot_maker(), plot_cross.data_maker())
