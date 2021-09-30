import numpy as np
import pandas as pd
from typing import Dict, Tuple

events_T = Dict[str, pd.DatetimeIndex]
mask_T = Dict[str, np.ndarray]
nested_T = Dict[str, pd.DataFrame]
outcomes_T = Dict[str, pd.Series]
pred_lcl_T = Dict[str, pd.Series]
pred_gbl_T = Dict[str, pd.DatetimeIndex]
prediction_T = Dict[str, Tuple[pd.Series, pd.DatetimeIndex]]
sensors_T = Dict[str, pd.DataFrame]
