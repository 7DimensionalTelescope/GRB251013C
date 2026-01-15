import pandas as pd
from .const import CIRCULAR_DATA_FILENAME

def read_circular_data():
    """Read circular data from Excel file.
    
    Returns:
        pd.DataFrame: The circular data as a pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the circular data file does not exist.
    """
    return pd.read_excel(CIRCULAR_DATA_FILENAME)
