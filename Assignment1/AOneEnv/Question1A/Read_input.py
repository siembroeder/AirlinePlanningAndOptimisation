from pathlib import Path
import pandas as pd


def read_excel_pandas(path: Path, sheets: list, indx = 0) -> list:
    """
    Reads input graph from an Excel file using pandas.read_excel().

    Alternatively, use [Polars](https://pola.rs/).
    """
    df_list = []
    # Read sheets into DataFrames
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name = sheet, index_col=indx)
        df_list.append(df)

    return df_list


# Dij = log(k)+b1log(pi*pj)+b2log(gi*gj)-b3log(f*dij)