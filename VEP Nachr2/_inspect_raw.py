import pandas as pd
# Inspect raw columns across the data files (for Idea B: disease/mechanism metadata)
for f in ["data/raw/final_mapped.xlsx", "data/raw/final.xlsx"]:
    try:
        xl = pd.ExcelFile(f)
        print(f"\n=== {f} sheets: {xl.sheet_names}")
        for sh in xl.sheet_names[:3]:
            df = pd.read_excel(f, sheet_name=sh, nrows=3)
            print(f"  [{sh}] cols ({len(df.columns)}): {list(df.columns)}")
    except Exception as e:
        print(f"\n=== {f}: ERROR {e}")
