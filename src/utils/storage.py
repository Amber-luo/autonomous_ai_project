from pathlib import Path
import pandas as pd


def read_parquet(file_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)

# 将DataFrame写入Parquet文件
def write_parquet(df: pd.DataFrame, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
