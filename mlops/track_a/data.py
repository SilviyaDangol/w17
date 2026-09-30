"""Download and validate the official IBM Telco Customer Churn dataset."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA = Path("data/Telco-Customer-Churn.csv")
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
REQUIRED = {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"}
EXPECTED_MINIMUM_ROWS = 7_000


def load_data() -> pd.DataFrame:
    """Return the official data; reject accidental toy/synthetic substitutes."""
    DATA.parent.mkdir(exist_ok=True)
    if not DATA.exists() or _row_count(DATA) < EXPECTED_MINIMUM_ROWS:
        temporary = DATA.with_suffix(".download")
        urlretrieve(URL, temporary)
        temporary.replace(DATA)
    df = pd.read_csv(DATA)
    if len(df) < EXPECTED_MINIMUM_ROWS or not REQUIRED.issubset(df.columns):
        raise ValueError("Expected the official 7,043-row IBM Telco Customer Churn CSV.")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)
