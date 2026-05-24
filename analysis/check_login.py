"""Quick login probe. Reads creds from env to avoid hardcoding."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stock_project_for_class"))
from stock_api import Get_User_Stocks  # noqa: E402

acct = os.environ["STOCK_ACCT"]
pw = os.environ["STOCK_PW"]

result = Get_User_Stocks(acct, pw)
print("type:", type(result).__name__)
print("repr:", repr(result)[:500])
