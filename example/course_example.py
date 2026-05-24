!git clone https://ciot.imis.ncku.edu.tw:25388/Amy/stock_project_for_class.git

%cd stock_project_for_class
!pip install -r requirements.txt


# 引入套件
from stock_api import get_taiwan_stock_data

# 取得台股資訊 (例如台積電 2330)
df = get_taiwan_stock_data("2330", "2024-01-01", "2026-03-31")

# 在 Colab 中直接呼叫變數 df，會自動渲染出漂亮的表格
df

# 1. 從套件中匯入買入 (Buy_Stock) 與賣出 (Sell_Stock) 的功能
# (同時匯入 Get_User_Stocks 可以用來確認目前資產)
from stock_api import Buy_Stock, Sell_Stock, Get_User_Stocks

# 2. 設定您的帳號與密碼 (請替換成您在課堂系統中的實際帳密)
account = "您的帳號"
password = "您的密碼"


# ====================================================
# 【功能一：預約購入股票】
# ====================================================
# 參數說明：Buy_Stock(帳號, 密碼, 股票代號, 購入張數, 購入價格)
target_stock = 2330   # 欲購買的股票代號（例如：2330 台積電）
buy_shares = 1        # 欲購入的張數
buy_price = 1975      # 預約購入的價格

print(f"正在預約買入股票 ({target_stock})...")
buy_success = Buy_Stock(account, password, target_stock, buy_shares, buy_price)

# 判斷是否成功
if buy_success == True:
    print("預約買入成功！")
else:
    print("預約買入失敗，請檢查帳號密碼或金額是否足夠。")


print("-" * 50)


# ====================================================
# 【功能二：預約售出股票】
# ====================================================
# 參數說明：Sell_Stock(帳號, 密碼, 股票代號, 售出張數, 售出價格)
sell_shares = 1       # 欲賣出的張數
sell_price = 1978     # 預約賣出的價格

print(f"正在預約賣出股票 ({target_stock})...")
sell_success = Sell_Stock(account, password, target_stock, sell_shares, sell_price)

# 判斷是否成功
if sell_success == True:
    print("預約賣出成功！")
else:
    print("預約賣出失敗，請檢查您是否確實持有該股票。")

簡易投資法：月線趨勢策略 Python 實作

買進規則：當「最新收盤價」大於「過去20天的平均價格」，代表趨勢向上，買進！

賣出規則：當「最新收盤價」跌破「過去20天的平均價格」，代表趨勢向下，賣出！

其他時間：什麼都不做，安心睡覺。

import pandas as pd
from datetime import datetime, timedelta
from stock_api import get_taiwan_stock_data, Buy_Stock, Sell_Stock, Get_User_Stocks

# ==========================================
# 1. 基本設定區 (請填入你的帳密)
# ==========================================
account = "您的帳號"
password = "您的密碼"
target_stock_str = "2330" # 抓取資料用的字串格式
target_stock_int = 2330   # 買賣下單用的數字格式

print(f"啟動傻瓜投資法，目標股票：{target_stock_str}")

# ==========================================
# 2. 獲取數據與計算指標
# ==========================================
# 設定時間範圍：抓取過去 40 天的資料 (為了確保能算出 20 日均線，扣除假日需多抓幾天)
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")

# 呼叫 API 取得歷史資料
df = get_taiwan_stock_data(target_stock_str, start_date, end_date)

# 確保資料是照日期排序的，然後取得最新收盤價與 20 日均線 (MA20)
latest_close = df['close'].iloc[-1]
ma20 = df['close'].tail(20).mean() # 取最後20筆收盤價算平均

print(f"最新收盤價: {latest_close:.2f} 元")
print(f"過去20日平均線(MA20): {ma20:.2f} 元")

# ==========================================
# 3. 檢查目前庫存狀態
# ==========================================
# 取得持有股票
user_stocks = Get_User_Stocks(account, password)
print(f"目前庫存: {user_stocks}")

# 判斷是否已經持有該股票 (註：需依照 Get_User_Stocks 實際回傳的格式調整，這裡假設回傳陣列包含代號)
# 如果 user_stocks 是一個包含字典的陣列，例如 [{'stock_code': 2330, 'shares': 1}]，你需要自己寫一個小迴圈去判斷
has_stock = False
if user_stocks:
    # 這邊提供一個防呆寫法，假設只要字串裡有出現代碼就算持有
    if str(target_stock_int) in str(user_stocks):
        has_stock = True

# ==========================================
# 4. 傻瓜決策機器人 (核心邏輯)
# ==========================================
print("-" * 30)
if latest_close > ma20 and not has_stock:
    print("傻瓜訊號：股價【突破】月線，趨勢向上，準備買進！")
    # 用最新收盤價掛單買進 1 張
    Buy_Stock(account, password, target_stock_int, 1, int(latest_close))

elif latest_close < ma20 and has_stock:
    print("傻瓜訊號：股價【跌破】月線，趨勢向下，準備停利/停損！")
    # 用最新收盤價掛單賣出 1 張
    Sell_Stock(account, password, target_stock_int, 1, int(latest_close))

else:
    print("💡 傻瓜訊號：沒有觸發條件，或是已經滿倉/空手。按兵不動！")