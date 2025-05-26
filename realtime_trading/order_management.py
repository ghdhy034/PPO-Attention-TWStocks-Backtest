# realtime_trading/order_management.py
def place_order(order_details):
    """
    模擬下單：order_details 為字典，包含股票代號、買賣方向、數量、價格等資訊。
    實際上線時請接入 shioaji API。
    """
    print(f"Placing order: {order_details}")
    return "ORDER12345"

def check_order_status(order_id):
    """
    模擬查詢訂單狀態。
    """
    print(f"Checking status for order: {order_id}")
    return "Filled"

def cancel_order(order_id):
    """
    模擬取消訂單。
    """
    print(f"Cancelling order: {order_id}")
    return True

if __name__ == "__main__":
    order = place_order({"symbol": "2330", "side": "buy", "quantity": 100, "price": 150})
    status = check_order_status(order)
    print(f"Order status: {status}")
    cancel_order(order)
