def get_rois(timeframe_mins):
    rois = {
        "0": 0.15,  # 5% for the first 3 candles
        str(timeframe_mins * 10): 0.10,  # 2% after 3 candles
        str(timeframe_mins * 20): 0.05,  # 1% After 6 candles
        str(timeframe_mins * 30): 0.03,  # 1% After 9 candles
        str(timeframe_mins * 40): 0.01  # 1% After 12 candles
    }
    return rois
