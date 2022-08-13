def get_rois(timeframe_mins):
    rois = {
        "0": 0.05,  # 5% for the first 3 candles
        str(timeframe_mins * 3): 0.02,  # 2% after 3 candles
        str(timeframe_mins * 6): 0.01,  # 1% After 6 candles
    }
    return rois
