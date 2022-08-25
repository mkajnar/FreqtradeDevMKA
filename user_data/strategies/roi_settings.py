def get_rois(timeframe_mins):
    rois = {
        "0": 0.01,
        str(timeframe_mins * 300): 0.005,
        str(timeframe_mins * 600): 0.003,
        str(timeframe_mins * 900): 0.001
    }
    return rois
