def get_rois():
    return {
        "60": 0.00,
        "50": 0.01,
        "40": 0.02,
        "30": 0.04,
        "20": 0.08,
        "0": 0.10
    }

    # roi_prefs = {}
    # p = 0.02
    # for i in range(0, 10000, 30):
    #     v = (100 - p * i) / 100
    #     if v >= 0:
    #         roi_prefs[f"{i}"] = v
    # return roi_prefs

