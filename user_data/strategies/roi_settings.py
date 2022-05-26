def get_rois():
    return {
        "40": 0.0,  # Exit after 40 minutes if the profit is not negative
        "30": 0.01,  # Exit after 30 minutes if there is at least 1% profit
        "20": 0.02,  # Exit after 20 minutes if there is at least 2% profit
        "0": 0.04  # Exit immediately if there is at least 4% profit
    }

    # roi_prefs = {}
    # p = 0.02
    # for i in range(0, 10000, 30):
    #     v = (100 - p * i) / 100
    #     if v >= 0:
    #         roi_prefs[f"{i}"] = v
    # return roi_prefs

