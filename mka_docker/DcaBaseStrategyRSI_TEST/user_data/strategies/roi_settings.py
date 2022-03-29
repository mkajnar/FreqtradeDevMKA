def get_rois():
    roi_prefs = {}
    p = 0.02
    for i in range(0, 10000, 30):
        v = (100 - p * i) / 100
        if v >= 0:
            roi_prefs[f"{i}"] = v
    return roi_prefs

