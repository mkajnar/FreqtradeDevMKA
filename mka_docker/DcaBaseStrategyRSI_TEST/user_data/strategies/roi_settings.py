def get_rois():
    roi_prefs = {}
    p = 0.2
    for i in range(0, 10000):
        v = (100 - p * i) / 100
        if v >= 0:
            roi_prefs[i * 1] = v
    return roi_prefs