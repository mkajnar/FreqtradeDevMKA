def fill(buckets, _from, _to, _l):
    buckets[f'{_from}-{_to}'] = [n for n in _l if _from < n < _to]
    pass

def histogram(s):
    l = s.tolist()[-200:]
    buckets = {}
    for i in range(0,100,5):
        fill(buckets,i,i+5,l)
    return [(k,len(buckets[k])) for k in buckets.keys()]