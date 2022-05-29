def fill(buckets, _from, _to, _l):
    buckets[f'{_from}-{_to}'] = [n for n in _l if _from < n < _to]
    pass

def histogram(s):
    l = s.tolist()
    buckets = {}
    fill(buckets,0,10,l)
    fill(buckets,10,20,l)
    fill(buckets,20,30,l)
    fill(buckets,30,40,l)
    fill(buckets,40,50,l)
    fill(buckets,50,60,l)
    fill(buckets,60,70,l)
    fill(buckets,70,80,l)
    fill(buckets,80,90,l)
    fill(buckets,90,100,l)
    return [(k,len(buckets[k])) for k in buckets.keys()]