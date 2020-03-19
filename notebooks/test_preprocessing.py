

def clean_dat(chunk):
    # allowable_chars = [ chr(i) for i in range(128) ]
    # allowable_chars = set(allowable_chars)
    print('chunk', chunk)
    # Read stopwords
    with open('datasets/stops.txt', 'r') as f:
        stops = f.read().split('\n')
    for w in chunk.split():
        print('w', w)
        if w not in set(stops):
            print('clean', w)
    tops = [ w for w in chunk.split() if w not in set(stops) and not w.isnumeric() ]
    print(tops)
    return ' '.join([ w for w in chunk.split() if w not in set(stops) and not w.isnumeric()])
print(clean_dat('dog had a 13 good day'))
