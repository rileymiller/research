

def clean_dat(chunk):
    allowable_chars = [ chr(i) for i in range(128) ]
    allowable_chars = set(allowable_chars)

    print('chunk before', chunk)

    clean_chunk = ''
    for word in chunk:
        isClean = True
        for char in word:
            if char not in allowable_chars:
                isClean = False
                break
        if isClean:
            clean_chunk = clean_chunk + word
    print('chunk after:', clean_chunk)

    return clean_chunk
clean_dat('I want to eat وسا')

clean_dat('لكل خبر معين، المهمة هي تحديد وتعيين وسم آو آوسام تمثل موضوع الخبر')
