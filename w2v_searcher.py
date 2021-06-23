from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("mov_w2v") # 모델 로드
while True :
    q = input('Keyword: ')
    if q == 'q' :
        break
    try :
        print(model.most_similar(q))
    except Exception :
        print('Word does not exist')