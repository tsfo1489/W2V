import pandas as pd
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
train_data = pd.read_csv('output.csv')
train_data = train_data.dropna(how = 'any')
train_data['comment'] = train_data['comment'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
tokenized_data = []
for sentence in tqdm(train_data['comment']):
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)
print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
model = Word2Vec(sentences = tokenized_data, vector_size = 300, window = 5, min_count = 5, workers = 4, sg = 0)
print(model.wv.vectors.shape)
model.wv.save_word2vec_format('scr_w2v')