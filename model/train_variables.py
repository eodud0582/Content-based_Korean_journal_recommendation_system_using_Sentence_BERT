import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Train 데이터 관련 준비
train = pd.read_csv('data/train.csv', index_col=0) # 전처리한 train 데이터(dataframe) 다시 읽어오기
train_y = train['journal'] # Train y; Train 데이터 doc id - journal
train_id_index = pd.Series(range(len(train.index)), index=train.index) # train 문서id-index; Train 데이터셋에서의 각 문서 위치
journ_order = train.groupby('journal').groups.keys()
journ_index = pd.Series(range(len(journ_order)), index=journ_order) # 저널명-index 매핑; 저널 고정 순서
print('Journal:', journ_index.shape)

# Train SBERT embedding
# def get_embed(abstract):
#     return model.encode(abstract)
# model = SentenceTransformer('smartmind/ko-sbert-augSTS-maxlength512')
# workers = -1 # os.cpu_count() * 2
# with parallel_backend(backend='threading', n_jobs=workers):
#     train_embed = list(Parallel()(delayed(get_embed)(train_abs) for train_abs in tqdm(train['abstract'], position=0, leave=True)))
train_embed = np.load('data/train_embed.npy')
print('[Train] Abstract embedded:', train_embed.shape)

# Journal-Title Matrix (JTM)
train_journ_tt = train.groupby('journal')['title_nn'].apply(' '.join) # journal별로 모든 논문들 title text 합치기
count_vect_tt = CountVectorizer(min_df=1, ngram_range=(1,1)) # unigram count_vectorize
jtm = count_vect_tt.fit_transform(train_journ_tt) # sparse matrix
print('[Train] JTM:', jtm.shape)

# Journal-Keyword Matrix (JKM)
train_journ_kw = train.groupby('journal')['keyword'].apply(' '.join) # journal별로 모든 논문들 keyword text 합치기
count_vect_kw = CountVectorizer(min_df=1, ngram_range=(1,1)) # unigram count_vectorize
jkm = count_vect_kw.fit_transform(train_journ_kw) # sparse matrix
print('[Train] JKM:', jkm.shape)