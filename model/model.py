import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import *
import train_variables

import os, random
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
#     tf.random.set_seed(seed) # Tensorflow 사용시 
SEED = 777
set_seeds(SEED)

# 입력 문서에 대한 Top K 저널 추천

# ################################################# #
# Phase 1: 입력 문서와 Abstract 유사도가 높은 순서로 저널 리스트 생성

def recommend_journals(sentences, input_title, input_keyword, K):
    # SBERT Embedding 모델 다운로드 및 임베딩
    model = SentenceTransformer('smartmind/ko-sbert-augSTS-maxlength512') # https://huggingface.co/smartmind/ko-sbert-augSTS-maxlength512 # http://knlp.snu.ac.kr/index.html
    embeddings = model.encode(sentences)
    test_embed = embeddings[np.newaxis, ...]
    print('[Test] Abstract embedded:', test_embed.shape)

    # 문서 vs. 문서 코사인 유사도
    s1 = cosine_similarity(test_embed, train_variables.train_embed) # test embed vs. train embed

    # customized function 적용 및 추천 저널 리스트(R) 생성
    r = get_all_sim_journ(s1[0], train_variables.train_id_index, train_variables.train_y, 0.7)

    # Top-(K-1) 추천 저널 리스트
    top_k_journs = r[:K-1]

    # ################################################# #
    # Phase 2: 추천 저널 리스트 재배치

    # 입력 문서의 Title, Keyword 전처리
    title_nn = extract_noun(input_title.strip()) # 입력된 title이 없다면 빈 string 반환
    keyword = extract_keyword(input_keyword.strip()) # 입력된 keyword가 없다면 빈 string 반환
    print('[Test] Title preprocessed:', title_nn)
    print('[Test] Keyword preprocessed:', keyword)
    print()

    # [수정]
    # 제목 and/or 키워드가 있다면, 입력 문서의 저널별 유사도 계산(Title 유사도 + Keyword 유사도) 및 유사도 높은 순서로 재정렬
    if title_nn or keyword:
        s2 = get_s2(keyword, train_variables.count_vect_kw, train_variables.jkm) # 빈 string 입력시 빈 string 반환
        s3 = get_s3(title_nn, train_variables.count_vect_tt, train_variables.jtm) # 빈 string 입력시 빈 string 반환
        # 최종 유사도: Keyword 유사도(S2) + Title 유사도(S3)
        s23 = s2 + s3
        # 높은 유사도 순으로 추천 저널 리스트 새로 정렬
        if np.sum(s23) > 0:
            r_resort = resort_r(s23[0], top_k_journs, train_variables.journ_index)
        # 하지만 유사도 모두 0이라면, 기존(Abstract 기준) 추천 저널 리스트 그대로 사용
        else:
            r_resort = top_k_journs
    # 제목, 키워드 모두 없는 경우, 기존(Abstract 기준) 추천 후보 리스트 그대로 사용
    else:
        r_resort = top_k_journs

    # ################################################# #
    # Phase 3: 상위 저널과 유사한 저널 탐색 및 추천 리스트에 추가
    top_k_final = find_other_sim_journ(r_resort, train_variables.jtm, train_variables.jkm, train_variables.journ_index)

    # 최종 추천 저널 리스트 출력
    print(f'Top-{K} journals recommended')
    print(top_k_final)