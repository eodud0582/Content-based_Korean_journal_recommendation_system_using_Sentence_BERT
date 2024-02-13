import pandas as pd
import numpy as np

import collections
from operator import itemgetter

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt, Mecab

from sklearn.metrics.pairwise import cosine_similarity

##################################################
# KoNLPY 및 Mecab 설치
'''
Homebrew 설치 필요 (MacOS)
'''

# !pip install konlpy
# !pip install mecab-python
# !bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

##################################################
# 제목(Title) 전처리

def extract_noun(sentence):
    '''
    Descriptions::
    - title에서 한글 명사/영문 용어 추출
    
    Parameters:
    - sentence: 문장 문자열
    '''
    mecab = Mecab()
    # 한글(일반 명사, 고유 명사), 외국어, 숫자 추출 (특수문자는 제외)
    nn_list = [re.sub(r'[^\w\s]','',text).strip().replace('   ', ' ').replace('  ', ' ').lower() for text, pos in mecab.pos(sentence) if pos in ('NNG','NNP','SL','SN')]
    # 중복 제거
    # nn_list = list(dict.fromkeys(nn_list).keys()) 
    # 문자열 결합
    nn_res = ' '.join(nn_list).replace('   ', ' ').replace('  ', ' ') 
    return nn_res

##################################################
# 키워드(Keyword) 전처리

def try_split(keywords):
    '''
    Descriptions::
    - ; 또는 , 로 keywords 분리 
    
    Parameters:
    - keywords: 문자열
    '''
    if keywords.find(';') == -1:
        kw_split = [x for x in keywords.split(',') if x != ''] # 마지막 element로 ''이 생성 될 수 있기 때문
        return kw_split
    else:
        kw_split = [x for x in keywords.split(';') if x != '']
        return kw_split

def separate_eng_kor(nlp_tool, text_in_kw):
    '''
    Descriptions:
    - 한 어절에서 영어와 한글을 분리하는 함수
    - 숫자와 한글이 붙어있는 경우는 '2단계','3차원' 등의 그 자체로 중요한 용어
    - 하지만 영어와 한글이 붙어있는 경우는  'f nucleatum속'처럼 분리를 해야 중요한 용어로 활용할 수 있을 것으로 보임 -> 'f nucleatum 속'
    
    Parameters:
    - nlp_tool: 사용할 형태소 분석기
    - text_in_kw: 한 어절
    '''
    splitted_word_segments = list()
    
    for each_word_seg in text_in_kw:
        
        kor_exist = False
        eng_exist = False
        temp_list = list()
        # 한 어절을 품사 태깅된 형태소로 분류하여 반환
        text_pos_list = nlp_tool.pos(each_word_seg)
        # 한 어절의 분류된 형태소별 영어-한글 여부 확인
        for text, pos in text_pos_list:
            # 해당 형태소가 NNG 또는 NNP면 한글을 의미
            if pos in ('NNG','NNP'):
                kor_exist = True
                temp_list.append(text.strip().lower())
            # 해당 형태소가 SL이면 영어를 의미
            elif pos == 'SL':
                eng_exist = True
                temp_list.append(text.strip().lower())
        # 하나의 어절에 한글과 영어가 모두 있다면, 한글과 영어가 붙어있음을 의미하며, 사이를 띄워서 반환
        if kor_exist and eng_exist:
            result_text = ' '.join(temp_list)
        # 아니라면, 해당 어절 그대로 반환
        else:
            result_text = each_word_seg
        
        splitted_word_segments.append(result_text)
    
    return ' '.join(splitted_word_segments)

def extract_keyword(keywords):
    '''
    Descriptions:
    - 중요한 단어를 의미하는 'keyword'라는 특성을 고려하여, 하나의 text로 된 keyword가 쪼개지지 않도록, keyword 처리시엔 mecab 형태소 분석 X
    - 단, 영어와 한글이 붙어있는 경우는 분리하고, 숫자와 한글/영어가 붙어있는 경우엔 그대로 유지
    
    Parameters:
    - keywords: 문자열
    '''
    mecab = Mecab()
    keyword_list = [separate_eng_kor(mecab, re.sub(r'[^\w\s]',' ',keyword).strip().replace('   ', ' ').replace('  ', ' ').split(' ')).lower() for keyword in try_split(keywords)] # 특수문자 제거
    #keyword_list = list(dict.fromkeys(keyword_list).keys()) # 처리한 키워드 내에서 중복 제거
    keyword_res = ' '.join(keyword_list).replace('   ', ' ').replace('  ', ' ') # 문자열 결합
    return keyword_res

##################################################
# Phase 1
import collections
def get_all_sim_journ(doc_sims, train_id_index, train_y, threshold=0.7):
    '''
    Parameters:
    - doc_sims : numpy array; 각 문서vs.문서 유사도 점수
    - threshold : "높은 유사도"의 기준 (Default: 0.7)
    '''
    # weighted similarity score 계산
    doc_and_scores = dict(zip(train_id_index.index, doc_sims)) # train_id_index: doc id - index
    journ_and_scores = list(zip(train_y.values, doc_sims)) # train_y: doc id - 저널명
    docs_over_th = list((dict(filter(lambda x: x[1] >= threshold, doc_and_scores.items())).keys()))
    journs_over_th = collections.Counter(train_y[docs_over_th].values)
    weight = [journs_over_th[journ_name] if journ_name in journs_over_th else 0 for journ_name in train_y.values]
    
    sim_weighted = ((0.135 * np.exp(0.0891 * (doc_sims*100))) + 0.5) + weight # document별 weighted similarity
    
    # 각 점수마다 해당 document id 붙이기 ([document id, weighted 유사도 점수] 형태로)
    sim_weighted = list(zip(train_id_index.index, sim_weighted)) # e.g. ('JAKO201610364779000', 0.5445481)
    
    # weighted 유사도 >= threshold
    # sim_weighted = list(filter(lambda x: x[1] >= threshold, sim_weighted))
    
    # [document id, 유사도 점수] -> 저널별 [journal 이름, 유사도 총 점수]
    sum_dict = collections.defaultdict(list)
    for doc_id, sim_score in sim_weighted:
        journ_name = train_y[doc_id] # 저널명 가져오기
        # sum_dict[journ_name] += sim_score
        sum_dict[journ_name].append(sim_score) # 저널별 유사도 점수 합치기
    
    mean_dict = {key: np.mean(values) for key, values in sum_dict.items()}
    
    # 총 유사도 점수가 높은 순서대로 저널 이름 정렬 (S-Sort)
    journ_sorted = list(dict(sorted(mean_dict.items(), key=itemgetter(1), reverse=True)).keys())
    
    return journ_sorted

##################################################
# Phase 2

def get_s2(keyword, count_vect_kw, jkm): # 수정
    '''
    Descriptions:
    - 입력 문서의 키워드 count vectorize
    - 그리고 입력 문서와 저널들의 키워드 유사도 계산
    
    Parameters:
    - keyword: 입력 문서 keyword에 대해 전처리 완료한 텍스트
    - count_vect_kw: Train 데이터의 keyword에 fit한 CountVectorizer() 객체
    - jkm: Train 데이터로 생성한 Journal-Keyword Matrix
    '''
    # Test 데이터의 Document-Keyword Matrix 생성 (Train JKM 생성때 fit된 count vectorizer 사용)
    test_doc_kw = [keyword] # CountVectorize를 위해 iterable이어야 하기 때문에 리스트 씌우기
    test_doc_kw_mat = count_vect_kw.transform(test_doc_kw)
    # keyword 사이의 document vs. journal 유사도 계산
    s2 = cosine_similarity(test_doc_kw_mat, jkm)
    return s2

def get_s3(title_nn, count_vect_tt, jtm): # 수정
    '''
    Descriptions:
    - 입력 문서의 제목 count vectorize
    - 입력 문서와 저널들의 제목 유사도 계산
    
    Parameters:
    - title_nn: 입력 문서 title에 대해 전처리 완료한 텍스트
    - count_vect_tt: Train 데이터의 title(title_nn)에 fit한 CountVectorizer() 객체
    - jtm: Train 데이터로 생성한 Journal-Keyword Matrix
    '''
    # Test 데이터의 Document-Title Matrix 생성 (Train JTM 생성때 fit된 count vectorizer 사용)
    test_doc_tt = [title_nn] # CountVectorize를 위해 iterable이어야 하기 때문에 리스트 씌우기
    test_doc_tt_mat = count_vect_tt.transform(test_doc_tt)
    # title 사이의 document vs. journal 유사도 계산
    s3 = cosine_similarity(test_doc_tt_mat, jtm)
    return s3

def resort_r(tt_kw_s, top_k_journs, journ_index, i=0): # 수정
    '''
    Descriptions:
    - 저널별 유사도(제목+키워드 유사도)에 따라 추천 저널 리스트 재배치/재정렬
    
    Parameters:
    - tt_kw_s: 저널별 유사도(제목+키워드 유사도)
    - top_k_journs: 현재 Top K-1 추천 저널 리스트
    - journ_index: 저널명-Index번호가 고정된 순서로 매핑 된 series; 저널명 매핑/불러오는데에 사용
    - i: index
    '''
    # 각 doc의 journal별 title+keyword 유사도 점수에 journal 이름 붙이기 
    journ_scores = dict(zip(journ_index.index, tt_kw_s)) # journal 이름 붙이기
    # 높은 유사도순으로 journal 재배치 (기존 list of journals에 있는 저널만 포함) -> 새로운 R 생성
    journ_sorted = list(dict(sorted(filter(lambda x: x[0] in top_k_journs, journ_scores.items()), key=itemgetter(1), reverse=True)).keys())
    return journ_sorted

##################################################
# Phase 3

def find_other_sim_journ(cur_top_journs, jtm, jkm, journ_index):
    '''
    Descriptions:
    - 추천 저널 후보 리스트에서, 맨 앞의 최상위 저널(R1) 1개에 대해 저널들과의 유사도를 분석 (저널 vs. 저널 유사도 분석)
    - 이후, 유사도 높은 저널부터 순차적으로 추천 저널 후보 리스트에 없는 저널을 찾기
    - 만약 찾았지만, 발견한 저널의 유사도 점수가 0보다 크지 않다면 (그 뒤 후보들의 유사도는 볼 필요 없이)
    - R1은 pass하고 다음 상위 저널(R2, R3, ... 순차적으로)로 같은 방법으로 탐색
    
    Parameters:
    - cur_top_journs: Top K-1 추천 저널 리스트
    - jtm: Train 데이터로 생성한 Journal-Title Matrix
    - jkm: Train 데이터로 생성한 Journal-Keyword Matrix
    - journ_index: 저널-Index번호가 고정된 순서로 매핑 된 series; 저널명 매핑/불러오는데에 사용
    '''
    def if_not_in_r(x):
        # list of journals에 이미 있는지/없는지 확인
        if x[0] in cur_top_journs:
            return False
        else:
            return True
    
    # list of journals이 애초에 비어있는 경우
    if not len(cur_top_journs):
        # 빈 list of journals 그대로 반환
        return cur_top_journs
    
    # 추가 후보 저널 탐색
    add_journ_found = False
    for i in range(len(cur_top_journs)):
        # 현재 선택된 상위 저널
        best_journ = cur_top_journs[i] # 맨 앞의(최상위) 후보 저널부터 하나씩

        # 해당 상위 저널과 다른 저널들의 title 유사도 구하기
        best_j_tt_mat = jtm[journ_index[best_journ]]
        best_tt_sim = cosine_similarity(best_j_tt_mat, jtm) # 상위 저널 JTM - train JTM 과의 title 코사인 유사도 (1, 268)

        # 해당 상위 저널과 다른 후보 저널들의 keyword 유사도 구하기
        best_j_kw_mat = jkm[journ_index[best_journ]]
        best_kw_sim = cosine_similarity(best_j_kw_mat, jkm) # 상위 저널 JKM - train JKM 과의 keyword 코사인 유사도

        # 해당 후보 저널의 다른 저널들과의 최종 유사도 구하기
        s4 = best_tt_sim + best_kw_sim # Title 코사인 유사도 + Keyword 코사인 유사도 더하기
        s4 = s4[0] # 268개 각 저널별 유사도 점수; test document 1개씩 처리 중이기 때문에 [i]가 아닌 [0] indexing
        
        # 높은 유사도 순으로 후보 저널 정렬
        best_jj_sim = sorted(dict(zip(journ_index.index, s4)).items(), key=itemgetter(1), reverse=True)

        # 그 중 기존 list of journals에 없는 저널 탐색 (유사도 높은 후보부터)
        new_journ_found = next(filter(if_not_in_r, best_jj_sim), None)
        if new_journ_found != None: # list of journals에 없는 후보 저널 1개를 찾았고,
            if new_journ_found[1] > 0: # 해당 저널의 유사도 점수가 0보다 크다면
                top_journs_added = cur_top_journs + [new_journ_found[0]] # list of journals에 추가
                add_journ_found = True
                return top_journs_added
            else: # 유사도 점수가 0이면, 다음 후보 저널로 탐색 (R2,R3,...)
                continue
        else: # list of journals에 없는 후보 저널이 없다면(모든 후보 저널이 이미 다 list of journals에 있다면)
            continue # 다음 후보 저널로 탐색 (R2,R3,...)
    
    # 모든 후보 저널을(R1,R2,R3,...) 다 탐색했지만, 적합한 후보 저널을 못 찾았다면
    if not add_journ_found:
        top_journs_added = cur_top_journs # 기존 list of journals 그대로 사용
        return top_journs_added

##################################################
# 추천 정확도 평가

def get_acc_micro(y_pred, y_true):
    '''
    Descriptions:
    - Micro accuracy : 전체 평균 accuracy; 전체를 기준으로 정확도를 평가하는 지표
    '''
    return np.mean(np.array([1 if y_true[i] in y_pred[i] else 0 for i in range(len(y_true))])) 

def get_acc_macro(y_pred, y_true):
    '''
    Descriptions:
    - Macro accuracy : 저널별 평균 accuracy의 전체 평균; 저널별 정확도를 기준으로 전체의 정확도를 평가하는 지표
    '''
    #global each_journ_ox, each_journ_mean_acc
    
    # 저널별로 맞은 것은 1, 틀린 것은 0으로 넣기
    each_journ_ox = collections.defaultdict(list)
    for journ_actual, journs_rec in list(zip(y_true, y_pred)):
        if journ_actual in journs_rec:
            each_journ_ox[journ_actual].append(1)
        else:
            each_journ_ox[journ_actual].append(0)
    # 저널별 평균 accuracy 계산
    each_journ_mean_acc = {key: np.mean(values) for key, values in each_journ_ox.items()} # dictionary comprehension
    # 저널별 평균 accuracy의 전체 평균 계산
    mean_of_mean_acc = np.mean(list(each_journ_mean_acc.values()))
    return mean_of_mean_acc

def get_mrr(y_pred, y_true):
    '''
    Descriptions:
    - MRR : 추천 저널 리스트에서 정답의 위치(순위)를 반영하여 정확도를 평가하는 지표
    '''
    # 각 예측별 reciprocal rank 구하기 (실제 저널이 추천 리스트에서 몇 번째에 위치하는지)
    recip_rank_computed = []
    for journ_actual, journs_rec in list(zip(y_true, y_pred)):
        try:
            # 추천 리스트에 있다면, 위치 번호에 역수 취하기 (앞에 나올수록 좋은 모델이니까)
            recip_rank = 1 / (journs_rec.index(journ_actual) + 1) # 1/(index + 1)
        except:
            # 추천 리스트에 없다면 0
            recip_rank = 0
        recip_rank_computed.append(recip_rank)
    # Mean Reciprocal Rank 계산
    mmr = np.mean(recip_rank_computed)
    return mmr