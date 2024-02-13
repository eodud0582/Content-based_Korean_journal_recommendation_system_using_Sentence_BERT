##################################
## DataFrame 전체 출력 함수
import pandas as pd
from IPython.display import display, HTML

def print_all(df): 
    with pd.option_context('display.max_rows',None):
        with pd.option_context('display.max_columns',None):
            with pd.option_context('display.float_format', '{:,.4f}'.format):
                display(df)

                

##################################
## 값 형태에 따른 날짜 형태 변환 함수

import pandas as pd
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta

def make_date(x):
    '''
    날짜 변수의 값 형태에 따라 날짜 형태로 변환
    e.g. 202103 -> 2021-03-01
         2013Q4 -> 2013-12-01
    
    Parameters
    ------------------
    x: 문자열(string) 타입의 날짜 관련 값
    '''
    # time 값이 8자리일 때 (형태: 연월일)
    if len(x) == 8:
        temp = x[:4] + '-' + x[4:6] + '-' + x[6:]
        temp = datetime.datetime.strptime(temp, '%Y-%m-%d').date()
        return temp

    # time 값이 6자리일 때 (형태: 연월, 연분기)
    elif len(x) == 6:
        first_part = x[:4]
        second_part = x[4:]
        
        # '연도+월'인 경우
        if first_part.isnumeric() and second_part.isnumeric():
            temp = first_part + '-' + second_part
            temp = pd.to_datetime(temp).date()
            #temp = datetime.strptime(temp, '%Y-%m').date()
            return temp
        
        # '연도'+'Q'인 경우
        elif first_part.isnumeric() and 'Q' in second_part:  # Q1, Q2 등 예외 처리
            temp = first_part + '-' + second_part
            temp = (pd.to_datetime('-'.join(temp.split()[::-1])) +
                    pd.offsets.QuarterBegin(0)).date()
            return temp

    # time 값이 4자리일 때 (형태: 년)
    elif len(x) == 4 and x.isnumeric():
        #temp = pd.to_datetime(x).date()
        temp = str(x) + '-12-31'
        temp = datetime.datetime.strptime(temp, '%Y-%m-%d').date()
        return temp

    # 이 외의 경우
    else:
        return x



##################################
## 데이터 프로파일링 함수

# 입력받은 dataframe에 대해 컬럼별 주요 통계량을 정리해서 dataframe으로 반환
import math
import string
import numpy as np
import pandas as pd

def get_profile(df):
    '''
    입력받은 DataFrame에 대해 컬럼별 주요 통계량을 정리해서 DataFrame으로 반환
    
    Parameters
    ------------------
    df: DataFrame
    
    Information
    ------------------
    문자열, 숫자(integer/float) 등 컬럼별 데이터 타입에 따라 맞춰 결과 return
    
    Return 정보
    - 레코드 수, 평균, 표준편차, 변동계수, 최소값, 1사분위수, 중앙값(2사분위수), 3사분위수, 최대값, Null 수, 공백값 수, 0값 수, 최대 빈도값, 최대 빈도수, 최소 빈도값, 최소 빈도수 등
    '''
    # 숫자형 값의 길이 확인 함수
    def number_length(n):
        if n > 0:
            if math.isinf(n):
                digits = math.inf
            else:
                digits = int(math.log10(n))+1
        elif n == 0: # handle 0
            digits = 1
        elif n < 0: # handle negative numbers
            if math.isinf(n):
                digits = math.inf
            else:
                digits = int(math.log10(-n))+1 # +1 if you don't count the '-' 
        else:
            digits = None
        return digits
    
    ## Overview
    num_observations = df.shape[0]
    num_vars = df.shape[1]
    total_null = df.isna().sum().sum()
    total_blank = np.sum(df == '').sum() + np.sum(df == ' ').sum() + np.sum(df == '  ').sum()
    #total_zero = np.sum(df == 0).sum() + np.sum(df == '0').sum()
    total_missing = total_null + total_blank
    total_missing_pct = round(total_missing / (num_observations * num_vars) * 100, 2)
    duplicate_cols = df.columns[df.T.duplicated(keep=False)].tolist()
    num_duplicate_cols = len(duplicate_cols)
    duplicate_rows = df.index[df.duplicated(keep=False)].to_list()
    num_duplicate_rows = len(duplicate_rows)

    print('## Overview \n')
    
    print(df.info(verbose=False) or '')
    print('Number of variables:', num_vars)
    print('Number of observations:', num_observations)
    print('Total missing cells:', total_missing)
    print('Total missing cells (%):', total_missing_pct)
    print('Duplicate columns:', duplicate_cols)
    print('Duplicate columns (count):', num_duplicate_cols)
    print('Duplicate rows:', duplicate_rows)
    print('Duplicate rows (count):', num_duplicate_rows)

    print('\n## Details')
    
    profiles_list = []

    for col in df.columns:
        #col_df = pd.to_numeric(df[col], errors='ignore') # 숫자 타입으로 바꿀 수 있는 값은 바꾸기
        col_df = df[col]

        col_data_type = col_df.dtype # 데이터 타입
        all_can_be_num = pd.to_numeric(col_df, errors='ignore').apply(lambda x: isinstance(x, (float, int))).all() # 모든 값 숫자(int,float) 형태 변경 가능 여부
        
        value_count = col_df.count() # 값 수
        unique_count = col_df.nunique() # 고유값 수
        
        null_count = col_df.isna().sum() # null 개수
        blank_count = ((col_df == '') | (col_df == ' ') | (col_df == '  ')).sum() # 공백 개수
        zero_count = ((col_df == 0) | (col_df == '0')).sum() # 0 개수
        
        # 현재 데이터 타입이 숫자형이라면
        if col_df.dtype == 'float' or col_df.dtype == 'int':
            special_chars = set(string.punctuation.replace('.','').replace('-',''))
            special_char_count = col_df.apply(lambda x: any(s for s in special_chars if s in str(x))).sum()
            negative_count = (col_df < 0).sum() # 음수값 수
            mean_value = col_df.mean() # 평균
            std_value = col_df.std() # 표준편차
            coeff_variation = std_value / mean_value # 변동계수; 표준편차를 같은 단위를 가지는 평균으로 나누어 표준화하므로 단위가 다른 속성을 비교할 수 있는 장점 존재
            min_value = col_df.min() # 최소값
            q1 = np.percentile(col_df, 25) # 1사분위수
            median_value = col_df.median() # 중앙값(2사분위수)
            q3 = np.percentile(col_df, 75) # 3사분위수
            max_value = col_df.max() # 최대값
            max_length = col_df.apply(lambda x: number_length(x)).max() # 값 최대 길이
            min_length = col_df.apply(lambda x: number_length(x)).min() # 값 최소 길이
        # 현재 그 외 데이터 타입이라면
        else:
            special_chars = set(string.punctuation)
            special_char_count = col_df.apply(lambda x: any(s for s in special_chars if s in str(x))).sum()
            negative_count = None # 음수값 수
            mean_value = None # 평균
            std_value = None # 표준편차
            coeff_variation = None
            min_value = None # 최소값
            q1 = None # 1사분위수
            median_value = None # 중앙값(2사분위수)
            q3 = None # 3사분위수
            max_value = None # 최대값
            max_length = col_df.apply(lambda x: len(str(x))).max() # 값 최대 길이
            min_length = col_df.apply(lambda x: len(str(x))).min() # 값 최소 길이
        
        most_frequent_value = col_df.value_counts().idxmax() # 최대 빈도값
        most_frequency = col_df.value_counts().max() # 최대 빈도수
        least_frequent_value = col_df.value_counts().idxmin() # 최소 빈도값
        least_frequency = col_df.value_counts().min() # 최소 빈도수

        profiles_list.append([col, col_data_type, all_can_be_num, value_count, unique_count, null_count, blank_count, zero_count, special_char_count, negative_count, mean_value, std_value, coeff_variation, min_value, q1, median_value, q3, max_value, most_frequent_value, most_frequency, least_frequent_value, least_frequency, max_length, min_length])

    profile_df = pd.DataFrame(profiles_list, columns=['col','dtype','all_can_be_num','cnt','nunique','null_cnt','blank_cnt','zero_cnt','spec_char_cnt','neg_cnt','mean','std','cv','min','25%','median','75%','max',
                                                      'most_freq_val','most_freq_cnt','least_freq_val','least_freq_cnt','max_length','min_length'])

    return profile_df



##################################
## 정상성 검정 함수

# Ljung-box 검정
from statsmodels.stats.diagnostic import acorr_ljungbox

def return_ljungbox_test(timeseries, lags=1, threshold=0.05):
    '''
    Ljung-box 검정: 시계열의 자기상관 그룹이 0과 다른지 여부에 대한 통계 검증 방법
    
    Parameters
    ------------------
    timeseries: array 형식의 시계열 데이터
    lags: 시차
          자기상관성(autocorrelation) 검정을 위해선 lags=1(default)로 설정
    threshold: 귀무가설 기각 기준 p-value
               (Default: p-value < 0.05)
    
    Information
    ------------------
    # 귀무가설: 시계열에 자기상관성이 없음(서로 독립적) (정상)
    # 대립가설: 시계열에 자기상관성이 있음(독립적이지 않음) (비정상)
    # p-value < 0.05 => 귀무가설 기각, 대립가설 채택 = 비정상 시계열을 의미 (p-value가 커야 좋다)
    '''
    ljngbox_result = acorr_ljungbox(timeseries, lags, boxpierce=True, return_df=True) # Box-Pierce Test도 함께 실행

    lb_pvalue_lag1 = ljngbox_result.loc[lags,'lb_pvalue'] # Ljung-Box Test at lag 1
    bp_pvalue_lag1 = ljngbox_result.loc[lags, 'bp_pvalue'] # Box-Pierce Test at lag 1
    
    if lb_pvalue_lag1 < threshold:
        result = [lb_pvalue_lag1, '비정상']
    elif lb_pvalue_lag1 >= threshold:
        result = [lb_pvalue_lag1, '정상']
    else:
        result = [np.nan, np.nan]

    if bp_pvalue_lag1 < threshold:
        result.extend([bp_pvalue_lag1, '비정상'])
    elif bp_pvalue_lag1 >= threshold:
        result.extend([bp_pvalue_lag1, '정상'])
    else:
        result.extend([np.nan, np.nan])
    
    return result

# KPSS(Kwiatowski-Phillips-Schmidt-Shin) 검정
from statsmodels.tsa.stattools import kpss

def return_kpss_test(timeseries, regression='c', threshold=0.05):
    '''
    KPSS(Kwiatowski-Phillips-Schmidt-Shin) 검정: 시계열이 평균 또는 선형 추세 주변에 고정되어 있는지 또는 단위근으로 인해 고정되지 않은지 검증하는 통계 방법
    
    Parameters
    ------------------
    timeseries: array 형식의 시계열 데이터
    regression: c, ct
        c: The data is stationary around a constant (데이터가 평균을 중심으로 일정할 때; 추세성을 고려하지 않음)(default)
        ct: The data is stationary around a trend (데이터가 선형 추세 주변으로 일정할 때; 추세성을 고려)
    threshold: 귀무가설 기각 기준 p-value
               (Default: p-value < 0.05)
    
    Information
    ------------------
    # 귀무가설: 시계열 과정이 정상적 (정상)
    # 대립가설: 시계열 과정이 비정상적 (비정상)
    # p-value < 0.05 => 귀무가설 기각, 대립가설 채택 = 비정상 시계열을 의미 (p-value가 커야 좋다)
    '''
    kpsstest = kpss(timeseries, regression=regression, nlags="auto")
    pvalue = kpsstest[1]
    
    if pvalue < threshold:
        return [pvalue,'비정상']
    elif pvalue >= threshold:
        return [pvalue,'정상']
    else:
        return [np.nan, np.nan]
    
# ADF(Augmented Dickey-Fuller) 검정
from statsmodels.tsa.stattools import adfuller

def return_adf_test(timeseries, regression='c', threshold=0.05):
    '''
    ADF(Augmented Dickey-Fuller) 검정: 단위근이 시계열 샘플에 존재한다는 귀무가설을 검증하는 통계 방법 
    
    Parameters
    ------------------
    timeseries: array 형식의 시계열 데이터
    regression: c, ct, ctt, n
        c : constant (상수항만 존재) (default)
        ct : constant and trend (상수항과 추세 존재)
        ctt : constant and linear and quardratic trend (상수항과 1차 2차 추세가 존재)
        n : no constant, no trend (상수항과 추세가 둘다 존재하지 않음)
    threshold: 귀무가설 기각 기준 p-value
               (Default: p-value < 0.05)
    
    Information
    ------------------
    # 귀무가설: 시계열에 단위근이 존재 (비정상)
    # 대립가설: 시계열이 정상성을 만족 (추세 정상성을 만족)
    # p-value < 0.05 => 귀무가설 기각, 대립가설 채택 = 정상 시계열을 의미 (p-value가 작아야 좋다)
    '''
    result = adfuller(timeseries, autolag='AIC', regression=regression)
    adf_statistic = result[0]
    p_value = result[1]
    num_lags = result[2]
    num_observations = result[3]
    critical_values = result[4]
    
    if p_value < threshold:
        return [p_value, '정상'] 
    elif p_value >= threshold:
        return [p_value, '비정상']
    else:
        return [np.nan, np.nan]
    
# 위 세 검정을 한꺼번에 수행
def return_stationarity(series_to_check, threshold=0.05):
    
    # Ljung-box Test
    try:
        ljungbox_result = return_ljungbox_test(series_to_check, 1, threshold) # lags=1
        result = ljungbox_result
    except:
        result = [np.nan]*4
        
    # KPSS Test
    try:
        kpss_result = return_kpss_test(series_to_check, 'c', threshold) # regression=c
        result.extend(kpss_result)
    except:
        result.extend([np.nan]*2)
    
    # ADF Test
    try:
        adf_result = return_adf_test(series_to_check, 'c', threshold) # regression=c
        result.extend(adf_result)
    except:
        result.extend([np.nan]*2)
        
    return result

# 정상성 검정 결과를 DataFrame으로 정리
def check_df_staionarity(df, threshold=0.05):
    combined_result = []
    for col in df.columns:
        
        test_result = return_stationarity(df[col], threshold)
        test_result = [col] + test_result
        combined_result.append(test_result)

    #result_df = pd.DataFrame(combined_result, columns=['variable','lbox_lag1_p','lbox_lag1_res','kpss_p','kpss_res','adf_p','adf_res'])
    result_df = pd.DataFrame(combined_result, columns=['variable','lbox_lag1_p','lbox_lag1_res','bp_lag1_p','bp_lag1_res','kpss_p','kpss_res','adf_p','adf_res'])
    
    return result_df

# 잔차 추출 함수
# Seasonal-Trend decomposition using LOESS (STL)
from statsmodels.tsa.seasonal import STL

def stl_err(ts):
    '''
    시계열 데이터를 분해한 후, 잔차(residual)만 추출
    현재 시계열 분해 설정은 월별 데이터에 맞춰져 있음
    
    Parameters
    ------------------
    ts: array 형식 또는 1차원 시계열 데이터
    '''
    stl = STL(ts, period=12, seasonal=13) # default: seasonal=7
    stl_result = stl.fit()
    ts_l = stl_result.resid
    return ts_l



##################################
## correlation 기반 변수 제거 함수

def calc_drop(res):
    # threshold보다 큰 변수 조합에 들어가는 모든 변수
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))

    # drop 변수, 즉 제거할 가능성이 있는 변수 (한 번이라도 drop으로 선택된 변수)
    poss_drop = list(set(res['drop'].tolist()))

    # drop에 포함되지 않은/된 적이 없는, 즉 꼭 keep 할 변수
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # keep 변수를 짝으로 가진 모든 변수 = 꼭 제거할 drop 변수
    p = res[ res['v1'].isin(keep) | res['v2'].isin(keep) ][['v1', 'v2']] # 어떤 변수가 keep 할 변수인지 안다면, 해당 변수의 pair(짝) 변수는 제거할 변수라는 것을 알 수 있겠지? keep 할 변수가 하나라도 포함 된 변수조합을 dataframe에서 추출
    
    # 위에서 추려진 조합들에서 하나는 keep 할 변수일 것이고, 다른 나머지 변수는 drop해야 할 변수일 것이다
    q = list(set(p['v1'].tolist() + p['v2'].tolist())) # drop할 변수들은 찾았지만, keep, drop 변수 모두 중복이 있을 수 있겠지? 위 변수조합의 모든 변수들(v1, v2)을 한줄로 쭉 줄세워 중복 제거; -> 이 리스트에는 keep할 변수도 있고, drop할 변수도 있다
    
    # 위 리스트에서 keep 할 변수가 아닌 변수들만 추려내면, (keep할 변수의 짝인) 제거할 변수들만 남게 된다
    drop = (list(set(q).difference(set(keep)))) # 이 drop 변수들은 '꼭 제거할 변수'들이 되겠다

    # '제거할 가능성이 있는 변수' 리스트에서 (위에서 추려낸) '꼭 제거할 변수'들은 빼기
    poss_drop = list(set(poss_drop).difference(set(drop))) # 방금 위에서 추려낸 drop 항목에는 포함되지 않은 제거 가능성 있는 변수들(poss_drop)을 추려내서 다시 이 poss_drop 리스트를 업데이트하겠다

    # (꼭 제거할 drop 변수들을 제외하고) 남은 제거 가능성 있는 변수(poss_drop)를 가진 변수조합을 dataframe에서 추출
    m = res[ res['v1'].isin(poss_drop) | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]

    # 남은 제거 가능성 있는 변수(poss_drop)가 포함된 변수조합 중에서, 추가적으로 제거 할 변수 추려내기
    '''
    변수조합/pair에서 '남은 제거 가능성 있는 변수'(poss_drop)의 나머지 짝이 '꼭 제거할 변수'(drop)가 아닌 '남은 제거 가능성 있는 변수' 및 해당 변수조합만 추려낼 것이다 (자기 짝이 '꼭 제거할' 변수라면, 이 짝이 제거되어야 하고, 이 '제거 가능성 있는 변수'는 제거 할게 아닐 테니까) 
    이렇게 추려낸다면, 짝이 'keep할 변수'이거나 '제거 가능성 있는 변수'일 제거 가능성 있는 변수들(poss_drop) 및 해당 변수조합들만 남게 된다 (제거 할 변수인지, v1,v2 모두 '제거 가능성 있는 변수'라면 둘 중 무엇을 제거할지 결정만 남는다)
    -> 이러한 변수의 변수조합 row에서 'drop'컬럼이 가리키는 변수, 제거해야 할 변수가 바로 추가적으로/또 다른 제거 할 변수가 되겠다
    '''
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop'])) # v1,v2 둘 다 '꼭 제거할(drop)'변수가 아닌 변수조합 m 에서 -> 각 변수조합마다 제거해야 할 변수(drop 컬럼 값)로 제시된 변수들만 추려내기

    # 꼭 제거할 변수 리스트에 방금 추가적으로 추려낸 제거할 변수(more_drop)들을 추가
    for item in more_drop:
        drop.append(item)

    return drop

def corr_drop_cols(df, cut=0.8):
    
    global corr_result 
    #res

    # 변수간 상관계수 구하기 및 상삼각행렬 선택
    corr_mtx = df.corr().abs() # (absolute) correlation matrix 생성
    avg_corr = corr_mtx.mean(axis=1) # 각 변수별 다른 변수들과의 평균 correlation
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(bool)) # correlation matrix의 upper triangle 영역 선택
    '''
    - np.ones(corr_mtx.shpae) 상관계수 매트릭스 모양/크기 그대로 '1'로 채워진 numpy 배열 생성 
    - np.triu(array, k=0(default)); 상삼각행렬에 해당하는 원소만 남길 수 있음; 가장 윗줄부터 '0'이 k개부터 시작/채워진다
    - np.where()을 사용하려면 True/False 형태여야 하여 .astype(bool)로 dtype 변경
    '''

    dropcols = list() # 제거할 변수 리스트

    res = pd.DataFrame(columns=(['v1','v2','v1_avg_corr','v2_avg_corr','corr','drop'])) # 빈 dataframe 생성

    # 변수별 평균 상관계수를 비교하며 제거할 변수 선택
    for row in range(len(up)-1): # 컬럼 개수-1 만큼 반복; 삼각행렬 기준 한 변수마다 또 아래에서 for문을 돌린다
        col_idx = row + 1 # 삼각행렬 기준으로 row번째 컬럼은 자기자신이니, +1해서 '그 다음 컬럼'을 향하도록; '그 다음 컬럼'의 index
        for col in range(col_idx, len(up)): # '그 다음 컬럼'부터 끝까지
            if corr_mtx.iloc[row, col] > cut: # 삼각행렬 기준, 현재 변수(row index)와 현재 다른 변수(col index)의 상관계수가 threshold보다 크다면
                # 두 변수 중 평균 상관계수 값이 큰 변수를 제거 대상으로
                if avg_corr.iloc[row] > avg_corr.iloc[col]: 
                    dropcols.append(row) # 현재 변수(row) index를 제거할 변수 리스트에 추가
                    drop = corr_mtx.columns[row] # 현재 변수(row) 이름을 drop_col에 할당 
                else:
                    dropcols.append(col) # 현재 변수의 짝인 다른 변수(col) index를 제거할 변수 리스트에 추가
                    drop = corr_mtx.columns[col] # 다른 변수(col) 이름을 drop_col에 할당

                s = pd.DataFrame(np.array([[corr_mtx.index[row], # 현재 변수 이름
                                            up.columns[col], # 다른 변수 이름
                                            avg_corr[row], # 현재 변수의 평균 상관계수
                                            avg_corr[col], # 다른 변수의 평균 상관계수
                                            up.iloc[row,col], # (upper triangle에서) 현재 변수와 다른 변수의 상관계수
                                            drop]]), # 현재 변수와 다른 변수 중 제거할 변수 이름
                                 columns=res.columns.tolist())

                res = pd.concat([res, s], ignore_index=True) # dataframe에 저장
                # 이 dataframe에는 threshold보다 큰 상관계수를 가진 변수 조합(현재 변수,다른 변수)의 정보가 들어간다

    dropcol_names = calc_drop(res)
    
    corr_result = res
    
    return dropcol_names

# 제거할 변수 및 남길 변수를 찾고 DataFrame으로 정리 (target 변수 제외)

def get_drop_keep_cols(df, threshold):
    
    #global dropcols, keepcols
    
    # 제거할 변수는?
    dropcols = corr_drop_cols(df, threshold)
    print('제거할 변수 수: ', len(dropcols))
    
    # 남길 변수는?
    keepcols = list(set(df.columns).difference(dropcols))
    print('남길 변수 수: ', len(keepcols))

    result_cols = pd.DataFrame({'drop_cols': dropcols}).join(pd.DataFrame({'keep_cols': keepcols}), how='outer')

    return result_cols


