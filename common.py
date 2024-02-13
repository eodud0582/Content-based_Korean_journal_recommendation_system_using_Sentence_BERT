import os
import datetime
import numpy as np
import pandas as pd
from IPython.display import display, HTML

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pandas.plotting import register_matplotlib_converters
#fm._load_fontmanager(try_read_cache=False)

import scipy.stats as stats
import statsmodels.api as sm

from timeit import default_timer as timer

# import boto3
# import psycopg2
# import awswrangler as wr
# from pyathena import connect
# from pyathena.pandas.util import as_pandas

import warnings
# warnings.filterwarnings('ignore')

import platform

# 한글 관련 폰트 있는지 확인
kr_list = list()
found = False
print('# Fonts found')
for font in font_manager.fontManager.ttflist:
    if any([k_font for k_font in ['Gothic','gothic','GOTHIC','Malgun','malgun','Nanum','nanum'] if k_font in font.name]):
        kr_list.append(font.name)
        print(font.name, font.fname)
        found = True
if not found:
    print('=> No Korean fonts found')

# 한글 폰트 설정
print('# Setting Korean')
common_kr = ['Malgun Gothic','AppleGothic','NanumGothic','NanumBarunGothic']
kr_set = False
for kr_font in common_kr:
    if kr_font in kr_list:
        rc('font', family=kr_font)
        print(f'=> {kr_font} set')
        kr_set = True
        break
if not kr_set:
    print('=> Common korean font not set')

# 운영체제별 한글 폰트 설정
# print('# Setting Korean')
# if platform.system() == 'Darwin': # Mac 환경 폰트 설정
#     font_path = '/usr/share/fonts/urw-base35/URWGothic-Demi.otf'
#     font_name = font_manager.FontProperties(fname=font_path).get_name()
#     rc('font', family=font_name) # 'AppleGothic'
#     try:
#         rc('font', family='AppleGothic')
#         print('=> Korean set in your Mac')
#     except:
#         print('=> No AppleGothic in your Mac')
# elif platform.system() == 'Windows': # Windows 환경 폰트 설정
#     font_path = 'C:/Windows/Fonts/malgun.ttf'
#     font_name = font_manager.FontProperties(fname=font_path).get_name()
#     rc('font', family=font_name) # 'Malgun Gothic'
#     try:
#         rc('font', family='Malgun Gothic')
#         print('=> Korean set in your Windows')
#     except:
#         print('=> No Malgun Gothic in your Windows')
# else:
#     print('=> Unknown system')

# font_manager.FontProperties에 직접 다운로드/저장한 한글 폰트 파일의 폰트 경로를 전달 (이후 시각화/차트 생성시 한글 출력을 위해 사용)
fontprop = font_manager.FontProperties(fname='./NanumGothic.ttf')  

# 마이너스 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# IPython 에서 제공하는 Rich output 대한 표현방식으로 도표와 같은 그림, 소리, 애니메이션들을 출력 (Rich output) 하는 것
#%matplotlib inline
# 레티나 설정 - 폰트 주변이 흐릿하게 보이는 것을 방지해 글씨가 좀 더 선명하게 출력 됨
#%config InlineBackend.figure_format='retina'

# matplotlib에 pandas 포맷터 및 변환기를 등록(pandas 객체를 matplotlib에 그려내려면 명시적으로 변환기 등록 필요)
register_matplotlib_converters()
# 시각화 스타일 설정
#sns.set(style='whitegrid', palette='muted', font_scale=1.0) # -> 이것땜에 matplotlib 한글 폰트 설정 오류남

# 지수표현식 없애기 
pd.options.display.float_format = '{:.4f}'.format
# 지수표현식 설정 초기화
#pd.reset_option('display.float_format')

# 랜덤 시드 고정
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)

print('\n# Compledted importing and setting')