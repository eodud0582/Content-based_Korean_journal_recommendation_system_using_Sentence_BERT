# Sentence BERT를 이용한 내용 기반 국문 저널추천 시스템 (Content-based Korean journal recommendation system using Sentence BERT)

## Publication information
- 지능정보연구 (Journal of Intelligence and Information Systems, JIIS, pISSN 2288-4866, eISSN 2288-4882)
- Publishing institution: 한국지능정보시스템학회 (Korea Intelligent Information System Society)
- Date: 2023.09
- Related links:
  - https://www.jiisonline.org/index.php?mnu=archive&archiveId=964&PHPSESSID=c87f33175f0c8cafc5ec8f868ac29721
  - https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003000595
  - https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11538427

## Abstract

전자저널의 발전과 다양한 융복합 연구들이 생겨나면서 연구를 게시할 저널의 선택은 신진 연구자들은 물론 기존 연구자들에게도 새로운 문제로 떠올랐다. 논문의 수준이 높더라도 논문의 주제와 저널 범위의 불일치로 인해 게재가 거부될 수있기 때문이다. 이러한 문제를 해결하기 위해 연구자의 저널 선정을 돕기 위한 연구는 영문 저널을 대상으로는 활발하게 이루어졌으나 한국어 저널을 대상으로 한 연구는 그렇지 못한 실정이다. 본 연구에서는 한국어 저널을 대상으로 투고할 저널을 추천하는 시스템을 제시한다. 첫 번째 단계는 과거 저널에 게재된 논문들의 초록을 SBERT (Sentence-BERT)를 이용하여 문서 단위로 임베딩하고 새로운 문서와 기존 게재논문의 유사도를 비교하여 저널을 추천하는 것이다. 다음으로 초록의 유사도 여부, 키워드 일치 여부, 제목 유사성을 고려하여 추천할 저널의 순서가 결정되고, 저널별로 구축된 단어 사전을 이용하여 선순위 추천 저널과 유사한 저널을 찾아 추천 리스트에 추가하여 추천 다양성을 높인다. 이러한 방식으로 구축된 추천 시스템을 평가한 결과Top-10 정확도 76.6% 수준으로 평가되었으며, 추천 결과에 대한 사용자의 평가를 요청하고 추천 결과의 유효성을 확인하였다. 또한, 제안된 프레임워크의 각 단계가 추천 정확도를 높이는 데에 도움이 된다는 결과를 확인하였다. 본 연구는 그동안 활발히 이루어지지 않았던 국문 학술지 추천에 대한 새로운 접근을 제시한다는 점에서 학술적 의의가 있으며, 제안된 기능을 문서와 저널 보유상태에 따라 변경하여 손쉽게 서비스에 적용할 수 있다는 점에서 실무적인 의의를 가진다.

With the development of electronic journals and the emergence of various interdisciplinary studies, the selection of journals for publication has become a new challenge for researchers. Even if a paper is of high quality, it may face rejection due to a mismatch between the paper’s topic and the scope of the journal. While research on assisting researchers in journal selection has been actively conducted in English, the same cannot be said for Korean journals. In this study, we propose a system that recommends Korean journals for submission. Firstly, we utilize SBERT (Sentence BERT) to embed abstracts of previously published papers at the document level, compare the similarity between new documents and published papers, and recommend journals accordingly. Next, the order of recommended journals is determined by considering the similarity of abstracts, keywords, and title. Subsequently, journals that are similar to the top recommended journal from previous stage are added by using a dictionary of words constructed for each journal, thereby enhancing recommendation diversity. The recommendation system, built using this approach, achieved a Top-10 accuracy level of 76.6%, and the validity of the recommendation results was confirmed through user feedback. Furthermore, it was found that each step of the proposed framework contributes to improving recommendation accuracy. This study provides a new approach to recommending academic journals in the Korean language, which has not been actively studied before, and it has also practical implications as the proposed framework can be easily applied to services.

## Keywords
Deep learning, Document similarity, Recommendation system, Research papers, SBERT(Sentence Bidirectional Encoder Representations from Transformers)
