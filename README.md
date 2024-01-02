# Sentence BERT를 이용한 내용 기반 국문 저널추천 시스템 (Content-based Korean journal recommendation system using Sentence BERT)

With the development of electronic journals and the emergence of various interdisciplinary studies, the selection of journals for publication has become a new challenge for researchers. Even if a paper is of high quality, it may face rejection due to a mismatch between the paper’s topic and the scope of the journal. While research on assisting researchers in journal selection has been actively conducted in English, the same cannot be said for Korean journals. In this study, we propose a system that recommends Korean journals for submission. Firstly, we utilize SBERT (Sentence BERT) to embed abstracts of previously published papers at the document level, compare the similarity between new documents and published papers, and recommend journals accordingly. Next, the order of recommended journals is determined by considering the similarity of abstracts, keywords, and title. Subsequently, journals that are similar to the top recommended journal from previous stage are added by using a dictionary of words constructed for each journal, thereby enhancing recommendation diversity. The recommendation system, built using this approach, achieved a Top-10 accuracy level of 76.6%, and the validity of the recommendation results was confirmed through user feedback. Furthermore, it was found that each step of the proposed framework contributes to improving recommendation accuracy. This study provides a new approach to recommending academic journals in the Korean language, which has not been actively studied before, and it has also practical implications as the proposed framework can be easily applied to services.
