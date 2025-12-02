순서.
파싱된 데이터 input-> 줄별로 읽음-> 줄별로 임베딩 모델로 임베딩 -> pkl 이나 이런 파일로 저장 -> 4개 테스트 실행 -> 결과 저장


테스트


(A) 동일 클러스터간 거리 조합
dist(cat1, cat2)
dist(rm1, rm2)
dist(systemctl1, systemctl2)

(B) 다른 클러스터간 거리
dist(cat /etc/passwd, rm -rf /tmp)
dist(cat /etc/passwd, systemctl restart nginx)

(C) kNN Retrieval 정확도
curl google.com 과 가장 가까운 5개 명령어를 뽑았을 때
 실제로 curl 명령어들만 나오는가?

(D) Silhouette Score 