# 검색 시스템 구현 보고서

## 1. 개요

이 프로젝트는 온라인 정보 공간을 넓은 탐색 공간으로 보고, 이를 트리 구조로 분할 정복하면서 탐색하는 검색 시스템을 구현한 것이다.

핵심 목표는 다음과 같다.

- 검색 결과를 단순 나열하지 않고 주제별로 분기한다.
- 임베딩과 밀도 기반 클러스터링으로 하위 토픽을 자동 생성한다.
- 중복, 저신뢰 정보, 무의미한 분기를 초기에 차단한다.
- 트리가 과도하게 커지지 않도록 pruning과 종료 조건을 둔다.
- 벡터 변화 추적을 통해 브랜치 품질을 점검한다.

## 2. 구현 범위

구현된 기능은 다음과 같다.

1. 검색 시작 노드 생성
2. 웹 검색 결과 수집
3. 문서 임베딩
4. 신뢰도 기반 필터링
5. 임베딩 기반 중복 제거
6. DBSCAN / HDBSCAN 클러스터링
7. 토픽 분할 및 하위 쿼리 생성
8. 브랜치 점수화 및 continue / stop 판정
9. frontier pruning
10. 깊이 제한 및 정보 부족 종료

## 3. 아키텍처

### 3.1 설정 계층

[`jakal_search/config.py`](jakal_search/config.py)에서 엔진 파라미터를 분리했다.

- `SearchLimits`: 최대 깊이, 최대 노드 수, frontier 폭, 결과 수 등 제어
- `SimilarityConfig`: 중복 제거와 주제 분리 임계값
- `TrustConfig`: 도메인 기반 신뢰도와 저신뢰 필터 규칙
- `ScoringConfig`: continue / stop 판단용 가중치와 임계값

### 3.2 임베딩 계층

[`jakal_search/embedding.py`](jakal_search/embedding.py)에서 두 가지 임베더를 제공한다.

- `SentenceTransformerEmbedder`: 트랜스포머 임베딩
- `HashingEmbedder`: 설치가 없을 때 쓰는 경량 fallback

즉, 외부 모델이 없어도 시스템이 동작하고, 환경이 갖춰지면 고품질 임베딩으로 자동 전환된다.

### 3.3 검색 제공자

[`jakal_search/providers.py`](jakal_search/providers.py)는 DuckDuckGo HTML 엔드포인트를 이용한 기본 검색 제공자를 담고 있다.

- 쿼리 입력
- 제목 / 스니펫 / URL 추출
- 도메인 정규화

이 계층은 나중에 구글, 사내 검색, 크롤러 기반 인덱스로 교체하기 쉽게 분리했다.

### 3.4 클러스터링

[`jakal_search/clustering.py`](jakal_search/clustering.py)에서 밀도 기반 클러스터링을 수행한다.

- 기본은 HDBSCAN 우선 사용
- HDBSCAN이 없으면 DBSCAN으로 fallback
- DBSCAN도 결과가 없을 경우 eps를 점진적으로 넓혀 재시도

이 방식은 검색 결과 밀도가 들쭉날쭉한 현실 웹 검색에 맞추기 위한 것이다.

### 3.5 신뢰도 필터

[`jakal_search/trust.py`](jakal_search/trust.py)는 문서별 신뢰도 점수를 계산하고 저신뢰 문서를 차단한다.

- 도메인 화이트/블랙 성격의 가중치
- suspicious prototype과의 유사도 검사
- 너무 낮은 신뢰도와 높은 semantic risk가 동시에 나타나면 제거

즉, 임베딩이 비슷해도 신뢰도가 불안정하면 브랜치에 넣지 않는다.

### 3.6 브랜치 스코어링

[`jakal_search/scoring.py`](jakal_search/scoring.py)는 브랜치 continuation 판단을 담당한다.

평가 요소:

- novelty
- trust
- scope
- support
- vector consistency
- drift

이 점수로 branch가 계속 확장될 가치가 있는지 판단한다.

### 3.7 검색 엔진

[`jakal_search/engine.py`](jakal_search/engine.py)가 전체 오케스트레이션을 담당한다.

흐름은 다음과 같다.

1. root query 임베딩 생성
2. 검색 결과 수집
3. 신뢰도 필터링
4. 임베딩 기반 dedupe
5. 클러스터링
6. 클러스터별 하위 쿼리 생성
7. 브랜치 품질 평가
8. frontier pruning
9. 깊이 / 노드 수 제한까지 반복

특징:

- 부모와 유사한 결과는 허용할 수 있도록 설계
- 트리 전체의 중복은 `DocumentMemoryItem`과 ancestry 체크로 억제
- cluster가 너무 작거나 novelty가 낮으면 생성하지 않음

## 4. 종료 및 차단 규칙

시스템은 다음 조건에서 분기를 멈춘다.

- 최대 깊이에 도달한 경우
- 검색 결과가 너무 적은 경우
- 신뢰도 필터 후 남는 문서가 너무 적은 경우
- dedupe 후 남는 문서가 너무 적은 경우
- 밀도 높은 클러스터가 없는 경우
- novelty가 너무 낮은 경우
- scope가 너무 낮은 경우
- continue classifier 점수가 임계값보다 낮은 경우
- frontier가 너무 커져 pruning 대상이 되는 경우

이 규칙들을 통해 무한 확장과 지수적 폭증을 막는다.

## 5. 실행 방법

기본 실행 예시는 다음과 같다.

```bash
python -m jakal_search "graph search algorithms"
```

JSON 결과를 저장하려면:

```bash
python -m jakal_search "graph search algorithms" --json-out tree.json
```

## 6. 검증 결과

테스트는 다음 기준으로 통과했다.

- dedupe 규칙 검증
- subtopic 분기 검증
- max depth 종료 검증

실행 확인도 완료했다.

- `python -m pytest -q`
- `python -m jakal_search "graph search algorithms" --max-depth 1 --max-nodes 4 --results-per-query 8`

## 7. 한계와 향후 개선

현재 구현은 실사용 가능한 골격을 갖췄지만, 아래는 추가 개선 여지가 있다.

- continue / stop 분류기를 실제 학습 모델로 교체
- 도메인 신뢰도 스코어를 외부 평판 데이터와 결합
- 검색 제공자를 다중 소스로 확장
- 브랜치별 요약과 메타 분석 추가
- 로그 기반 탐색 품질 리포트 생성

## 8. 결론

이번 구현은 검색을 단순 키워드 매칭이 아니라, 트리 기반 정보 탐색 문제로 다루는 구조를 코드로 옮긴 것이다.

핵심은 다음 세 가지다.

- 불필요한 확장 억제
- 유의미한 주제 분할
- 신뢰도와 벡터 일관성 기반의 선택적 탐색

이 구조를 기반으로 이후에는 학습형 스코어러, 다중 검색 소스, 브랜치 요약을 얹어 더 강한 탐색 시스템으로 발전시킬 수 있다.
