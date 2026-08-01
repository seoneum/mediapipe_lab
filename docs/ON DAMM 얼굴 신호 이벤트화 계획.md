---
type: project_note
status: implemented_slice
created: 2026-07-26
updated: 2026-07-27
tags: [project, planning, mediapipe, privacy, machine-learning]
lang: ko
project: ON DAMM
---

# ON DAMM 얼굴 신호 이벤트화 계획

## 현재 상태

2026-07-26 기준 전체 얼굴 센싱·아동 대상 촬영 시스템은 여전히 계획·안전 검토 단계다. 다만 첫 구현 슬라이스로 **오프라인 개인화 학습 baseline**을 구현했다.

- 기존 전체 설계의 Architect `BLOCK / REQUEST CHANGES`와 Critic `ITERATE` 기록은 역사적 안전 검토로 보존한다.
- 첫 슬라이스는 `app/ondamm_personalization.py`, `app/ondamm_personalization_cli.py`, `app/ondamm_learning.py`와 focused tests에 반영했다.
- 기존 작업 트리는 기준선으로 보존하며 reset/clean/전체 덮어쓰기를 하지 않았다.

## 구현된 첫 슬라이스: 개인화 학습 baseline

- 카메라·raw media 없이 교사 라벨이 붙은 안전한 event summary만 입력한다.
- 한 개인의 승인된 positive outcome과 품질 게이트를 사용해 deterministic nearest-centroid 모델을 학습한다.
- `grouped_person_split`과 `split_person_events`로 person/event leakage를 막고, sparse·저품질·불일치 입력은 abstain하여 baseline 계획을 유지한다.
- 모델 JSON은 승인 event ID, support label, row fingerprint, SHA-256 manifest digest를 보존한다. 추천은 모델이 발급한 authorization과 교사 승인 근거가 모두 있을 때만 생성된다.
- 추천은 `visual_schedule`, `short_prompt`, `transition_preview`, `brief_break`, `reinforcement` 중 승인된 지원 힌트만 기존 학습 계획의 prompt/transition/reinforcement에 제한적으로 추가한다. 진단, 집중도, 선호도, 감정, 상태 점수는 산출하지 않는다.
- CLI demo와 learning-plan 통합, focused unittest, py_compile, diff check, adversarial QA를 통과했다.

## 문헌 요약을 반영한 학습 방향

사용자가 제공한 두 논문의 ASD 관련 요약을 방향으로 삼아, 단일 기능 목표, 예측 가능한 구조, 시각적 지원, 짧은 prompt, 전환 예고, 짧은 휴식, 강화, 교사 피드백, 실제 환경 일반화를 우선한다. 이는 논문 PDF를 이 저장소에서 독립적으로 재검증했다는 의미가 아니라 현재 설계 원칙으로 반영한 것이다.

## 보류한 후속 작업

다음은 rough follow-up으로 기록만 하고 이번 슬라이스에서는 구현하지 않는다.

- camera/MediaPipe live capture와 얼굴·홍채 신호 eventization
- child media retention, point-specific recording, purge
- GPT extraction/network inference
- consent, revocation, lifecycle hardening, retention enforcement
- full API/UI workflow 및 production review surface

## 목표 흐름

성인 운영자 자기검증 → 동일 런타임 검증 → 카메라 상대적 얼굴 신호 이벤트화 → 교사 리뷰 → 명시적 승인 → 승인된 근거만 학습 프로그램에 반영

## 이벤트 설계

MediaPipe의 연속 랜드마크를 저장하지 않고 다음 제한된 이벤트와 품질 정보만 저장한다.

- `iris_zone_transition`: 눈 코너 기준 카메라 상대 홍채 구역 변화
- `iris_landmark_stability_window`: 홍채 랜드마크 안정 구간 proxy
- `head_orientation_transition`: 카메라 상대 머리 방향 proxy
- `facial_movement_window`: 표정이 아니라 얼굴 움직임 패턴 구간
- `sensor_unavailable_window`: 얼굴·홍채·랜드마크 품질 부족 상태

“동공 응시 시간”은 임상적 응시 측정이나 동공 측정이 아니라 `iris_landmark_stability_window`라는 **홍채 랜드마크 안정 proxy**로만 표현한다.

## 안전·개인정보 경계

- 실제 카메라 실행 전에 성인 자기검증이 필요하다.
- 자기검증은 카메라·모델·해상도·설정의 동작 확인이며, 성인의 생체 기준을 아동에게 적용하지 않는다.
- 전체 세션 녹화는 금지한다.
- 녹화는 기본 비활성화이며, 명시적 동의·보존기간이 있을 때만 이벤트 전후 제한 구간을 저장한다.
- 철회·만료 시 촬영, 재생, 분석, 리뷰, 승인, 학습 반영을 차단하고 원본·캐시·파생물을 정책에 따라 삭제한다.
- 진단, 집중도·선호도·감정·순응도 자동 점수화, 교육 효과 판정, 정밀 시선추적을 하지 않는다.
- MediaPipe 결과가 학습 프로그램이나 지원 기록철을 자동 변경하지 않는다.

## 교사 리뷰와 개인화

1. 교사가 `accepted / rejected / uncertain`을 판단한다.
2. 관찰 사실과 교사 해석을 분리한다.
3. 통제된 지원 라벨만 선택한다.
4. 별도 승인 작업이 있을 때만 `SessionSummary`에 근거를 promotion한다.
5. 학습 프로그램은 선택된 승인 세션과 지원 라벨만 사용한다.

센서 이벤트의 빈도, 표정, 시선, 머리 방향, 시간값은 직접 학습 목표나 선호도 판정의 입력이 되지 않는다.

## 구현 예정 모듈

- `app/ondamm_facial_features.py`: 순수 이벤트 정책·상태 머신·proxy 용어
- `app/ondamm_self_validation.py`: 성인 자기검증 리포트와 런타임 fingerprint
- `app/ondamm_lifecycle.py`: 동의·철회·보존·권한·삭제의 단일 소유자
- `app/ondamm_contracts.py`: 이벤트/리뷰/승인/API/dossier의 단일 스키마 소유자
- `app/ondamm_event_recording.py`: v2 이벤트 메타데이터와 제한 녹화
- `app/ondamm_review.py`, `app/ondamm_web.py`: 교사 리뷰·승인·근거 선택 API
- `app/ondamm_learning.py`, `app/ondamm_learning_cli.py`: 승인 근거 기반 학습 프로그램
- `ui/`: 자기검증 상태, 이벤트 품질, 리뷰, 승인, 보존/삭제 표시

기존 v1 이벤트와 dossier는 읽을 수 있게 유지하되, legacy 얼굴 라벨은 v2 이벤트·GPT·개인화에 사용하지 않는다.

## 현재 보류 이유

최종 검토에서 다음 계약을 더 명확히 해야 한다.

- 철회 시 활성 lease·완료된 run·cache·review·승인 근거를 fail-closed로 무효화하는 순서
- 동의 철회 journal의 fsync/recovery와 보존기간 상한
- source snapshot의 불변성 및 promotion hash
- `Path`가 노출되지 않는 MediaPipe/OpenCV용 안전한 media handle
- purge 작업의 재시작·실패·재시도 상태
- 두 카메라 CLI/API가 동일한 capture authorizer를 사용하는지 검증

## 다음 검증 원칙

구현 후에도 전체 세션 영상이 아니라 결정론적 synthetic observation으로 먼저 검증한다.

- 순수 이벤트 상태 머신 단위 테스트
- 자기검증 만료·fingerprint 불일치 테스트
- 동의·철회·삭제·캐시 만료 테스트
- 리뷰만으로 dossier가 변하지 않는지 테스트
- 승인된 support label만 학습 preview를 바꾸는지 테스트
- legacy 얼굴 이벤트와 facial run이 GPT/범용 media route로 가지 않는지 테스트
- API/UI는 품질·근거·철회 상태를 표시하고 `no-store`를 유지하는지 테스트

> 기존 `PRJ - ON DAMM 방향 및 설계 정리`와 `PRJ - ON DAMM MVP 구현 현황`의 “카메라는 보조 기능이며 최종 반영은 사람 승인” 원칙을 구체화한 계획 문서다.
## 2026-07-27 구현 업데이트: 얼굴 신호 후보 이벤트 학습

이번 구현은 **얼굴 원본을 자동 해석하는 시스템이 아니라, 구조화된 관찰 요약에서 검토할 만한 후보 이벤트를 만들고, 사람의 승인 라벨을 이용해 개인별 지원전략 후보를 찾는 오프라인 baseline**이다.

```text
구조화된 MediaPipe 관찰 요약
→ bounded event window 생성
→ 후보 이벤트 저장/직렬화
→ expert/teacher/parent 라벨
→ 승인된 helpful 사례만 per-person prototype 학습
→ 새 관찰과 prototype 거리 비교
→ confidence 기준 미달이면 abstain
→ 통과해도 human-review candidate로만 출력
```

### 구현 파일

| 파일 | 역할 |
|---|---|
| `app/ondamm_face_event_learning.py` | 데이터 계약, 후보 이벤트화, 라벨 검증, prototype 학습, 매칭, JSON 직렬화 |
| `app/ondamm_face_event_learning_cli.py` | bounded JSON 입력과 `--demo`, `--train`, `--match` CLI 래퍼 |
| `tests/test_ondamm_face_event_learning.py` | threshold, 승인, provenance, 직렬화, abstention, 자원 한계, CLI 경계 테스트 |
| `outputs/ondamm/artifacts/face-event-g002-test-report.json` | 32개 focused test 및 adversarial 검증 기록 |
| `outputs/ondamm/artifacts/face-event-ai-slop-cleanup.md` | 변경 파일 cleanup 결과 |
| `outputs/ondamm/artifacts/face-event-g002-quality-gate.json` | architect/QA/iteration/terminal critic 품질 게이트 |

### 1. 무엇을 측정하고 무엇을 측정하지 않는가

현재 `ObservationSample`은 다음과 같은 **관찰 proxy**만 받는다.

- `facial_movement_proxy_values`: 얼굴 랜드마크에서 계산된 움직임 크기 요약값
- `gaze_zone`, `gaze_dwell_seconds`: 카메라 기준 구역과 해당 구역에 머문 시간
- `head_orientation_zone`, `head_transition_count`: 머리 방향 구역과 전환 횟수
- `quality_score`, `quality_flags`: 얼굴 검출·랜드마크 품질의 메타데이터
- `person_id`, `session_id`, `context_id`, `timestamp`: 데이터 분할과 provenance에 필요한 식별자

다음은 schema에서 거부한다.

- `emotion`, `concentration`, `attention`, `preference`
- `ASD`, `autism`, `diagnosis`, `compliance`
- `image`, `video`, `media`, `frame`, `raw_image`, `raw_video`

따라서 “고개를 돌렸다 = 집중력이 낮다”, “시선이 이탈했다 = 흥미가 없다” 같은 결론은 코드가 내리지 않는다. 이벤트는 **교사·전문가·부모가 확인할 장면을 찾는 표식**이다.

### 2. 이벤트화 이론: 연속 신호를 짧은 관찰 단위로 바꾸기

연속적인 랜드마크 시계열을 그대로 학습시키지 않고, 최대 3개 sample의 짧은 window로 묶는다.

- 얼굴 움직임: 인접 sample 간 proxy 평균 절대 변화가 `0.20` 이상
- 시선: dwell이 `2.0초` 이상이거나 gaze zone이 바뀜
- 머리 방향: window 안 전환 횟수가 `2` 이상
- 품질: 평균 quality가 `0.60` 미만이거나 quality flag가 존재함
- 이벤트 span: 최대 `30초`

경계값은 임의의 진단 기준이 아니라, **결정론적 테스트를 위한 정책 threshold**다. 같은 입력은 항상 같은 candidate ID와 같은 JSON을 만든다.

### 3. prototype 학습 이론

지원전략을 `s`라고 하고, 승인된 후보들의 feature vector를 `x_1, ..., x_n`이라 하면 prototype은 성분별 평균이다.

$$
p_s = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

새 후보 `x`와 prototype의 유클리드 거리는 다음과 같다.

$$
d(x,p_s)=\sqrt{\sum_j(x_j-p_{s,j})^2}
$$

현재 baseline confidence는 bounded heuristic이다.

$$
confidence = \max(0, 1-\frac{d}{2})
$$

가장 가까운 전략을 고르되 `confidence <= 0.70`이면 결과를 내지 않는다. 이것은 “전략이 정답”이라는 뜻이 아니라 **검토 우선순위를 정하는 후보 점수**다.

### 4. 사람 라벨과 승인 diode

각 `LabelRecord`에는 reviewer role(`expert`/`teacher`/`parent`), reviewer ID, candidate ID, 관찰 context, support strategy, outcome, explicit approval이 있다.

학습 조건은 다음과 같다.

1. candidate의 person/context와 label이 일치해야 한다.
2. 동일 reviewer의 중복 라벨은 거부한다.
3. positive candidate마다 서로 다른 reviewer ID 2개 이상의 승인 필요.
4. 승인된 `helpful`만 prototype에 포함한다.
5. `not_helpful`, `uncertain`, `not_observed`는 positive prototype을 만들지 않는다.
6. 저품질 후보·혼합 person·reviewer disagreement는 거부하거나 abstain한다.

이 구조는 센서값이 직접 교육 결정을 내리는 것을 막는 **human-in-the-loop data diode**다. 모델은 “이 얼굴은 무엇이다”를 배우지 않고, “이 관찰 후보가 나타난 학습 맥락에서 어떤 교사 승인 support strategy가 검토 후보가 되었는가”만 다룬다.

### 5. provenance와 leakage 방지

`EventCandidate`는 `source_sample_digest`, provenance sample ID, evidence ID, candidate fingerprint를 가진다. 학습 때 입력 candidate를 원본 `source_samples`에서 다시 eventize하고 fingerprint를 비교한다.

모델은 다음 manifest를 저장한다.

- training candidate ID
- 각 candidate fingerprint
- label provenance digest
- strategy prototype
- model digest

그래서 ID나 digest만 다시 계산해 변경된 입력을 숨기는 경로를 차단한다. `grouped_person_split`/event split 원칙과 함께, 같은 사람 또는 같은 event가 train/validation 양쪽에 섞이지 않게 해야 한다.

### 6. 코드 읽는 순서

1. `ObservationSample.__post_init__` — 입력 범위와 금지 필드의 시작점
2. `_check_forbidden`, `_check_json_numbers` — cycle/depth/breadth/finite-number 방어
3. `extract_event_candidates` — window와 threshold가 후보가 되는 과정
4. `EventCandidate.__post_init__` — 후보의 불변식·notice·provenance
5. `LabelRecord`와 `train_per_person_model` — 승인 라벨이 prototype으로 제한되는 과정
6. `PrototypeModel.__post_init__`, `_verify_model` — manifest와 digest 검증
7. `match_reviewable_candidates` — 거리, confidence, abstention, human review 출력
8. `dumps`, `loads_*`, CLI `main` — deterministic JSON과 오류 경계

### 7. 직접 실행 예시

```bash
PYTHONPATH=. .venv/bin/python -m app.ondamm_face_event_learning_cli --demo
PYTHONPATH=. .venv/bin/python app/ondamm_face_event_learning.py --demo
PYTHONPATH=. .venv/bin/python -m unittest tests/test_ondamm_face_event_learning.py
```

`--demo` 두 경로의 출력은 byte-identical이어야 한다. 실제 입력에서는 `--train`에 `samples`, `labels`, 선택적 `person_id`를 주고, `--match`에는 `samples`, `model`을 준다. 모든 오류는 고정된 `notice`와 함께 반환되며, malformed/cyclic/oversized 입력은 성공적인 학습 결과로 변환되지 않는다.

### 8. 현재 구현의 명시적 한계

- 카메라 capture, child capture, raw media retention, GPT/network, consent/revocation/lifecycle, production API/UI는 이 slice에 없다.
- 실제 아동 대상 사용 전에는 adult/self validation, synthetic/offline 검증, 별도 privacy·윤리·철회·삭제 계약이 선행되어야 한다.
- prototype은 교육적 진단 모델이 아니며, ASD 상태·집중도·감정·선호도·순응도를 추정하지 않는다.
- 매칭 결과는 자동 학습계획 변경이 아니라 사람이 확인하는 `EventCandidate`다.
- SHA-256 manifest는 오프라인 self-consistency 검증이며, 외부 서명/키 관리가 포함된 production authenticity가 아니다.
