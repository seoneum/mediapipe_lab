---
type: project_note
status: implemented_slice
created: 2026-07-26
updated: 2026-07-26
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
