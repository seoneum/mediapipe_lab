# ON DAMM 아동별 temporal 개인화 계약

## 제품 목표

최종 목표는 cross-person generalization이 아니라 **특정 아동 한 명에게 강하게 개인화된 temporal movement model**이다. 제품 성능은 다른 사람에게 유지될 필요가 없다. Target child 성능을 높이는 과정에서 P1/P2/P3 성능이 저하되어도 제품 목표상 실패가 아니다.

P1/P2/P3는 다음 용도로만 사용한다.

1. 초기 TCN temporal representation 개발
2. pipeline/debug
3. pretrained initialization
4. embedding 구조 sanity check

P4 신규 수집이나 LOSO 확대는 이 제품 경로에 포함하지 않는다. 기존 LOSO 코드는 초기 연구 진단 자료일 뿐 최종 제품 metric이나 모델 선택 기준이 아니다.

## 단계별 개인화

### 초기 단계

공통 개발 데이터로 만든 frozen TCN encoder와 child-specific prototype memory를 사용한다.

```text
P1/P2/P3 development data
  → pretrained causal TCN encoder
  → target child episode embedding
  → target child 전용 known / unknown / suppression prototype memory
```

### Target child 데이터 축적 후

동일 아동의 과거 세션만 사용해 TCN 전체를 fine-tune할 수 있다. 일부 layer를 반드시 freeze할 필요가 없으며 P1/P2/P3 catastrophic forgetting을 허용한다.

```text
Child A session 1 + session 2
  → normalization / epoch selection / threshold selection
  → pretrained TCN 전체 fine-tuning

Child A session 3
  → 모든 train-time 선택이 끝난 뒤 단 한 번 여는 held-out future-session test
```

`scripts/train_child_personalized_tcn.py`는 `--future-session`과 `--train-sessions`가 겹치면 즉시 실패한다. Future session은 normalization, epoch, threshold, checkpoint weight 선택에 사용할 수 없다.

```bash
.venv/bin/python scripts/train_child_personalized_tcn.py \
  --child-id child-a \
  --train-sessions s01 s02 \
  --future-session s03 \
  --pretrained-checkpoint outputs/micro_expression/v4_tcn/encoder_product.pt \
  --device mps
```

출력은 기본적으로 `outputs/micro_expression/children/<child-id>/temporal/`에 생성된다.

## Encoder 전환과 prototype memory

Fine-tuning된 encoder는 기존 encoder와 embedding 공간이 다르다. 따라서 기존 encoder digest로 만든 prototype vector를 새 checkpoint에 그대로 연결하지 않는다.

- child-personalized checkpoint는 자동 활성화하지 않는다.
- 새 encoder를 활성화하기 전에 승인된 source episode를 새 encoder로 다시 embedding하거나 child memory를 명시적으로 새로 구축한다.
- runtime의 encoder digest와 pattern-memory digest가 다르면 fail-closed 한다.
- fine-tuned checkpoint 사용 시 `--temporal-checkpoint`로 명시한다.

## 미래 세션 최종 지표

`app/ondamm_child_temporal_evaluation.py`와 `scripts/evaluate_child_future_session.py`가 다음 지표를 고정한다.

| 지표 | 정의 |
|---|---|
| known pattern event recall | 미래 세션 시작 전에 known이었던 정답 occurrence 중 시간 매칭과 pattern ID가 모두 맞은 비율 |
| known pattern precision | `KNOWN_OCCURRENCE` detection 중 정답 occurrence 및 pattern ID가 맞은 비율 |
| false activations/min | 어떤 정답 occurrence와도 시간 매칭되지 않은 active detection 수 / 미래 세션 분 |
| unknown repeated-pattern discovery precision | discovery 상태에 도달한 candidate cluster 중 하나의 반복 unknown pattern에만 대응한 cluster 비율 |
| duplicate cluster rate | 동일 unknown pattern에 둘 이상 만들어진 중복 discovery cluster 비율 |
| false merge rate | 둘 이상의 실제 pattern occurrence를 하나로 합친 discovery cluster 비율 |
| occurrences required until discovery | 올바른 candidate가 처음 `REPEATING_CANDIDATE`가 됐을 때의 occurrence count |
| first observation → eventization latency | 해당 unknown pattern의 첫 실제 occurrence 시작부터 최초 올바른 discovery eventization까지의 시간 |
| future-session stability | 미래 세션에 나타난 known pattern별 identity-correct recall의 macro average |

시간 매칭은 동일 미래 세션 안에서 temporal IoU 기반 one-to-one matching을 사용한다. `known_at_session_start`는 미래 세션을 보기 전에 고정하며, test session을 본 뒤 known/unknown 정답을 다시 정의하지 않는다.

### 평가 입력

Ground truth CSV 필수 열:

```text
child_id,session_id,event_id,pattern_id,start_timestamp,end_timestamp,known_at_session_start
```

Detection CSV 필수 열:

```text
child_id,session_id,detection_id,start_timestamp,end_timestamp,lifecycle,
pattern_id,candidate_id,occurrence_count,eventized_timestamp
```

Temporal runtime은 각 실행 출력 디렉터리의 `temporal_detections.csv`에 이 detection 계약을 자동 기록한다. Raw frame은 이 로그에 포함되지 않는다. `session_id`는 실행 출력 디렉터리 이름(`--run-id`를 주면 그 값)으로 고정되므로 미래 세션 평가에서는 `--run-id s03`처럼 명시적인 세션 ID를 사용한다.

평가 명령:

```bash
.venv/bin/python scripts/evaluate_child_future_session.py \
  --child-id child-a \
  --future-session s03 \
  --ground-truth path/to/s03_ground_truth.csv \
  --detections path/to/s03_detections.csv \
  --session-duration-seconds 1800
```

프로토콜 촬영에서 계산하는 binary action/control 결과는 pretrained movement representation을 점검하는 보조 지표다. 위 pattern discovery 지표를 대신하는 최종 제품 metric으로 사용하지 않는다.
