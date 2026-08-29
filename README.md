# mediapipe_lab · ON DAMM

Apple Silicon Mac에서 MediaPipe 기반 얼굴·시선·자세 신호를 실험하고, ON DAMM의 local-first 학습지원 흐름과 개인화 얼굴 움직임 연구를 검증하는 저장소입니다.

현재 핵심은 다음 세 가지입니다.

- 미세 움직임 연구: 통제 촬영, 프레임별 478 landmarks·52 blendshapes·head transform 추출, causal TCN과 label-free temporal embedding 실험
- ON DAMM MVP: dossier, 승인된 수업 기록, 학습 프로그램 초안, handoff, 철회 잠금, 개인별 temporal pattern memory와 검토형 승격
- 오프라인 영상 분석: 사람 추적, 얼굴 신호·행동 proxy 지표, 자막 MP4와 JSON/CSV 생성

> 이 프로젝트는 표정으로 감정, 집중도, ASD 여부, 순응도 또는 내적 상태를 진단하지 않습니다. 출력은 관찰 가능한 얼굴·시선·자세 변화의 보조 신호이며, 공식 기록이나 교육적 판단에는 사람의 검토와 승인이 필요합니다.

## 빠른 시작

요구 환경은 macOS, Python 3.12, 카메라 권한입니다. Apple GPU를 사용할 수 있으면 PyTorch가 MPS를 사용하고, 그렇지 않으면 CPU로 동작합니다.

```bash
git clone https://github.com/seoneum/mediapipe_lab.git
cd mediapipe_lab
bash scripts/setup_env.sh
```

설치와 기본 MediaPipe 모델을 확인합니다.

```bash
.venv/bin/python app/check_install.py
bash scripts/camera_probe.sh
```

카메라가 열리지 않으면 macOS의 **시스템 설정 → 개인정보 보호 및 보안 → 카메라**에서 Terminal 또는 사용하는 IDE 권한을 확인하고 `--camera 0`, `--camera 1` 순서로 시도하세요.

## 미세 움직임 데이터 수집

### 1. 통제 조건 촬영

`app/micro_expression_record_control.py`는 baseline, blink, gaze, head movement로 구성된 통제 프로토콜을 실행합니다. 준비 화면에서 `SPACE`로 시작하고 `Q` 또는 `ESC`로 중단합니다.

```bash
.venv/bin/python app/micro_expression_record_control.py \
  --participant p1 \
  --session s01 \
  --camera 0 \
  --fps 60
```

출력은 로컬 전용이며 Git에 포함되지 않습니다.

```text
data/micro_expression/recordings/<participant>/<session>/
├── control_gaze.mp4
├── control_gaze_labels.csv
└── metadata.json
```

- 화면 안내문·target·progress bar는 preview에만 표시됩니다.
- `control_gaze.mp4`에는 overlay를 그리기 전의 raw frame이 저장됩니다.
- CSV에는 frame index, capture/video timestamp, phase label, target 위치와 안내문이 기록됩니다.
- 카메라가 실제로 보고한 FPS가 유효하면 writer FPS로 사용하고, 그렇지 않으면 `--fps` 값을 사용합니다.

기존 P1/P2 촬영본을 정말 폐기하고 다시 찍을 때만 아래 명령을 사용하세요. 이 데이터는 Git에 없으므로 삭제 후 복구할 수 없습니다.

```bash
rm -rf data/micro_expression/recordings/p1 data/micro_expression/recordings/p2
```

### 2. 녹화 영상에서 수치 추출

먼저 Face Landmarker를 준비합니다.

```bash
bash scripts/download_micro_expression_models.sh
```

그다음 촬영 영상을 VIDEO 모드로 분석합니다.

```bash
.venv/bin/python app/micro_expression_video.py \
  --input data/micro_expression/recordings/p1/s01/control_gaze.mp4 \
  --output outputs/micro_expression/p1-s01-landmarks.mp4 \
  --csv outputs/micro_expression/p1-s01-signals.csv
```

CSV에는 얼굴 검출 여부와 함께 다음 값이 저장됩니다.

- `lm_0_*`~`lm_477_*`: 478개 얼굴 landmark의 x/y/z
- `bs_*`: MediaPipe Face Landmarker의 52개 blendshape coefficient
- `T_00`~`T_33`: facial transformation matrix

결과 MP4는 landmark 확인용 overlay 영상이며, 학습 원본은 별도의 raw `control_gaze.mp4`입니다.

### 3. 실시간 변화 확인

현재 공식 실시간 경로는 `micro_expression_camera.py`와 `micro_expression_signals.py`입니다.

```bash
.venv/bin/python app/micro_expression_camera.py --camera 0 --dino-every 3
```

키:

- `B`: 현재 DINO patch feature를 neutral baseline으로 설정
- `R`: baseline 초기화
- `Q` 또는 `ESC`: 종료

화면에는 blendshape, 정렬된 landmark displacement, 눈·눈썹·입 ROI motion, head yaw/pitch/roll, coarse gaze, DINO patch 변화 heatmap이 표시됩니다.

DINOv3 ViT-S/16은 gated 모델이어서 저장소에 포함하지 않습니다. Hugging Face에서 `facebook/dinov3-vits16-pretrain-lvd1689m` 라이선스에 동의한 뒤 전체 모델 파일을 `models/dinov3/vits16/`에 내려받아야 합니다. DINO 없이 프레임별 MediaPipe CSV만 필요하면 `micro_expression_video.py`를 사용하세요.

## 미세 움직임 학습 설계

연구 경로에는 person-held-out causal TCN이, ON DAMM 제품 경로에는 frozen encoder를 이용한 개인별 temporal pattern memory가 구현되어 있습니다. 첫 제품 encoder는 DINO나 시선·머리 자세 같은 nuisance 신호를 섞지 않고 다음 79개 label-free feature만 받습니다.

```text
52 × bs_*           MediaPipe blendshape
18 × geom_abs_*     canonical face geometry
 9 × motion_*       generic facial motion
────────────────────────────────────────
79 features × 60 frames(약 2초) → causal TCN → 64-D L2 embedding
```

`scripts/train_v4_tcn.py`는 각 held-out fold의 학습 완료 시 분류 head를 제외한 `tcn.*` 가중치, feature 순서, robust normalization 통계, split provenance를 제품용 checkpoint로 내보냅니다.

```bash
.venv/bin/python scripts/train_v4_tcn.py

# 예: outputs/micro_expression/v4_tcn/encoder_held_out_p1.pt
```

checkpoint가 없거나 feature 순서·TCN shape이 맞지 않으면 runtime은 즉시 실패합니다. 학습되지 않은 random encoder로 조용히 fallback하지 않습니다. checkpoint 파일 자체는 실행 결과이므로 저장소에 포함하지 않습니다.

TCN stride 5의 겹치는 window는 반복 횟수로 세지 않습니다. `MicroMotionEpisodeDetector`가 onset/offset hysteresis, 최소 지속시간, refractory를 적용해 독립 episode로 합친 뒤 `PatternMemoryStore`가 child별 known prototype과 unknown micro-cluster를 비교합니다.

```text
KNOWN match             → KNOWN_OCCURRENCE
UNKNOWN 1~2회           → metadata + embedding만 로컬 저장, MP4 없음
동일 UNKNOWN 3회째       → 현재 episode만 pre 1.5초 + post 1.0초 MP4 저장
사람 검토 accepted       → 별도 이름/승인으로 KNOWN pattern 승격
사람 검토 rejected       → suppression memory 등록
uncertain               → watch 상태로 계속 관찰
```

초기 runtime은 `Frozen TCN + Dynamic Prototype Memory`만 사용합니다. online TCN 재학습은 꺼져 있으며, 승인된 후보 centroid만 known prototype으로 추가합니다. prototype vector는 `outputs/ondamm/pattern-memory/<child_id>/vectors.npz`에 남고 dossier audit에는 encoder/prototype digest와 source event ID만 기록합니다.

핵심 구현은 다음 모듈로 분리되어 있습니다.

- `app/ondamm_temporal_encoder.py`: checkpoint 검증·export·64-D embedding
- `app/ondamm_micro_motion.py`: 겹치는 TCN endpoint의 episode segmentation
- `app/ondamm_pattern_memory.py`: known/unknown/suppression memory와 recurrence policy
- `app/ondamm_micro_motion_runtime.py`: feature source → episode → memory → clip/UI orchestration
- `app/ondamm_event_recording.py`: ephemeral RAM buffer와 disk persistence 분리, post-tail delayed finalize

새 runtime은 카메라에 종속되지 않습니다. live MediaPipe adapter나 offline extractor가 checkpoint의 정확한 feature 순서로 frame feature와 원본 frame을 `MicroMotionRuntime.add_observation()`에 전달해야 합니다. 같은 영상의 인접 frame/window를 train/test에 무작위로 섞지 말고 session 또는 source video 단위로 분리하며, 별도의 재촬영 세션을 최종 확인용으로 남겨 두세요.

## ON DAMM 로컬 웹 MVP

```bash
bash scripts/ondamm_web.sh
open http://127.0.0.1:8765
```

주요 기능:

- 개인별 local dossier와 승인된 session summary
- 추천·학습 프로그램 초안과 명시적 승인
- 사람에게 읽히는 handoff export와 수동 재구성
- 동의 철회 시 `withdrawn_locked` 전이
- MediaPipe 관찰 보조와 local clip 검토
- 지정 움직임·시선·자세 이벤트의 짧은 영상 자동 저장
- 개인별 known pattern 지속 검출과 unknown 반복 후보 발견
- 반복 3회째의 현재 episode만 저장하는 privacy-aware clip policy
- 보호자·교사·기관 사회복지사의 역할별 독립 검토와 의견 일치·불일치 표시
- 승인된 session ID를 근거로 한 개인별 얼굴 움직임 규칙

원격 GPT 프레임 검토는 선택 기능입니다. 전체 영상을 보내지 않고 제한된 축소 JPEG frame만 전송하며, UI에서 매번 명시적으로 동의해야 합니다.

```bash
export OPENAI_API_KEY='your-api-key'
export ONDAMM_GPT_MODEL='gpt-5.6'
bash scripts/ondamm_web.sh
```

키, dossier, 촬영 영상과 실행 결과는 저장소에 커밋하지 마세요.

### 지정 움직임 이벤트 자동 저장과 교차 검토

승인된 얼굴 움직임 프로필 또는 기본 움직임 라벨을 지정하면, 해당 움직임이 설정 시간 이상 지속될 때 이벤트 클립을 로컬에 저장할 수 있습니다. 예를 들어 `mouth_dimple`을 0.4초 이상 관찰할 때 자동 저장하려면 다음처럼 실행합니다.

```bash
bash scripts/ondamm_learning.sh \
  --child-id child-a \
  --duration-seconds 120 \
  --record-events \
  --movement-label mouth_dimple \
  --movement-min-seconds 0.4
```

여러 라벨은 `--movement-label`을 반복해 지정합니다. 라벨은 감정명이 아니라 `mouth_dimple`, `brow_raise`, `lip_press`처럼 관찰 가능한 움직임 이름이어야 합니다.

웹 UI의 **관찰 보조 → 미세 움직임 이벤트 검토**에서 자동 저장된 MP4를 확인할 수 있습니다. 각 이벤트에는 다음 세 역할이 독립적으로 의견을 남깁니다.

- 보호자
- 교사
- 기관 사회복지사

각 검토는 `의미 있는 움직임 후보 / 이벤트 아님 / 추가 맥락 필요`, 영상에서 직접 확인한 사실, 상황 코멘트를 분리해 저장합니다. 세 역할의 최신 의견이 일치하는지 표시하지만, 합의가 생겨도 dossier나 수업 기록에는 자동 반영하지 않습니다. 별도의 사람 승인 단계가 필요합니다.

현재 로컬 MVP에는 계정 인증과 기관별 권한 관리가 없으므로 검토자가 UI에서 역할과 이름을 직접 선택합니다. 실제 기관 배포 전에는 사용자 인증, 역할 기반 접근 제어, 전자서명과 보존기간 정책을 추가해야 합니다.

### 제출·발표용 live demo capture

`ondamm_learning_cli.py`의 카메라 모드는 `MicroExpressionSignalExtractor` 하나로 얼굴을 분석합니다. 별도 카메라 viewer를 동시에 실행하지 않고, 같은 frame에서 다음 작업을 한 프로세스로 처리합니다.

```text
camera frame
  → 478 landmark + mouth/eyes/brow motion
  → checkpoint 순서의 79-D temporal feature
  → personal neutral motion calibration
  → frozen causal TCN + episode recurrence
  → debug overlay preview
  → 3번째 독립 episode의 overlay MP4만 저장
  → 기존 ON DAMM UI에서 재생·MediaPipe 분석·교차 검토
```

먼저 TCN 학습을 한 번 실행해 제품용 frozen encoder checkpoint를 만듭니다. 이미 `encoder_*.pt`가 있으면 생략합니다.

```bash
.venv/bin/python scripts/train_v4_tcn.py
```

터미널 1에서 UI를 실행합니다.

```bash
bash scripts/ondamm_web.sh
open http://127.0.0.1:8765
```

UI에서 선택할 `demo-child` dossier가 존재하는지 확인한 뒤 터미널 2에서 live demo를 실행합니다.

```bash
bash scripts/ondamm_learning.sh \
  --child-id demo-child \
  --duration-seconds 60 \
  --record-events \
  --debug-overlay \
  --require-temporal
```

checkpoint를 명시하려면 다음 옵션을 붙입니다.

```bash
--temporal-checkpoint outputs/micro_expression/v4_tcn/encoder_held_out_p1.pt
```

`--debug-overlay`는 preview뿐 아니라 새 temporal event MP4에도 skeleton, mouth/eyes/brow motion, candidate ID, `REPEAT n / 3`, `EVENT SAVED` 상태를 넣습니다. 앞의 1~2회 episode는 여전히 MP4를 쓰지 않습니다. 처음 3초는 개인 neutral motion calibration이므로 얼굴을 편하게 유지한 뒤 같은 짧은 움직임을 각각 중립 구간을 사이에 두고 세 번 수행합니다.

checkpoint를 지정하지 않으면 가장 최근 `outputs/micro_expression/v4_tcn/encoder_*.pt`를 자동 선택합니다. `--require-temporal`은 checkpoint가 없을 때 skeleton만 보여 주며 계속 진행하는 대신 즉시 실패하게 하므로 제출 촬영에 권장합니다. DINO는 기본적으로 꺼져 있으며 꼭 필요할 때만 `--demo-dino`를 사용합니다.

실제 headless 운용은 동일 명령에 `--headless`를 추가합니다. detector와 저장 정책은 그대로이고 OpenCV 창만 표시하지 않습니다.

촬영 중 overlay 상태는 다음 순서로 바뀝니다.

```text
CALIBRATING NEUTRAL
→ READY / OBSERVING
→ UNKNOWN_OCCURRENCE · REPEAT 1 / 3
→ UNKNOWN_OCCURRENCE · REPEAT 2 / 3
→ REPEATING_CANDIDATE · REPEAT 3 / 3
→ REPEATING PATTERN DETECTED · EVENT SAVED
```

마지막 메시지가 나온 뒤 UI의 **영상 목록 새로고침**을 누르면 overlay가 포함된 event clip이 나타납니다. 이후 영상 재생 → **MediaPipe로 분석** → 역할별 독립 검토 저장 → 검토 카드 반영 흐름을 그대로 촬영할 수 있습니다.

### Temporal pattern memory 검토와 승격

패턴 메모리가 생성된 아동을 선택하면 **관찰 보조 → 개인별 temporal pattern memory**에 known pattern과 unknown 반복 후보가 표시됩니다. 3회 미만 후보는 영상 없이 횟수·지속시간·품질·nearest-known 거리만 보여 줍니다. 3회째 저장된 event clip을 세 역할이 모두 `accepted`로 검토한 경우에만 별도 패턴 이름과 승인자를 입력해 known pattern으로 승격할 수 있습니다. 세 역할이 모두 `rejected`인 경우에만 suppression memory 등록이 가능합니다.

```text
GET  /api/dossiers/{child_id}/patterns
POST /api/dossiers/{child_id}/patterns/candidates/{candidate_id}/promote
POST /api/dossiers/{child_id}/patterns/candidates/{candidate_id}/suppress
POST /api/dossiers/{child_id}/patterns/candidates/{candidate_id}/watch
```

승격은 기존 event review와 별개의 명시적 행위입니다. 합의가 생겼다는 이유로 dossier나 TCN을 자동 변경하지 않습니다.

## 오프라인 영상 분석기

모델과 환경을 준비합니다.

```bash
bash scripts/download_video_models.sh
.venv/bin/python -m app.ondamm_video_env --check
```

영상 하나를 분석해 자막 MP4와 개인별 JSON/CSV를 만듭니다.

```bash
bash scripts/ondamm_video_analyzer.sh \
  --input input.mp4 \
  --output outputs/ondamm/video/result.mp4 \
  --device auto \
  --sample-every 3 \
  --metrics-json outputs/ondamm/video/metrics.json \
  --metrics-csv outputs/ondamm/video/metrics.csv
```

진입점은 `app/ondamm_video_analyzer_cli.py`, 셸 wrapper는 `scripts/ondamm_video_analyzer.sh`입니다.

| 플래그 | 의미 |
|---|---|
| `--input PATH` | 입력 영상 |
| `--output PATH` | 자막이 새겨진 결과 MP4 |
| `--device {auto,cpu,mps,cuda}` | 추론 장치 |
| `--sample-every N` | N frame마다 얼굴 신호 샘플링 |
| `--metrics-json PATH` | 개인별 지표 JSON |
| `--metrics-csv PATH` | 개인별 지표 CSV |

`auto`는 macOS 환경 보고서에 따라 검증된 MPS 또는 CPU만 선택하며 **절대 cuda를 고르지 않습니다**. Ubuntu+CUDA에서는 `--device cuda`를 명시하세요.

종료 코드:

- `0`: 성공
- `2`: 입력 영상 열기 실패
- `3`: 모델 파일 없음
- `4`: 렌더 실패

사람별 지표 스키마:

| 필드 | 의미 |
|---|---|
| `global_id` | 재연관된 사람 ID |
| `attention_pct` | 구현된 행동 proxy의 비율 |
| `focus_seconds` | 해당 proxy의 누적 초 |
| `interest` | `낮음` / `중간` / `높음` 연구용 구간 |
| `expression_timeline` | `{t_sec, label}` 목록 |
| `frames_covered` | 얼굴 신호가 있는 frame 수 |
| `total_frames` | 전체 frame 수 |
| `low_confidence` | 신원 재연관 확신 부족 여부 |

### unknown_N 표기의 의미

얼굴을 볼 수 없거나 기존 ID와의 연결이 애매하면 `unknown_N`으로 남기고 `low_confidence=true`를 기록합니다. 추정으로 기존 사람에게 강제 병합하지 않습니다.

결과 영상에는 **행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다**라는 자막이 burned-in으로 포함됩니다. 60초 1080p/30fps 영상의 참고 처리 시간은 M 시리즈 Mac에서 약 10분 안쪽이지만, 사람 수와 장치·샘플링 간격에 따라 달라집니다.

`attention_pct`, `interest`, expression timeline은 행동 proxy 구현을 검증하기 위한 연구 출력이며 의학적·교육적 진단값이 아닙니다. 신원 재연관 확신이 부족하면 `unknown_N`과 `low_confidence=true`로 남깁니다.

Ubuntu 22.04 설정은 [`docs/ondamm-video-ubuntu22.md`](docs/ondamm-video-ubuntu22.md)를 참고하세요.

## SMIRK 연구 경로

SMIRK adapter는 person/session/source-video 누수를 막는 manifest, Track A/B runner, checkpoint feature export, MediaPipe+SMIRK fusion baseline을 제공합니다.

```text
app/ondamm_smirk_manifest.py
smirk_ondamm/dataset.py
smirk_ondamm/train_ondamm.py
smirk_ondamm/export_features.py
app/ondamm_smirk_fusion_train.py
```

이 경로는 코드 계약과 단위 테스트가 구현된 연구용 adapter입니다. 공식 SMIRK checkout, FLAME 자산, GPU 실행, 자체 checkpoint 학습과 held-out 평가는 별도이며, 실제 실행 전에는 `GPU PENDING / TRAINING NOT RUN` 상태로 해석해야 합니다.

## 프로젝트 구조

```text
app/
  micro_expression_*        미세 움직임 촬영·분석·실시간 신호(호환용 파일명 유지)
  holistic_camera.py        얼굴·포즈·손·시선 실시간 preview
  ondamm_*                  ON DAMM 도메인·센싱·웹·영상 분석
smirk_ondamm/               SMIRK dataset/training/export adapter
scripts/                    설치·실행 wrapper
configs/                    tracker·미세 움직임 capture 설정
docs/                       설계와 환경 문서
tests/                      자동 회귀 테스트
ui/                         ON DAMM 로컬 웹 UI
```

다운로드 모델, 개인 dossier, 촬영 영상, key, 출력물과 로컬 캐시는 `.gitignore` 대상입니다.

## 검증

```bash
.venv/bin/python -m py_compile app/*.py smirk_ondamm/*.py
PYTHONPATH=. .venv/bin/python -m pytest -m 'not smoke' -q
```

카메라·MPS·ffmpeg·실제 모델을 쓰는 smoke test는 자동 단위 테스트와 별도로 실행해야 합니다. 자동 테스트 통과는 실제 카메라 품질이나 사람 대상 일반화를 보장하지 않습니다.

## 라이선스와 데이터 경계

- MediaPipe: Apache-2.0
- Ultralytics: AGPL-3.0. 배포·서비스 방식에 따른 의무를 별도로 확인해야 합니다.
- InsightFace 코드: MIT. `buffalo_l` weights는 **non-commercial research purposes only** 제한이 있으므로 상업 사용 금지
- EmotiEffLib/HSEmotion 계열: 각 프로젝트 고지 확인
- DINOv3: Meta DINOv3 License 동의가 필요한 gated model이며 weight를 이 저장소에서 재배포하지 않음
- SMIRK/FLAME: 공식 저장소와 자산별 라이선스·접근 조건을 별도로 준수
- 포함되지 않은 프로젝트: LibreFace와 sixdrepnet은 비교 검토 대상일 뿐 이 저장소 의존성에 포함하지 않음

얼굴 영상, landmarks, blendshapes, 3D geometry와 개인화 모델은 재식별 가능성이 있는 민감 데이터로 취급하세요. 수집 목적, 동의, 보존 기간, 철회 및 파기 절차가 정해지기 전에는 아동 데이터를 수집하지 않는 것이 이 저장소의 기본 운영 원칙입니다.
