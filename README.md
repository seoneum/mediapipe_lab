# mediapipe_lab · ON DAMM

Apple Silicon Mac에서 MediaPipe 기반 얼굴·시선·자세 신호를 실험하고, ON DAMM의 local-first 학습지원 흐름과 개인화 얼굴 움직임 연구를 검증하는 저장소입니다.

현재 핵심은 다음 세 가지입니다.

- 미세 움직임 연구: 통제 촬영, 프레임별 478 landmarks·52 blendshapes·head transform 추출, optical-flow·DINOv3 변화 시각화
- ON DAMM MVP: dossier, 승인된 수업 기록, 학습 프로그램 초안, handoff, 철회 잠금, 검토형 얼굴 움직임 프로필
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

이 저장소는 현재 **수집·신호 추출·시각화까지 구현**되어 있고, 1D CNN/TCN temporal classifier는 아직 구현하지 않았습니다. 모델 입력의 권장 형태는 프레임별 절대값, neutral 대비 변화량, 1차 시간차를 함께 쓰는 것입니다.

```text
b_t                 절대 blendshape
b_t - b_neutral     개인별 neutral 대비 변화
b_t - b_(t-1)       순간 변화
15~30 frame window  약 0.25~0.5초 시계열(60 fps 기준)
```

첫 모델은 특정 사람·특정 촬영 환경에 맞춘 이진 분류부터 시작하는 것이 안전합니다.

```text
neutral vs target_movement
→ 작은 MLP / 1D CNN / TCN
→ confidence가 낮으면 abstain
```

특정 사람에게 overfitting하는 것은 이 연구 목적에서는 personalization으로 사용할 수 있습니다. 다만 같은 영상의 인접 프레임을 train/test에 무작위로 섞으면 성능이 과대평가되므로, 최소한 session 또는 source video 단위로 분리하고 별도의 재촬영 세션을 최종 확인용으로 남겨 두세요.

MediaPipe 수치에서 목표 변화가 noise와 구분되지 않을 때는 얼굴 정렬에 MediaPipe를 유지한 채 눈·눈썹·입 ROI의 frame difference, optical flow 또는 DINO feature를 추가합니다.

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
