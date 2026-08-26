# ON DAMM 영상 분석기 - Ubuntu 22.04 환경 안내

macOS(MPS) 기준 안내는 README.md의 "ON DAMM 영상 분석기 (offline video analyzer)" 섹션에 있습니다. 이 문서는 같은 파이프라인을 Ubuntu 22.04에서, 특히 CUDA GPU가 있는 서버나 워크스테이션에서 돌릴 때의 대안 환경을 다룹니다. CLI 명령어는 macOS와 완전히 동일합니다.

## 1. 시스템 패키지

ffmpeg와 OpenCV/MediaPipe가 런타임에 요구하는 공유 라이브러리를 먼저 설치합니다.

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg libgl1 libglib2.0-0
```

- `ffmpeg`: 결과 MP4 인코딩에 필수입니다. 없으면 분석 CLI가 종료 코드 4로 실패하고 설치 안내를 출력합니다.
- `libgl1`, `libglib2.0-0`: cv2와 mediapipe 임포트에 필요한 런타임 의존성입니다. 모니터 없는 headless 서버에서도 필요합니다.
- `fonts-nanum`(선택): 한글 자막 렌더링용 나눔고딕 TTF를 제공합니다. 시스템에 한글 폰트가 없다면 함께 설치하세요.

```bash
sudo apt install -y fonts-nanum
```

## 2. venv 생성 (python3.10)

Ubuntu 22.04의 기본 python3은 3.10입니다. 저장소 루트에서:

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

requirements.txt의 `# --- ON DAMM video analyzer ---` 섹션이 이때 함께 설치됩니다.

## 3. torch CUDA wheel

CUDA GPU를 쓴다면 PyTorch 공식 CUDA wheel 인덱스에서 torch를 설치합니다. 인덱스 태그(`cu126` 등)는 쓰려는 torch 버전과 CUDA 드라이버 조합에 맞춰 https://pytorch.org 에서 고르면 됩니다. 아래는 그 패턴입니다.

```bash
.venv/bin/pip install "torch>=2.7,<2.14" --index-url https://download.pytorch.org/whl/cu126
```

설치 확인:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True`가 출력되면 CUDA 사용 준비가 끝난 것입니다.

## 4. 모델 내려받기와 환경 검증

명령어는 macOS와 동일합니다.

```bash
bash scripts/download_video_models.sh
.venv/bin/python -m app.ondamm_video_env --check
```

Ubuntu에서는 `env_report.json`의 `mps_available`이 항상 `false`로 기록됩니다. 이건 오류가 아니라 플랫폼 차이입니다.

## 5. 분석 실행

CLI 명령어도 macOS와 완전히 동일합니다.

```bash
bash scripts/ondamm_video_analyzer.sh \
  --input input.mp4 \
  --output outputs/ondamm/video/result.mp4 \
  --device auto \
  --sample-every 3 \
  --metrics-json outputs/ondamm/video/metrics.json \
  --metrics-csv outputs/ondamm/video/metrics.csv
```

직접 실행:

```bash
.venv/bin/python -m app.ondamm_video_analyzer_cli \
  --input input.mp4 \
  --output outputs/ondamm/video/result.mp4 \
  --metrics-json outputs/ondamm/video/metrics.json \
  --metrics-csv outputs/ondamm/video/metrics.csv
```

장치 선택 참고:
- `--device cuda`: CUDA GPU가 있을 때 사용합니다.
- `--device mps`: Apple Silicon 전용입니다. Linux에는 MPS가 없으므로 MPS 전용 코드 경로는 자동으로 CPU 또는 CUDA로 폴백(fallback)됩니다. 별도 설정이 필요 없습니다.
- `--device auto`: 감지된 환경 기준으로 최적 장치를 고릅니다. Ubuntu + CUDA GPU라면 cuda, 아니면 cpu입니다.
- 종료 코드 체계(2 입력 실패 / 3 모델 없음 / 4 렌더 실패)도 README와 동일합니다.
