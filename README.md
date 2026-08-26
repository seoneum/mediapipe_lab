# MediaPipe Lab

macOS Apple Silicon 기준 MediaPipe Tasks 실습 프로젝트입니다.
Obsidian 기준 가이드: `💡 Notes/Reference/Engineering/N - MediaPipe 완전 가이드 - MacBook Pro M5 Pro 기준.md`

## 1. 설치

```bash
cd /Users/seoneum/ai/mediapipe_lab
bash scripts/setup_env.sh
```

## 2. 실행

```bash
# 사용 가능한 카메라 인덱스 찾기
bash scripts/camera_probe.sh

# Holistic 실시간 카메라. 1번이 안 되면 --camera 0부터 다시 시도
bash scripts/holistic_camera.sh --camera 1

# 선 없이 점만 보기 / 얼굴 점 숨기기
bash scripts/holistic_camera.sh --camera 1 --no-lines
bash scripts/holistic_camera.sh --camera 1 --no-face

# 점과 선 크기 조정
bash scripts/holistic_camera.sh --camera 1 --point-radius 3 --line-thickness 3

# 표정 추정 또는 iris/gaze 표시 끄기
bash scripts/holistic_camera.sh --camera 1 --no-expression
bash scripts/holistic_camera.sh --camera 1 --no-iris

# 인식 민감도 조정. 낮추면 더 잘 잡지만 오탐 증가, 높이면 확실한 것만 표시
bash scripts/holistic_camera.sh --camera 1 --hand-landmarks-confidence 0.35
bash scripts/holistic_camera.sh --camera 1 --pose-detection-confidence 0.65

# 정적 이미지 Object Detector. 이미지가 없으면 데모 이미지를 자동 생성
bash scripts/object_image.sh

# 이미지에서 detection 최대 개수와 score 기준 조정
bash scripts/object_image.sh --image data/images/my_photo.jpg --output outputs/my_photo_detected.jpg --max-results 20 --score 0.35

# 핵심 패키지 버전 확인
bash scripts/package_versions.sh
```

## 2-1. ON DAMM MVP 시작

이 저장소는 이제 MediaPipe 실습 코드 위에 **ON DAMM 개인맞춤 학습지원 MVP**를 함께 담고 있습니다.

### 웹 UI 실행

명령어 대신 로컬 웹 화면에서 지원 기록철, 승인 수업 기록, 활동 추천 초안/승인, 학습 프로그램 미리보기, 관찰 보조 데모 초안, 서명된 인수인계를 사용할 수 있습니다.

```bash
bash scripts/ondamm_web.sh
```

브라우저에서 `http://127.0.0.1:8765`를 엽니다. 다른 포트는 `bash scripts/ondamm_web.sh --port 9000`처럼 지정할 수 있습니다.

- 서버는 기본적으로 `127.0.0.1`에만 바인딩됩니다.
- 기존 `data/ondamm/dossiers/` JSON을 그대로 사용합니다.
- 추천과 관찰 결과는 승인 전까지 공식 기록철에 자동 저장되지 않습니다.
- 실제 카메라 미리보기는 기존 `scripts/ondamm_sensing.sh` 경로를 유지합니다.
- `outputs/ondamm/**/event_recording.json`에 연결된 MP4는 관찰 보조 화면에서 로컬 재생됩니다.
- Google MediaPipe 분석은 선택 영상의 샘플 프레임을 이 Mac 안에서 처리하고, 얼굴 blendshape를 감정이 아닌 **표정 움직임 힌트**로만 표시합니다.
- GPT 검토는 선택 사항입니다. 전체 MP4가 아니라 최대 3장의 축소 JPEG 프레임만, UI에서 매번 명시적으로 동의한 경우에 OpenAI Responses API로 전송합니다.

GPT 검토를 사용할 때만 서버 실행 전에 키를 환경 변수로 전달합니다. 키를 소스나 브라우저에 입력하지 마세요.

```bash
export OPENAI_API_KEY='your-api-key'
export ONDAMM_GPT_MODEL='gpt-5.6'  # 생략 시 gpt-5.6
bash scripts/ondamm_web.sh
```

키가 없으면 로컬 영상 재생과 MediaPipe 분석은 그대로 작동하고 GPT 버튼만 비활성 상태로 표시됩니다. GPT 결과도 비권위 초안이며 공식 기록철에 자동 저장되지 않습니다.

핵심 원칙:
- 진단 도구가 아닙니다.
- 센서는 보조 신호일 뿐입니다.
- continuity dossier가 중심입니다.
- transfer는 import가 아니라 human-readable handoff artifact를 참고한 수동 재구성입니다.
- 아동 데이터 처리, 센싱 해석, 추천 생성, 저장은 local-first가 기본이며, 별도 문서화된 동의 기반 export 흐름 없이는 원격 추론·원격 동기화·텔레메트리 전송을 하지 않습니다.

### 2-1A. ASD 학습지원 근거와 ON DAMM 학습 프로그램 초안 (업데이트: 2026-07-11)

ON DAMM의 학습 프로그램은 **시각적 구조화 + 짧은 과제 블록 + 체계적 프롬프팅 + 강화 + 전환 지원**을 기본 축으로 잡습니다.

문헌 근거 요약:
- Knight, Sartini, Spriggs (2015): visual activity schedule은 ASD 학습자에게 evidence-based practice로 볼 수 있고, **systematic instructional procedure**와 함께 쓸 때 효과가 큽니다.
- Steinbrenner et al. (2020): visual supports, prompting, reinforcement, task analysis는 ASD 아동·청소년 대상 evidence-based practice 묶음입니다.
- NPDC/AFIRM Visual Support Brief (2016): visual support는 그림, 글자, object, schedule, timeline, script처럼 **말 대신/말과 함께 쓰는 구체적 단서**입니다.
- Waters, Lerman, Hovanetz (2009): visual schedule만으로는 전환 시 문제행동이 줄지 않을 수 있으므로, **강화와 기능 기반 지원**을 함께 써야 합니다.

따라서 ON DAMM 학습 프로그램은 다음 순서로 구성하는 것이 안전합니다.
1. **도입**: 오늘의 목표, 순서, 끝나는 조건을 시각 카드로 먼저 보여 준다.
2. **짧은 과제 블록**: 3~10분 내 과제를 한 번에 하나씩 제시한다.
3. **프롬프팅**: 처음에는 시각 단서/제스처/짧은 구두 지시를 쓰고, 반응이 안정되면 prompt를 점진적으로 줄인다.
4. **강화와 전환 지원**: 완료 즉시 짧은 칭찬, 선호 자극, 쉬는 시간, next-step 예고를 연결한다.
5. **기록**: 정답률만이 아니라, 어떤 prompt가 먹혔는지, 어떤 전환에서 저항이 있었는지, 어떤 강화가 안정화에 도움 되었는지를 session summary로 남긴다.

운영 계약:
- **교사/보호자 역할**: 목표 설정, prompt 선택, 강화 방식 승인, session summary 검토를 담당한다.
- **아동 역할**: 시각 일정표를 보고 한 번에 하나의 과제를 수행한다.
- **1회 세션 종료 기준**: 과제 1~3개 완료, 예정 시간 도달, 피로/거부 신호 확인 중 하나가 충족되면 종료한다.
- **실패/저항 시 fallback**: 더 쉬운 단계로 낮추기, first-then 형태로 축소하기, prompt 강도 일시 상향 후 다시 줄이기, 짧은 휴식 후 재시도, 그날 세션 종료 중 하나를 선택한다.
- **기록 최소 단위**: 사용한 prompt, 성공한 강화, 어려웠던 전환, 다음 담당자에게 넘길 메모를 남긴다.

ON DAMM 구현 방향:
- 학습 프로그램은 아동을 진단하거나 평가하지 않는다.
- 추천은 **다음 활동 초안**과 **전환/강화 메모**를 제안하는 수준으로 제한한다.
- 카메라 센싱은 점수화가 아니라, **특이 이벤트 전후의 관찰 보조 증거**와 reviewed note draft 생성에만 사용한다.

참고 링크:
- https://pubmed.ncbi.nlm.nih.gov/25081593/
- https://eric.ed.gov/?id=ED609029
- https://eric.ed.gov/?id=ED595398
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2695333/

```bash
# dossier 생성
bash scripts/ondamm_mvp.sh create-dossier \
  --child-id child-demo \
  --name "Demo Child" \
  --age-band "초등 저학년" \
  --communication-modality "시각 단서 + 짧은 구두 지시" \
  --preference "동물 카드" \
  --strategy "짧은 단계 제시" \
  --support "전환 전에 예고하기"

# dossier 조회
bash scripts/ondamm_mvp.sh show-dossier --child-id child-demo --json

# 승인된 세션 요약 추가
bash scripts/ondamm_mvp.sh add-session-summary \
  --child-id child-demo \
  --title "동물 분류 활동 1회차" \
  --activity "동물 분류" \
  --response "시작 1분 안에 카드 선택 반응이 안정화됨" \
  --interpretation "선호 자극을 먼저 제시하면 전환 저항이 낮아짐" \
  --approved-by "teacher-a" \
  --tag "동물" --tag "전환성공"

# baseline 추천 초안 생성 + 승인 저장
bash scripts/ondamm_mvp.sh recommend-baseline \
  --child-id child-demo \
  --goal "분류 활동 5분 유지" \
  --caregiver-input "짧은 피드백이 있을 때 더 잘 참여함" \
  --drafted-by "teacher-a" \
  --approved-by "teacher-a" \
  --output baseline-demo.md

# human-readable handoff artifact 생성
bash scripts/ondamm_mvp.sh handoff-brief --child-id child-demo

# 문헌 근거 기반 학습 프로그램 초안 + 실행 요약 생성
bash scripts/ondamm_learning.sh \
  --child-id child-demo \
  --demo \
  --headless \
  --output-dir outputs/ondamm/artifacts/g002-learning-demo-parent-no-record

# 특이 이벤트(local clip)까지 같이 남기는 데모 실행
bash scripts/ondamm_learning.sh \
  --child-id child-demo \
  --demo \
  --record-events \
  --headless \
  --output-dir outputs/ondamm/artifacts/g002-learning-demo-parent-record
```

생성 위치:
- dossier JSON: `data/ondamm/dossiers/`
- handoff / recommendation markdown: `outputs/ondamm/`
- learning plan / run summary / manifest: `outputs/ondamm/artifacts/...`
- event clip outputs when `--record-events`: `<output-dir>/event-clips/`

### 2-1B. 얼굴 신호 이벤트화·개인화 계획

구현 현황과 후속 안전 경계는 [`docs/ON DAMM 얼굴 신호 이벤트화 계획.md`](docs/ON%20DAMM%20얼굴%20신호%20이벤트화%20계획.md)에 정리되어 있습니다.
## 2-2. ON DAMM sensing assist

Option C는 **baseline dossier를 대체하지 않는 보조 lane**입니다.
센싱 결과는 dossier에 자동 저장되지 않고, 교사가 검토할 **reviewed note draft**만 생성합니다.
승인 상태 전이:
- `reviewed_note_draft -> human review -> approved session summary` 또는 `discard`
- reviewed note draft는 교사/보호자 검토 전까지 canonical dossier 기록이 아닙니다.

```bash
# deterministic demo mode: camera 없이 sensing draft 출력
bash scripts/ondamm_sensing.sh \
  --child-id child-demo \
  --duration-seconds 8 \
  --audio-presence-note "짧은 발성이 들렸음" \
  --demo

# 실제 카메라 기반 headless sensing draft
bash scripts/ondamm_sensing.sh \
  --child-id child-demo \
  --camera 1 \
  --duration-seconds 8 \
  --headless

# 실제 카메라 기반 live preview
# 2026-07-11 업데이트: --headless 없이 실행하면 얼굴 점, 손/포즈 overlay,
# iris marker, expression / gaze / posture 텍스트를 같이 볼 수 있다.
bash scripts/ondamm_sensing.sh \
  --child-id child-demo \
  --camera 1 \
  --duration-seconds 8
```

출력 원칙:
- raw media 저장 안 함
- dossier 자동 writeback 안 함
- face presence / coarse gaze zone / posture proxy만 요약
- 진단 / 집중도 점수 / 순응도 점수로 해석 금지

## 2-3. Continuity export / manual re-establishment

G003에서는 recipient-side import를 만들지 않고, **human-readable export + origin-side integrity evidence + manual re-establishment template**만 제공합니다.

```bash
# signed handoff export + manifest 생성
bash scripts/ondamm_mvp.sh export-handoff \
  --child-id child-demo \
  --output export-child-demo.md

# recipient 쪽 수동 continuity 재구성용 template 생성
bash scripts/ondamm_mvp.sh prepare-reestablishment \
  --manifest outputs/ondamm/export-child-demo.md.manifest.json

# 보호자/관리자 withdrawal lock
bash scripts/ondamm_mvp.sh withdraw-dossier \
  --child-id child-demo \
  --reason-code consent_withdrawn \
  --reason "보호자 요청"
```

제약:
- export는 snapshot이다.
- recipient-side import / promotion 없음.
- continuity는 수동 재작성이다.
- withdrawn_locked 상태에서는 handoff/export/recommend/session write가 막힌다.

## 3. 모델

- `models/holistic_landmarker.task`
  - 얼굴, 포즈, 왼손, 오른손 landmark를 한 번에 반환합니다.
  - 현재 Python Holistic Task는 기본적으로 주 피사체 1명 기준으로 이해하는 것이 맞습니다.
- `models/efficientdet_lite2.tflite`
  - COCO object detection 모델입니다.
  - 여러 사람 수를 제대로 세려면 이 계열 detector에서 `person` detection을 세는 방식이 필요합니다.

## 4. 어디를 고치면 되는가

대부분의 실시간 카메라 화면 동작은 `app/holistic_camera.py`에 있습니다.

| 바꾸고 싶은 것 | 수정 위치 |
| --- | --- |
| 기본 카메라 번호 | `DEFAULT_CAMERA_INDEX` 또는 `--camera` |
| 기본 해상도 | `DEFAULT_WIDTH`, `DEFAULT_HEIGHT` 또는 `--width`, `--height` |
| 점 크기 | `DEFAULT_POINT_RADIUS` 또는 `--point-radius` |
| 선 두께 | `DEFAULT_LINE_THICKNESS` 또는 `--line-thickness` |
| 몸 관절 연결 | `POSE_CONNECTIONS` |
| 손 관절 연결 | `HAND_CONNECTIONS` |
| 점/선 색상 | `draw_points(...)`, `draw_connections(...)` 호출부의 BGR 색상값 |
| 얼굴 점 표시 끄기 | `--no-face` |
| skeleton line 끄기 | `--no-lines` |
| 표정 추정 표시 끄기 | `--no-expression` |
| iris/gaze 표시 끄기 | `--no-iris` |
| 표정 규칙 | `EXPRESSION_RULES` |
| 표정 표시 개수 | `DEFAULT_TOP_EXPRESSIONS` 또는 `--top-expressions` |
| 시선 판정 민감도 | `estimate_gaze(...)` 안의 threshold |
| face/pose/hand 인식 민감도 | `--face-detection-confidence`, `--pose-detection-confidence`, `--hand-landmarks-confidence` 등 |
| 모델 파일 경로 | `app/paths.py` |
| 설치 패키지 | `requirements.txt` |
| 모델 다운로드 URL | `scripts/download_models.sh` |

## 5. 사람 수와 손 개수에 대한 핵심 주의

현재 `app/holistic_camera.py`의 화면 표시는 다음 의미입니다.

- `people=0/1`
  - Holistic이 주 피사체 pose landmark를 잡았는지 여부입니다.
  - 실제 화면 안의 전체 사람 수가 아닙니다.
- `hands=0~2`
  - 왼손 landmark가 있으면 1, 오른손 landmark가 있으면 1을 더한 값입니다.
  - Holistic 구조상 왼손 1개 + 오른손 1개까지 보는 것으로 이해하면 됩니다.
- `pose_pts`, `face_pts`, `left_pts`, `right_pts`
  - 사람/손 개수가 아니라 잡힌 landmark 점 개수입니다.

여러 사람을 세고 싶으면:

1. `ObjectDetector`에서 `person` bounding box 개수를 센다.
2. 필요하면 각 person 영역을 잘라 `PoseLandmarker` 또는 별도 파이프라인에 넣는다.
3. 그 다음에 특정 사람 1명을 골라 Holistic으로 얼굴/손/포즈를 자세히 본다.

즉, `Holistic = 한 명을 자세히 보기`, `Object/Pose Detector = 여러 명을 찾기`로 나누는 것이 안전합니다.

## 6. 코드 읽는 순서

1. `app/paths.py`
   - 프로젝트 루트, 모델 폴더, `BaseOptions` 생성 위치
2. `app/check_install.py`
   - 모델 파일이 MediaPipe Task로 열리는지 확인
3. `app/object_image.py`
   - 정적 이미지 `IMAGE` 모드, bounding box, `person_count`
4. `app/camera_probe.py`
   - macOS 카메라 인덱스 확인
5. `app/holistic_camera.py`
   - 실시간 `VIDEO` 모드, timestamp, landmark 표시, 표정/gaze 표시
6. `scripts/*.sh`
   - 위 Python 파일들을 편하게 실행하는 wrapper

## 7. 주의

MediaPipe Tasks는 CPU inference를 선택해도 macOS GL/Metal 컨텍스트를 만들 수 있습니다.
따라서 카메라 앱과 설치 검증은 제한된 샌드박스보다 일반 Terminal 또는 PyCharm에서 실행하는 편이 안정적입니다.

## ON DAMM 영상 분석기 (offline video analyzer)

영상 파일 하나를 넣으면, 그 안의 사람마다 고유 ID가 유지된 채 집중 시간, 흥미도, 표정(8종 + 감정가/각성)이 한글 자막으로 새겨진 결과 영상과 개인별 지표 파일(JSON/CSV)을 만들어 줍니다. 웹캠 실시간 처리가 아니라 오프라인 파일 처리만 하고, 추론 시점에는 클라우드 API를 쓰지 않습니다. 모든 처리는 이 Mac 안에서 끝납니다.

### 빠른 시작 (macOS)

```bash
cd /Users/seoneum/ai/mediapipe_lab
bash scripts/setup_env.sh
bash scripts/download_video_models.sh
.venv/bin/python -m app.ondamm_video_env --check
```

`--check`는 임포트, ffmpeg 존재, 모델 파일, MPS 사용 가능 여부를 검증하고 결과를 `outputs/ondamm/video/env_report.json`에 기록합니다. MPS fp32 검증에서 허용 오차(로그잇 최대 절대 오차 1e-3)를 넘기면 기본 장치가 CPU로 기록되고, 분석기도 그 값을 따릅니다.

### 분석 CLI

CLI 구현은 `app/ondamm_video_analyzer_cli.py`이고, 셸 wrapper는 `scripts/ondamm_video_analyzer.sh`입니다.

```bash
bash scripts/ondamm_video_analyzer.sh \
  --input input.mp4 \
  --output outputs/ondamm/video/result.mp4 \
  --device auto \
  --sample-every 3 \
  --metrics-json outputs/ondamm/video/metrics.json \
  --metrics-csv outputs/ondamm/video/metrics.csv
```

wrapper 없이 직접 실행할 수도 있습니다.

```bash
.venv/bin/python -m app.ondamm_video_analyzer_cli \
  --input input.mp4 \
  --output outputs/ondamm/video/result.mp4
```

| 플래그 | 의미 |
| --- | --- |
| `--input PATH` | 입력 영상 파일. 열지 못하면 종료 코드 2 |
| `--output PATH` | 자막이 새겨진 결과 MP4 저장 경로 |
| `--device {auto,cpu,mps,cuda}` | 연산 장치. `auto`는 환경에서 감지된 최적 장치를 고릅니다 |
| `--sample-every N` | N 프레임마다 얼굴 신호를 샘플링합니다. 기본값은 3 |
| `--metrics-json PATH` | 개인별 지표 JSON 출력 경로 |
| `--metrics-csv PATH` | 개인별 지표 CSV 출력 경로 |

종료 코드:
- `0`: 성공
- `2`: 입력 영상 열기 실패
- `3`: 모델 파일 없음. `bash scripts/download_video_models.sh`를 먼저 실행하세요.
- `4`: 렌더 실패. 대부분 ffmpeg 부재입니다. macOS는 `brew install ffmpeg`, Ubuntu는 `sudo apt install ffmpeg`.

### 결과물

모든 산출물은 `outputs/ondamm/video/` 아래에 모입니다.

결과 MP4:
- 사람마다 색 있는 박스와 `{global_id} · 집중 {attention_pct}% · 흥미 {interest} · {dominant expression}` 라벨이 붙습니다.
- 모든 프레임 하단에 반투명 자막 **행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다**가 새겨집니다(burned-in). 이 문구는 결과 영상과 함께 움직입니다. 영상만 공유해도 비진단 문구가 빠지지 않습니다.

지표 JSON/CSV는 사람별로 한 행(객체)씩 기록됩니다. 필드는 PersonMetrics 스키마로 고정되어 있습니다.

| 필드 | 의미 |
| --- | --- |
| `global_id` | ArcFace 임베딩 재연관으로 유지되는 개인 고유 ID |
| `attention_pct` | 집중 비율(0~100) |
| `focus_seconds` | 누적 집중 초 |
| `interest` | 흥미도 수준. `낮음` / `중간` / `높음` |
| `expression_timeline` | `{t_sec, label}` 목록. 시간별 표정 타임라인 |
| `frames_covered` | 얼굴 신호가 덮인 프레임 수 |
| `total_frames` | 영상 전체 프레임 수 |
| `low_confidence` | 신원 확신이 낮으면 `true` |

### unknown_N 표기의 의미

얼굴을 볼 수 없거나 임베딩이 애매해서 기존 개인 ID에 묶을 수 없는(low-confidence) 사람은 `unknown_1`, `unknown_2` 같은 임시 ID로 표기되고 `low_confidence=true`가 함께 기록됩니다. 이후 프레임에서 확신 기준(코사인 유사도와 마진)을 넘으면 같은 사람이 원래 ID로 다시 묶일 수 있습니다. unknown_N은 실패 표시가 아니라 "아직 확신 없음"이라는 상태의 정직한 표시입니다.

### 처리 속도 참고치

60초 1080p/30fps 클립 하나를 M 시리즈 Mac 노트북(CPU 또는 MPS)에서 끝까지 처리하는 데 약 10분 안쪽이 걸립니다. 영상 길이, 사람 수, `--sample-every` 값에 따라 달라집니다.

### 캘리브레이션

data/calib 구조 안내는 캘리브레이션 워커 산출물 문서 참조.

## 라이선스 고지 (NOTICE)

이 프로젝트는 사전학습 모델과 외부 라이브러리를 조합합니다. 각 구성요소의 라이선스와 배포 조건은 다음과 같습니다.

- **ultralytics (YOLO26)**: AGPL-3.0으로 배포됩니다. 학술 목적으로 본 프로젝트의 결과물을 배포할 때는 ultralytics의 AGPL-3.0 내용을 명시하고 해당 소스 링크(https://github.com/ultralytics/ultralytics )를 함께 제공해야 합니다.
- **insightface buffalo_l**: insightface 코드 자체는 MIT입니다. 그러나 buffalo_l 사전학습 가중치는 비상업 연구(non-commercial research purposes only) 전용 약관으로 배포됩니다. 상업적 용도로는 가중치를 사용할 수 없습니다.
- **MediaPipe**: Apache-2.0. Copyright The MediaPipe Authors.
- **EmotiEffLib 및 HSEmotion 계열**: Apache-2.0 표기를 따릅니다.
- **포함되지 않은 프로젝트**: LibreFace(USC academic license)와 sixdrepnet은 이 프로젝트에 포함되어 있지 않습니다. 문헌 검토 단계에서 비교한 백업 후보였을 뿐, 의존성으로 넣지 않았습니다.
