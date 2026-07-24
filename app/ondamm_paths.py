from pathlib import Path


# ON DAMM MVP는 MediaPipe sensor stack과 분리된 로컬 dossier 경로를 사용한다.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

ONDAMM_DATA = DATA / "ondamm"
ONDAMM_DOSSIERS = ONDAMM_DATA / "dossiers"
ONDAMM_SECRETS = ONDAMM_DATA / "secrets"
ONDAMM_EXPORTS = OUTPUTS / "ondamm"
ONDAMM_LEARNING_EXPORTS = ONDAMM_EXPORTS / "learning"
ONDAMM_EVENT_CLIPS = ONDAMM_LEARNING_EXPORTS / "event-clips"
