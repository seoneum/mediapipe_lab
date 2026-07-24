import argparse

import cv2


def main() -> None:
    # macOS에서 카메라 번호는 환경마다 달라질 수 있다.
    # 이 스크립트는 0번부터 --max-index까지 실제 프레임이 읽히는지 확인한다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index", type=int, default=4)
    args = parser.parse_args()

    for index in range(args.max_index + 1):
        # AVFoundation은 macOS 기본 카메라 백엔드다.
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # ok=True여도 빈 frame이면 실제 사용 불가로 본다.
        status = "ok" if ok and frame is not None and frame.size else "no frame"
        print(f"camera {index}: {status} {width}x{height}")


if __name__ == "__main__":
    main()
