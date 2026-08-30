from __future__ import annotations

import argparse
import json

from ondamm_purge import execute_purge, preview_purge


def main() -> int:
    parser = argparse.ArgumentParser(description="철회된 아동의 로컬 자료를 안전하게 확인하고 삭제합니다.")
    parser.add_argument("--child-id", required=True, help="철회로 잠긴 지원 기록철의 로컬 ID")
    parser.add_argument("--execute", action="store_true", help="미리보기가 아니라 실제 삭제를 실행합니다.")
    parser.add_argument("--confirmation", help="실제 삭제 확인 문구: 삭제 <아동 ID>")
    parser.add_argument("--actor-id", default="guardian-admin", help="삭제 실행 담당자")
    args = parser.parse_args()
    try:
        result = (
            execute_purge(
                args.child_id,
                confirmation=args.confirmation or "",
                actor_id=args.actor_id,
            )
            if args.execute
            else preview_purge(args.child_id)
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"실행할 수 없습니다: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        print(f"\n실제 삭제하려면 --execute --confirmation '삭제 {args.child_id}'를 추가해 주세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
