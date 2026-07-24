## Summary
개정안은 이전 지적이었던 motion authority, 센서 책임, 전원 정지 체인, phase gate, 파일럿 경계를 대부분 실질적으로 보완했다. 특히 옵션 B를 최종 제품이 아니라 safety mule로 재정의하고, 재시작 권한과 degraded mode를 명문화한 점은 안전 아키텍처를 한 단계 성숙시켰다.

다만 아직 승인하기에는 두 군데가 남아 있다. 첫째, Phase 2 safety mule에서 얻은 안전 증거를 실제 자가균형 휠레그드 본체로 어떻게 재검증할지 전이 게이트가 없다. 둘째, supervisor stop 1초 이내 정지 개시 계약은 0.8 m 최소 이격과 0.35 m 정지거리 목표와 동시에 잡히지 않는다. 그래서 권고 경로 자체는 유지하되 이번 개정안은 `WATCH`와 `REQUEST CHANGES`가 맞다.

## Analysis
- 이전 핵심 이슈 해소 여부:
  - motion authority는 §4에서 permit, veto, restart authority를 분리해 상당 부분 해소됐다.
  - 센서 책임은 §5에서 D435와 전방 하단 2D 라이다로 분리되고 degraded mode까지 적혀 있어 이전보다 명확하다.
  - power stop 체인은 §6에서 active braking, torque off, contactor 개방의 우선순위를 명시해 구조가 좋아졌다.
  - phase gate는 §7과 §10에서 수치 계약과 pass fail을 넣어 실행 가능성이 올라갔다.
  - pilot boundary는 §8에서 포함 제외 기준과 misuse 종료 규칙을 넣어 충분히 좁혀졌다.
- 가장 강한 steelman 반론:
  - safety mule 경로는 안전 인터페이스를 먼저 고정한다는 장점이 있지만, 동시에 최종 휠레그드의 핵심 위험인 자가균형, 무게중심 이동, 낙상 모드, 제동시 pitch 거동을 뒤로 미룬다. 이 반론은 강하다. 잘못 운영하면 mule에서 통과한 안전 논리가 최종 플랫폼에도 그대로 전이된다는 착시를 만들 수 있다.
- 실제 tradeoff tension:
  - 더 보수적인 mule 경로는 취약 사용자 근접 리스크를 낮추지만, 최종 제품 정체성 검증 속도와 증거의 전이성을 잃는다.
  - 측면 센서를 늦추고 절차로 막는 현재 선택은 BOM과 복잡도를 줄이지만, 운영 가능 환경을 강하게 제한한다.
- synthesis:
  - 추천 경로 자체는 유지하는 편이 맞다. 대신 mule에서 최종 휠레그드로 넘어가는 순간 safety evidence를 재사용할 수 있는 항목과 재검증해야 하는 항목을 분리하는 전이 게이트를 추가하면 보수성과 제품 정체성 검증을 동시에 잡을 수 있다.
- 명시적 principle violation check:
  - Principle 2, 3, 5는 이번 개정에서 대체로 충족된다. 접촉 없는 거리 유지, 단일 센서 비의존, 감독자 필수 조건이 문서에 박혔다.
  - 다만 Principle 1은 수치 계약에서 약해진다. supervisor stop 1초 이내 정지 개시는 안전보다 운용 편의에 가까운 숫자이며, 현재 이격 거리 계약과 함께 두면 방어 여유가 너무 작다.

## Root Cause
이번 개정은 책임 소유와 금지 범위를 잘 채웠지만, 증거의 전이와 수치 계약의 정합성 관리가 아직 부족하다. 즉 정적인 경계는 좋아졌지만, 플랫폼이 바뀌는 순간과 사람이 멈춤을 요구하는 순간의 동적 안전 계약이 아직 덜 닫혀 있다.

## Findings
### 1. HIGH — safety mule 통과 증거를 실제 휠레그드로 이전하는 재검증 게이트가 없다
- Reference: `.gjc/_session-019ef0be-28b8-7000-a7e1-5b3d641eaf60/plans/ralplan/019ef0be-28b8-7000-a7e1-5b3d641eaf60/stage-02-revision.md` §2 권고 선택, §7 Phase 2 go no go gate, Sequencing.
- Impact: 옵션 B는 문서 스스로 최종 제품 아키텍처가 아니라고 적고 있는데, Sequencing은 Phase 2 go 후 바로 사람추종 검증으로 넘어간다. 이 상태면 mule에서 입증한 deterministic stop, anti roll, brownout 응답을 자가균형 휠레그드에도 그대로 적용해도 된다는 잘못된 해석을 허용한다.
- Fix suggestion: Phase 2와 Phase 3 사이에 명시적 전이 게이트를 추가하라. 예를 들어 Phase 2.5 Final platform requalification을 두고, 자가균형 휠레그드로 넘어가는 순간 최소한 E stop, brownout, watchdog, D435 fault, 라이다 fault, 정지거리, anti fall 또는 anti roll, restart inhibit를 최종 동역학에서 다시 100퍼센트 검증한 뒤에만 사람 대상 시험으로 진입하도록 적어야 한다.

### 2. MEDIUM — supervisor stop 지연 예산이 최소 이격과 정지거리 계약과 정합되지 않는다
- Reference: 같은 파일 §10 공통 수치 계약과 Phase 3 pass fail.
- Impact: 문서는 0.4 m/s 속도 상한, 0.8 m 최소 이격, 0.35 m 정지거리, supervisor stop 요청 후 1초 이내 정지 개시를 동시에 둔다. 이 조합은 비정상 상황에서 supervisor stop이 최후 방어선일 때 남는 여유를 지나치게 줄인다. 현재 숫자만으로는 사람이 멈춤을 요구한 시점 이후에도 과도한 추가 접근을 허용할 수 있다.
- Fix suggestion: supervisor stop을 두 단계로 나눠 적어라. 사람 접근 가능한 즉시 정지 수단은 하드와이어드 E stop 또는 동등한 bounded stop path로 두고, 소프트 stop은 보조 채널로 격하하라. 동시에 latency 목표를 정지 개시가 아니라 safe state 도달 시간이나 추가 접근 거리 예산으로 다시 정의하라.

## Recommendations
1. 가장 먼저 Phase 2와 Phase 3 사이에 최종 플랫폼 재검증 게이트를 추가하라.
2. supervisor stop 계약을 거리 예산 기반으로 다시 쓰고, 인간 개입 채널과 E stop 채널의 역할을 분리하라.
3. 그 다음 단계에서만 현재 수치 계약을 유지한 채 옵션 B safety mule 경로를 승인하라.

## Architectural Status
`WATCH`

## Code Review Recommendation
`REQUEST CHANGES`

## Trade-offs
- 옵션 B 유지
  - 장점: 취약 사용자 근접 리스크를 가장 잘 낮춘다.
  - 단점: 최종 휠레그드 증거 전이성이 약하다.
- 더 이른 최종 플랫폼 통합
  - 장점: 제품 정체성과 실제 동역학을 빨리 검증한다.
  - 단점: 균형 실패와 추종 실패가 초기에 결합되어 위험이 급증한다.
- 합성안
  - 옵션 B는 유지하되, 최종 플랫폼 재검증 게이트를 강제해 보수성과 전이성을 함께 확보한다.
