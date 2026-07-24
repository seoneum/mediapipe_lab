## Summary
계획은 옵션 B 선택, 노인 우선 타기팅, Phase 0~4 게이팅이 전반적으로 보수적이고 방향성도 타당하다. 다만 취약 사용자 근접 시스템에 필요한 제어 권한 경계, 센서 커버리지 책임, 전원에서 정지까지의 안전 체인이 아직 인터페이스 수준으로 내려오지 않아 현재 상태로는 실행 승인보다 계획 수정이 먼저다.

## Analysis
- 강점: 단계 분리 자체는 좋다. 벤치 FOC 검증에서 모바일 베이스를 거쳐 사람추종과 파일럿으로 이동하는 순서가 추종 지능과 동역학 리스크를 분리한다 (`stage-01-planner.md:203-229`). 노인 우선, 영아 직접 추종 금지, Linux 런타임 고정도 현실적이다 (`stage-01-planner.md:84-88`, `stage-01-planner.md:213-215`, `stage-01-planner.md:232-254`, `stage-01-planner.md:320`).
- 가장 강한 steelman antithesis: 옵션 B는 안전 데모에는 강하지만, 자가균형 휠레그드라는 핵심 차별점을 뒤로 미뤄 제품 정체성을 흐릴 수 있다. 2D 라이다와 비자가균형 베이스에 맞춘 포장, 전력, HRI 가정이 나중의 진짜 휠레그드 아키텍처에 재사용되지 않을 수 있어, 안전성은 높여도 개념 검증은 지연될 수 있다 (`stage-01-planner.md:63-66`, `stage-01-planner.md:84-88`).
- Principle check:
  - 원칙 1은 대체로 준수한다. 안전 경계 우선 원칙과 단계 게이팅이 일관된다 (`stage-01-planner.md:38-44`, `stage-01-planner.md:227-229`).
  - 원칙 2는 부분 위반이다. 저속 실내, 비접촉을 선언하지만 속도 상한, 가감속, 최소 이격거리, 정지거리 예산이 수치로 고정되지 않았다 (`stage-01-planner.md:39-44`, `stage-01-planner.md:209-210`).
  - 원칙 3은 부분 위반이다. 센서 단일 의존 금지는 맞지만 실제 안전층은 2D 라이다 또는 초음파 보조로 열려 있어 상호 대체 가능한 수준으로 표현된다 (`stage-01-planner.md:41`, `stage-01-planner.md:137`, `stage-01-planner.md:208`, `stage-01-planner.md:283`).
  - 원칙 4는 대체로 준수한다. 관측성, 정지, 수동 개입, 사고 전 버퍼 로그가 명시되어 있다 (`stage-01-planner.md:42`, `stage-01-planner.md:306-309`).
  - 원칙 5는 부분 위반이다. 감독자 존재는 말하지만 시작, 재시작, 대상 전환, 대상 상실 후 재출발 권한 모델이 빠져 있다 (`stage-01-planner.md:43`, `stage-01-planner.md:214-216`, `stage-01-planner.md:273`).
- 실제 tradeoff tension: 안전 여유와 감사 가능성을 키울수록 제품 충실도와 학습 속도는 느려진다. 반대로 초기에 휠레그드 균형까지 같이 검증하면 제품 적합성은 빨리 보이지만 실패 모드가 결합되어 취약 사용자 근접 리스크가 급증한다.
- Synthesis: 옵션 B를 유지하되 최종 제품 아키텍처가 아니라 안전 검증용 mule로 명시하는 편이 낫다. Phase 2 종료 시점에 휠레그드 차별성, 사용자 가치, 전력 여유가 입증되지 않으면 비자가균형 보조 운반 플랫폼으로 피벗하거나 프로그램을 중단하는 명시적 go or no-go 게이트를 넣어야 한다.

## Root Cause
핵심 문제는 계획이 부품 선정과 단계 순서에는 강하지만, 취약 사용자 근접 로봇에 필수적인 안전 인터페이스를 아직 권한 행렬 수준으로 정의하지 않았다는 점이다. 즉 무엇이 움직임을 허가하고, 무엇이 언제 무조건 정지시키며, 저전압, 가림, 통신 지연 시 어느 채널이 최종 권한을 갖는지가 암묵적이다.

## Findings
1. HIGH — 제어 권한 경계와 재시작 권한 모델이 부족함
   - 근거: 상위 인지와 하위 FOC 분리는 언급되지만 (`stage-01-planner.md:56`), 실제 안전 정지 체인은 D435 없이도 안전 정지, watchdog 안정화, 재식별 불확실성 시 즉시 정지, E stop 상태 전이 수준에 머문다 (`stage-01-planner.md:203`, `stage-01-planner.md:228-229`, `stage-01-planner.md:273`, `stage-01-planner.md:289`).
   - 영향: perception, supervisor, watchdog, E stop, contactor 중 누가 최종 motion veto를 갖는지 불명확해 단일 고장 안전성을 검증할 수 없다.
   - 계획 변경 요청: Phase 1 종료 기준 전에 motion authority matrix, safe state 정의, restart inhibit 조건, supervisor 승인 절차, target reacquire 승인 절차, 속도와 이격거리 예산을 별도 섹션으로 추가.
2. HIGH — 센서 중복이 원칙 수준에 머물고 커버리지 명세가 없음
   - 근거: 센서 단일 의존 금지를 말하지만 실제 구현 항목은 2D 라이다 또는 초음파 보조로 열려 있다 (`stage-01-planner.md:137`, `stage-01-planner.md:208`, `stage-01-planner.md:283`).
   - 영향: 라이다와 초음파는 취약 사용자 발, 보행기 다리, 유리, 저반사 장애물에 대한 실패 양상이 달라 상호 대체 관계로 두기 어렵다. 현재 문장으로는 중복이 아니라 선택지 나열이다.
   - 계획 변경 요청: 최소 감지 구역 표와 센서 책임 분할을 추가하고, 전방 저위치 발과 발판 존, 측면 근접 존, 투명 또는 반사 물체 대응, 센서 단일고장 시 degraded mode를 명시.
3. HIGH — 전원, 제동, E stop 체인이 충분히 구체화되지 않음
   - 근거: contactor, 충전 중 구동 금지, E stop 2개, 저전압 시 성능 저하 모드는 있으나 brownout 시 제동 방식, freewheel 허용 여부, 재기동 금지 조건, BMS fault 전이, 기계식 브레이크 필요성은 없다 (`stage-01-planner.md:140`, `stage-01-planner.md:150-151`, `stage-01-planner.md:278-289`).
   - 영향: 저전압이나 비상정지 순간에 차체가 coasting 또는 pitch forward로 넘어가면 노인 낙상 유발 가능성이 남는다.
   - 계획 변경 요청: 전원 차단이 torque off인지 active braking인지, 저전압 임계값별 동작, BMS fault, 수동 밀기 허용 조건, 기계적 구름 방지 전략을 Phase 1에서 2 산출물에 포함.
4. MEDIUM — 노인 우선 타기팅은 타당하지만 파일럿 사용자 경계가 아직 얕음
   - 근거: 보행 보조가 아니고 직접 접촉 없이도 가치가 있다는 정의는 적절하나 (`stage-01-planner.md:233-241`), 실제 파일럿 조건은 보호자 관찰 체크리스트와 짧은 세션 수준이다 (`stage-01-planner.md:214-216`). 보행기, 지팡이, 반려동물, 인지저하, 사용자의 잡아당김 같은 misuse 모델은 빠져 있다.
   - 영향: 취약 사용자 파일럿에서 가장 흔한 비의도 사용을 놓치면 계획은 실행 가능해 보여도 현장에서는 곧바로 금지될 수 있다.
   - 계획 변경 요청: 파일럿 포함과 제외 기준, mobility aid 상호작용 시나리오, 사용자 grabbing 금지 가정, supervisor 1인당 관리 한계, 반려동물 및 clutter 환경 제외 조건을 추가.

## Recommendations
1. 실행 승인 전에 safety controller, motion authority, restart contract를 문서화한다.
2. 2D 라이다 또는 초음파를 최소 커버리지 명세와 fault matrix가 있는 구체 설계 결정으로 바꾼다.
3. Phase 2 이전에 power to stop chain과 brownout behavior를 산출물로 승격한다.
4. Phase 4 이전에 vulnerable user pilot inclusion and exclusion과 misuse scenarios를 체크리스트가 아닌 gate criteria로 승격한다.
5. 옵션 B는 유지하되 Phase 2 종료 시 제품 정체성 검증 go or no-go를 넣는다.

## Architectural Status
`BLOCK`

## Code Review Recommendation
`REQUEST CHANGES`

## Trade-offs
- 옵션 B 유지: 안전성, 감사 가능성, 단계 검증성은 높다. 대신 휠레그드 차별성 검증은 늦어진다.
- 옵션 A 전진: 제품 적합성과 동역학 학습은 빠르다. 대신 균형 실패와 추종 실패가 결합되어 취약 사용자 근접 리스크가 커진다.
- 옵션 C 우선: 가림 내성은 높다. 대신 태그 휴대 부담 때문에 자연스러운 보조 경험과 채택성이 떨어진다.
