**[ITERATE]**

**Review Result**: COMMENT

**Justification**: 옵션 B 선택과 단계 게이팅은 안전 우선 원칙과 대체로 맞지만, 실행자가 추정 없이 움직이기에는 핵심 안전 계약이 아직 비어 있다. 특히 계획의 Acceptance criteria는 문서 산출 여부 위주라 안전 게이트로 테스트할 수 없고, Expanded test plan도 시나리오 이름은 있으나 수치 임계값과 pass/fail 판정이 없다. Architect 리뷰는 제어 권한, 센서 책임, 전원 정지 체인을 정확히 잡았지만 이 Acceptance criteria 및 검증 정량화 부족을 명시적으로 승격하지 않아 보완 코멘트가 필요하다.

**Summary**:
- Clarity: 단계와 범위는 명확하지만 motion authority, safe state, restart contract가 빠져 실행 해석 여지가 크다.
- Verifiability: README.md의 Obsidian 관례와 저장소 내 RealSense 또는 moteus 통합 코드 부재는 확인했다. 외부 제품 사양과 공급성은 이번 비평에서 재검증하지 않았다.
- Completeness: 노인 우선과 영아 제한은 타당하나 파일럿 포함 또는 제외 기준, misuse 경계, degraded mode가 비어 있다.
- Big Picture: 옵션 B는 안전 검증용 mule로는 타당하지만 진짜 휠레그드 정체성 유지용 Phase 2 go or no-go가 필요하다.
- Principle/Option Consistency: 안전 우선 원칙과 옵션 B 선택은 맞지만 2D 라이다 또는 초음파 식 개방형 선택은 원칙 3과 충돌한다.
- Alternatives Depth: 계획의 대안 비교는 기본 수준은 충족하고 Architect의 steelman이 균형을 보강한다. 다만 각 대안의 제품 정체성 손실과 재사용성 비용을 같은 해상도로 비교해야 공정하다.
- Risk/Verification Rigor: 사전 실패 분석 3개와 시험 범주 구조는 출발점으로 충분하지만 deliberate mode 기준으로는 불충분하다. 이유는 정지거리, 최소 이격, 가감속, confidence threshold, brownout 동작, sensor disagreement, restart 승인 조건이 수치화되지 않았기 때문이다.
- Deliberate-mode sufficiency: 불충분. 프리모템은 3개로 끝나도 되지만 각 시나리오마다 관측 신호, 차단 조건, 복귀 조건, 로그 증거가 붙어야 하고 확장 시험 계획은 phase gate별 계량 exit criteria가 필요하다.

**Representative task simulation**:
1. Phase 3 안전 반경 유지 구현: 현재 최소 거리와 정지거리 예산이 없어 구현자와 시험자가 서로 다른 기준을 세울 수 있다.
2. 센서 고장 주입 검증: 2D 라이다 또는 초음파가 모두 열려 있어 커버리지 책임과 degraded mode를 설계 없이 시험할 수 없다.
3. 저전압 및 E-stop 복귀 절차 구현: torque off인지 active braking인지, 누가 재시작을 승인하는지 없어 안전 체인 테스트 케이스가 잠기지 않는다.

**Minimum revision set for next planner pass**:
1. 옵션 B를 safety mule로 명시하고 Phase 2 종료 게이트에 휠레그드 정체성 go or no-go 기준을 추가한다.
2. motion authority matrix를 추가해 perception, supervisor, watchdog, E-stop, contactor의 move permit, veto, restart 권한을 정의한다.
3. 센서 책임 표를 추가해 D435, 2D 라이다, 초음파 중 무엇을 채택할지와 전방 저위치, 측면 근접, 반사체, 가림 상황별 minimum coverage 및 single-fault degraded mode를 고정한다.
4. Acceptance criteria를 문서 산출물 기준에서 실행 게이트 기준으로 바꾼다. 각 phase마다 속도 상한, 최소 이격, 정지거리, confidence threshold, brownout response, supervisor intervention latency 같은 pass/fail 수치를 넣는다.
5. Expanded test plan에 concrete verification steps를 넣는다. 각 핵심 시나리오별 setup, fault injection, expected stop behavior, required logs, release blocker를 명시한다.
6. 노인 파일럿의 포함 또는 제외 기준과 misuse 모델을 추가한다. 보행기, 지팡이, grabbing, 반려동물, clutter, 인지저하, supervisor ratio를 최소한 gate 수준으로 적는다.
