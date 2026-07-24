## Summary
이번 3차 개정은 직전 리뷰에서 남겨졌던 두 핵심 결손을 실질적으로 닫았다. Phase 2.5 Final-platform requalification gate가 safety mule 증거의 무비판적 전이를 금지하고, stop 계약이 human-immediate-stop과 soft supervisor stop으로 분리되어 거리 예산 기반 acceptance로 재정의됐다. 현재 deliberate-mode 계획은 취약 사용자 근접 운용 전제에서 승인 가능한 수준이다.

## Analysis
- 최종 플랫폼 재적격화 이슈 해소: 직전 아키텍트 리뷰는 mule 통과 증거를 자가균형 최종 플랫폼으로 그대로 넘기는 해석을 막는 전이 게이트 부재를 HIGH로 지적했다 (.gjc/_session-019ef0cb-a3b8-7000-8fe8-994d5058aa61/plans/ralplan/019ef0cb-a3b8-7000-8fe8-994d5058aa61/stage-02-architect.md:29-31). 이번 개정은 Phase 2.5를 신설해 증거 전이 금지 원칙, 8개 재검증 항목, 100퍼센트 재현 검증, 사람 없는 시험 release blocker 0건, 미충족 시 인간 대상 시험 금지를 명시했다 (.gjc/_session-019ef0be-28b8-7000-a7e1-5b3d641eaf60/plans/ralplan/019ef0be-28b8-7000-a7e1-5b3d641eaf60/stage-03-revision.md:258-278). 또한 Sequencing에 Phase 2.5를 삽입해 사람 대상 추종보다 앞에 두었다 (stage-03-revision.md:290-293). 이로써 safety mule과 최종 자가균형 동역학 사이의 안전 증거 경계가 문서상 충분히 잠겼다.
- 거리 예산 기반 stop 계약 이슈 해소: 직전 리뷰는 0.4 m/s, 0.8 m 이격, 0.35 m 정지거리와 함께 supervisor stop 1초 이내 정지 개시만 두면 여유가 부족하다고 봤다 (stage-02-architect.md:33-36). 이번 개정은 stop 채널을 human-immediate-stop과 soft supervisor stop으로 분리하고, 각각 추가 접근 0.15 m 상한 및 1.0초 이내 또는 추가 접근 0.25 m 이하 중 더 보수적인 조건으로 고정했다 (stage-03-revision.md:166-169, stage-03-revision.md:373-377). 더 중요한 점은 이 계약이 선언으로 끝나지 않고 Phase 1, 2, 2.5, 3 acceptance와 시나리오 D, F의 expected behavior 및 release blocker로 전파됐다는 것이다 (stage-03-revision.md:383-426, stage-03-revision.md:458-476). 이전의 지연 중심 계약이 이제 거리 예산 중심 안전 계약으로 정합화됐다.
- deliberate-mode 적합성: 실패 시나리오, fault injection, required logs, release blocker가 재적격화와 stop 계약을 따라 구체화되어 실행자가 추측 없이 검증 프로토콜로 내릴 수 있다 (stage-03-revision.md:389-476). 특히 인간 대상 시험 진입 전 사람 없는 최종 플랫폼 검증을 강제한 점이 안전 케이스를 단계적으로 분리한다.
- 명시적 principle violation check:
  - Principle 1 위반 없음. stop 계약이 운용 편의형 latency 목표에서 거리 예산형 bounded stop으로 바뀌어 안전 우선 원칙과 정렬됐다 (stage-03-revision.md:166-169, stage-03-revision.md:373-377).
  - Principle 2 위반 없음. 속도, 최소 이격, 금지 거리, 정지거리 계약이 함께 유지된다 (stage-03-revision.md:372-377, stage-03-revision.md:417-426).
  - Principle 3 위반 없음. D435와 전방 하단 2D 라이다의 책임 분할 및 degraded mode가 계속 고정돼 있다 (stage-03-revision.md:171-183).
  - Principle 4 위반 없음. watchdog, E-stop, brownout, required logs, release blocker가 관측 가능성과 정지 가능성을 우선한다 (stage-03-revision.md:152-169, stage-03-revision.md:449-476).
  - Principle 5 위반 없음. supervisor 승인, 재시작 승인, 인간 대상 시험 전 재적격화, 파일럿 감독 조건이 일관된다 (stage-03-revision.md:154-169, stage-03-revision.md:273-283, stage-03-revision.md:428-433).

## Steelman Antithesis
가장 강한 반론은 그대로 유효하다. safety mule 접근은 여전히 최종 휠레그드의 자가균형, pitch 거동, 낙상 모드, 무게중심 이동 같은 본질적 위험을 뒤로 미룬다. 즉 이번 개정이 좋아진 이유는 그 반론이 사라져서가 아니라, 그 반론이 실제로 맞을 수 있음을 인정한 뒤 재적격화 없이는 인간 대상 시험 금지라는 하드 게이트를 박았기 때문이다.

## Tradeoff Tension
- 보수적 safety mule 경로는 초기 근접 리스크를 낮추지만, 최종 플랫폼에서 동일 시험을 다시 해야 하므로 일정과 검증 비용이 증가한다.
- 더 이른 최종 플랫폼 통합은 제품 정체성을 빨리 검증하지만, 균형 실패와 추종 실패가 초기에 결합되어 실패 비용이 급격히 커진다.
- 이번 개정은 전자를 택하되, 후자의 증거 공백을 Phase 2.5 재적격화로 메운다.

## Synthesis
옵션 B safety mule 권고는 그대로 유지하는 편이 맞다. 다만 그 권고가 정당하려면 mule의 안전 인터페이스와 최종 플랫폼의 동역학을 동일시하지 않아야 하는데, 이번 개정은 바로 그 경계를 명시적 재적격화와 거리 예산형 stop 계약으로 고정했다. 그래서 현재 문서는 보수성과 제품 정체성 검증 가능성을 함께 잡는 쪽으로 수렴했다.

## Root Cause
이전 결손의 근본 원인은 두 가지였다. 첫째, safety evidence의 전이 경계가 문서상 잠겨 있지 않았다. 둘째, supervisor stop이 인간 개입 채널의 역할 구분 없이 지연 목표 하나로 표현되어 있었다. 이번 개정은 이를 각각 Phase 2.5 재적격화 게이트와 채널 분리 + 추가 접근 거리 예산으로 치환해 근본 원인 수준에서 닫았다.

## Findings
남은 블로킹 또는 하이 시급도 이슈는 없다. 직전 리뷰의 미해결 두 건은 현재 문서에서 충분히 해소됐다.

## Recommendations
1. 현 artifact는 승인한다.
2. 후속 실행 문서에서는 Phase 2.5와 시나리오 D/F의 로그 스키마를 그대로 테스트 프로토콜 템플릿으로 승격하면 된다.
3. 이후 개정에서도 Phase 2.5를 우회하는 문구나 soft stop을 E-stop 대체로 격상시키는 문구는 금지해야 한다.

## Architectural Status
CLEAR

## Code Review Recommendation
APPROVE

## Trade-offs
- 옵션 B + Phase 2.5 재적격화
  - 장점: 취약 사용자 근접 리스크를 낮추면서도 최종 플랫폼 증거 전이를 통제한다.
  - 단점: 동일 안전 계약을 두 플랫폼에서 반복 검증해야 한다.
- 조기 최종 플랫폼 통합
  - 장점: 실제 동역학과 제품 정체성을 더 빨리 검증한다.
  - 단점: 초기 단계부터 균형 실패와 추종 실패가 결합된다.
