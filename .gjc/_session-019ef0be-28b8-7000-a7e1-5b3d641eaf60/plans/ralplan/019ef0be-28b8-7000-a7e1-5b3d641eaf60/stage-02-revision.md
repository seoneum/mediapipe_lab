# 개정 RALPLAN 합의 계획: RealSense D435 + moteus 기반 사람추종 2륜 휠레그드 보조 로봇

## Summary
이 문서는 취약 사용자 가까이에서 운용될 수 있는 사람추종 보조 로봇 개념의 **개정판 초기 합의 계획**이다. 구현이 아니라 안전 경계, 제품 스택, 학습 순서, 단계별 실행 게이트, 취약 사용자 파일럿 경계, 향후 Obsidian 정리 구조를 고정한다. 권고 방향은 계속해서 **노인 우선, 영아 직접 추종 금지**다. 또한 이전 계획의 옵션 B를 최종 제품 아키텍처가 아니라 **안전 검증용 safety mule**로 명시하고, Phase 2 종료 시점에 진짜 휠레그드 제품으로 계속 갈지 비자가균형 보조 운반 플랫폼으로 피벗할지 go/no-go 게이트를 추가한다.

## 1. 문제 정의와 명시적 범위
### 문제 정의
목표는 실내에서 지정된 보호 대상자 또는 보호자의 승인을 받은 사용자를 저속으로 따라가며, 물건 동반 이동, 호출 응답, 생활 동선 동반, 비진단적 패턴 요약을 제공할 수 있는 보조 로봇의 **안전중심 초기 계획**을 수립하는 것이다. 본 문서는 사람추종 지능 자체보다 **움직여도 되는 조건, 멈춰야 하는 조건, 다시 움직여도 되는 조건**을 먼저 정의한다.

### 명시적 범위
- RealSense D435 기반 근거리 사람추종 전략과 제한 사항 정리
- BLDC + FOC + moteus 기반 구동계 후보 선정
- 휠레그드 또는 비자가균형 보조 운반 플랫폼으로 이어질 수 있는 단계 게이트 설계
- 센서 책임 분할, motion authority matrix, 전원-정지 체인 정의
- 취약 사용자 대상 활용 시나리오와 금지 범위 구분
- Phase 0~4 순차 개발 및 검증 로드맵 작성
- 향후 Obsidian 저장용 노트 경로와 개요 제안

### 제외 범위
- 제품 소스 구현, 펌웨어 작성, 제어기 튜닝 실행
- 실제 Obsidian 파일 작성
- 의료 진단, 낙상 판정 확정, 영유아 직접 접촉 보조 자동화
- 무인 야외 주행, 계단 주행, 문개폐 자동화, 비승인 추종

## In scope / Out of scope
### In scope
- 안전중요 보조 로봇의 초기 아키텍처 옵션 비교
- 제품 스택 권고와 선구매 대 지연구매 구분
- 실행 가능한 phase gate와 pass/fail 기준
- 노인 우선, 영아 제한 시나리오 정리
- 취약 사용자 파일럿 포함/제외 경계

### Out of scope
- 상세 기구 설계 도면
- 법규 인증 패키지 작성
- 양산 BOM 확정
- 개인정보 정책 최종안

## 2. RALPLAN-DR 요약
### Principles
1. 안전 경계가 성능보다 우선이다.
2. 사람추종은 저속 실내 보조로 제한하고, 접촉 없는 거리 유지가 기본이다.
3. 센서 하나의 낙관적 해석에 의존하지 않는다.
4. 초기 단계는 화려한 AI보다 관측 가능성, 정지 가능성, 수동 개입성을 우선한다.
5. 취약 사용자 대상 기능은 반드시 보호자 또는 운영자 감독 흐름 안에 둔다.

### Decision Drivers 상위 3
1. 취약 사용자 근접 운용에서의 안전성과 실패 시 피해 크기
2. 부품 공급 가능성, 문서 성숙도, 재현 가능한 실험성
3. 사람추종 가치 검증과 하부 동역학 검증을 단계적으로 분리할 수 있는지 여부

### Viable Options
#### 옵션 A: D435 + 단일 보조 컴퓨터 + moteus r4.11 + 조기 자가균형 휠레그드 프로토타입
- 구성: Linux 기반 런타임 컴퓨터, D435 depth, moteus r4.11, IMU/엔코더 융합, 초기부터 자가균형까지 포함
- 장점:
  - 진짜 휠레그드 정체성을 빨리 검증한다.
  - 상위 perception과 하위 제어를 같은 아키텍처에서 조기 통합할 수 있다.
- 단점:
  - 균형 실패와 추종 실패가 결합되어 취약 사용자 근접 리스크가 급증한다.
  - Phase 1~2에서 안전 케이스를 분리 검증하기 어렵다.
  - 실험 중단 조건이 늦게 드러날 수 있다.

#### 옵션 B: D435 + 전방 저위치 대응 2D 안전 라이다 + moteus r4.11 + 초기 비자가균형 저속 베이스 **(safety mule)**
- 구성: D435는 사람 인지와 거리 유지, 2D 안전 라이다는 전방 저위치 장애물/발/발판 검출, moteus r4.11은 구동 제어, 초기 섀시는 저중심 비자가균형 또는 보조 안정 구조
- 장점:
  - 센서 책임 분리가 가능하고 single-fault degraded mode를 설계하기 쉽다.
  - 사람추종 가치 검증과 휠레그드 동역학 검증을 분리할 수 있다.
  - 취약 사용자 대상 시연에서 안전 설명과 정지 계약이 명확하다.
- 단점:
  - 최종 휠레그드 제품 아키텍처가 아니다.
  - 부품 수, 배선, 통합 복잡도가 증가한다.
  - 휠레그드 차별성 검증은 Phase 2 go/no-go 이후로 미뤄진다.

#### 옵션 C: UWB 태그 또는 리모컨 중심 추종 + 비전은 보조
- 구성: 핵심 추종은 태그 기반, D435는 장애물 회피와 사용자 확인 보조
- 장점:
  - 가림과 조명 변화에 덜 민감하다.
  - 안전 반경 제어가 단순하다.
- 단점:
  - 사용자 태그 휴대 부담이 생긴다.
  - 무태그 자연스러운 추종 목표와 거리가 있다.
  - 취약 사용자 채택성이 낮아질 수 있다.

### 권고 선택
현재는 **옵션 B를 safety mule로 권고**한다.
- 이유:
  - 취약 사용자 가까이에서 바로 최종 제품 정체성을 밀어붙이는 것보다, 안전 인터페이스를 고정하는 실험용 mule이 더 타당하다.
  - D435 단독이 아니라 D435 + 안전 라이다의 책임 분할로 failure containment가 가능하다.
  - Phase 2까지 하부 안전 주행, 정지 체인, brownout 동작, 감독자 개입성을 검증한 뒤에만 휠레그드 고유 난제로 올라간다.

### 대안이 지금 덜 적합한 이유
- 옵션 A 비선호 이유: 기술적으로 매력적이지만 취약 사용자 근접 환경에서 균형 실패와 추종 실패가 결합된다.
- 옵션 C 비선호 이유: 안전성은 높일 수 있으나 자연스러운 assistive following 경험과 제품 방향성이 약해진다.

## 3. 권장 제품 스택
### Compute platform
- 런타임 메인: Ubuntu 기반 미니 PC 또는 NVIDIA Jetson Orin Nano 급
- 개발 보조: 현재 macOS Apple Silicon 저장소는 알고리즘 탐색, 로그 분석, 문서 정리에 사용
- 이유: D435는 macOS에서 컴파일 가능해도 검증 플랫폼이 아니므로 실제 로봇 런타임은 Linux 우선으로 고정

### Depth camera
- 권장: Intel RealSense D435 1대
- 역할: 사람 인지, 근거리 거리 유지, 대상 재획득 보조
- 한계: 유리/반사체/검은 저위치 장애물/근접 가림/좁은 FOV

### Motor controllers
- 권장: mjbots moteus r4.11
- 이유: 10~44V, CAN FD, 비교적 높은 전류 처리, 공급/문서 가시성

### Actuators / motors / gear / encoders strategy
- 바퀴 구동: 저 KV BLDC + 감속기 + 절대 또는 고해상도 증분 엔코더
- 초기 자세 또는 다리 축: 최소 자유도만 두고 나중에 확장
- 직접구동보다 저속 고토크 감속형 우선
- 이유: 취약 사용자 근접에서는 예측 가능한 감속 응답과 정지 재현성이 우선

### Base locomotion sensors and safety hardware
- 필수: IMU, 휠 엔코더, 드라이브/모터 온도, 범퍼 또는 접촉 스위치, 하드웨어 E-stop, 소프트웨어 watchdog, contactor, 상태등, 부저
- 추가 안전 센서 최소안: **2D 라이다를 Phase 1 선구매 대상으로 고정**
- 초음파는 대체재가 아니라 보완재로만 검토하며, Phase 2 종료 시 반사/투명 물체 실패가 반복되면 추가 여부 결정

### Battery / power / docking
- 권장 전압: 24V 또는 36V 계열 우선 검토
- 충전 중 구동 금지 인터락 필수
- 물리 E-stop 2개 이상: 사용자 접근 위치 1개, 운영자 후면 1개
- 초기는 자동 도킹보다 수동 충전 우선

### Must buy now
- D435
- moteus r4.11 최소 2개
- 시험용 저속 BLDC + 감속기 + 엔코더 세트
- IMU, 휠 엔코더, E-stop, contactor, 퓨즈, DC 전원 계측 장비
- Linux 런타임 컴퓨트 플랫폼
- CAN FD 인터페이스
- **2D 안전 라이다 1대**
- 벤치 전원공급기와 절연 보호 장비

### Defer until prototype valid
- 자동 도킹 하드웨어
- 추가 상부 카메라
- 다자유도 다리 액추에이터 확장
- 대용량 배터리 팩 확대
- 외장 하우징 고도화
- 초음파 또는 추가 근접 센서 보강

## File-level changes
- 현재 저장소 제품 소스 변경 없음
- 현재 저장소 내 RealSense, moteus, 로봇 통합 코드 없음이 확인됨
- README.md에서 Obsidian 노트 관례 확인
- 실제 Obsidian 파일 생성은 이번 범위에 포함하지 않음

## 4. Motion authority matrix
| 구성요소 | Move permit 권한 | Move veto 권한 | Restart authority | 비고 |
|---|---|---|---|---|
| Perception(D435 추종 스택) | 제한적 있음. 대상 식별 confidence와 거리 조건을 만족할 때만 상위 추종 명령 제안 가능 | 있음. 대상 상실, confidence 저하, depth 품질 저하 시 정지 요청 | 없음 | 단독으로 재시작 불가 |
| Supervisor(운영자/보호자) | 있음. 세션 시작 승인, 대상 선택 승인, 추종 재개 승인 | 있음. 언제든 정지 명령 가능 | 있음. 단 E-stop/contactor fault 해제 후에만 | 취약 사용자 세션 필수 |
| Software watchdog | 없음 | 있음. heartbeat 상실, 지연 초과 시 즉시 정지 | 없음 | 자동 복귀 금지 |
| Hardware E-stop | 없음 | 절대적 있음. 즉시 motion kill | 없음 | 물리 리셋 후에도 supervisor 재승인 필요 |
| Contactor/BMS fault chain | 없음 | 절대적 있음. 전력 차단 또는 inhibit | 없음 | fault clear와 전원 건전성 확인 필요 |
| Low-level controller(moteus + 하위 제어) | 제한적 있음. 상위 허가 범위 내 저수준 실행 | 있음. 엔코더 불일치, 과전류, 과열, 제어 불안정 시 정지 | 제한적 없음. fault clear 후 supervisor 승인 필요 | 활성 제동/토크오프 정책 집행 주체 |

### Safe state 정의
- 기본 safe state는 **능동 감속 후 정지(active braking to stop)** 이다.
- 제어 불안정, 드라이브 fault, contactor 개방이 필요한 경우에만 **torque-off**로 전환한다.
- safe state 진입 후에는 supervisor가 명시적으로 재시작 승인하기 전까지 자동 재출발 금지.

## 5. 센서 최소 커버리지 / 책임 표
### 프로토타입 기본 선택
- **기본 선택: D435 + 전방 하단 장착 2D 안전 라이다**
- 초음파는 기본 선택이 아니라 **Phase 2 decision gate 항목**이다.

| 위험/책임 영역 | 최소 커버리지 책임 | 1차 센서 | 2차/보조 | degraded mode 규칙 |
|---|---|---|---|---|
| 전방 저위치 장애물/발/발판 | 바닥에서 낮은 높이의 발, 발판, 보행기 하단 감지 | 2D 안전 라이다 | 범퍼 스위치 | 라이다 fault 시 추종 금지, 수동 저속 이동만 허용 |
| 사용자와 최소 이격 거리 유지 | 추종 대상과 0.8m 이상 간격 유지 | D435 | 휠 속도/IMU 추정 | D435 confidence 임계 미만이면 즉시 정지 |
| 측면 근접 | 측면 0.4m 이내 접근 방지 | 범퍼 + 운영 절차 | 필요 시 Phase 2 이후 초음파 검토 | 측면 커버리지 부재 상태에서는 좁은 통로 파일럿 금지 |
| 반사/투명 물체 | 유리, 반사체, 광택 가구 근처 감속/정지 | 2D 라이다 우선, 실패 시 운영자 개입 | 범퍼 | 반복 실패 1회라도 안전 회피 실패 시 release blocker |
| 가림/occlusion | 대상 상실 또는 타인 가림 시 오추종 방지 | D435 재식별 confidence | supervisor 승인 | confidence 미달 300ms 초과 시 정지 |
| single-fault degraded mode | 센서 하나 고장 시 안전 상태 유지 | watchdog + 하위 제어 | supervisor | D435 fault: 추종 금지. 라이다 fault: 사람 없는 수동 이동만 허용. 둘 다 fault: contactor inhibit 또는 정지 유지 |

### 초음파 추가 decision gate
- Phase 2 종료 시 다음 중 하나라도 발생하면 초음파 추가 검토를 **의무화**:
  1. 투명/반사체 근접 실패가 2회 이상 재현됨
  2. 측면 근접 경고 부재로 supervisor 개입이 세션당 1회 초과
  3. 좁은 통로에서 측면 clearance 불확실성 때문에 실험 금지 구간이 과도하게 커짐

## 6. 전원-정지 체인 상세
### 정지 방식 우선순위
1. 정상 정지: low-level controller가 **active braking**으로 감속 후 정지
2. 비정상 정지: 제어 불안정/과전류/통신상실 시 active braking 시도 후 실패하면 torque-off
3. 절대 차단: E-stop 또는 BMS severe fault 시 contactor 개방

### Brownout threshold / behavior
- 경고 임계: 정격 저전압 경고 구간 진입 시 속도 상한을 즉시 0.3 m/s 이하로 제한
- 차단 임계: 구동 유지가 불안정해지는 저전압 구간 진입 시 추종 중지, 능동 감속 정지, restart inhibit
- brownout 중에는 자동 추종 재개 금지, supervisor 현장 확인 후에만 재시작 가능

### BMS fault behavior
- BMS 경고: 신규 추종 시작 금지, 현재 세션 종료 후 수동 복귀만 허용
- BMS 심각 fault: contactor 개방, motion kill, 이벤트 로그 필수

### Restart inhibit conditions
- E-stop 해제 직후
- BMS severe fault 해제 직후
- watchdog timeout 이후 최초 복귀 시
- perception confidence threshold 미회복 상태
- 엔코더/IMU 불일치 해소 전
- 전복 가능 자세 또는 anti-roll 미확인 상태

### Manual push conditions
- 전원 차단 후 구동 토크 0 확인
- 기울기 없는 평지에서만 허용
- supervisor 1명 이상 동반
- 취약 사용자 근접 구간에서는 수동 밀기 금지

### Anti-roll strategy
- 기본: 저중심 섀시 + 정지 시 미끄럼/구름 방지 구조
- 경사면 실험은 초기 전 단계에서 금지
- 수동 정지 후에도 바퀴 자유구름으로 굴러가면 Phase 2 release blocker

## 7. 학습 로드맵
### Phase 0: Desk study
- FOC 기초: dq 변환, 전류 루프, 속도 루프, 토크 상수, back EMF, 전류 제한
- BLDC 실무: 모터 상수 읽기, 발열, 감속기 백래시, 엔코더 정렬
- D435 실무: 깊이 노이즈, 반사체 실패 모드, 캘리브레이션
- HRI: 감독자 개입 설계, 비진단 원칙, 취약 사용자 설명 책임
- 산출물: 위험 목록 초안, 부품 shortlist, 측정 지표 목록

### Phase 1: Benchtop actuator + FOC
- 1축 또는 2축 벤치 리그 구성
- moteus 전류 제한, 속도 제한, 열 관측, watchdog 검증
- 엔코더 정렬 및 저속 추종 오차 측정
- E-stop, contactor, brownout, restart inhibit 절차 검증

### Phase 2: Mobile base safety mule
- 저중심 비자가균형 또는 보조 안정 구조 베이스
- IMU + 엔코더 융합, 저속 주행, 제동 거리 측정
- D435 없이도 안전 정지와 원격 수동 제어 성립 확인
- D435 + 라이다 결합 전까지는 사람 추종 금지

### Phase 2 go / no-go gate
다음 조건을 모두 만족하면 **진짜 휠레그드 제품 경로로 go**:
1. 안전 mule이 반복 실험에서 제동, brownout, E-stop, 라이다 fault, D435 fault에 대해 모두 deterministic stop을 보임
2. 속도/이격/정지거리 목표를 충족함
3. 사용자 가치 가설이 노인 우선 시나리오에서 유지됨
4. 전력/발열/중량 예산이 휠레그드 확장 가능 범위 안에 있음

다음 중 하나라도 해당하면 **비자가균형 assistive carrier로 pivot 또는 프로그램 중단**:
1. 안전 정지 체인이 반복적으로 흔들림
2. 자가균형 추가 시 낙상 위험을 수용 가능 수준으로 낮출 근거가 없음
3. 센서 degraded mode가 현장 운영을 지나치게 제한함
4. 사용자 가치가 단순 운반 플랫폼만으로도 충분히 충족됨

### Phase 3: Person following
- D435 기반 대상 추적, 거리 유지, 재식별 실패 시 정지
- 라이다와 충돌 방지 결합
- supervisor 승인 없는 대상 전환 금지
- 측면 접근, 후방 접근, 좁은 통로 자동 진입 금지

### Phase 4: Assistive pilot
- 노인 우선 파일럿: 물건 동반 이동, 호출 응답, 생활 동선 동반
- 보호자/운영자 감독 하 짧은 세션
- 영아 관련은 직접 추종이 아니라 보호자 보조 시나리오만 제한 검토

## Sequencing and dependencies
1. 안전 요구와 비목표 확정
2. Linux 런타임 고정
3. moteus + BLDC 벤치 리그로 FOC와 정지 체인 검증
4. safety mule 모바일 베이스 검증
5. Phase 2 go/no-go 판단
6. 성인 감독자 대상 사람추종 검증
7. 노인 우선 보조 시나리오 파일럿
8. 영아 관련은 보호자 지원 범위만 별도 심사

## 8. 취약 사용자 적용 분석
### 노인 우선 사용 사례
1. 실내 저속 동행: 물건 운반 동행, 호출 응답, 지정 위치 따라오기
2. 생활 지원 보조: 물, 리모컨, 약 상자 운반 보조. 단 복약 판단은 하지 않음
3. 패턴 알림: 이동 감소, 호출 후 응답 지연, 특정 구역 반복 같은 비진단 요약
4. 원격 케어 보조: 보호자가 로봇 위치와 마지막 상태를 확인

### 왜 노인 우선인가
- 과업 정의가 비교적 명확하다.
- 직접 접촉 없이도 가치가 나온다.
- 실패 시 즉시 정지와 supervisor 개입으로 완화 가능하다.
- 영아는 급접근, 바닥 자세 변화, 접촉 위험 때문에 같은 추종 개념을 적용하기 어렵다.

### 영아 및 아기 관련 명시적 안전 한계
#### 제한적으로 가능한 범위
- 보호자 동반 환경에서 방 안 반대편까지 물건 운반
- 주변 환경 관찰 보조 메모 초안 생성
- 보호자 호출 시 시야 공유 보조

#### 금지 또는 강한 제한
- 아기를 직접 따라다니는 자동 추종
- 침대, 유아 매트, 좁은 놀이 공간 근접 주행
- 울음, 질식, 건강 상태를 진단처럼 판단하는 알림
- 보호자 부재 상황에서 자동 감시 대체

### 파일럿 포함 / 제외 기준
#### 포함 기준
- 보행 보조기기 없이 실내 단거리 이동이 가능한 노인 또는 보호자 동반 사용 환경
- supervisor 1명 이상이 즉시 개입 가능한 세션
- 반려동물과 바닥 clutter가 통제된 공간

#### 제외 기준
- 보행기/지팡이와 로봇이 동일한 좁은 통로를 동시에 지나야 하는 환경
- 반려동물이 자유롭게 이동하는 환경
- 바닥 clutter, 전선, 반사체가 많은 환경
- 사용자가 로봇을 지지물처럼 잡아당길 가능성이 높은 환경
- 급성 인지 혼란으로 stop 지시나 supervisor 지시 이해가 어려운 상태

### Cognitive impairment / misuse 고려
- 경도 인지저하 사용자는 보호자 동반 하에서만 검토
- 로봇을 지지대처럼 잡는 행위는 **명시적 misuse**로 간주하고 해당 세션 종료
- supervisor 비율: **취약 사용자 1명당 supervisor 최소 1명**
- 보행기, 지팡이, 반려동물, clutter는 기본 파일럿에서 제외 후 별도 안전 케이스 필요

### AI가 현실적으로 할 수 있는 것
- 사람 재식별 보조와 거리 유지 상태 추정
- 이동 패턴, 호출 빈도, 구역 체류 같은 비진단 요약
- 위험 이벤트 후보 로그 분류와 검토 우선순위 제안
- 관찰 기록 초안 작성과 보호자용 요약 생성

### AI가 해서는 안 되는 것
- 질병, 인지저하, 통증, 낙상 여부를 단정하는 판단
- 영아 상태를 의료적으로 해석하는 알림
- supervisor 승인 없이 추종 대상 자율 변경
- 불확실한 상황에서 이동 지속 선택

## 9. Deliberate mode
### 사전 실패 분석: 정확히 3개 시나리오
1. **가림 후 오추종 시나리오**
   - 복도에서 대상 사용자가 코너를 돌고 다른 사람이 시야에 들어오자 로봇이 잘못된 사람을 따라감
   - 결과: 사생활 침해, 충돌, 보호자 신뢰 상실
   - 방지: confidence 임계 미만 300ms 초과 시 즉시 정지, supervisor 재승인 전 재출발 금지

2. **균형 또는 제동 실패 시나리오**
   - 배터리 전압 저하나 제어 불안정으로 급정지 시 차체가 앞쪽으로 기울며 사용자 다리에 접촉
   - 결과: 낙상 유발 가능성
   - 방지: safety mule 단계에서 비자가균형 유지, brownout 시 추종 중지, active braking 우선, anti-roll 구조 필수

3. **센서 맹점 접근 시나리오**
   - 검은색 낮은 발판 또는 반사체를 센서가 안정적으로 보지 못해 근접 충돌
   - 결과: 발 끼임, 물건 파손, 사용자 놀람
   - 방지: 라이다 책임 고정, 반복 실패 시 release blocker, 필요 시 초음파 추가 gate 발동

## 10. 확장 검증 및 시험 계획
### 공통 수치 계약
- 사람 근접 주행 속도 상한: **0.4 m/s 이하**
- 수동 복귀 저속 한계: **0.2 m/s 이하**
- 최소 이격 거리 목표: **0.8 m 이상 유지**, 0.6 m 미만 진입 금지
- 사람 근접 정지거리 목표: **명령 후 0.35 m 이내**
- perception confidence threshold: **0.85 미만이 300ms 지속되면 정지**
- supervisor intervention latency 목표: **1초 이내 정지 개시**
- brownout 경고 응답: **2초 이내 속도 상한 0.3 m/s 이하로 축소**, 차단 임계 진입 시 추종 중지

### Phase gate acceptance criteria
#### Phase 0 pass/fail
- 통과:
  - 부품 shortlist, 위험 목록, 센서 책임 표, motion authority matrix가 완성됨
  - D435, moteus, Linux 런타임, safety mule 개념에 대한 비목표가 명시됨
- 실패:
  - 센서 대체재가 열려 있거나 restart authority가 불명확함

#### Phase 1 pass/fail
- 통과:
  - 벤치 리그에서 50회 반복 정지 중 50회 모두 명령 후 정지 체인이 동작
  - brownout 주입 시 2초 이내 속도 축소, 차단 임계 시 추종 금지 전이 확인
  - E-stop 후 자동 재시작 0회, supervisor 재승인 없이는 구동 재개 불가
- 실패:
  - 1회라도 uncontrolled coast 또는 restart inhibit 누락 발생

#### Phase 2 pass/fail
- 통과:
  - safety mule이 사람 없는 공간에서 속도 0.4 m/s 이하로 20회 반복 주행 성공
  - 장애물 앞 정지거리 0.35 m 이내 달성
  - 라이다 fault, D435 fault, watchdog timeout 각각에서 deterministic stop 100% 달성
  - 수동 밀기 조건과 anti-roll 확인 완료
- 실패:
  - 라이다 fault 시 계속 자율 주행하거나, 정지 후 재시작 권한이 모호함

#### Phase 3 pass/fail
- 통과:
  - 성인 감독자 대상 추종 세션 20회 중 20회에서 최소 이격 0.8 m 유지
  - confidence <0.85가 300ms 지속될 때 100% 정지
  - supervisor stop 요청 후 1초 이내 정지 개시
  - 오추종 0건
- 실패:
  - 이격 0.6 m 미만 진입 1회라도 발생
  - supervisor 개입보다 perception 재획득이 우선하는 동작이 발생

#### Phase 4 pass/fail
- 통과:
  - 노인 우선 파일럿 10세션에서 grabbing misuse 0건, 직접 접촉 유발 0건
  - supervisor 1:1 비율 유지
  - 반려동물/좁은 통로/보행기 혼재 환경을 파일럿 범위에서 제외함
- 실패:
  - 사용자나 보호자가 로봇을 지지대로 사용하려 하거나, exclusion 기준을 무시한 세션이 1회라도 실행됨

### 시나리오별 concrete verification
#### 시나리오 A: 가림 후 재식별 실패
- Setup: 복도형 환경, 대상자 1명 + 비대상자 1명, 코너 가림 발생
- Fault injection: 대상자가 코너 뒤로 사라진 직후 비대상자가 전방 시야 진입
- Expected stop behavior: confidence 0.85 미만 300ms 지속 시 정지, 자동 재추종 금지
- Required logs: confidence 시계열, target ID 전환 후보, 속도 명령, 정지 트리거, supervisor 승인 이벤트
- Release blocker: 오추종 1회라도 발생하면 Phase 3 차단

#### 시나리오 B: brownout / 저전압
- Setup: 벤치 리그 또는 사람 없는 mule 주행
- Fault injection: 전압 강하 조건 주입
- Expected stop behavior: 경고 구간에서 2초 이내 0.3 m/s 이하 제한, 차단 구간에서 추종 종료 후 active braking 정지, restart inhibit 유지
- Required logs: 배터리 전압, 전류, 속도 상한 전이, stop state, restart inhibit flag
- Release blocker: uncontrolled coast 또는 자동 재가동 발생 시 Phase 1~2 차단

#### 시나리오 C: 라이다 fault
- Setup: mule 주행 중 전방 저위치 장애물 배치
- Fault injection: 라이다 데이터 스트림 중단 또는 유효범위 이상치 주입
- Expected stop behavior: 자율 추종 즉시 금지, active braking 정지, 이후 supervisor 수동 저속 이동만 허용
- Required logs: 라이다 health, watchdog 상태, 주행 모드 전환, 정지 명령 시점
- Release blocker: 라이다 fault 상태에서 자율 주행 지속 시 Phase 2 차단

#### 시나리오 D: E-stop
- Setup: 사람 없는 공간, 저속 주행
- Fault injection: 전면 또는 후면 E-stop 수동 작동
- Expected stop behavior: 즉시 motion kill, 이후 supervisor와 fault clear 전까지 재시작 금지
- Required logs: E-stop 입력, contactor 상태, 토크 상태, 재시작 시도 로그
- Release blocker: E-stop 후 자동 복귀 1회라도 발생 시 즉시 차단

#### 시나리오 E: grabbing misuse
- Setup: 파일럿 모의 세션, supervisor 동반
- Fault injection: 사용자가 손잡이 아닌 차체를 지지대처럼 잡으려 함
- Expected stop behavior: 즉시 정지, 세션 종료, misuse 기록
- Required logs: 접촉 스위치/범퍼 이벤트, supervisor stop, 세션 종료 코드
- Release blocker: grabbing 발생 후 이동 지속 시 Phase 4 차단

## 11. ADR stub
### Decision
취약 사용자 대상 초기 사람추종 보조 로봇은 **노인 우선**, **옵션 B safety mule**, **D435 + 전방 저위치 대응 2D 안전 라이다 + moteus r4.11 + Linux 런타임** 조합으로 단계 검증한다.

### Drivers
- 안전 우선과 deterministic stop
- 공급 가능성과 문서 성숙도
- 추종 지능과 하부 안정화의 단계 분리

### Alternatives considered
- 옵션 A: D435 단독 중심 조기 자가균형 휠레그드
- 옵션 C: UWB 또는 태그 중심 추종

### Why chosen
- safety mule 접근이 가장 보수적이며 취약 사용자 근접 실패 비용을 줄인다.
- 휠레그드 정체성은 Phase 2 go/no-go 이후로 판단해도 된다.

### Consequences
- 초기 BOM 증가
- 최종 휠레그드 제품 아키텍처 확정은 늦어진다.
- 대신 전원-정지 체인, 센서 책임, restart contract를 먼저 고정할 수 있다.

### Follow ups
- D435 Linux 런타임 실증
- moteus 벤치 리그 부품 확정
- safety mule 센서 배치 상세화
- 취약 사용자 로그 보관 원칙 정의

## 12. Obsidian 저장용 노트 권고
### 제안 경로와 제목
- 경로: `/Users/seoneum/Library/Mobile Documents/iCloud~md~obsidian/Documents/every_thing/💡 Notes/Reference/Engineering/`
- 제목 제안: `N - 사람추종 휠레그드 보조로봇 안전중심 초기 설계 계획 - D435 moteus 기준.md`

### 노트 개요 제안
1. 목적과 비목표
2. 핵심 결정 요약
3. safety mule와 최종 제품 경계
4. 부품 스택과 선구매 목록
5. motion authority matrix
6. 센서 책임 표
7. 전원-정지 체인
8. Phase 0~4 게이트
9. 노인 우선 파일럿 경계
10. 영아 관련 금지 범위
11. 사전 실패 시나리오 3개
12. 검증 로그 템플릿
13. ADR 요약

## Verification
- README.md에서 Obsidian 노트 관례 확인
- 저장소 내 RealSense, moteus, 로봇 통합 코드 부재 확인
- 기존 planner, architect, critic 아티팩트를 반영해 safety mule, motion authority, 센서 책임, 전원-정지 체인, phase gate 기준을 개정함

## Risks and mitigations
- 위험: safety mule이 최종 제품으로 오인될 수 있음
  - 완화: 옵션 B를 명시적으로 mule로 표기하고 Phase 2 go/no-go 추가
- 위험: 센서 책임이 다시 모호해질 수 있음
  - 완화: D435 + 2D 라이다 기본 선택 고정, 초음파는 decision gate로만 취급
- 위험: 취약 사용자 파일럿이 범위를 넘을 수 있음
  - 완화: inclusion/exclusion, supervisor 1:1, grabbing misuse 종료 규칙 고정
