# KOSPI Donchian 20 Channel — 매일 자동 신호 시스템

매일 장 마감 후 자동으로 코스피 데이터를 수집하고 돈치안 20일 채널 신호를
계산해서 **텔레그램 푸시 + HTML 대시보드 갱신**을 수행하는 스크립트입니다.

---

## 📂 파일 구성

| 파일 | 용도 |
|---|---|
| `donchian_kospi_daily.py` | 메인 스크립트 |
| `dashboard_template.html` | HTML 대시보드 템플릿 (필수) |
| `run_donchian.bat` | 윈도우 자동 실행용 배치 파일 |
| `requirements.txt` | 필요한 라이브러리 목록 |
| `SETUP_TELEGRAM.md` | 텔레그램 봇 설정 가이드 |
| `SETUP_SCHEDULER.md` | 윈도우 작업 스케줄러 설정 가이드 |

실행 후 자동 생성:
| 파일 | 용도 |
|---|---|
| `output/dashboard.html` | 매일 갱신되는 대시보드 |
| `output/signals_log.csv` | 모든 매매 신호 이력 |
| `output/donchian_daily.log` | 실행 로그 |
| `output/cron.log` | 작업 스케줄러 실행 로그 |

---

## 🚀 빠른 시작 (3단계)

### 1) 라이브러리 설치
```powershell
pip install -r requirements.txt
```

### 2) 텔레그램 설정
`SETUP_TELEGRAM.md` 참고해서 봇 토큰과 chat_id를 환경변수로 등록

### 3) 수동 실행 테스트
```powershell
python donchian_kospi_daily.py
```

콘솔에 신호 출력 + 텔레그램 푸시 + `output/dashboard.html` 생성되면 성공

---

## ⏰ 자동 실행 (선택)

`SETUP_SCHEDULER.md` 참고해서 윈도우 작업 스케줄러에 등록.
매일 오후 4:30에 자동 실행됩니다.

---

## 📊 실제 사용 흐름

```
매일 오후 4:30 — GitHub Actions 자동 실행
   ↓
매도/매수 조건 충족일마다 — 텔레그램 알림 도착
   ↓
알림 확인 — GitHub Pages 대시보드에서 상세 확인
   ↓
다음 거래일 아침 — 관리 종목 매수 또는 매도 주문
```

### 관리 종목

신호는 **코스피 지수(1001) 하나로만** 계산하고, 그 신호를 아래 종목에 적용합니다.
코스피와의 상관이 0.9 이상인 종목만 대상으로 합니다 (`CONFIG.MANAGED`).

| 종목 | 코스피 상관 | 베타 |
|---|---|---|
| RISE 200 (148020) | 0.99 | 1.06 |
| SOL AI반도체TOP2+ (0167A0) | 0.94 | 1.30 |
| TIGER 코리아TOP10 (292150) | 0.94 | 1.08 |

`CONFIG.REFERENCE`는 지수 신호를 적용하지 않고 참고용으로만 표시합니다.

> **알림은 포지션 전환일이 아니라 조건 충족일마다** 나갑니다.
> 돈치안은 상태가 바뀌는 순간에만 신호를 내므로, 매도 신호를 한 번 놓치면
> 그 뒤로는 조건이 계속 참이어도 다시 알려주지 않습니다. 이 구멍을 막기 위한 설계입니다.

---

## ⚙️ 설정 변경

`donchian_kospi_daily.py` 상단 `CONFIG` 딕셔너리에서 조정 가능:

```python
'DC_WINDOW_ENTRY': 20,    # 매수 채널 (40, 60으로 늘리면 거래 감소)
'DC_WINDOW_EXIT':  20,    # 매도 채널 (10으로 줄이면 빠른 청산)
'NOTIFY_ALWAYS':   False, # True면 신호 없는 날도 매일 푸시
```

---

## ⚠️ 주의사항

- **자동 매매가 아닙니다.** 신호 알림만 보내며, 실제 매수/매도 주문은 직접 해야 합니다.
- **신호는 다음 거래일 시가 기준** 매매를 가정합니다 (백테스트 가정과 일치).
- pykrx가 KRX 정책 변경으로 가끔 동작 안 하면, FinanceDataReader로 자동 폴백됩니다.
- 단, FDR도 안 되면 KRX 직접 다운로드를 권장 (data.krx.co.kr).
- 본 스크립트는 **참고 신호 생성용**이며, 투자 결과에 대한 책임은 사용자에게 있습니다.

---

## 🔧 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `KRX 로그인 실패` 오류 | pykrx 최신 버전이 KRX 계정 요구. FDR 폴백 자동 작동 |
| `dashboard_template.html이 없습니다` | 같은 폴더에 두 파일이 함께 있어야 함 |
| 텔레그램 푸시 안 옴 | SETUP_TELEGRAM.md 4단계 확인 / NOTIFY_ALWAYS 옵션 |
| HTML이 깨져 보임 | UTF-8 인코딩 문제. 브라우저에서 인코딩 강제 변경 |
