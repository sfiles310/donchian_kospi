# KOSPI Donchian 20 Channel — 매일 자동 신호 시스템

매일 장 마감 후 자동으로 코스피 데이터를 수집하고 돈치안 20일 채널 신호를
계산해서 **텔레그램 푸시 + HTML 대시보드 갱신**을 수행하는 스크립트입니다.

---

## 🔁 정기 점검 — 5개월마다 (지수 거래량 이력)

FTD 판정에는 지수 거래량이 필요합니다. 네이버 일봉이 **최근 110거래일(약 5.2개월)**만
주기 때문에, 그보다 오래된 구간은 `data/kospi_index_history.csv`가 채웁니다.
이 파일이 5개월 넘게 낡으면 두 구간 사이에 구멍이 생겨 FTD 판정 구간이 줄어듭니다.

```powershell
python refresh_index_history.py
git add data/kospi_index_history.csv
git commit -m "지수 거래량 이력 갱신"
```

`KIS_APP_KEY`, `KIS_APP_SECRET` 환경변수가 필요합니다. 로컬에서만 실행하며
GitHub Actions는 이 키를 쓰지 않습니다.

> 잊어도 됩니다. 갱신할 때가 되면 **매일 알림 하단에 경고가 붙습니다.**
> 구멍이 실제로 생기기 전에 먼저 알리고, 생긴 뒤에는 어디가 비었는지까지 알려줍니다.

---

## 📂 파일 구성

| 파일 | 용도 |
|---|---|
| `donchian_kospi_daily.py` | 메인 스크립트 |
| `dashboard_template.html` | HTML 대시보드 템플릿 (필수) |
| `kis/` | KIS Open API 수집 계층 (인증·호출·정규화·저장) |
| `collect_kis_flow.py` | KIS 수급 패널 수집기 진입점 |
| `ftd_signal.py` | 오닐 FTD 재진입 판정 (알림·검증 공용) |
| `ftd_page_template.html` | FTD 재진입 점검 화면 템플릿 |
| `data/kospi_index_history.csv` | 지수 거래량 이력. FTD 판정에 필요 |
| `refresh_index_history.py` | 위 이력 갱신 (로컬 전용, 몇 달에 한 번) |
| `foreign_flow_validation.py` | 외국인 수급·환율 비교 검증 스크립트 |
| `foreign_flow_dashboard_template.html` | 외국인 수급 검증 대시보드 템플릿 |
| `run_donchian.bat` | 윈도우 자동 실행용 배치 파일 |
| `requirements.txt` | 필요한 라이브러리 목록 |
| `SETUP_TELEGRAM.md` | 텔레그램 봇 설정 가이드 |
| `SETUP_SCHEDULER.md` | 윈도우 작업 스케줄러 설정 가이드 |

실행 후 자동 생성:
| 파일 | 용도 |
|---|---|
| `output/dashboard.html` | 매일 갱신되는 대시보드 |
| `output/ftd.html` | FTD 재진입 점검 화면. 알림 하단 링크가 가리킨다 |
| `output/foreign_flow_validation.html` | 기존 차트와 분리된 외국인 수급 검증 화면 |
| `output/foreign_flow_*.csv` | 결합 데이터·전략·조건·상관·검색 결과 |
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

같은 날 알림을 한 번 더 확인할 때는 셸에서 한글을 조합하지 말고 전용 옵션을 사용합니다.

```powershell
python donchian_kospi_daily.py --test-notification
```

---

## ⏰ 자동 실행 (선택)

`SETUP_SCHEDULER.md` 참고해서 윈도우 작업 스케줄러에 등록.
매일 오후 4:30에 자동 실행됩니다.

---

## 📊 실제 사용 흐름

```
매일 오후 4:30 — Windows 스케줄러가 확정 종가를 검증
   ↓
오후 4:40 이후 — GitHub Actions가 차트를 먼저 갱신한 뒤 미발송 알림 전송
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

## 외국인 수급 비교 검증

기존 돈치안20 차트와 산출물을 변경하지 않고 별도 화면을 생성합니다.

```powershell
python foreign_flow_validation.py --tickers "005930,000660" --start 2024-01-01
```

기본 검증 항목:

- 외국인 단독, 돈치안20 단독, 돈치안20+외국인 전략 비교
- 다음 거래일 시가 진입 기준 1·3·5·10·20일 이후 수익
- 외국인 순매수 강도, 순매수 지속일, 보유율 변화, 기관 수급, 거래량, 환율 상관
- 외국인 5일 매수강도 분위별 성과
- 외국인 신호를 5일 블록으로 섞은 위약 전략 비교
- 조건부 초과수익의 95% 블록 부트스트랩 신뢰구간
- 왕복비용 1배·1.5배·2배 스트레스
- KOSPI 시가총액 상위 50개 종목의 최근 외국인 매수강도 검색

왕복 거래비용은 실제 증권사 조건에 맞게 bp 단위로 지정합니다.

```powershell
python foreign_flow_validation.py --round-trip-cost-bps 25 --scan-top 100
```

외국인 확정 수급은 `output/foreign_flow_cache`에 저장되어 다음 실행부터 증분
갱신됩니다. 장중 잠정 수급은 검증 데이터에 사용하지 않습니다.

---

## 🔧 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `KRX 로그인 실패` 오류 | pykrx 최신 버전이 KRX 계정 요구. FDR 폴백 자동 작동 |
| `dashboard_template.html이 없습니다` | 같은 폴더에 두 파일이 함께 있어야 함 |
| 텔레그램 푸시 안 옴 | SETUP_TELEGRAM.md 4단계 확인 / NOTIFY_ALWAYS 옵션 |
| HTML이 깨져 보임 | UTF-8 인코딩 문제. 브라우저에서 인코딩 강제 변경 |

---

## KIS 수급 패널 수집

투자 판단에 쓸 정보를 한 저장소로 모읍니다. 외국인 순매수 하나가 아니라, 같은
(날짜, 코드) 키로 7개 데이터셋을 붙입니다.

| 데이터셋 | 키 | 내용 | 공개 시점 |
|---|---|---|---|
| `market_investor_flow_daily` | 지수 | **코스피 지수 자체의** 투자자별 순매수 + 지수 OHLC | 당일 18시 |
| `investor_flow_daily` | 종목 | 투자자 11주체 순매수 수량·대금 (외국인/개인/기관/증권/투신/사모/은행/보험/종금/연기금/기타법인) | 당일 18시 |
| `price_daily` | 종목 | 수정주가 일봉 | 당일 16시 |
| `short_sale_daily` | 종목 | 공매도 체결량·비중 | 당일 18시 |
| `program_trade_daily` | 종목 | 프로그램매매 순매수 | 당일 18시 |
| `credit_balance_daily` | 종목 | 신용 융자·대주 잔고 | 익일 9시 |
| `loan_trans_daily` | 종목 | 대차거래 체결·상환·잔고 | 익일 9시 |

기관을 뭉뚱그리지 않는 것이 요점입니다. 연기금 순매수와 사모펀드 순매수는 성격이
반대라 합치면 정보가 사라집니다.

### 사용법

인증키는 환경변수로만 읽습니다.

```powershell
$env:KIS_APP_KEY = "..."      # 셸 세션에만 두고 파일에 쓰지 않습니다
$env:KIS_APP_SECRET = "..."
```

기본은 **모의 응답 모드**입니다. 실제 호출 없이 정규화·검사·저장 경로를 전부 확인합니다.

```powershell
python collect_kis_flow.py --tickers 005930,000660 --start 2026-06-01
```

실제 호출은 `--live`와 확인 절차를 거칩니다. 조회 API만 쓰며 주문은 보내지 않습니다.

```powershell
python collect_kis_flow.py --live --tickers 005930 --start 2024-01-01 --end 2026-08-05
```

모인 범위 확인:

```powershell
python collect_kis_flow.py --status
```

### 저장 구조

`data/kis_panel.sqlite` 한 파일에 데이터셋별 표로 들어가고, 기본키는 `(date, ticker)`
입니다. 같은 구간을 다시 수집하면 덮어쓰므로 중단 후 재실행이 언제나 안전합니다.

모든 행은 `data_available_at`(그 값을 실제로 볼 수 있었던 시각)을 함께 저장합니다.
확정 전 값은 `is_provisional=1`로 표시되고 기본 조회에서 빠집니다. 미래 정보로 과거를
판단하는 실수를 스키마 차원에서 막기 위한 장치입니다.

```python
from kis import PanelStore, get_dataset

store = PanelStore()
flow = store.read(get_dataset("investor_flow_daily"), tickers=["005930"], start="2024-01-01")
```

### 엔드포인트 추가

`kis/endpoints.py`에 경로·TR ID를, `kis/datasets.py`에 필드 매핑을,
`kis/collector.py`에 기간 조회 방식을 각각 한 줄씩 추가하면 끝입니다. 모의 응답과
저장 스키마는 정의에서 자동으로 따라옵니다.

---

## KRX 전 종목 외국인 수급 가설 검증

현재 시총 상위 종목을 과거로 거슬러 올라가는 방식은 생존편향이 생기므로, 실전 판단에는
그날 실제 상장·거래 가능했던 KOSPI 전 종목 패널을 사용합니다. 입력 형식은
`krx_foreign_flow_panel_template.csv`의 첫 줄과 같습니다.

필수 열:

- `date,ticker,open,close,volume,trading_value,foreign_net_value,market_cap`
- `is_listed,is_tradable`

권장 열:

- `data_available_at`: 해당 수급 수치를 실제로 이용할 수 있게 된 시각. 없으면 당일 확정으로
  가정하지 않고 진입을 T+2 시가로 늦춥니다.
- `sector_code`: 당시 시점의 업종 코드. 있어야 섹터중립 가설 H3를 검사합니다.
- `exit_reason,last_trading_date`: 상폐·합병·이전상장을 단순 -100%로 만들지 않기 위한 기록입니다.

먼저 데이터 없이 상태를 확인할 수 있습니다.

```powershell
python krx_foreign_flow_panel.py
```

공식 KRX 자료를 정규화한 뒤 연구/검증/최종 홀드아웃 경계를 실행 전에 고정합니다.

과거 패널 수집기는 pykrx에 기대고 있어 제거했습니다. 입력은 공식 KRX 자료를 직접
정규화하거나, `collect_kis_flow.py`로 모은 `data/kis_panel.sqlite`에서 만들어 씁니다.

```powershell
python krx_foreign_flow_panel.py `
  --input data/krx_kospi_panel.csv `
  --data-kind krx-real `
  --research-end 2021-12-30 `
  --validation-end 2023-12-28 `
  --max-allowed-mdd 0.10
```

판정은 화면과 `output/krx_foreign_flow_panel_summary.json`에서 먼저 확인합니다.

- `PASS`: 2배 비용에서 절대수익·시장 대비 우위·수급 신호 우위·MDD 한도·최종 홀드아웃을 모두 통과
- `FAIL`: 데이터는 충분하지만 사전 기준을 통과하지 못함
- `NOT_READY`: 실데이터·표본·업종·실제 청산가격 경로 등이 부족해 판정하면 안 됨

실데이터가 아니거나 출처가 확인되지 않은 입력은 성과 수치를 생성하지 않습니다. 이 도구는
주문을 전송하지 않으며, PASS도 수익을 보장하지 않습니다.
