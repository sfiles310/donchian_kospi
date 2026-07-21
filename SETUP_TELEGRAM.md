# 텔레그램 봇 5분 설정 가이드

매일 코스피 신호를 텔레그램으로 받기 위한 일회성 설정입니다.

---

## 1단계 — 봇 만들기 (1분)

1. 텔레그램에서 **@BotFather** 검색 → 채팅 시작
2. `/newbot` 입력
3. 봇 이름 입력 (예: `코스피 돈치안 봇`)
4. 봇 사용자명 입력 (예: `kospi_donchian_bot` — 마지막에 `bot` 필수)
5. BotFather가 알려주는 **TOKEN 복사**
   ```
   예: 1234567890:AAH1234567890abcdefghijklmnopqrstuvwx 
   ```

<나의 봇>
봇 이름 : 코스피 돈치안 봇
봇 사용자명 : kospi_donchian_bot
Token : (BotFather에서 발급 — 이 파일에 적지 말 것)
Chat ID : (getUpdates로 확인 — 이 파일에 적지 말 것)
---
※ 토큰은 아래처럼 환경변수로만 보관합니다. 문서·스크립트에 직접 적으면
   깃 저장소에 올라가는 순간 봇이 탈취됩니다.

[System.Environment]::SetEnvironmentVariable('TG_BOT_TOKEN', '<발급받은토큰>', 'User')
[System.Environment]::SetEnvironmentVariable('TG_CHAT_ID',   '<본인chat_id>',  'User')



## 2단계 — 본인 chat_id 알아내기 (2분)

1. 방금 만든 봇 이름으로 검색해서 **대화창 열기**
2. 봇에게 `/start` 또는 아무 메시지나 1번 보내기
3. 웹 브라우저에서 아래 URL 접속 (TOKEN 부분만 본인 것으로 교체)
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
4. JSON 결과에서 `"chat":{"id": 숫자` 부분 확인 → **chat_id 복사**
   ```json
   {"message":{"chat":{"id":987654321,"first_name":"...
   ```
   여기서 `987654321`이 chat_id

---

## 3단계 — 환경변수 설정 (윈도우 기준, 2분)

### 방법 A. 환경변수 (권장 — 보안상 안전)

PowerShell을 **관리자 권한**으로 열고:

```powershell
[System.Environment]::SetEnvironmentVariable('TG_BOT_TOKEN', '여기에_토큰', 'User')
[System.Environment]::SetEnvironmentVariable('TG_CHAT_ID',   '여기에_chat_id', 'User')
```

설정 후 **PowerShell 창 다시 열기** (그래야 적용됨)

확인:
```powershell
echo $env:TG_BOT_TOKEN
echo $env:TG_CHAT_ID
```

### 방법 B. 코드에 직접 (간단하지만 비추천)

`donchian_kospi_daily.py` 파일을 열어 CONFIG 부분에 직접 입력:

```python
CONFIG = {
    'TELEGRAM_BOT_TOKEN': '1234567890:AAH...',
    'TELEGRAM_CHAT_ID':   '987654321',
    ...
}
```

⚠️ 이 방법은 GitHub 등에 코드를 올릴 때 토큰이 노출되니 주의

---

## 4단계 — 테스트

스크립트 실행:
```powershell
python donchian_kospi_daily.py
```

실행 후 텔레그램에 알림이 오는지 확인.

신호가 없는 날에는 푸시가 오지 않습니다 (`NOTIFY_ALWAYS=False` 기본값).
매일 무조건 받고 싶다면 코드 CONFIG에서 `NOTIFY_ALWAYS=True`로 변경.

---

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `텔레그램 토큰/chat_id 미설정` 메시지 | 환경변수 미적용. PowerShell 새 창 열고 재시도 |
| 푸시는 안 오는데 콘솔엔 결과 정상 출력 | 신호 변화가 없는 날 → 정상 동작. `NOTIFY_ALWAYS=True`로 매일 받기 가능 |
| `getUpdates` 결과 비어있음 | 봇에게 메시지를 한 번 보내야 함 (2단계의 `/start`) |
| 한국 IP 차단 시 | 거의 없지만 발생하면 환경변수 `HTTPS_PROXY` 설정 필요 |
