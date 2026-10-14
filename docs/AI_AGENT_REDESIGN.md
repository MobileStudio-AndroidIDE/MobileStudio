# MobileStudio AI — 에이전트 시스템 재설계 문서

이 문서는 실제 저장소 조사 결과를 근거로 작성되었습니다. 추측한 내용과
실제로 확인한 내용을 구분하며, 조사 시점은 2026-10-13 (master, HEAD `d7d38c2`,
작업 트리에 기존 사용자 변경 사항 다수 존재) 입니다.

---

## 0. 조사 방법 / 검증 환경 (실측)

- 빌드/컴파일: `./gradlew :app:compileDebugKotlin --offline` → **성공 (exit 0)**
- 단위 테스트: `./gradlew :app:testDebugUnitTest --offline` → **성공 (exit 0)**
- JDK: OpenJDK 21, Android SDK: `/home/limsemin/Android/Sdk` (`local.properties`)
- minSdk 26 / targetSdk 34 / compileSdk 34, NDK `arm64-v8a`, Compose BOM 2024.09.00
- 기존 작업 트리는 이미 수정된 파일이 많으므로(예: `AiSettings.kt`,
  `OcEngine.kt`, `strings.xml`, `i18n/`) **되돌리거나 덮어쓰지 않는다.**

---

## 1. Android 프로젝트 구조 (실측)

- 단일 모듈: `:app` (`settings.gradle.kts`, `rootProject.name = "MobileStudio"`).
- 소스 루트: `app/src/main/java/com/aide/mobilestudio/` (수백 개 파일, View 기반 UI 중심).
- Compose 의존성은 존재하나 AI 패널은 View 기반으로 구현되어 있음.
- 시작 Activity: `MainActivity` (`AndroidManifest` 기준, `currentProjectDir` 보유).
- 아키텍처: Activity 중심 + 다수 `object` 싱글턴 + `AnalysisEngine`, `BuildScheduler` 등.
- UI 문자열: `res/values/strings.xml` + `AppLanguage`(assets/i18n/*.json, 30여 개 언어).

## 2. 기존 AI 시스템 (실측)

AI 관련 코드는 **두 개의 평행한 시스템**으로 존재한다.

### (A) 네이티브 에이전트 (`ai/opencode/*` + `ai/agent/*`, `ai/api/*`) — 주력

- 엔진: `ai/opencode/agent/OcEngine.kt` (약 987줄).
  - `EngineBridge` 구현. 세션/스냅샷/지침/스킬/커맨드/프로바이더/MCP를 조합.
  - 에이전트 루프: `runLoop()` → 스트리밍 추론 → 툴 호출 → 결과 저장 → 반복(기본 80 스텝).
  - 서브에이전트: `spawnSubagent()` / `runSubagent()` (깊이 제한 `config.subagentDepth`).
  - 압축(compaction), 타이틀 자동 생성, 플랜 모드(`plan` 에이전트) 지원.
  - 권한: `ai/opencode/permission/Permission.kt` (wildcard, deny/ask/allow).
  - 사후 검증: `handlePostWriteAnalysis()` — `MainActivity.triggerLiveDiagnosticsAndWait()`
    후 `getErrorFingerprints()` 비교. (기존 구현 재사용 대상)
- 세션: `ai/opencode/session/OcSessionStore.kt`, `OcSnapshot.kt`(revert),
  `OcCompaction.kt`, `OcMessage.kt`, `OcInstructions.kt`.
- 툴: `ai/opencode/tool/*` — `bash/read/write/edit/glob/grep/patch/todowrite/task/
  question/skill/webfetch/websearch` (+`OcGlob`, `OcMcpToolAdapter`).
- 커맨드/스킬/MCP: `OcCommandRegistry`, `OcSkill`, `OcMcpClient`.
- 프로바이더: `ai/opencode/provider/OcProviders.kt` (`OcProviderRegistry`,
  `gemini/openai/anthropic/local` + config 기반 커스텀 프로바이더).
  - 개별 클라이언트: `ai/api/GeminiClient.kt`, `OpenAiClient.kt`, `AnthropicClient.kt`,
    `LocalLlamaProvider.kt`, `LlamaRuntimeBridge.kt`, `LlamaAndroid.kt`.
- 구(舊) 엔진(별도): `ai/agent/AiAgentEngine.kt` + `ai/agent/ToolRegistry.kt`(1573줄)
  + `ai/agent/ModelManager.kt`, `AiConversationStorage.kt`, `HardwareManager.kt`.
  → `AiAgentPanel`이 `OcEngine`을 `AiAgentEngine`으로 alias import 하므로,
  실제 UI는 신형(`OcEngine`)을 사용. 구형 `ai/agent/AiAgentEngine`는 별도 잔존.

### (B) opencode Serve API 클라이언트

- `ai/AiActivity.kt` + `ai/AiService.kt` + `ai/OpencodeEnvironment.kt`.
  - opencode 바이너리를 설치/실행하고 HTTP + SSE(`/event`)로 대화.
  - ToolsActivity에서 진입(`AiActivity.start`). 세션/에이전트/모델/커맨드 피커, 권한 다이얼로그.

### 현재 AI UI

- 주 진입: `MainActivityLayoutFactory.kt:1307` → `ai/ui/AiAgentPanel.kt`(2137줄)를
  하단 패널에 삽입. View 기반 채팅 UI(스트리밍, 추론 카드, 툴 승인, 슬래시 커맨드,
  모델/런타임 선택, Plan 표시).
- 보조 진입: `ToolsActivity` → `AiActivity`(opencode serve).

## 3. 로컬 AI (llama.cpp) (실측)

- 네이티브 소스 트리: 루트 `llama.cpp/` (CMake). JNI/브리지:
  `ai/api/LlamaRuntimeBridge.kt`, `LlamaRuntimeManager.kt`, `api/LlamaAndroid.kt`,
  `LocalLlamaProvider.kt`, `NnapiHelper.kt`.
- 모델: `ModelManager`(GGUF 목록/다운로드/저장공간 검사, `contextLength`, 양자화 태그).
- 하드웨어: `HardwareManager`(Vulkan/NNAPI 감지, 스레드 권장) — 매우 단순.
- 설정: `AiSettings`에 `nGpuLayers`, `nThreads`, `contextLength`.
- 판정: 로컬 추론 경로는 실존하나, 메모리 추정/자원 정책/동시성 제어는 미흡.

## 4. 빌드/컴파일 (실측) — 라이브 검증의 핵심 자산

- **`analysis/RealtimeCompiler.kt` 가 이미 존재**: Gradle 없이
  Kotlin(`K2JVMCompiler` jar, `KotlinDaemon`), Java(`jdk.compiler` 모듈),
  C/C++(ninja 또는 NDK clang)를 직접 컴파일하고 raw 출력 라인을 반환.
- `BuildProblemParser`(BuildProblem.kt, 순수 Kotlin)가 kotlinc/javac/cpp/cmake/
  ninja/aapt/Gradle 형식 파싱 → `BuildProblem`.
- `analysis/Diagnostic.kt`, `DiagnosticCollector.kt`, `ErrorFingerprint.kt` 존재.
- `MainActivity`: `getTotalErrorCount()`, `getErrorsAsText()`, `getErrorFingerprints()`,
  `triggerLiveDiagnosticsAndWait()` 제공(현재 오류만 노출, 경고/정보는 미노출).
- 결론: **새 컴파일러 시스템을 만들지 않는다.** 위 자산을 어댑터로 감싸 재사용한다.

## 5. Shell 및 파일 도구 (실측)

- `ShellRunner.runSh()` (object, `/system/bin/sh`), `ShellEnvironment.buildToolEnvironment()`.
- BashTool: 기본 타임아웃 120s, 최대 600s, 위험도 판정은 ad-hoc 문자열 매칭.
- 파일 도구: `read/write/edit/glob/grep/patch` + 경로 workspace 검증(`ToolContext`).
- 위험 명령 차단은 있으나 체계적 분류기/정책 테이블은 없음.

## 6. 유지 / 교체 결정

### 유지 (재사용)
- `analysis/*` 컴파일러/진단/프로바이더 연동 (핵심 자산).
- `ai/opencode/*` 세션/툴/권한/스킬/MCP 구조.
- `ai/api/*` 프로바이더 클라이언트.
- `ModelManager`, `LlamaRuntime*`.
- `ShellRunner`, `ShellEnvironment`, 파일 도구.
- `highlight.LanguageDetector` + 하이라이터 (확장 대상).
- `MainActivity`의 진단 노출 API.

### 교체/개선
- Live 검증: `ErrorFingerprint` 단순 개수 비교 → 구조적 `DiagnosticDiffEngine`.
  (새/해결/기존/경고 분류, 이동 라인 보정, 불확실성 표시)
- Universal language support: `highlight.Language`(하이라이트 전용) 위에
  확장 가능한 `LanguageRegistry`(감지·툴 어댑터·capability) 추가.
- JSON 검증/자동복구: 신규.
- 자원/병렬성 정책: `HardwareManager` 확장한 `ResourceManager` 신규.
- Shell 위험 분류: 신규 `CommandRiskClassifier`로 통합.
- UI: 최종적으로 `AiAgentPanel`을 작업공간(Chat/Plan/Changes/Search/Diagnostics/
  Sessions/Models/Settings) 구조로 재구성 대상. **이번 세션에서는 미완.**

## 7. 목표 아키텍처 매핑 (29 역할 → 현실)

| 역할 | 상태 | 구현 위치 |
|---|---|---|
| Agent Orchestrator | 존재 | `OcEngine` |
| Session Manager | 존재 | `OcSessionStore` |
| Plan Manager | 부분 | `OcAgentRegistry` plan 에이전트 + `todowrite` |
| Provider Registry | 존재 | `OcProviderRegistry` |
| API Adapter | 존재 | `ai/api/*` |
| Local LLM Provider | 존재 | `LocalLlamaProvider`, `LlamaRuntime*` |
| Tool Registry/Engine | 존재 | `OcToolRegistry`, `OcEngine.executeToolCall` |
| Permission Manager | 존재 | `Permission` (+ 신규 `CommandRiskClassifier`) |
| Subagent Manager | 존재 | `OcEngine.spawnSubagent` |
| Resource Manager | **신규** | `ai/resource/ResourceManager.kt` |
| **Language Registry** | **신규** | `ai/lang/LanguageRegistry.kt` |
| File Search | 존재 | `glob/grep` |
| Code Edit/Diff/Undo | 존재 | `edit/write/patch`, `OcSnapshot` |
| Compiler Adapter | 존재→래핑 | `analysis/RealtimeCompiler` + **신규** `LiveCompilerAdapter` |
| **Live Validation / Diagnostic Diff** | **신규** | `ai/validation/*` |
| **JSON Validation & Repair** | **신규** | `ai/validation/JsonValidator.kt` |
| Shell Execution | 존재 | `ShellRunner` + 신규 위험 분류 |
| Error Repair Loop | 존재(개수 비교) → 개선 | `OcEngine.handlePostWriteAnalysis` |
| Session Persistence | 존재 | `OcSessionStore` |
| Event/Status | 존재 | `OcEngine.Listener` |

## 8. 이번 세션 구현 범위 (정직한 상태)

상세 파일/테스트는 §10 참고. 상태 표기는 Implemented / Integrated / Tested /
Build Verified / Planned 를 구분한다.

- `ai/lang/LanguageRegistry.kt` — 다국어 레지스트리 + 환경 툴 probe. **Implemented/Tested**
- `ai/validation/JsonValidator.kt` — JSON 검증·규칙기반 자동복구. **Implemented/Tested**
- `ai/validation/LiveDiagnostics.kt` — 정규화 + `DiagnosticDiffEngine`(15 시나리오).
  **Implemented/Tested**
- `ai/validation/LiveValidationService.kt` — 기준선/변경추적/검증 오케스트레이션.
  **Implemented/Tested**
- `ai/resource/ResourceManager.kt` — 적응형 병렬성/메모리 정책. **Implemented/Tested**
- `ai/shell/CommandRiskClassifier.kt` — 명령 위험 분류 + `BashTool` 통합. **Integrated/Tested**
- `MainActivity.getAllProblems()` — 경고 포함 진단 노출(가산적). **Implemented**
- `OcEngine.handlePostWriteAnalysis` → diff 엔진 사용. **Integrated**
- 새 UI 전면 교체, 세션/Changes/Diagnostics 화면군, MCP/LSP UI — **Planned (미완)**

## 9. 구현 순서 (실제 조정)

Phase 0 조사 → Phase 1 문서(이 문서) → Phase 2 핵심 인터페이스/언어레지스트리/
검증 → Phase 4 툴/쉘 → Phase 5 라이브검증/JSON → Phase 3 자원 →
Phase 6 UI(미완) → Phase 8 테스트/빌드(수행).

## 10. 검증 방법

- `./gradlew :app:compileDebugKotlin --offline`
- `./gradlew :app:testDebugUnitTest --offline`
- 신규 테스트: `LanguageRegistryTest`, `JsonValidatorTest`, `DiagnosticDiffEngineTest`,
  `LiveValidationServiceTest`, `ResourceManagerTest`, `CommandRiskClassifierTest`.
- 실제 기기에서 llama.cpp 추론/API 호출 테스트는 이 환경(휴대폰 없음/키 없음)에서
  수행 불가 → 미수행으로 명시.

## 11. 남은 문제 (우선순위)

1. AI UI 전면 교체(작업공간 구조) — 최우선 잔여.
2. 구형 `ai/agent/AiAgentEngine`/`ToolRegistry` 중복 정리(참조 조사 후).
3. 경고/정보 포함 end-to-end 검증 파이프라인 UI 연결.
4. LSP 어댑터, 모델 메모리 실측 기반 자원 정책 고도화.

---

## 12. 실행 결과 (실측, 2026-10-13)

### 빌드
- `./gradlew :app:compileDebugKotlin --offline` → **성공**
- `./gradlew :app:assembleDebug --offline` → **성공 (APK 생성)**
- 단위 테스트 컴파일/실행: `./gradlew :app:testDebugUnitTest --offline`

### 테스트
- 신규 테스트 6개 클래스, **78개 케이스** 전부 통과:
  `LanguageRegistryTest`(11), `JsonValidatorTest`(16), `DiagnosticDiffEngineTest`(17),
  `LiveValidationServiceTest`(7), `ResourceManagerTest`(13), `CommandRiskClassifierTest`(14).
- 전체 테스트 202개, 실패 0개 (아래 전제 조건 하).
- **전제/한계**: 기존(사용자 미커밋) 테스트 파일
  `app/src/test/java/com/aide/mobilestudio/extension/ExtensionInstallGateTest.kt` 는
  `ExtensionInstallGate` API와 타입이 맞지 않아 **컴파일 자체가 실패**한다. 이 파일은
  이번 작업과 무관한 기존 WIP이며 수정/삭제하지 않았다. 해당 파일을 소스셋에서
  제외하면 나머지 전체가 컴파일·통과한다. (파일은 원본 그대로 복원됨)

### 실제 통합 지점 (UI만이 아니라 엔진에 연결됨)
- `BashTool.riskHint` → `CommandRiskClassifier` 로 위험도 판정 통합.
- `write`/`edit` 툴 → JSON/JSONC 저장 직후 `JsonValidator`+`JsonRepairer` 실행,
  결과를 툴 출력에 포함(자동 복구 시 파일에 반영, 실패 시 위치 보고).
- `OcEngine.handlePostWriteAnalysis` → `DiagnosticDiffEngine` 로 교체하여
  새/해결/기존/경고 및 불확실 매칭을 분류(기존 `MainActivity` 진단 API 재사용).
- `MainActivity.getAllProblems()` 추가(경고/정보 포함 노출, 가산적 변경).

### 이 환경에서 검증 불가한 항목 (미수행으로 명시)
- 실제 기기에서의 llama.cpp 추론, 외부 API 호출(키 없음), NDK/실제 컴파일러 동작.
- 새 AI UI 렌더링/회전/다크모드 등 UI 검증(새 UI 미구현).

---

## 13. 새 AI UI 구현 결과 (2차 작업)

기존 AI UI(`AiActivity` opencode serve 클라이언트, `AiAgentPanel` 하단 패널)는
삭제하지 않고 유지했으며, **새 전체화면 작업공간을 1차 진입점으로 연결**했다.

### 새 화면
`ai/ui/AiWorkspaceActivity.kt` — 탭 기반 AI 작업공간:
- **Chat**: 스트리밍 응답, 추론 표시, 도구 실행 상태, 승인/질문 다이얼로그, 중지.
- **Plan**: `todowrite` 결과(todo 상태/우선순위), "계획 요청" 버튼.
- **Changes**: 세션 중 변경 파일 목록, before/after Diff(`DiffViewerActivity`), 전체 되돌리기(`engine.revert()`).
- **Diagnostics**: `WorkspaceValidation`(기존 `RealtimeCompiler` + `LiveValidationService`)로
  새/기존/경고/해결 분류. 항목 탭 → 상세 + "AI에게 수정 요청", 재검사.
- **Sessions**: 목록/새 세션/열기/이름변경/분기/보관/삭제.
- **Models**: 공급자/모델 목록, 선택(로컬은 GGUF 필요 안내).
- **Settings**: API 키·모델·스레드·컨텍스트·자동편집·웹검색, 자원 상태(여유 MB/힙/병렬 한도).
- **Terminal**: 셸 명령 실행(출력/종료코드).

### 연결/구현
- 엔진: 기존 `OcEngine` 재사용. `projectDir` 오버라이드, `onFileChanged` 이벤트,
  `listProviders/listProviderModels/currentModelLabel/currentTodos/latestBackupForCurrentSession` 추가(가산적).
- 검증: `WorkspaceValidation` + `RealtimeCompilerRunner` (MainActivity 비의존).
- 자원: `AndroidMemoryProbe` + `ResourceManager`로 설정 탭에 실제 측정/정책 표시.
- 진입점: `MainActivity.openAiChatFromEditor()` → `openAiWorkspace()`(현재 프로젝트 전달),
  `ToolsActivity` "Open AI Workspace". **manifest에 AI Activity가 누락돼 있던 기존 문제도 함께 등록**.
- i18n: UI 라벨은 `AiWorkspaceStrings`(EN 기본/KO) — 코드·경로·로그는 번역하지 않음.

### 검증
- `:app:compileDebugKotlin`, `:app:assembleDebug`, `:app:testDebugUnitTest` 모두 오프라인 성공.
- 컴파일 오류: 기존 stale 증분 컴파일 아티팩트는 `--rerun-tasks`로 해소되었고,
  `ExtensionBuilder.kt:144`의 nullable 수신자 오류(`androidJar`)를 수정.
- 남은 한계: 실제 기기 UI 동작(회전/키보드/런타임)은 미검증.

---

## 14. 3차 작업 — UI 고도화 + 추가 기능

### UI 고도화 (`고급스럽게`)
- `ai/ui/AiUiKit.kt`: 테마 인식 카드/필/원형 아바타/리플 버튼/칩/구분선/섹션헤더/빈 상태 킷.
- `ai/ui/AiMarkdown.kt`: 코드블록·인라인 코드·**굵게**·제목·불릿 렌더러(스팬 기반, 의존성 없음).
- 상단바: 원형 아바타 + 프로젝트 칩 + 모델 + 상태 필(색상: info/accent/warning/success/error/muted).
- 탭: 알약형 탭 + 선택 강조 + 리플.
- 채팅: 사용자/에이전트 버블 분리, 에이전트 카드에 헤더·마크다운·복사/다시생성 액션, 툴 타임라인(⚙/✓), 시스템 알림 필.
- 입력바: 라운드 필드 + 강조 전송 버튼. 빈 상태 제공.
- **테마 연동**: `UiTheme.resolve(this)`로 앱의 다크/라이트/사용자 테마를 반영.

### 새 탭/기능
- **Search 탭 신규**: 파일명 + 내용(정규식) 검색, 제외 디렉터리 필터, 결과 클릭 복사/롱프레스 "AI에게 설명 요청".
- **Sessions 검색** 추가.
- **Plan 편집**: 각 todo "재시도", "계획 편집" 요청.
- **Models**: 공급자별 "연결 테스트" 및 "모델 목록" 가져오기.
- **Settings**: 프로젝트 언어 감지 + 컴파일러 가용성, LSP 서버 가용성(PATH probe), MCP 서버 목록/테스트, 자원 상태.

### 신규 모듈 (모두 테스트 포함)
- `ai/provider/ConnectionTester.kt` + `AndroidHttpTransport.kt`: 주입형 transport 기반 연결 테스트.
  상태코드(200/401/403/404/429/5xx/예외) 해석, 키 마스킹.
- `ai/provider/ProviderModelList.kt`: OpenAI/Gemini/배열 형식 모델 목록 파서.
- `ai/lsp/LspProtocol.kt`: Content-Length 프레이밍(스트리밍 디코더) + publishDiagnostics → `LiveDiagnostic` 변환.
- `ai/lsp/LspServerRegistry.kt`: 언어별 서버 사양 + PATH 가용성 probe.
- `ai/resource/TaskScheduler.kt`: `ResourceManager` 기반 적응형 병렬 실행.
- `OcEngine`: permission 판정 공용화, **자동 허용된 read-only 툴 병렬 실행**(승인 우회 없음, 그 외엔 순차), `mcpServerNames/testMcpServer`.

### 검증
- 테스트 총 **269개, 실패 0**. 신규 모듈 테스트 99+개.
- `assembleDebug` APK 생성 성공.
- 빌드 메모리 부족으로 `gradle.properties`의 Kotlin/Gradle 힙을 1.5G→2.5G로 상향.
- 실제 기기 런타임, 실제 네트워크 API 호출, 실제 LSP/MCP 서버 실행은 미검증(환경 제약).

---

## 15. 4차 작업 — 실기기 불필요 항목 마무리

### LSP 클라이언트 (핵심 코어 + 테스트)
- `ai/lsp/LspClient.kt`: `LspTransport` 인터페이스, `StreamLspTransport`(스트림 프레이밍),
  `LspClient`(id 상관, `initialize`/`didOpen`/`didClose`/`shutdown`,
  `publishDiagnostics` → `LiveDiagnostic` 저장·콜백), `LspServerProcess`(프로세스 기동 best-effort).
- `LspClientTest`: 초기화 응답 상관 + 알림 수신, 에러 응답 예외, didOpen 프레이밍, EOF.

### MCP 서버 관리
- `OpencodeConfig.upsertMcpServer/removeMcpServer/globalConfigFile` (앱 전역 config에 저장).
- 엔진 `addMcpServer/removeMcpServer`(reloadConfig 반영).
- Settings 탭: **서버 추가(이름+명령) / 삭제 / 테스트**.

### Plan 구조적 편집
- 엔진 `updateTodos`. Plan 탭: **단계 추가**, 롱프레스 메뉴(**재시도/위로/아래로/완료 토글/삭제**) — 세션에 영속.

### 워크스페이스 화면 완성
- **Tools 탭**(등록 도구 + 읽기 전용 표시), **Resources 탭**(모델/메모리/병렬성/모델 로드 가능/빌드+추론).
- 탭 구성: Chat·Plan·Changes·Search·Diagnostics·Tools·Resources·Sessions·Models·Settings·Terminal.

### 정리
- 구형 `ai/agent/AiAgentEngine`에 `@Deprecated`(OcEngine으로 대체, 실제 소스셋 미참조) 명시.

### 검증
- 테스트 총 **288개, 실패 0** (LspClientTest 등 추가). `assembleDebug` 성공.
- 동시 편집 중 발생한 외부 컴파일 오류 2건(`BuildApkLocator` 시퀀스→리스트,
  `ExtensionBuilder` nullable)도 수정. Kotlin 증분 캐시 충돌은 캐시 정리로 해소.
- 실제 언어 서버 프로세스 실행/실기기 검증은 미수행.




