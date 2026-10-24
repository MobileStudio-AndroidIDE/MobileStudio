"""오프라인 지식베이스 — 모델·네트워크 없이 동작하는 한국어 개념 사전.

LLM 가중치나 외부 API를 쓰지 않는다.  사람이 검수한 구조적 지식(정의 ·
하는 일 · 사용 예 · 주의 · 관련 항목)을 담고, 질문에서 용어를 찾아
한국어 설명을 조립한다.  웹이 꺼져 있어도 기본 개념은 항상 답할 수 있고,
웹을 쓸 때도 먼저 이 지식으로 한국어 맥락을 세운다.

공개 API::

    lookup(text)        -> Entry | None
    render(entry)       -> str          (한국어 설명 조립)
    answer(text)        -> str | None   (찾으면 바로 답)
    web_context(text)   -> str | None   (웹 답변 앞에 붙일 한국어 맥락)
    translate_terms(s)  -> str          (용어 글로서리 치환)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Entry:
    term: str
    summary: str
    details: tuple[str, ...] = ()
    category: str = "general"
    aliases: tuple[str, ...] = ()
    example: str = ""
    caution: str = ""
    related: tuple[str, ...] = ()
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "summary": self.summary,
            "details": list(self.details),
            "category": self.category,
            "aliases": list(self.aliases),
            "example": self.example,
            "caution": self.caution,
            "related": list(self.related),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# 지식 항목
# ---------------------------------------------------------------------------

_ENTRIES: tuple[Entry, ...] = (
    # ── 셸 기본 명령 ────────────────────────────────────────────────
    Entry(
        term="pwd",
        aliases=("print working directory",),
        category="shell",
        summary="pwd는 현재 작업 디렉터리의 절대 경로를 출력하는 셸 명령입니다.",
        details=(
            "앞으로 실행할 명령들이 어느 디렉터리를 기준으로 동작하는지 확인할 때 씁니다.",
            "경로를 바꾸지는 않고, 지금 위치만 알려줍니다.",
        ),
        example="pwd",
        caution="셸 내장 명령이라 별도 프로그램이 없어도 대부분 동작합니다.",
        related=("ls", "cd"),
    ),
    Entry(
        term="ls",
        aliases=("list", "디렉터리 목록"),
        category="shell",
        summary="ls는 디렉터리 안의 파일과 하위 디렉터리 목록을 보여주는 명령입니다.",
        details=(
            "-l은 권한·크기·수정시각까지 자세히, -a는 숨김 파일까지 보여줍니다.",
            "결과를 정렬하거나 시간순으로 보려면 옵션을 더합니다.",
        ),
        example="ls -la",
        caution="파일을 지우거나 만들지 않습니다. 조회 전용입니다.",
        related=("pwd", "cd", "find"),
    ),
    Entry(
        term="cd",
        aliases=("change directory",),
        category="shell",
        summary="cd는 현재 작업 디렉터리를 다른 경로로 옮기는 셸 내장 명령입니다.",
        details=(
            "cd ..는 한 단계 위, cd ~는 홈 디렉터리, cd -는 직전 디렉터리로 갑니다.",
            "이동은 셸 안에서만 유지되며, 프로세스를 새로 띄우면 초기화됩니다.",
        ),
        example="cd /var/log",
        related=("pwd", "ls"),
    ),
    Entry(
        term="cp",
        aliases=("copy", "복사"),
        category="shell",
        summary="cp는 파일이나 디렉터리를 복사하는 명령입니다.",
        details=(
            "디렉터리는 -r을 붙여야 재귀적으로 복사됩니다.",
            "덮어쓰기를 막으려면 -n, 확인하려면 -i를 씁니다.",
        ),
        example="cp -r src backup/",
        caution="같은 이름이 있으면 기본적으로 덮어씁니다.",
        related=("mv", "rm"),
    ),
    Entry(
        term="mv",
        aliases=("move", "이동", "이름 변경"),
        category="shell",
        summary="mv는 파일·디렉터리를 옮기거나 이름을 바꾸는 명령입니다.",
        details=(
            "같은 디렉터리에서 다른 이름을 주면 이름 변경이 됩니다.",
            "다른 파일시스템으로 옮길 때는 복사 후 삭제로 동작합니다.",
        ),
        example="mv notes.txt docs/notes.md",
        related=("cp", "rm"),
    ),
    Entry(
        term="rm",
        aliases=("remove", "삭제"),
        category="shell",
        summary="rm은 파일을 지우는 명령입니다. 디렉터리는 -r이 필요합니다.",
        details=(
            "삭제한 파일은 휴지통을 거치지 않고 즉시 사라집니다.",
            "-i를 붙이면 하나씩 확인하며 지웁니다.",
        ),
        example="rm -i old.log",
        caution="rm -rf / 같은 명령은 시스템을 파괴합니다. 경로를 항상 확인하세요.",
        related=("cp", "mv"),
    ),
    Entry(
        term="mkdir",
        aliases=("make directory", "폴더 생성"),
        category="shell",
        summary="mkdir는 새 디렉터리를 만드는 명령입니다.",
        details=(
            "-p를 붙이면 중간 경로까지 한 번에 만들고, 이미 있어도 오류를 내지 않습니다.",
        ),
        example="mkdir -p build/classes",
        related=("cd", "ls"),
    ),
    Entry(
        term="cat",
        aliases=("concatenate", "파일 내용 보기"),
        category="shell",
        summary="cat은 파일 내용을 그대로 표준 출력으로 내보내는 명령입니다.",
        details=(
            "여러 파일을 이어붙일 수도 있습니다.",
            "긴 파일은 화면이 넘치므로 less나 head/tail이 편합니다.",
        ),
        example="cat README.md",
        related=("less", "head", "tail"),
    ),
    Entry(
        term="head",
        aliases=("앞부분",),
        category="shell",
        summary="head는 파일의 앞부분(기본 10줄)을 보여주는 명령입니다.",
        details=("-n 옵션으로 줄 수를 바꿉니다.",),
        example="head -n 20 app.log",
        related=("tail", "cat"),
    ),
    Entry(
        term="tail",
        aliases=("뒷부분",),
        category="shell",
        summary="tail은 파일의 뒷부분을 보여주며, -f로 실시간 추적이 가능합니다.",
        details=(
            "tail -f는 로그 파일에 새로 추가되는 줄을 계속 출력합니다.",
        ),
        example="tail -f server.log",
        related=("head", "cat"),
    ),
    Entry(
        term="grep",
        aliases=("검색",),
        category="shell",
        summary="grep은 텍스트에서 패턴과 일치하는 줄을 찾아 출력하는 명령입니다.",
        details=(
            "-r은 디렉터리 재귀, -n은 줄 번호, -i는 대소문자 무시입니다.",
            "정규표현식을 지원하며, 파이프와 함께 쓰면 다른 명령의 출력도 걸러냅니다.",
        ),
        example="grep -rn TODO src/",
        related=("find", "sed", "awk"),
    ),
    Entry(
        term="find",
        aliases=("파일 찾기",),
        category="shell",
        summary="find는 이름·크기·수정시각 같은 조건으로 파일을 찾는 명령입니다.",
        details=(
            "-name은 이름, -size는 크기, -mtime은 수정 시각 조건입니다.",
            "-exec를 붙이면 찾은 파일에 다른 명령을 실행할 수 있습니다.",
        ),
        example="find . -name '*.py' -mtime -7",
        related=("grep", "ls"),
    ),
    Entry(
        term="chmod",
        aliases=("권한 변경",),
        category="shell",
        summary="chmod는 파일·디렉터리의 읽기/쓰기/실행 권한을 바꾸는 명령입니다.",
        details=(
            "기호식(u+x)과 숫자식(755) 두 가지 표기가 있습니다.",
            "755는 소유자 전체 권한, 나머지 읽기·실행을 뜻합니다.",
        ),
        example="chmod +x run.sh",
        caution="777은 모든 사용자에게 쓰기 권한을 주므로 가급적 피합니다.",
        related=("ls",),
    ),
    Entry(
        term="ps",
        aliases=("프로세스 목록",),
        category="shell",
        summary="ps는 현재 실행 중인 프로세스 목록을 보여주는 명령입니다.",
        details=(
            "ps aux는 모든 사용자의 프로세스를 자세히 보여줍니다.",
            "특정 프로세스는 grep과 조합해 찾습니다.",
        ),
        example="ps aux | grep python",
        related=("kill", "top"),
    ),
    Entry(
        term="kill",
        aliases=("프로세스 종료",),
        category="shell",
        summary="kill은 실행 중인 프로세스에 신호를 보내 종료시키는 명령입니다.",
        details=(
            "기본 신호는 정상 종료(SIGTERM)이며, -9는 강제 종료입니다.",
            "대상은 프로세스 ID(PID)로 지정합니다.",
        ),
        example="kill 1234",
        caution="-9는 저장 없이 즉시 끊으므로 마지막 수단으로 씁니다.",
        related=("ps",),
    ),
    Entry(
        term="df",
        aliases=("디스크 여유",),
        category="shell",
        summary="df는 파일시스템별 전체·사용·여유 공간을 보여주는 명령입니다.",
        details=("-h를 붙이면 사람이 읽기 쉬운 단위로 표시합니다.",),
        example="df -h",
        related=("du",),
    ),
    Entry(
        term="du",
        aliases=("용량", "디렉터리 크기"),
        category="shell",
        summary="du는 디렉터리나 파일이 차지하는 용량을 계산하는 명령입니다.",
        details=(
            "-sh는 현재 디렉터리 합계만, -ah는 항목별로 보여줍니다.",
            "큰 디렉터리를 찾을 때 sort -rh와 함께 씁니다.",
        ),
        example="du -sh * | sort -rh | head",
        related=("df",),
    ),
    Entry(
        term="tar",
        aliases=("압축", "아카이브"),
        category="shell",
        summary="tar는 여러 파일을 하나의 아카이브로 묶는 명령입니다.",
        details=(
            "-czf는 gzip 압축 생성, -xzf는 해제, -tzf는 내용 목록입니다.",
            "압축 자체는 gzip이 담당하고 tar는 묶는 역할입니다.",
        ),
        example="tar -czf backup.tar.gz src/",
        related=("zip", "gzip"),
    ),
    Entry(
        term="ssh",
        aliases=("원격 접속",),
        category="shell",
        summary="ssh는 암호화된 채널로 원격 서버에 접속하거나 명령을 실행하는 도구입니다.",
        details=(
            "키 기반 인증이 비밀번호보다 안전합니다.",
            "scp·rsync와 함께 쓰면 파일 전송도 보안 채널로 합니다.",
        ),
        example="ssh user@host",
        related=("scp", "rsync"),
    ),
    Entry(
        term="curl",
        aliases=("http 요청",),
        category="shell",
        summary="curl은 명령줄에서 HTTP·HTTPS 요청을 보내고 응답을 출력하는 도구입니다.",
        details=(
            "-o로 저장, -L은 리다이렉트 추적, -H는 헤더 추가입니다.",
            "API를 시험하거나 파일을 내려받을 때 자주 씁니다.",
        ),
        example="curl -L -o file.zip https://example.com/file.zip",
        related=("wget", "jq"),
    ),
    Entry(
        term="awk",
        aliases=(),
        category="shell",
        summary="awk는 텍스트를 열(필드) 단위로 처리하는 스크립트형 도구입니다.",
        details=(
            "공백으로 나뉜 열을 $1, $2로 참조합니다.",
            "합계·평균 같은 집계는 END 블록에서 처리합니다.",
        ),
        example="awk '{s+=$1} END {print s}' data.txt",
        related=("sed", "cut"),
    ),
    Entry(
        term="sed",
        aliases=(),
        category="shell",
        summary="sed는 텍스트를 줄 단위로 치환·삭제하는 스트림 편집기입니다.",
        details=(
            "s/찾기/바꾸기/ 형태로 치환하고, g를 붙이면 줄 전체를 반복 치환합니다.",
            "-i로 파일을 직접 수정할 수 있습니다.",
        ),
        example="sed 's/old/new/g' file.txt",
        caution="-i는 원본을 바로 덮어씁니다. 백업 옵션을 함께 쓰는 편이 안전합니다.",
        related=("grep", "awk"),
    ),
    Entry(
        term="sort",
        aliases=("정렬",),
        category="shell",
        summary="sort는 텍스트 줄을 사전순·숫자순으로 정렬하는 명령입니다.",
        details=(
            "-n은 숫자 정렬, -r은 역순, -h는 1K/2M 같은 단위 인식입니다.",
            "-k로 특정 열을 기준으로 정렬합니다.",
        ),
        example="sort -n scores.txt",
        related=("uniq", "awk"),
    ),
    Entry(
        term="uniq",
        aliases=("중복 제거",),
        category="shell",
        summary="uniq는 연속으로 반복된 줄을 하나로 합치는 명령입니다.",
        details=(
            "반드시 정렬된 입력에서 동작하므로 보통 sort와 함께 씁니다.",
            "-c는 각 줄의 반복 횟수를 함께 출력합니다.",
        ),
        example="sort names.txt | uniq -c",
        related=("sort", "awk"),
    ),
    Entry(
        term="wc",
        aliases=("줄 수", "단어 수"),
        category="shell",
        summary="wc는 파일의 줄·단어·바이트 수를 세는 명령입니다.",
        details=("-l은 줄, -w는 단어, -c는 바이트 수만 셉니다.",),
        example="wc -l main.py",
        related=("cat", "grep"),
    ),
    Entry(
        term="less",
        aliases=("페이지 보기",),
        category="shell",
        summary="less는 긴 텍스트를 스크롤하며 읽는 페이지 뷰어입니다.",
        details=(
            "방향키·PageUp/PageDown으로 이동하고 q로 종료합니다.",
            "/검색어로 문서 안을 검색할 수 있습니다.",
        ),
        example="less /var/log/syslog",
        related=("cat", "head"),
    ),
    Entry(
        term="man",
        aliases=("매뉴얼",),
        category="shell",
        summary="man은 명령의 공식 매뉴얼 페이지를 보여주는 도구입니다.",
        details=(
            "특정 절은 man 5 passwd처럼 번호로 지정합니다.",
            "man -k 키워드로 관련 문서를 검색합니다.",
        ),
        example="man grep",
        related=("less",),
    ),
    Entry(
        term="echo",
        aliases=(),
        category="shell",
        summary="echo는 인자를 표준 출력으로 그대로 내보내는 셸 내장 명령입니다.",
        details=(
            ">로 파일에 쓰고, >>로 이어 붙일 수 있습니다.",
            "스크립트에서 변수 값을 확인할 때 자주 씁니다.",
        ),
        example="echo \"$PATH\"",
        related=("cat", "export"),
    ),
    Entry(
        term="export",
        aliases=("환경변수 설정",),
        category="shell",
        summary="export는 셸 변수를 자식 프로세스가 물려받는 환경변수로 만드는 명령입니다.",
        details=(
            "export VAR=값 형태로 설정합니다.",
            "설정은 현재 셸 세션에만 유지되며, 영구 적용은 셸 설정 파일에 적습니다.",
        ),
        example="export PATH=$PATH:/opt/bin",
        related=("env", "echo"),
    ),
    Entry(
        term="env",
        aliases=("환경변수 보기",),
        category="shell",
        summary="env는 현재 환경변수 목록을 출력하거나, 다른 환경으로 명령을 실행합니다.",
        details=("-i는 빈 환경에서 실행해 재현성을 확인할 때 씁니다.",),
        example="env | sort",
        related=("export", "echo"),
    ),
    Entry(
        term="alias",
        aliases=(),
        category="shell",
        summary="alias는 긴 명령에 짧은 이름을 붙이는 셸 기능입니다.",
        details=(
            "현재 세션에만 적용되며, 영구 사용은 셸 설정 파일에 적습니다.",
            "등록된 목록은 인자 없이 alias만 실행해 확인합니다.",
        ),
        example="alias ll='ls -la'",
        related=("export",),
    ),

    # ── 셸 개념 ────────────────────────────────────────────────────
    Entry(
        term="셸",
        aliases=("shell", "bash", "zsh", "쉘"),
        category="shell",
        summary="셸은 사용자가 입력한 명령을 해석해 운영체제에 전달하는 프로그램입니다.",
        details=(
            "bash·zsh·fish가 대표적이며, 스크립트 언어로도 쓰입니다.",
            "파이프·리다이렉션 같은 연결 기능을 제공합니다.",
        ),
        example="echo $SHELL",
        related=("터미널", "파이프"),
    ),
    Entry(
        term="터미널",
        aliases=("terminal", "콘솔", "console"),
        category="shell",
        summary="터미널은 텍스트로 셸과 상호작용하는 화면 프로그램입니다.",
        details=(
            "터미널이 셸을 실행하고, 셸이 명령을 해석합니다.",
        ),
        related=("셸",),
    ),
    Entry(
        term="파이프",
        aliases=("pipe", "|"),
        category="shell",
        summary="파이프는 한 명령의 출력을 다음 명령의 입력으로 연결하는 셸 기능입니다.",
        details=(
            "`|` 기호로 연결하며, 조회·필터·집계를 조합할 때 씁니다.",
            "각 단계는 동시에 실행되어 스트림처럼 흐릅니다.",
        ),
        example="ps aux | grep python | wc -l",
        related=("셸", "grep"),
    ),
    Entry(
        term="리다이렉션",
        aliases=("redirection", "표준출력"),
        category="shell",
        summary="리다이렉션은 표준 입출력을 파일이나 다른 스트림으로 돌리는 셸 기능입니다.",
        details=(
            "`>`는 덮어쓰기, `>>`는 이어쓰기, `2>`는 오류 출력 지정입니다.",
        ),
        example="python app.py > out.log 2>&1",
        related=("파이프", "echo"),
    ),
    Entry(
        term="환경변수",
        aliases=("environment variable",),
        category="shell",
        summary="환경변수는 프로세스가 물려받는 이름-값 쌍의 설정입니다.",
        details=(
            "PATH처럼 실행 파일 탐색 경로를 정하는 변수도 있습니다.",
            "셸에서 export로 설정하면 자식 프로세스가 물려받습니다.",
        ),
        example="echo $HOME",
        related=("export", "env"),
    ),
    Entry(
        term="권한",
        aliases=("permission", "퍼미션"),
        category="os",
        summary="권한은 파일·디렉터리에 누가 읽기/쓰기/실행을 할 수 있는지 정하는 규칙입니다.",
        details=(
            "소유자·그룹·기타 사용자 세 범위로 나뉩니다.",
            "ls -l의 rwxrwxrwx 표기와 숫자 표기(755 등)가 대응합니다.",
        ),
        example="ls -l",
        related=("chmod",),
    ),
    Entry(
        term="파일시스템",
        aliases=("filesystem",),
        category="os",
        summary="파일시스템은 저장 장치에 파일과 디렉터리를 조직하는 규칙과 구조입니다.",
        details=(
            "리눅스는 / 아래에 모든 장치를 트리로 붙입니다.",
            "ext4·xfs·btrfs·ntfs 등 형식이 있습니다.",
        ),
        related=("마운트", "df"),
    ),
    Entry(
        term="마운트",
        aliases=("mount",),
        category="os",
        summary="마운트는 저장 장치나 원격 자원을 디렉터리 트리의 한 지점에 연결하는 일입니다.",
        details=("리눅스에서는 모든 파일이 단일 트리에 나타나도록 붙입니다.",),
        related=("파일시스템",),
    ),
    Entry(
        term="프로세스",
        aliases=("process",),
        category="os",
        summary="프로세스는 실행 중인 프로그램의 인스턴스입니다.",
        details=(
            "각 프로세스는 고유한 PID와 자체 메모리 공간을 가집니다.",
            "부모-자식 관계로 생성되며, 종료 시 자원이 회수됩니다.",
        ),
        related=("ps", "kill"),
    ),

    # ── 깃 ────────────────────────────────────────────────────────
    Entry(
        term="깃",
        aliases=("git",),
        category="vcs",
        summary="깃은 소스 코드의 변경 이력을 기록·추적하는 분산 버전 관리 시스템입니다.",
        details=(
            "각 커밋은 스냅샷과 부모 링크를 가진 객체로 저장됩니다.",
            "브랜치로 갈래를 만들고, 병합으로 다시 합칩니다.",
        ),
        example="git status",
        related=("커밋", "브랜치", "깃 저장소"),
    ),
    Entry(
        term="깃 저장소",
        aliases=("repository", "repo", "리포지토리"),
        category="vcs",
        summary="깃 저장소는 이력·브랜치·설정을 담는 저장 공간이며 .git에 들어 있습니다.",
        details=(
            "git init으로 새로 만들고, git clone으로 복제합니다.",
            "작업 트리와 이력 저장소로 나뉩니다.",
        ),
        example="git init",
        related=("깃",),
    ),
    Entry(
        term="커밋",
        aliases=("commit",),
        category="vcs",
        summary="커밋은 스테이징된 변경을 하나의 이력 단위로 확정하는 작업입니다.",
        details=(
            "메시지에 변경의 목적을 남깁니다.",
            "커밋마다 해시 ID가 생기고 이전 커밋과 연결됩니다.",
        ),
        example="git commit -m \"fix: 정렬 버그\"",
        related=("git add", "깃"),
    ),
    Entry(
        term="git status",
        aliases=("깃 상태",),
        category="vcs",
        summary="git status는 작업 트리와 스테이징 영역의 현재 상태를 요약합니다.",
        details=(
            "변경·추가·삭제된 파일과 아직 추적되지 않은 파일을 보여줍니다.",
            "커밋 전에 무엇이 반영될지 확인하는 첫 명령입니다.",
        ),
        example="git status",
        related=("깃", "커밋"),
    ),
    Entry(
        term="git add",
        aliases=("스테이징", "stage"),
        category="vcs",
        summary="git add는 변경한 파일을 다음 커밋에 포함되도록 스테이징 영역에 올립니다.",
        details=(
            "git add -p는 부분 단위로 골라 담습니다.",
            "스테이징은 커밋 내용을 미리 확정하는 단계입니다.",
        ),
        example="git add src/app.py",
        related=("커밋",),
    ),
    Entry(
        term="git commit",
        aliases=(),
        category="vcs",
        summary="git commit은 스테이징된 변경을 이력에 기록합니다.",
        details=(
            "-m으로 메시지를 지정하고, -a는 추적 중 파일의 변경을 함께 담습니다.",
        ),
        example="git commit -m \"feat: 로그인 추가\"",
        related=("커밋", "git add"),
    ),
    Entry(
        term="git push",
        aliases=(),
        category="vcs",
        summary="git push는 로컬 커밋을 원격 저장소로 올리는 명령입니다.",
        details=("-u로 기본 원격/브랜치를 연결해 두면 이후 생략할 수 있습니다.",),
        example="git push -u origin main",
        related=("git pull",),
    ),
    Entry(
        term="git pull",
        aliases=(),
        category="vcs",
        summary="git pull은 원격 변경을 내려받아 현재 브랜치에 합치는 명령입니다.",
        details=("내부적으로 fetch 후 merge(또는 rebase)를 수행합니다.",),
        example="git pull --rebase",
        related=("git push",),
    ),
    Entry(
        term="git log",
        aliases=("깃 로그",),
        category="vcs",
        summary="git log는 커밋 이력을 최신순으로 보여줍니다.",
        details=(
            "--oneline은 한 줄 요약, --stat은 변경 통계를 함께 보여줍니다.",
        ),
        example="git log --oneline -10",
        related=("커밋",),
    ),
    Entry(
        term="git diff",
        aliases=(),
        category="vcs",
        summary="git diff는 작업 트리·스테이징·커밋 사이의 차이를 보여줍니다.",
        details=(
            "인자 없이는 아직 스테이징하지 않은 변경을 보여줍니다.",
            "--staged는 스테이징된 변경을 봅니다.",
        ),
        example="git diff --stat",
        related=("git status",),
    ),
    Entry(
        term="브랜치",
        aliases=("branch",),
        category="vcs",
        summary="브랜치는 커밋 갈래에 붙인 이름으로, 독립적으로 작업하는 단위입니다.",
        details=(
            "git branch는 목록, git switch -c는 새 브랜치 생성입니다.",
            "브랜치는 가벼운 포인터일 뿐 파일을 복사하지 않습니다.",
        ),
        example="git switch -c feature/login",
        related=("깃", "병합"),
    ),
    Entry(
        term="병합",
        aliases=("merge",),
        category="vcs",
        summary="병합은 서로 다른 브랜치의 이력을 하나로 합치는 작업입니다.",
        details=(
            "같은 부분을 다르게 고쳤다면 충돌이 나고, 사람이 해결합니다.",
        ),
        example="git merge feature/login",
        related=("브랜치",),
    ),

    # ── 언어·런타임 ───────────────────────────────────────────────
    Entry(
        term="파이썬",
        aliases=("python",),
        category="language",
        summary="파이썬은 읽기 쉬운 문법과 풍부한 표준 라이브러리를 가진 범용 프로그래밍 언어입니다.",
        details=(
            "인터프리터가 소스를 바이트코드로 컴파일해 가상머신에서 실행합니다.",
            "데이터 분석·자동화·웹 백엔드·인공지능 분야에서 널리 쓰입니다.",
        ),
        example="python3 -c \"print('hello')\"",
        related=("인터프리터", "pip", "가상환경"),
    ),
    Entry(
        term="pip",
        aliases=(),
        category="language",
        summary="pip은 파이썬 패키지를 설치·관리하는 도구입니다.",
        details=(
            "pip install로 설치하고, requirements.txt로 목록을 고정합니다.",
        ),
        example="pip install requests",
        related=("파이썬", "가상환경"),
    ),
    Entry(
        term="가상환경",
        aliases=("venv", "virtualenv"),
        category="language",
        summary="가상환경은 프로젝트별로 독립된 파이썬 패키지 공간을 만드는 기능입니다.",
        details=(
            "프로젝트 간 의존성 충돌을 막습니다.",
            "python3 -m venv .venv로 만들고 activate로 활성화합니다.",
        ),
        example="source .venv/bin/activate",
        related=("파이썬", "pip"),
    ),
    Entry(
        term="인터프리터",
        aliases=("interpreter",),
        category="language",
        summary="인터프리터는 소스 코드를 한 번에 기계어로 번역하지 않고 실행 시점에 해석·실행합니다.",
        details=(
            "수정 후 즉시 실행할 수 있어 실험이 빠릅니다.",
            "파이썬·자바스크립트가 대표적입니다.",
        ),
        related=("컴파일러", "파이썬"),
    ),
    Entry(
        term="컴파일러",
        aliases=("compiler",),
        category="language",
        summary="컴파일러는 소스 코드를 실행 전에 기계어나 다른 언어로 번역하는 프로그램입니다.",
        details=(
            "미리 번역하므로 실행이 빠르지만, 수정마다 다시 빌드해야 합니다.",
            "C·C++·Rust가 대표적입니다.",
        ),
        related=("인터프리터",),
    ),
    Entry(
        term="자바스크립트",
        aliases=("javascript", "js"),
        category="language",
        summary="자바스크립트는 웹 브라우저에서 시작해 서버까지 쓰이는 동적 프로그래밍 언어입니다.",
        details=(
            "이벤트 기반·비동기 실행 모델을 가집니다.",
            "Node.js로 브라우저 밖에서도 실행합니다.",
        ),
        related=("인터프리터",),
    ),

    # ── 운영체제·웹·데이터 ────────────────────────────────────────
    Entry(
        term="리눅스",
        aliases=("linux",),
        category="os",
        summary="리눅스는 오픈소스 커널을 중심으로 한 운영체제 계열입니다.",
        details=(
            "커널이 하드웨어와 프로세스·메모리·파일시스템을 관리합니다.",
            "우분투·데비안·페도라 같은 배포판이 커널과 도구를 묶어 제공합니다.",
        ),
        related=("커널", "우분투", "배포판"),
    ),
    Entry(
        term="커널",
        aliases=("kernel",),
        category="os",
        summary="커널은 하드웨어 자원을 관리하고 프로그램에 안전한 실행 환경을 제공하는 운영체제 핵심입니다.",
        details=(
            "프로세스 스케줄링, 메모리 관리, 장치 제어, 시스템 호출을 담당합니다.",
        ),
        related=("리눅스", "프로세스"),
    ),
    Entry(
        term="우분투",
        aliases=("ubuntu",),
        category="os",
        summary="우분투는 데비안을 기반으로 한 대표적인 리눅스 배포판입니다.",
        details=(
            "설치와 사용이 쉬워 서버·개발 환경에서 널리 쓰입니다.",
            "LTS 버전은 장기 지원을 제공합니다.",
        ),
        related=("리눅스", "배포판"),
    ),
    Entry(
        term="배포판",
        aliases=("distribution", "distro"),
        category="os",
        summary="배포판은 리눅스 커널에 패키지 관리자·유틸리티·데스크톱을 묶어 설치 가능하게 만든 묶음입니다.",
        details=("우분투·데비안은 apt, 페도라는 dnf를 씁니다.",),
        related=("리눅스", "우분투"),
    ),
    Entry(
        term="http",
        aliases=("https",),
        category="web",
        summary="HTTP는 클라이언트와 서버가 요청과 응답으로 데이터를 주고받는 프로토콜입니다.",
        details=(
            "메서드(GET·POST 등)와 상태 코드(200·404·500)로 대화합니다.",
            "HTTPS는 TLS로 내용을 암호화합니다.",
        ),
        related=("api", "rest"),
    ),
    Entry(
        term="api",
        aliases=("API",),
        category="web",
        summary="API는 프로그램끼리 정해진 규칙으로 기능을 주고받는 인터페이스입니다.",
        details=(
            "함수 시그니처, HTTP 엔드포인트, 라이브러리 인터페이스 등이 모두 API입니다.",
            "잘 설계된 API는 내부 구현을 숨기고 계약만 노출합니다.",
        ),
        related=("http", "rest"),
    ),
    Entry(
        term="json",
        aliases=("JSON",),
        category="data",
        summary="JSON은 이름-값 쌍과 배열로 데이터를 표현하는 텍스트 형식입니다.",
        details=(
            "사람이 읽기 쉽고 언어에 독립적이라 API 응답에 널리 쓰입니다.",
            "주석과 함수는 표현할 수 없습니다.",
        ),
        example='{"name": "rena", "version": 1}',
        related=("yaml", "api"),
    ),
    Entry(
        term="yaml",
        aliases=("YAML",),
        category="data",
        summary="YAML은 들여쓰기로 구조를 표현하는 사람 친화적 데이터 형식입니다.",
        details=(
            "설정 파일에 널리 쓰이며, JSON의 상위 호환 표현도 담을 수 있습니다.",
        ),
        related=("json",),
    ),
    Entry(
        term="도커",
        aliases=("docker", "컨테이너"),
        category="infra",
        summary="도커는 애플리케이션과 의존성을 컨테이너로 묶어 어디서든 같은 환경으로 실행하는 도구입니다.",
        details=(
            "이미지를 빌드하고 컨테이너로 실행합니다.",
            "가상머신보다 가볍지만 커널은 호스트와 공유합니다.",
        ),
        example="docker run -it python:3 bash",
        related=("커널",),
    ),
    Entry(
        term="알고리즘",
        aliases=("algorithm",),
        category="cs",
        summary="알고리즘은 문제를 해결하는 유한한 단계의 절차입니다.",
        details=(
            "정확성과 함께 시간·공간 비용으로 평가합니다.",
            "같은 문제도 입력 특성에 따라 유리한 알고리즘이 다릅니다.",
        ),
        related=("자료구조", "빅오"),
    ),
    Entry(
        term="자료구조",
        aliases=("data structure",),
        category="cs",
        summary="자료구조는 데이터를 저장하고 꺼내는 방식을 정한 구조입니다.",
        details=(
            "배열·연결 리스트·스택·큐·해시 테이블·트리·그래프가 기본입니다.",
            "연산별 비용이 달라 문제에 맞게 고릅니다.",
        ),
        related=("알고리즘", "빅오"),
    ),
    Entry(
        term="빅오",
        aliases=("big-o", "시간복잡도", "복잡도"),
        category="cs",
        summary="빅오 표기법은 입력이 커질 때 알고리즘 비용이 증가하는 상한을 나타냅니다.",
        details=(
            "O(1) < O(log n) < O(n) < O(n log n) < O(n²) 순으로 증가합니다.",
            "상수와 낮은 차수는 무시하고 증가율만 봅니다.",
        ),
        example="O(n log n)",
        related=("알고리즘", "자료구조"),
    ),

    # ── 셸 확장 ──────────────────────────────────────────────────
    Entry(
        term="nano",
        aliases=("나노",),
        category="shell",
        summary="nano는 터미널에서 쓰는 간단한 텍스트 편집기입니다.",
        details=(
            "단축키가 화면 아래에 표시돼 처음 쓰기 쉽습니다.",
            "저장은 Ctrl+O, 종료는 Ctrl+X입니다.",
        ),
        example="nano notes.txt",
        related=("vim", "cat"),
    ),
    Entry(
        term="vim",
        aliases=("vi",),
        category="shell",
        summary="vim은 모드가 있는 강력한 터미널 편집기입니다.",
        details=(
            "일반 모드·입력 모드·명령 모드로 나뉘며 i로 입력, Esc로 복귀합니다.",
            "저장 후 종료는 :wq, 저장 없이 종료는 :q!입니다.",
        ),
        example="vim main.py",
        caution="처음에는 모드 개념이 헷갈려 저장 없이 빠져나가기 어렵습니다.",
        related=("nano", "man"),
    ),
    Entry(
        term="ln",
        aliases=("링크",),
        category="shell",
        summary="ln은 파일에 하드 링크나 심볼릭 링크를 만드는 명령입니다.",
        details=(
            "-s는 심볼릭(경로를 가리키는) 링크를 만듭니다.",
            "설정 파일을 여러 곳에서 공유할 때 자주 씁니다.",
        ),
        example="ln -s /opt/app/current /usr/local/bin/app",
        related=("cp", "mv"),
    ),
    Entry(
        term="whoami",
        aliases=(),
        category="shell",
        summary="whoami는 현재 명령을 실행하는 사용자 이름을 출력합니다.",
        details=("권한 문제를 진단할 때 가장 먼저 확인합니다.",),
        example="whoami",
        related=("id", "권한"),
    ),
    Entry(
        term="systemctl",
        aliases=(),
        category="shell",
        summary="systemctl은 systemd 기반 시스템에서 서비스(유닛)를 제어하는 명령입니다.",
        details=(
            "start/stop/restart/status/enable로 서비스를 다룹니다.",
            "enable은 부팅 시 자동 시작을 등록합니다.",
        ),
        example="systemctl status ssh",
        related=("ps", "로그"),
    ),
    Entry(
        term="apt",
        aliases=("apt-get",),
        category="shell",
        summary="apt는 데비안·우분투 계열에서 패키지를 설치·갱신하는 도구입니다.",
        details=(
            "install/remove/update/upgrade가 기본 동작입니다.",
            "저장소 목록이 오래되면 update 후 upgrade를 합니다.",
        ),
        example="sudo apt update && sudo apt install git",
        caution="설치·삭제는 sudo 권한이 필요하고 시스템 전역에 영향을 줍니다.",
        related=("배포판", "우분투"),
    ),
    Entry(
        term="top",
        aliases=("htop",),
        category="shell",
        summary="top은 실행 중인 프로세스와 자원 사용량을 실시간으로 보여줍니다.",
        details=(
            "CPU·메모리 사용률 기준으로 정렬되며 q로 종료합니다.",
            "htop은 색과 조작이 개선된 대안입니다.",
        ),
        example="top",
        related=("ps", "메모리"),
    ),
    Entry(
        term="캐시",
        aliases=("cache",),
        category="cs",
        summary="캐시는 느린 저장소의 값을 빠른 곳에 임시로 담아 재사용하는 계층입니다.",
        details=(
            "속도를 위해 정확성의 일부(최신성)를 포기합니다.",
            "캐시 무효화 전략이 성능과 정합성을 좌우합니다.",
        ),
        related=("메모리", "CPU"),
    ),
    Entry(
        term="메모리",
        aliases=("memory", "ram"),
        category="os",
        summary="메모리는 실행 중인 프로그램이 값을 담아 두는 주 저장 공간입니다.",
        details=(
            "프로세스마다 가상 주소 공간을 받고 커널이 실제 물리 메모리를 연결합니다.",
            "램이 부족하면 디스크를 빌려 쓰는 스왑이 일어나 느려집니다.",
        ),
        related=("CPU", "프로세스", "캐시"),
    ),
    Entry(
        term="cpu",
        aliases=("프로세서", "processor"),
        category="os",
        summary="CPU는 명령을 해석해 실행하는 중앙 처리 장치입니다.",
        details=(
            "여러 코어가 동시에 실행하며, 스케줄러가 프로세스·스레드를 배분합니다.",
        ),
        related=("메모리", "스레드", "프로세스"),
    ),
    Entry(
        term="스레드",
        aliases=("thread",),
        category="os",
        summary="스레드는 한 프로세스 안에서 실행되는 흐름의 단위입니다.",
        details=(
            "같은 주소 공간을 공유해 데이터를 빠르게 주고받지만, 동기화가 필요합니다.",
            "잘못 동기화하면 경쟁 상태나 교착 상태가 생깁니다.",
        ),
        related=("프로세스", "cpu"),
    ),
    Entry(
        term="해시",
        aliases=("hash", "해시함수"),
        category="cs",
        summary="해시는 임의 길이 입력을 고정 길이 값으로 바꾸는 함수입니다.",
        details=(
            "같은 입력은 항상 같은 값을 내고, 조금만 달라도 크게 달라집니다.",
            "무결성 확인·인덱싱·비밀번호 저장에 쓰이며, 단방향이라 되돌릴 수 없습니다.",
        ),
        example="sha256sum file.iso",
        related=("암호화", "json"),
    ),
    Entry(
        term="정규표현식",
        aliases=("regex", "정규식", "regexp"),
        category="cs",
        summary="정규표현식은 문자열의 패턴을 표현하는 간결한 문법입니다.",
        details=(
            "문자 클래스·반복·그룹으로 패턴을 조합합니다.",
            "grep·sed·언어 표준 라이브러리에서 두루 씁니다.",
        ),
        example="^\\d{3}-\\d{4}$",
        caution="복잡한 패턴은 가독성이 급격히 떨어지고 백트래킹으로 느려질 수 있습니다.",
        related=("grep", "sed", "awk"),
    ),
    Entry(
        term="로그",
        aliases=("log",),
        category="os",
        summary="로그는 프로그램이 실행 중 남기는 기록입니다.",
        details=(
            "무엇이 언제 왜 일어났는지 재구성하는 근거가 됩니다.",
            "레벨(debug/info/warn/error)로 중요도를 나눠 남깁니다.",
        ),
        example="tail -f /var/log/syslog",
        related=("tail", "systemctl"),
    ),
    Entry(
        term="dns",
        aliases=("도메인",),
        category="web",
        summary="DNS는 사람이 읽는 도메인 이름을 IP 주소로 바꾸는 시스템입니다.",
        details=(
            "계층적 이름 서버가 조회를 이어받아 최종 주소를 알려줍니다.",
            "결과는 TTL 동안 캐시됩니다.",
        ),
        example="dig example.com",
        related=("http", "캐시"),
    ),
    Entry(
        term="tcp",
        aliases=("tcp/ip",),
        category="web",
        summary="TCP는 데이터를 순서와 오류를 보정하며 안정적으로 전달하는 전송 프로토콜입니다.",
        details=(
            "연결을 맺고(핸드셰이크) 재전송과 흐름 제어로 신뢰성을 보장합니다.",
            "HTTP/HTTPS가 TCP 위에서 동작합니다.",
        ),
        related=("http", "dns"),
    ),

    # ── 깃 확장 ──────────────────────────────────────────────────
    Entry(
        term="git switch",
        aliases=("git checkout", "체크아웃"),
        category="vcs",
        summary="git switch는 브랜치를 바꾸거나 새로 만드는 명령입니다.",
        details=(
            "-c를 붙이면 새 브랜치를 만들며 이동합니다.",
            "예전에는 같은 일을 git checkout이 했습니다.",
        ),
        example="git switch -c feature/login",
        related=("브랜치", "git restore"),
    ),
    Entry(
        term="git restore",
        aliases=(),
        category="vcs",
        summary="git restore는 파일의 변경을 되돌리는 명령입니다.",
        details=(
            "기본은 작업 트리의 변경을 버리는 것이고, --staged는 스테이징을 되돌립니다.",
        ),
        example="git restore src/app.py",
        caution="작업 트리의 변경을 버리면 복구가 어렵습니다.",
        related=("git switch", "커밋"),
    ),
    Entry(
        term="git stash",
        aliases=("스태시",),
        category="vcs",
        summary="git stash는 아직 커밋하지 않은 변경을 잠시 치워 두는 기능입니다.",
        details=(
            "급히 다른 브랜치로 옮겨야 할 때 씁니다.",
            "pop으로 다시 꺼내고, list로 목록을 봅니다.",
        ),
        example="git stash push -m \"wip\"",
        related=("브랜치", "커밋"),
    ),
)

def _has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지 (조사 선택용)."""
    if not word:
        return False
    last = word[-1]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28 != 0
    return True


_JOSA = {
    "은/는": ("은", "는"),
    "이/가": ("이", "가"),
    "을/를": ("을", "를"),
    "과/와": ("과", "와"),
    "으로/로": ("으로", "로"),
}


def with_josa(word: str, pair: str) -> str:
    """조사 쌍('받침있을때/받침없을때')을 받침에 맞게 붙인다."""
    final, plain = _JOSA.get(pair, pair.split("/"))
    return word + (final if _has_batchim(word) else plain)


def particle_pair(word: str, pair: str) -> str:
    final, plain = _JOSA.get(pair, pair.split("/"))
    return final if _has_batchim(word) else plain
    final, plain = pair.split("/")
    return final if _has_batchim(word) else plain


def _normalize_alias(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower().strip()


_BY_ALIAS: dict[str, Entry] = {}
for _entry in _ENTRIES:
    for _alias in (_entry.term,) + _entry.aliases:
        _BY_ALIAS.setdefault(_normalize_alias(_alias), _entry)


_PARTICLE_SUFFIXES = ("은", "는", "이", "가", "을", "를", "의", "에", "에서",
                      "로", "으로", "와", "과", "도", "만", "란", "이란",
                      "라는", "이라는", "라고", "이라고")
_TAIL_WORDS = ("알려줘", "알려", "뭐야", "무엇", "뭔가요", "설명해줘", "설명",
               "뜻", "의미", "이란", "란", "인가요", "인가", "이야", "야",
               "좀", "해줘", "주세요", "please", "what", "is")


def _normalize_query(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[?!.,;:()\[\]{}'\"]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _candidates(text: str) -> list[str]:
    """질문에서 용어로 볼 수 있는 후보 문자열들을 뽑는다."""
    normalized = _normalize_query(text)
    if not normalized:
        return []
    out: list[str] = []
    words = normalized.split()

    def stem_of(word: str) -> str:
        stem = word
        for tail in _TAIL_WORDS:
            if stem.endswith(tail) and len(stem) > len(tail):
                stem = stem[: -len(tail)]
        for suffix in _PARTICLE_SUFFIXES:
            if stem.endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem

    stems = [stem_of(word) for word in words]
    # 원형과 줄기 양쪽에서 인접 1~3개 단어 묶음을 만든다.
    # ("큐브 스토어가" -> "큐브 스토어" 같은 띄어쓰기 용어를 잡기 위해서)
    for sequence in (words, stems):
        for size in (3, 2, 1):
            for start in range(0, len(sequence) - size + 1):
                out.append(" ".join(sequence[start:start + size]))
    out.extend(stems)
    return [c for c in out if c]


_LEARNED_CACHE: Optional[dict[str, Entry]] = None
_LEARNED_PATH: str = ""


def set_learned_store(path: str) -> None:
    """학습 지식 저장소 경로를 바꾼다 (테스트/설정용)."""
    global _LEARNED_CACHE, _LEARNED_PATH
    _LEARNED_PATH = str(path or "")
    _LEARNED_CACHE = None


def disable_learned() -> None:
    """학습 지식을 끈다 (결정적 테스트/격리용)."""
    set_learned_store("off")


_LEARNED_OFF = "off"

# 학습 지식이 가로채면 안 되는 작업 요청 표시
_CODE_TASK_MARKERS = ("짜줘", "짜 줘", "만들", "구현", "작성", "함수",
                      "코드", "프로그램", "리팩터", "테스트 코드", "def ")


def _learned_index() -> dict[str, Entry]:
    """문서에서 추출한 학습 개념들을 Entry 로 읽는다 (없으면 빈 dict)."""
    global _LEARNED_CACHE
    if _LEARNED_PATH == _LEARNED_OFF:
        return {}
    if _LEARNED_CACHE is not None:
        return _LEARNED_CACHE
    index: dict[str, Entry] = {}
    try:
        from knowledge import extract
        for concept in extract.learned_entries(_LEARNED_PATH):
            term = _normalize_alias(concept.term)
            if not term:
                continue
            index[term] = Entry(
                term=concept.term, summary=concept.summary,
                details=tuple(concept.details[:4]), category="learned",
                source=concept.origin)
    except Exception:
        index = {}
    _LEARNED_CACHE = index
    return index


def lookup(text: str, use_lexicon: bool = False) -> Optional[Entry]:
    """질문에서 지식 항목을 찾는다. 못 찾으면 None.

    ``use_lexicon`` 이 참이면, 직접 일치가 없을 때만 분포 통계 사전의
    유사어를 거쳐 한 번 더 찾는다 (별칭을 손으로 쓰지 않아도 표기
    차이를 이어준다).  직접 일치는 절대 흔들리지 않는다.
    """
    candidates = _candidates(text)
    # 긴 후보 우선 (구체적 용어가 일반 용어를 이기도록)
    for candidate in sorted(set(candidates), key=len, reverse=True):
        entry = _BY_ALIAS.get(candidate)
        if entry is not None:
            return entry
    # 문서에서 배운 개념 (직접 일치 다음 순위).
    # 단, 코드 작업 요청은 학습 지식이 가로채지 않는다.
    lowered_query = _normalize_query(text)
    if not any(marker in lowered_query for marker in _CODE_TASK_MARKERS):
        learned = _learned_index()
        if learned:
            for candidate in sorted(set(candidates), key=len, reverse=True):
                entry = learned.get(candidate)
                if entry is not None:
                    return entry
    if not use_lexicon:
        return None
    try:
        from knowledge import lexicon
        lex = lexicon.load()
    except Exception:
        lex = None
    if lex is None:
        return None
    for candidate in sorted(set(candidates), key=len, reverse=True):
        if candidate not in lex:
            continue
        try:
            synonyms = lex.synonyms(candidate, k=3, threshold=0.7)
        except Exception:
            return None
        for synonym in synonyms:
            for probe in (synonym, _strip_particles(synonym)):
                entry = _BY_ALIAS.get(probe)
                if entry is not None:
                    return entry
    return None


def _strip_particles(word: str) -> str:
    stem = str(word or "")
    for suffix in _PARTICLE_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return stem


def render(entry: Entry, include_related: bool = True) -> str:
    """항목을 한국어 설명으로 조립한다 (그대로 복사한 문장이 아니라 조립)."""
    parts = [entry.summary]
    for detail in entry.details:
        parts.append(f"- {detail}")
    if entry.example:
        parts.append("")
        parts.append("예시:")
        parts.append(f"  {entry.example}")
    if entry.caution:
        parts.append("")
        parts.append(f"주의: {entry.caution}")
    if include_related and entry.related:
        parts.append("")
        parts.append("관련: " + ", ".join(entry.related))
    parts.append("")
    parts.append("(출처: 오프라인 지식베이스)")
    return "\n".join(parts)


def answer(text: str) -> Optional[str]:
    """질문에 바로 답할 수 있으면 한국어 설명을 반환."""
    entry = lookup(text)
    if entry is None:
        return None
    return render(entry)


def web_context(text: str) -> Optional[str]:
    """웹 답변 앞에 붙일 한국어 맥락 (찾은 항목의 요약 + 핵심)."""
    entry = lookup(text)
    if entry is None:
        return None
    lines = [entry.summary]
    lines.extend(f"- {d}" for d in entry.details[:3])
    return "\n".join(lines)


# 자주 등장하는 영어 용어 -> 한국어.  완전한 번역기가 아니라 용어 대응표다.
_GLOSSARY = {
    "working directory": "작업 디렉터리",
    "command": "명령",
    "shell": "셸",
    "directory": "디렉터리",
    "file": "파일",
    "path": "경로",
    "standard output": "표준 출력",
    "standard input": "표준 입력",
    "reports": "출력합니다",
    "reports the": "출력합니다:",
    "print": "출력",
    "version control": "버전 관리",
    "repository": "저장소",
    "commit": "커밋",
    "branch": "브랜치",
    "operating system": "운영체제",
    "kernel": "커널",
    "programming language": "프로그래밍 언어",
    "function": "함수",
    "variable": "변수",
    "array": "배열",
    "list": "리스트",
    "object": "객체",
    "class": "클래스",
    "algorithm": "알고리즘",
    "data structure": "자료구조",
    "database": "데이터베이스",
    "server": "서버",
    "client": "클라이언트",
    "request": "요청",
    "response": "응답",
    "network": "네트워크",
    "security": "보안",
    "memory": "메모리",
    "process": "프로세스",
    "thread": "스레드",
    "compiler": "컴파일러",
    "interpreter": "인터프리터",
    "library": "라이브러리",
    "framework": "프레임워크",
    "interface": "인터페이스",
    "protocol": "프로토콜",
    "encryption": "암호화",
    "authentication": "인증",
    "package": "패키지",
    "install": "설치",
    "developer": "개발자",
}


def translate_terms(text: str) -> str:
    """문장에서 알려진 영어 용어만 한국어로 바꾼다 (기계 번역이 아님)."""
    result = str(text or "")
    for english, korean in sorted(_GLOSSARY.items(), key=lambda kv: -len(kv[0])):
        result = re.sub(re.escape(english), korean, result, flags=re.IGNORECASE)
    return result


def korean_ratio(text: str) -> float:
    """문자열에서 한글이 차지하는 비율."""
    letters = [c for c in str(text or "") if c.isalpha() or
               ("가" <= c <= "힣")]
    if not letters:
        return 0.0
    korean = sum(1 for c in letters if "가" <= c <= "힣")
    return korean / len(letters)


_BOILERPLATE = re.compile(
    r"(\[\d+\]|\[\s*(edit|citation needed|source)\s*\]|"
    r"위키\s*백과|위키피디아|wikipedia|from wikipedia)",
    re.IGNORECASE)


def strip_boilerplate(text: str) -> str:
    """위키 각주·편집 표시·출처 꼬리표를 제거한다."""
    value = _BOILERPLATE.sub(" ", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _first_sentences(text: str, limit: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return " ".join(p for p in parts[:limit] if p).strip()


def matched_terms(text: str) -> list[tuple[str, str]]:
    """문장에서 한국어 대응이 있는 영어 용어를 찾는다."""
    found: list[tuple[str, str]] = []
    lowered = str(text or "").lower()
    for english, korean in _GLOSSARY.items():
        if english in lowered:
            found.append((english, korean))
    return sorted(set(found), key=lambda item: -len(item[0]))


def korean_framing(text: str, topic: str = "") -> str:
    """영어 자료를 그대로 내보내지 않도록 한국어 틀을 만든다.

    자료가 이미 한국어면 빈 문자열을 돌려준다 (바꿀 것이 없음).
    완전한 기계 번역이 아니라, 아는 맥락 + 용어 대응 + 원문 발췌를
    분리해 보여주는 방식이다.
    """
    body = strip_boilerplate(text)
    if not body or korean_ratio(body) >= 0.35:
        return ""
    lines: list[str] = []
    context = web_context(topic) if topic else None
    if context:
        lines.append("먼저 아는 범위에서 정리하면:")
        lines.append(context)
        lines.append("")
    lines.append("찾은 자료가 영어라 그대로 옮기지 않았어요. "
                 "아래는 한국어 용어 대응과 영어 원문 핵심이에요.")
    terms = matched_terms(body)
    if terms:
        lines.append("")
        lines.append("용어 대응:")
        for english, korean in terms[:8]:
            lines.append(f"- {english} → {korean}")
    lines.append("")
    lines.append("영어 원문 핵심:")
    lines.append(_first_sentences(body, 2)[:280])
    return "\n".join(lines)


def fact_pool(text: str, use_lexicon: bool = True):
    """질문에 대한 **참인 사실 풀**을 만든다 (내용 샘플링용).

    반환: ``(entry, FactPool)`` 또는 None.
    """
    entry = lookup(text, use_lexicon=use_lexicon)
    if entry is None:
        return None
    return entry, pool_for_entry(entry)


def pool_for_entry(entry: Entry):
    """한 항목의 참인 사실 풀 (그래프 추론 포함)."""
    from knowledge.realizer import FactPool, Segment
    term = entry.term
    pool = FactPool()
    pool.must.append(Segment(
        kind="definition", text=entry.summary, terms=(term,),
        order=0, group="must", must=True))
    for detail in entry.details:
        pool.details.append(Segment("detail", f"- {detail}", (term,),
                                    order=10, group="detail"))
    if entry.example:
        pool.examples.append(Segment(
            "detail", f"- 사용 예: {entry.example}", (term,),
            order=20, group="example"))
    if entry.caution:
        pool.details.append(Segment(
            "detail", f"- 주의: {entry.caution}", (term,),
            order=15, group="caution"))
    for related in entry.related:
        pool.related.append(Segment(
            "detail", f"- 관련: {related}", (term,), order=40,
            group="related"))

    # 그래프 추론에서 참인 사실을 더 가져온다 (있으면)
    try:
        from knowledge import concept_graph
        graph = concept_graph.graph()
        for ancestor, path in graph.ancestors(term):
            if len(path) < 3:
                continue
            pool.inferred.append(Segment(
                "detail", f"- 추론: {with_josa(term, '은/는')} {ancestor}의 일종",
                (term,), order=30, group="inferred"))
        for prop, _ in graph.properties(term):
            pool.properties.append(Segment(
                "detail", f"- 속성: {prop}", (term,), order=35,
                group="property"))
    except Exception:
        pass
    return pool


@dataclass
class Match:
    """질문 해석 결과.  직접 일치가 아니면 신뢰도와 근거를 반드시 남긴다."""
    entry: Optional[Entry] = None
    match_type: str = "none"          # direct | synonym | none
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    ambiguous: bool = False
    alternatives: tuple[tuple[str, float], ...] = ()
    query: str = ""
    candidate: str = ""

    def needs_confirmation(self) -> bool:
        return self.ambiguous and self.entry is None


SYNONYM_CONFIRM = 0.75      # 분포 유사도: 이 위라야 후보로 본다
SYNONYM_AMBIGUOUS_MARGIN = 0.05
SURFACE_SIMILAR = 0.6       # 표기 유사도: 이 위여야 '같은 말'로 채택
                            # (함께 자주 나옴 != 같은 개념)


def _surface_ratio(left: str, right: str) -> float:
    """두 표기가 얼마나 닮았는지 (0~1).  오타·변형 판정에 쓴다."""
    import difflib
    return difflib.SequenceMatcher(None, str(left), str(right)).ratio()


def _spelling_candidate(query: str) -> Optional[Match]:
    """철자만 닮은 후보(오타·변형)를 찾는다.  뜻이 아니라 표기를 본다.

    한글 단어에만 적용한다 (영어 약어 오폭 방지).  경쟁 후보와 차이가
    충분할 때만 채택하고, 근거를 남긴다.
    """
    for candidate in sorted(set(_candidates(query)), key=len, reverse=True):
        if len(candidate) < 3 or candidate in _BY_ALIAS:
            continue
        if not all("가" <= ch <= "힣" for ch in candidate):
            continue
        scored: list[tuple[float, str]] = []
        for alias, entry in _BY_ALIAS.items():
            if len(alias) < 2 or not all("가" <= ch <= "힣" for ch in alias):
                continue
            ratio = _surface_ratio(candidate, alias)
            if ratio >= SURFACE_SIMILAR:
                scored.append((ratio, entry.term))
        if not scored:
            continue
        scored.sort(key=lambda item: -item[0])
        best_ratio, best_term = scored[0]
        rival = next((r for r, t in scored if t != best_term), 0.0)
        if best_ratio - rival < 0.10:
            continue        # 여러 표기와 비슷하면 단정하지 않는다
        entry = _BY_ALIAS.get(best_term) or next(
            (e for e in _ENTRIES if e.term == best_term), None)
        if entry is None:
            continue
        return Match(
            entry=entry, match_type="synonym",
            confidence=round(best_ratio, 3), candidate=candidate, query=query,
            evidence=(f"'{candidate}'는 '{best_term}'의 표기 변형으로 보임 "
                      f"(철자 유사도 {best_ratio:.2f})",))
    return None


def resolve(text: str, use_lexicon: bool = True) -> Match:
    """질문을 해석한다.

    - 정확 일치 -> 그대로 (신뢰도 1.0)
    - 유사 표기 -> **후보**로만 연결. 신뢰도·근거를 붙이고, 여러 개념과
      비슷하면 스스로 단정하지 않고 되묻는다.
    - 없음 -> none (호출자가 근거 부족을 알린다)
    """
    query = str(text or "")
    direct = lookup(query)
    if direct is not None:
        return Match(entry=direct, match_type="direct", confidence=1.0,
                     evidence=("정확 일치",), query=query)

    # 1) 표기 변형(오타) 후보 — 분포 통계와 무관하게, 철자가 닮았을 때만.
    spelling = _spelling_candidate(query)
    if spelling is not None:
        return spelling

    if not use_lexicon:
        return Match(query=query)
    try:
        from knowledge import lexicon
        lex = lexicon.load()
    except Exception:
        lex = None
    if lex is None:
        return Match(query=query)

    scored: list[tuple[float, str, str, str]] = []   # (score, cand, syn, term)
    for candidate in sorted(set(_candidates(query)), key=len, reverse=True):
        if candidate not in lex:
            continue
        try:
            synonyms = lex.neighbors(candidate, k=5)
        except Exception:
            continue
        for synonym, score in synonyms:
            if score < 0.6:
                continue
            entry = _BY_ALIAS.get(synonym) or _BY_ALIAS.get(
                _strip_particles(synonym))
            if entry is not None:
                scored.append((float(score), candidate, synonym, entry.term))
    if not scored:
        return Match(query=query)

    scored.sort(key=lambda item: -item[0])
    best_score, best_cand, best_syn, best_term = scored[0]
    distinct: list[tuple[str, float]] = []
    for score, _, _, term in scored:
        if all(term != existing for existing, _ in distinct):
            distinct.append((term, round(score, 3)))

    rival = next((s for t, s in distinct if t != best_term), 0.0)
    surface = _surface_ratio(best_cand, best_syn)

    # 함께 자주 나온다는 것만으로 같은 개념이라 단정하지 않는다.
    # 표기 자체가 비슷할 때(오타·변형)만 '후보 채택'한다.
    if (best_score >= SYNONYM_CONFIRM
            and best_score - rival >= SYNONYM_AMBIGUOUS_MARGIN
            and surface >= SURFACE_SIMILAR):
        entry = _BY_ALIAS.get(best_syn) or _BY_ALIAS.get(
            _strip_particles(best_syn))
        return Match(
            entry=entry, match_type="synonym", confidence=round(best_score, 3),
            evidence=(f"'{best_cand}' ~ '{best_syn}' "
                      f"표기 유사도 {surface:.2f} · 분포 유사도 "
                      f"{best_score:.2f}",),
            query=query)

    # 의미는 가깝지만 표기가 다른 경우: 동일시하지 않고 확인을 구한다.
    related = best_score >= SYNONYM_CONFIRM
    return Match(match_type="none", confidence=round(best_score, 3),
                 ambiguous=True, alternatives=tuple(distinct[:3]),
                 candidate=best_cand,
                 evidence=((f"'{best_cand}'는 '{best_syn}'와 의미가 가깝지만 "
                            f"표기가 달라 동일시하지 않음") if related else
                           (f"'{best_cand}'가 여러 개념과 비슷함",)),
                 query=query)


def answer_entry(entry: Entry, seed=None, note: str = "",
                 opener: bool = False, closer: bool = False) -> str:
    """항목 하나를 사실 고정·표현 가변으로 실현한다."""
    from knowledge.realizer import Realizer
    pool = pool_for_entry(entry)
    realizer = Realizer(seed=seed)
    body = realizer.realize_pool(
        pool, salt=str(seed if seed is not None else entry.term),
        opener=opener, closer=closer, key_terms=(entry.term,))
    if note:
        body = note + "\n\n" + body
    return body + "\n\n(출처: 오프라인 지식베이스)"


def answer_with_confidence(text: str, seed=None,
                           use_lexicon: bool = True) -> dict:
    """신뢰도·근거를 붙여 답한다.  모호하면 단정하지 않고 되묻는다."""
    match = resolve(text, use_lexicon=use_lexicon)
    if match.match_type == "direct" and match.entry is not None:
        return {"status": "direct", "confidence": 1.0, "evidence": [],
                "alternatives": [],
                "response": answer_varied(text, seed=seed)}
    if match.match_type == "synonym" and match.entry is not None:
        note = ("정확히 아는 표현은 아니지만, 가까운 후보로 보고 답할게요 "
                "(근거: " + "; ".join(match.evidence) + ")")
        return {"status": "synonym", "confidence": match.confidence,
                "evidence": list(match.evidence), "alternatives": [],
                "response": answer_entry(match.entry, seed=seed, note=note)}
    if match.ambiguous:
        subject = match.candidate or match.query
        if len(match.alternatives) == 1:
            term, score = match.alternatives[0]
            return {"status": "ambiguous", "confidence": match.confidence,
                    "evidence": list(match.evidence),
                    "alternatives": [term],
                    "response": (
                        f"'{subject}'는 제 지식에 없고, '{term}'" +
                        particle_pair(str(term), "과/와") + " 의미가 "
                        f"가까워 보여요 (분포 유사도 {score:.2f}). "
                        f"'{term}'를 말하는 건가요? 다른 뜻이면 알려주세요.")}
        options = ", ".join(f"{term}({score:.2f})"
                            for term, score in match.alternatives)
        return {"status": "ambiguous", "confidence": match.confidence,
                "evidence": list(match.evidence),
                "alternatives": [term for term, _ in match.alternatives],
                "response": (f"'{subject}'가 여러 개념과 비슷해서 바로 "
                             f"단정하지 않았어요: {options}\n"
                             "어느 쪽을 말하는지 알려주시면 정확히 "
                             "답할게요.")}
    return {"status": "unknown", "confidence": 0.0, "evidence": [],
            "alternatives": [], "response": ""}


INSUFFICIENT_KO = ("이 주제는 제 오프라인 지식에 없어서 확실한 근거가 "
                   "없어요. 추측으로 답하지 않겠습니다.")


def unified_answer(text: str, seed=None, use_lexicon: bool = True) -> dict:
    """지식 답변의 단일 통로.

    순서: ① 그래프로 추론 가능한가 → ② 지식베이스 직접 일치 →
    ③ 유사/모호 후보(신뢰도·근거) → ④ 없으면 unknown (근거 부족).
    그래프가 아는 개념(예: '운영체제')은 통계 이웃으로 답하지 않고
    분류·종류·용도로 답한다.
    """
    graph_response = None
    try:
        from knowledge import concept_graph
        graph_response = concept_graph.answer(text)
    except Exception:
        graph_response = None

    info = answer_with_confidence(text, seed=seed, use_lexicon=use_lexicon)
    status = str(info.get("status") or "unknown")
    match = resolve(text, use_lexicon=use_lexicon)
    term = match.entry.term if match.entry is not None else ""
    if not term:
        try:
            from knowledge import concept_graph
            term = concept_graph.first_term(text) or ""
        except Exception:
            term = ""

    # 구조 질문(종류/용도/공통/관계/예-아니오)은 그래프 답을 우선한다.
    structural = ("종류", "용도", "공통", "관계", "차이", "인가", "맞나",
                  "뭘 할 수 있", "어디에 쓰")
    prefer_graph = bool(graph_response) and any(
        marker in str(text) for marker in structural)

    if status == "direct" and not prefer_graph:
        response = graph_response or info.get("response") or ""
        return {"status": "direct", "response": response, "confidence": 1.0,
                "term": term,
                "evidence": ["정확 일치"] + list(info.get("evidence") or []),
                "source": "kb+graph" if graph_response else "kb"}
    if graph_response:
        return {"status": "graph", "response": graph_response,
                "confidence": 0.8, "term": term,
                "evidence": list(info.get("evidence") or [])
                            + ["개념 그래프 추론"], "source": "graph"}
    if status in ("synonym", "ambiguous"):
        return {"status": status, "response": info.get("response") or "",
                "confidence": info.get("confidence", 0.0), "term": term,
                "evidence": list(info.get("evidence") or []), "source": "kb"}
    return {"status": "unknown", "response": "", "confidence": 0.0,
            "term": term, "evidence": [], "source": "none"}


# ---- 되묻기/후속 질문용 표면 판정 (모델 없이 규칙으로) ----

_AFFIRMATIVE = ("응", "어", "네", "예", "맞아", "맞아요", "맞습니다", "ㅇㅇ",
                "그래", "그렇지", "당연", "yes", "y", "ok", "오케이", "콜")
_NEGATIVE = ("아니", "아니야", "아뇨", "노", "no", "n", "아니에요", "틀려")
_FOLLOWUP = ("그럼", "그럼?", "그건", "그거", "그것", "왜", "왜?", "더",
             "더 자세히", "자세히", "이어서", "그다음", "그러면", "계속",
             "더 알려줘", "더 보여줘")


def looks_affirmative(text: str) -> bool:
    value = re.sub(r"[\s?!.,~]+", "", _normalize_query(text))
    return value in _AFFIRMATIVE if value else False


def looks_negative(text: str) -> bool:
    value = re.sub(r"[\s?!.,~]+", "", _normalize_query(text))
    return value in _NEGATIVE if value else False


def looks_followup(text: str) -> bool:
    value = re.sub(r"[\s?!.,~]+", "", _normalize_query(text))
    if not value:
        return False
    if looks_affirmative(text) or looks_negative(text):
        return False       # 단순 긍정/부정은 후속 질문이 아니다
    if value in {re.sub(r"[\s?!.,~]+", "", item) for item in _FOLLOWUP}:
        return True
    return len(value) <= 4


def answer_varied(text: str, seed=None, use_lexicon: bool = True,
                  opener: bool = False, closer: bool = False) -> Optional[str]:
    """같은 질문에도 매번 조금씩 다른 답 (내용·표현 모두 가변, 사실 불변).

    seed 를 주면 재현 가능, 주지 않으면 매번 다르다.
    """
    built = fact_pool(text, use_lexicon=use_lexicon)
    if built is None:
        return None
    entry, pool = built
    from knowledge.realizer import Realizer
    realizer = Realizer(seed=seed)
    body = realizer.realize_pool(
        pool, salt=str(seed if seed is not None else text),
        opener=opener, closer=closer, key_terms=(entry.term,))
    return body + "\n\n(출처: 오프라인 지식베이스)"


def categories() -> list[str]:
    return sorted({entry.category for entry in _ENTRIES})


def entries() -> tuple[Entry, ...]:
    return _ENTRIES


__all__ = ["Entry", "lookup", "render", "answer", "answer_varied",
           "fact_pool", "web_context", "translate_terms", "korean_ratio",
           "categories", "entries", "strip_boilerplate", "korean_framing"]
