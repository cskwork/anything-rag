#!/bin/bash

# 실시간 임베딩 모니터링 스크립트
# 사용법: ./monitor_embedding.sh [명령어]

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 기본 명령어 (인수가 없으면 문서 로딩)
COMMAND=${1:-"python main.py load"}

print_header "실시간 임베딩 모니터링"
print_info "실행 명령어: $COMMAND"
print_info "Ctrl+C로 중단하세요"
echo ""

# 가상환경 활성화 시도
if [[ -d "$SCRIPT_DIR/.venv" && -z "$VIRTUAL_ENV" ]]; then
    print_info "가상환경 활성화 중..."
    source "$SCRIPT_DIR/.venv/bin/activate" 2>/dev/null || {
        echo "가상환경 활성화 실패, 전역 환경 사용"
    }
fi

# 명령어 실행하면서 임베딩 관련 로그만 실시간 필터링
cd "$SCRIPT_DIR"
$COMMAND 2>&1 | while IFS= read -r line; do
    # 타임스탬프 추출 (있다면)
    timestamp=""
    if [[ "$line" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}[[:space:]]+[0-9]{2}:[0-9]{2}:[0-9]{2} ]]; then
        timestamp=$(echo "$line" | cut -d'|' -f1 | xargs)
        line=$(echo "$line" | cut -d'|' -f3- | xargs)
    elif [[ "$line" =~ [0-9]{2}:[0-9]{2}:[0-9]{2} ]]; then
        timestamp=$(echo "$line" | grep -o '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]')
        line=$(echo "$line" | sed 's/[0-9][0-9]:[0-9][0-9]:[0-9][0-9][^|]*|[^|]*|//')
    fi
    
    # 임베딩 관련 로그만 표시
    if [[ "$line" =~ (📚|📄|🔄|✅|❌|📝|🕐|⏱️|🎉|📊|📈|⚡|💡|📏) ]] ||
       [[ "$line" =~ "임베딩.*시작|임베딩.*완료|임베딩.*중|청크.*완료|진행률" ]] ||
       [[ "$line" =~ "문서.*개.*시작|문서.*개.*완료|청크.*개.*처리" ]] ||
       [[ "$line" =~ "소요시간|남은.*시간|예상.*시간|평균.*시간" ]] ||
       [[ "$line" =~ "ERROR|WARN|실패|오류" ]]; then
        
        # 시간 표시 (있다면)
        if [[ -n "$timestamp" ]]; then
            echo -e "${PURPLE}[$timestamp]${NC} $line"
        else
            echo "$line"
        fi
    fi
done

echo ""
print_info "모니터링 완료"