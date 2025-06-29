#!/bin/bash

# 임베딩 테스트 스위트 - 모든 임베딩 관련 테스트를 실행하는 스크립트
# 사용법: ./test_embedding_suite.sh [옵션]
# 옵션: --verbose (상세 출력), --quick (빠른 테스트만)

set -e  # 오류 발생시 스크립트 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 설정
VERBOSE=false
QUICK=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_LOG="$SCRIPT_DIR/test_results.log"

# 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --quick|-q)
            QUICK=true
            shift
            ;;
        --help|-h)
            echo "임베딩 테스트 스위트"
            echo ""
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --verbose, -v    상세 출력 모드"
            echo "  --quick, -q      빠른 테스트만 실행"
            echo "  --help, -h       이 도움말 표시"
            echo ""
            echo "테스트 순서:"
            echo "  1. 환경 검사"
            echo "  2. Ollama 연결 테스트"
            echo "  3. 임베딩 기본 기능 테스트"
            echo "  4. RAG 워크플로우 테스트 (quick 모드에서는 제외)"
            echo "  5. 벡터 저장소 진단"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말을 보려면 $0 --help를 실행하세요"
            exit 1
            ;;
    esac
done

# 유틸리티 함수들
print_header() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

print_step() {
    echo -e "${BLUE}[단계 $1/$2]${NC} $3"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

# 로그 함수
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$TEST_LOG"
}

# 환경 검사 함수
check_environment() {
    print_step 1 5 "환경 검사 중..."
    
    # Python 버전 확인
    if ! command -v python3 &> /dev/null; then
        print_error "Python3가 설치되어 있지 않습니다"
        return 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_info "Python 버전: $PYTHON_VERSION"
    
    # 가상환경 확인
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "가상환경 활성화됨: $(basename $VIRTUAL_ENV)"
    else
        print_warning "가상환경이 활성화되지 않음"
        
        # .venv 디렉토리가 있는지 확인
        if [[ -d "$SCRIPT_DIR/.venv" ]]; then
            print_info "가상환경 활성화를 시도합니다..."
            source "$SCRIPT_DIR/.venv/bin/activate" 2>/dev/null || {
                print_error "가상환경 활성화 실패"
                return 1
            }
            print_success "가상환경 활성화 완료"
        else
            print_warning "가상환경이 설정되지 않음. 전역 Python 환경 사용"
        fi
    fi
    
    # 필수 패키지 확인
    print_info "필수 패키지 확인 중..."
    local missing_packages=()
    
    for package in "asyncio" "pathlib" "loguru" "numpy"; do
        if ! python3 -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_error "누락된 패키지: ${missing_packages[*]}"
        print_info "패키지 설치: pip install ${missing_packages[*]}"
        return 1
    fi
    
    # 테스트 스크립트 존재 확인
    local test_scripts=("test_embedding.py" "diagnose_vector_storage.py")
    if [[ "$QUICK" == false ]]; then
        test_scripts+=("test_rag_workflow.py")
    fi
    
    for script in "${test_scripts[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            print_error "테스트 스크립트를 찾을 수 없습니다: $script"
            return 1
        fi
    done
    
    print_success "환경 검사 완료"
    return 0
}

# Ollama 연결 테스트
test_ollama_connection() {
    print_step 2 5 "Ollama 연결 테스트 중..."
    
    # Ollama 서버 연결 확인
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        print_error "Ollama 서버에 연결할 수 없습니다"
        print_info "Ollama를 시작하세요: ollama serve"
        return 1
    fi
    
    local ollama_version=$(curl -s http://localhost:11434/api/version | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])" 2>/dev/null)
    print_success "Ollama 서버 연결됨 (버전: $ollama_version)"
    
    # 필요한 모델 확인
    local models_response=$(curl -s http://localhost:11434/api/tags)
    local has_embedding_model=false
    local has_chat_model=false
    
    if echo "$models_response" | grep -q "bge-m3"; then
        has_embedding_model=true
        print_success "임베딩 모델 (bge-m3) 설치됨"
    else
        print_warning "임베딩 모델 (bge-m3)이 설치되지 않음"
        print_info "설치 명령: ollama pull bge-m3:latest"
    fi
    
    if echo "$models_response" | grep -q "gemma3"; then
        has_chat_model=true
        print_success "채팅 모델 (gemma3) 설치됨"
    else
        print_warning "채팅 모델 (gemma3)이 설치되지 않음"
        print_info "설치 명령: ollama pull gemma3:latest"
    fi
    
    if [[ "$has_embedding_model" == false ]]; then
        print_error "임베딩 모델이 없어 테스트를 계속할 수 없습니다"
        return 1
    fi
    
    return 0
}

# 개별 테스트 실행 함수
run_test() {
    local test_name="$1"
    local script_path="$2"
    local description="$3"
    
    echo ""
    print_info "테스트 시작: $description"
    
    local start_time=$(date +%s)
    local output
    local exit_code
    
    if [[ "$VERBOSE" == true ]]; then
        # 상세 모드: 실시간 출력
        python3 "$script_path"
        exit_code=$?
    else
        # 간단 모드: 실시간 출력하되 진행률만 필터링
        python3 "$script_path" 2>&1 | while IFS= read -r line; do
            # 진행률, 문서 처리, 청크 임베딩 관련 로그는 항상 표시
            if [[ "$line" =~ (📚|📄|🔄|✅|❌|📝|🕐|⏱️|🎉|📊|📈|⚡) ]] || 
               [[ "$line" =~ "임베딩.*시작|임베딩.*완료|청크.*완료|진행률" ]] ||
               [[ "$line" =~ "ERROR|WARN|FAIL" ]]; then
                echo "$line"
            # DEBUG나 INFO 중에서도 중요한 것들만
            elif [[ "$line" =~ "INFO.*문서.*개|INFO.*청크.*개|INFO.*차원" ]]; then
                echo "$line"
            fi
        done
        exit_code=${PIPESTATUS[0]}  # python3 명령의 실제 exit code
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "$test_name 테스트 통과 (${duration}초)"
        log_message "SUCCESS: $test_name completed in ${duration}s"
        return 0
    else
        print_error "$test_name 테스트 실패 (${duration}초)"
        log_message "FAILED: $test_name failed in ${duration}s"
        
        if [[ "$VERBOSE" == false && -n "$output" ]]; then
            echo "오류 출력:"
            echo "$output" | tail -10  # 마지막 10줄만 표시
        fi
        return 1
    fi
}

# 메인 테스트 실행
main() {
    local start_time=$(date +%s)
    local total_tests=5
    local passed_tests=0
    local failed_tests=0
    
    if [[ "$QUICK" == true ]]; then
        total_tests=4
    fi
    
    print_header "임베딩 테스트 스위트 시작"
    print_info "테스트 모드: $([ "$QUICK" == true ] && echo "빠른 테스트" || echo "전체 테스트")"
    print_info "상세 출력: $([ "$VERBOSE" == true ] && echo "활성화" || echo "비활성화")"
    print_info "로그 파일: $TEST_LOG"
    
    # 로그 파일 초기화
    echo "=== 임베딩 테스트 스위트 시작 ===" > "$TEST_LOG"
    log_message "Test mode: $([ "$QUICK" == true ] && echo "quick" || echo "full")"
    log_message "Verbose: $([ "$VERBOSE" == true ] && echo "enabled" || echo "disabled")"
    
    # 테스트 실행
    if check_environment; then
        ((passed_tests++))
    else
        ((failed_tests++))
        print_error "환경 검사 실패. 테스트를 중단합니다."
        exit 1
    fi
    
    if test_ollama_connection; then
        ((passed_tests++))
    else
        ((failed_tests++))
        print_error "Ollama 연결 실패. 테스트를 중단합니다."
        exit 1
    fi
    
    # 3. 임베딩 기본 기능 테스트
    print_step 3 $total_tests "임베딩 기본 기능 테스트 중..."
    if run_test "임베딩 기본 기능" "$SCRIPT_DIR/test_embedding.py" "임베딩 생성 및 검증"; then
        ((passed_tests++))
    else
        ((failed_tests++))
    fi
    
    # 4. RAG 워크플로우 테스트 (빠른 모드가 아닐 때만)
    if [[ "$QUICK" == false ]]; then
        print_step 4 $total_tests "RAG 워크플로우 테스트 중..."
        if run_test "RAG 워크플로우" "$SCRIPT_DIR/test_rag_workflow.py" "문서 로딩부터 쿼리까지 전체 과정"; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
    fi
    
    # 5. 벡터 저장소 진단
    print_step $total_tests $total_tests "벡터 저장소 진단 중..."
    if run_test "벡터 저장소 진단" "$SCRIPT_DIR/diagnose_vector_storage.py" "저장소 상태 및 일관성 검사"; then
        ((passed_tests++))
    else
        ((failed_tests++))
    fi
    
    # 결과 요약
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    echo ""
    print_header "테스트 결과 요약"
    
    echo -e "${BLUE}총 실행 시간:${NC} ${total_duration}초"
    echo -e "${GREEN}통과한 테스트:${NC} $passed_tests/$total_tests"
    echo -e "${RED}실패한 테스트:${NC} $failed_tests/$total_tests"
    
    log_message "Test summary: $passed_tests/$total_tests passed, total time: ${total_duration}s"
    
    if [[ $failed_tests -eq 0 ]]; then
        print_success "모든 테스트가 통과했습니다! 🎉"
        echo ""
        print_info "임베딩 시스템이 정상적으로 작동하고 있습니다."
        
        if [[ "$QUICK" == true ]]; then
            print_info "전체 테스트를 실행하려면: $0 (--quick 옵션 제거)"
        fi
        
        log_message "All tests passed successfully"
        exit 0
    else
        print_error "일부 테스트가 실패했습니다."
        echo ""
        print_info "해결 방법:"
        echo "  • 상세 출력으로 재실행: $0 --verbose"
        echo "  • 로그 파일 확인: cat $TEST_LOG"
        echo "  • 개별 테스트 실행: python3 test_embedding.py"
        echo "  • 문서를 다시 로딩: python3 main.py load"
        
        log_message "Some tests failed"
        exit 1
    fi
}

# 스크립트 실행
main "$@"