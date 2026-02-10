# 설치 가이드 (Installation Guide)

## 필요한 패키지 설치

하이퍼볼릭 하이퍼그래프 모델을 실행하기 위해 다음 패키지들이 필요합니다.

### 방법 1: requirements.txt 사용 (권장)

```bash
pip install -r requirements_hyperbolic.txt
```

### 방법 2: 개별 설치

```bash
# Core packages
pip install torch torch-geometric
pip install geoopt
pip install torch-scatter ogb
pip install fasttext numpy tqdm scikit-learn
```

### 방법 3: PyTorch Geometric 설치 (CUDA 버전)

CUDA를 사용하는 경우, PyTorch Geometric을 CUDA 버전에 맞게 설치해야 합니다:

```bash
# CUDA 11.8
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CUDA 12.1
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

# CPU only
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## 패키지별 설명

- **torch**: PyTorch 딥러닝 프레임워크
- **torch-geometric**: 그래프 신경망 라이브러리
- **geoopt**: 하이퍼볼릭 기하학 연산 라이브러리
- **torch-scatter**: 분산 연산 유틸리티
- **ogb**: Open Graph Benchmark (SMILES 변환용)
- **fasttext**: FastText 임베딩 (선택적이지만 권장)
- **numpy**: 수치 연산
- **tqdm**: 진행 표시줄
- **scikit-learn**: 머신러닝 유틸리티

## 문제 해결

### geoopt 설치 오류

```bash
pip install geoopt --upgrade
```

### torch-scatter 설치 오류 (선택적 패키지)

`torch-scatter`는 **선택적 패키지**입니다. 설치가 실패해도 모델은 정상 작동합니다.

**설치 시도 방법:**

```bash
# 방법 1: 자동 설치 스크립트 사용 (권장)
bash install_packages.sh

# 방법 2: 수동 설치
# 먼저 PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# CPU 버전 (macOS/Linux)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# CUDA 버전 (CUDA 11.8 예시)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CUDA 버전 (CUDA 12.1 예시)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

**참고**: `torch-scatter` 설치가 실패해도 모델은 정상 작동합니다. PyTorch Geometric이 내부적으로 처리합니다.

### FastText 설치 오류

```bash
# macOS/Linux
pip install fasttext

# Windows에서는 추가 설정이 필요할 수 있습니다
# https://fasttext.cc/docs/en/support.html 참고
```

## 확인

설치가 완료되었는지 확인:

```bash
python -c "import torch; import torch_geometric; import geoopt; print('All packages installed successfully!')"
```

