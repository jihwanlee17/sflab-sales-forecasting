# 👕 F-MINT (Fashion Multimodal Internal Trend)

본 연구는 과거 판매 이력이 없는 패션 신제품의 12주 수요 예측을 위해 기존 [GTM(Google Trends Multimodal)-Transformer](https://github.com/HumaticsLAB/GTM-Transformer) 아키텍처를 변형한 **F-MINT 프레임워크**를 제안합니다.

---

## 🎯 Research Background

의류 기업의 상품기획자(MD)는 매 시즌 수만 개의 신제품 생산 수량을 결정해야 하는 난제에 직면합니다. 기존의 주관적 감이나 단순 통계 의존 방식은 인기 상품의 조기 품절 혹은 비인기 상품의 과잉 생산(재고 리스크)을 야기합니다. 본 연구는 다중 모달 정보를 통합하여 데이터 기반의 정밀한 수요 예측 가이드라인을 제공함으로써 재고 효율성을 극대화하고자 하였습니다.

---

## 🚀 Core Contributions

1. **Multimodal Architecture**: 제품 이미지(Fashion-CLIP), 상세 텍스트 설명, 메타데이터, 출시 시점 정보를 통합적으로 반영하는 인코더-디코더 구조를 설계했습니다.

2. **Internal Trend Signal (ITS)**: 외부 검색어 데이터의 희소성(Sparsity) 문제를 해결하기 위해, 네이버 트렌드와 같은 외부 지표 대신 Brand & Fabric 등 핵심 메타데이터 조합의 **주차별 판매량 중앙값(Median)** 을 트렌드 신호로 활용하도록 아키텍처를 변형했습니다.

3. **Advanced Modeling**: 기존 연구의 아키텍처를 기반으로 하되, 데이터의 특성에 최적화된 내부 통계 정보 피드백 구조를 도입하여 신제품 예측의 정밀도를 높였습니다.

---

## 📏 Evaluation Metrics: Adjusted SMAPE (AD_SMAPE)

본 연구는 모델의 성능을 정밀하게 평가하기 위해 **Adjusted SMAPE**를 주요 지표로 사용합니다.

### 수학적 정의

$$AD\_SMAPE = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|y_t - \hat{y}_t|}{y_t + \hat{y}_t}$$

표준 SMAPE 공식에서 분모의 평균 연산($/2$)을 생략(또는 표준 SMAPE 값에 $0.5$를 곱함)하여 0\~100% 스케일로 산출합니다.

### 지표 도입 배경

- **실무적 직관성**: 표준 SMAPE(0\~200% 범위)와 달리 0\~100% 스케일을 적용하여, 현업 MD가 "예측 오차가 실제 대비 몇 %인가"를 한눈에 파악할 수 있도록 설계했습니다.
- **Scale Invariance**: 판매 규모가 상이한 다양한 의류 아이템을 동일한 비중으로 공정하게 평가하기 위해 비율 기반의 지표를 채택했습니다.

---

## ⚙️ Experimental Setup

모델의 학습 및 검증을 위해 다음과 같은 환경을 설정하였습니다.

| 항목 | 설정 |
|------|------|
| Optimizer | AdaBelief |
| Epochs | 10 |
| Batch Size | 128 |
| Dataset Split | Train : Validation : Test = 8 : 1 : 1 |

### Dataset
[TBH Global사](https://www.tbhglobal.co.kr/MAIN_W.php)의 의류 브랜드인 'MindBridge' 및 'JucyJudy'의 13가지(청바지, 스커트, 블라우스 등) 의류 데이터 총 19,498개를 연구에 사용하였습니다.
| 구분 | 샘플 수 |
|------|---------|
| Total | 19,498개 |
| Train | 15,598개 |
| Validation | 1,950개 |
| Test | 1,950개 |

---

## 📊 Performance Results

제안된 F-MINT 모델은 외부 트렌드 데이터를 사용하는 기존 GTM 모델 대비 정량 지표에서 우수한 성과를 기록했습니다.

| 모델 및 데이터 세팅 | AD_SMAPE (↓) |
|---|---|
| GTM (Google Trends) | 0.454 |
| **F-MINT (Brand & Fabric)** | **0.427** |

- **성능 개선**: Brand & Fabric 메타데이터 조합과 AdaBelief 옵티마이저를 결합한 F-MINT 모델이 AD_SMAPE **0.427**을 기록하며 기존 GTM-Transformer 성능 대비 **2.7%** 만큼 유의미하게 개선되었습니다.
- **분석**: 불확실한 외부 데이터 의존성을 탈피하고, 내부 메타데이터 기반의 통계적 트렌드를 학습에 활용함으로써 예측의 안정성을 확보했습니다.
