#
# Black-Scholes-Merton 모형을 사용한 European Call Option 가격 결정
# 베가 계산 함수와 implied vol 추정 함수 포함
#

# Black-Scholes-Merton Model

def bsm_call_value(S0, K, T, r, sigma):
    '''Black-Scholes-Merton Model을 활용한 European Call Option 가격 결정 공식

    인수
    ===
    S0: 초기 주가 혹은 지수(float)
    K: 행사가(float)
    T: 만기까지 남은 시간(연 단위, float)
    r: 고정 무위험 단기 이자율(float)
    sigma: 변동성 파라미터(float)

    반환값
    ===
    value: European Call Option의 현재 가격(float)
    '''

    from math import log, sqrt, exp
    from scipy import stats

    S0=float(S0)
    d1 = (log(S0/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0/K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    # stats.norm.cdf: normal distribution의 cumulative distribution function 계산

    return value

# Vega Calculation Function

def bsm_vega(S0, K, T, r, sigma):
    '''Black-Scholes-Merton Model을 활용한 European Call Option의 Vega 계산

    인수
    ===
    S0: 초기 주가 혹은 지수(float)
    K: 행사가(float)
    T: 만기까지 남은 시간(연 단위, float)
    r: 고정 무위험 단기 이자율(float)
    sigma: 변동성 파라미터(float)

    반환값
    ===
    vega: Black-Scholes-Merton 공식을 변동성에 대해 1차 미분한 값. 베가 (float)
    '''   
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)

# Implied Volatility 계산 함수

def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    '''Black-Scholes-Merton Model을 활용한 European Call Option의 Implied Vol 계산

    인수
    ===
    S0: 초기 주가 혹은 지수(float)
    K: 행사가(float)
    T: 만기까지 남은 시간(연 단위, float)
    r: 고정 무위험 단기 이자율(float)
    C0: 주어진 European Call Option의 가격(float)
    sigma_est: 변동성 파라미터 초기 추정치(float)
    it: 반복 계산 횟수(integer)

    반환값
    ===
    sigma_est : 수치적으로 측정한 내재변동성 (float)
    '''       

    for i in range(it):
        sigma_est -= ((bsm_call_value(S0,K,T,r,sigma_est)-C0) / bsm_vega(S0,K,T,r,sigma_est))

    return sigma_est
    


