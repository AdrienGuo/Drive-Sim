# RBF Model introduction

RBF Model 的全名是 Radial basis function network, 簡稱 RBFN，中文叫做 "放射狀基底函數網路"

<img src="https://1.bp.blogspot.com/-hjh4bhL9iaQ/WZe1M_QmlbI/AAAAAAAAEIc/N4LnOTV-KLoP2w8NLBXfHQIs3NRuRRDigCLcBGAs/s640/ML-T-14-1.png" width="400px">

<img src="https://i.imgur.com/Sg6OQHg.png" title="source: imgur.com" width="450px">

RBF Network 的數學公式可以表示成 $F(x)= Output(\sum^{M}_{m=1}w_mRBF(x, \mu_m) + b)$
- $w_m$ 代表第 m 個 RBF 神經元道類神經元的權重
- $RBF(x, \mu_m)$ 代表第 m 的 RBF 神經元輸出值的基底函數
- b 代表可調整的偏移量

---
RBF Network 的演算法通常會分成兩個階段的學習過程  
1. 選擇 RBF 群 (非監督式學習)
2. RBF 群的權重學習 (監督式學習)

### 1. 選擇 RBF 群 (非監督式學習)
RBF 全名是 Radial basis function，中文叫做 "放射狀基底函數"，而這個函數最常是以 Gaussian Function 來表示 (在我的專案裡面也是)  
RBF Function: $RBF(x, \mu_m) = exp(- \frac{ \lvert x-\mu_m \rvert ^2}{2 {\sigma_m}^2})$  

來詳細分解內部的步驟  
總共會選 M 組 $RBF(x, \mu_m)$，$\mu_m$ 就是這組 RBF 的中心點，資料點 x 離 $\mu_m$ 越近，受到的影響就會越大 (Gaussian Function 的關係)

接著是討論，要怎麼決定每個 $RBF(x, \mu_m)$ 的中心點 $\mu_m$，最簡單的作法是直接將每一個資料點 x 都當作一個中心點，這叫做 Full RBF Network，但這樣的作法很容易會 overfitting，而且運算量會超大，因此通常也不會這麼做。

另外可以使用歸納的演算法來找出 $\mu_m$，像是專案裡面我所使用的 K-Means，先找出 k 個 $\mu_m$ 再去做計算與更新，這樣的 RBF Network 稱為 K Nearest Neighbor RBF Network。

#### K-Means
首先第一步是要決定要有幾個「中心點」，每一個中心點就是一個群，假設有 k 個  
接著從資料中隨機選取 k 個當作初始的中心點  
根據每筆資料與這 k 個中心點的歐基里德距離來歸類，會分給歐基里德距離最小的那一類，也就是最近的那一類  
分類完後，將每一類的每個資料做平均，找出這個群新的中心點，然後再根據這些新的中心點，再歸類一次  
循環多次，一直到收斂為止

K-Means Algorithm
```python=
initialize k central point: say, as k randomly chosen from X
alternating optimization of E: repeatedly
    optimize S (group): each x "optimally partitioned" using its closest µ
    optimize µ: each µ "optimally computed" as consensus within S
    until converge
```
converge 的條件: S(群) 內的點不再改變，或是 E(中心點的更動誤差) 的變化小於某個閥值


### 2. RBF 群的權重學習 (監督式學習)
這個步驟是要來學習 $w_m$  
這裡採用 LMS Algorithm 來更新 $w_m$  
誤差的公式為: $$E(n) = \frac{1}{2} (y_n - F(x_n))^2$$  
接著各別對要更新的參數做偏微分  
$$w_m(n+1) = w_m(n) - \alpha \frac{\partial E(n)}{\partial w_m(n)} = w_m(n) + \alpha (y_n - F(x_n)) RBF_m(x_n, \mu_m)$$
$$b(n+1) = b(n) - \alpha \frac{\partial E(n)}{\partial b(n)} = b(n) + \alpha (y_n - F(x_n))$$
一直不斷地重複這個步驟，就可以不斷地更新參數，看你要更新幾次 (epoch)

因為專案的最後是要輸出一個角度，屬於回歸分析，最後的輸出只需要一個數值就好。  


