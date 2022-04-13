# RBF Model introduction

RBF Model 的全名是 Radial basis function network, 簡稱 RBFN，中文叫做 "放射狀基底函數網路"

<img src="https://1.bp.blogspot.com/-hjh4bhL9iaQ/WZe1M_QmlbI/AAAAAAAAEIc/N4LnOTV-KLoP2w8NLBXfHQIs3NRuRRDigCLcBGAs/s640/ML-T-14-1.png" width="400px">

RBF Network 的數學公式可以表示成 $F(x)= \sum^{J}_{j=1}w_jRBF(x, \mu_j) + b$
- $w_j$ 代表第 j 個 RBF 神經元道類神經元的權重
- $RBF(x, \mu_j)$ 代表第 j 的 RBF 神經元輸出值的基底函數
- b 代表可調整的偏移量

---
RBF Network 的演算法通常會分成兩個階段的學習過程
首先是
