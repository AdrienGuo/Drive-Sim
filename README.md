# Drive-Sim
自駕車模擬。

利用深度學習 RBF 模型，讓自駕車根據過往的行駛資料來分析與學習，要在什麼時後進行轉彎，順利走出道路。

---
## 1. Project Description

#### 道路示意圖
<img src="https://i.imgur.com/YPbvKNk.png" width="350px">

圓形是車子<br>
左上角顯示是前右左3個距離的方向<br>
紅色區域是終點<br>

#### 軌道說明
軌道檔案在 "data/軌道座標點.txt"

第一行為車體中心起始的（x,y,φdegree)  
二，三行標示出終點區域位置  
第二行為區域左上角（x,y）  
第三行為區域右下角（x,y）  
第四行以後為軌道邊界節點（x,y）  
直到最後一行  
最後一行與第四行數值相同  
形成一個封閉的跑道  
軌道於起點線右下角為(-6,-3);左下角為(6,-3)  
起點線為(-6,0) -> (6,0)  

#### 訓練資料
檔案有兩種 "data/train4dAll.txt", "data/train4dAll.txt"  
- train4dAll.txt 格式: 前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)
- train6dAll.txt 格式: X 座標、Y 座標、前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)

---

## 2. Code Explanation
我分成 <u>車子與牆壁距離計算</u> 和 <u>RBF Model</u> 兩部分說明
### 2.1 計算車子到牆壁的距離
##### def vehicle_move(x, y, theta, empty)  
目的：這裡描述模擬車的運動方程式  

方程式如下  $$x(t+1) = x(t) + cos(\phi(t) + \theta(t)) + sin(\theta(t)) sin(\phi(t))$$ $$y(t+1) = y(t) + sin(\phi(t) + \theta(t)) - sin(\theta(t)) cos(\phi(t))$$ $$\phi(t+1) = \phi(t) - arcsin(\frac{2sin(\theta(t))}{b})$$ x: 車子所在的 x 座標  
y: 車子所在的 y 座標  
theta: 車子要轉的角度(θ)  
empty: 車子與水平線的夾角(ϕ)  

##### def distance_cal(vehicle_x, vehicle_y, empty)
目的：計算車子的三個方向到牆壁的距離  
vehicle_x: 車子所在的 x 座標  
vehicle_y: 車子所在的 y 座標  
empty: 車子與水平線的夾角  
這裡會計算車子和前、右、左三個方向的距離  
計算方式分成五個步驟  
