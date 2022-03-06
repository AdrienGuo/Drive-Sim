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
#### ```def vehicle_move(x, y, theta, empty)```
  
目的：這裡描述模擬車的運動方程式  

方程式如下

![](http://latex.codecogs.com/svg.latex?x(t&plus;1)&space;=&space;x(t)&space;&plus;&space;cos(\phi(t)&space;&plus;&space;\theta(t))&space;&plus;&space;sin(\theta(t))&space;sin(\phi(t)))

![](http://latex.codecogs.com/svg.latex?y(t&plus;1)&space;=&space;y(t)&space;&plus;&space;sin(\phi(t)&space;&plus;&space;\theta(t))&space;-&space;sin(\theta(t))&space;cos(\phi(t)))

![](http://latex.codecogs.com/svg.latex?\phi(t&plus;1)&space;=&space;\phi(t)&space;-&space;arcsin(\frac{2sin(\theta(t))}{b}))

x: 車子所在的 x 座標  
y: 車子所在的 y 座標  
theta: 車子要轉的角度(θ)  
empty: 車子與水平線的夾角(ϕ)  

#### ```def distance_cal(vehicle_x, vehicle_y, empty)```

目的：計算車子的三個方向到牆壁的距離  
vehicle_x: 車子所在的 x 座標  
vehicle_y: 車子所在的 y 座標  
empty: 車子與水平線的夾角  
這裡會計算車子和前、右、左三個方向的距離  

計算方式分成五個步驟  

1. ```def find_vehicle_front_all_wall_point(line)```  
車子這三個方向的直線可以算出一個直線方程式，將方程式和所有牆壁的方程式會算出很多交點。

2. ```def find_point_on_wall(points)```  
但這些交點並不是真的都在那些牆壁上(有些是在那些牆壁的延長線上)，因此這個步驟要去判斷這些點是不是都在真的牆壁上。

3. ```def find_correct_direction(points, vehicle_x, vehicle_y, empty)```  
接著要分析車子現在方向是朝向上面還是朝向下面，以及是朝向正右方(水平夾角=0)，或是朝向正左方(水平夾角=180)。

4. ```def find_closest_point(points, vehicle_x, vehicle_y)```  
這個方向仍然可能有多個交點，因此要找離車子最近的那一個，這個交點即是正確的那個交點。

5. ```def cal_dist(point, vehicle_x, vehicle_y)```  
將這個交點和車子的中心去算距離，回傳算出來的距離以及交點。

### 2.2 RBF Model
因規定不能使用現成的套件 ex: pytorch, tensorflow 等  
因此這裡自己做一個 RBF Model  

> 這裡就不說明原理，只說明我怎麼做的

RBF Model 示意圖:  
<img src="https://visualstudiomagazine.com/articles/2020/03/19/~/media/ECG/visualstudiomagazine/Images/2020/03/radial_basis_train_3.asxh" width="500px">  
> 我們的輸出只會有一個，因為是要預測 "方向盤要轉多少角度"，是屬於 regression 的任務

RBF 共分成兩個步驟，第一步要先分群，第二步才是真的訓練模型  
分群的方式我採用的是 K-means  

#### ```def kmeans(X, k, max_iters)```
目的：將資料分成 k 群，算出每一群的中心點與標準差。  
X: 所有的資料點  
k: 分成 k 群  
max_iters: 最大迭代次數  

#### ```class RBFNN```
- ```RBFNN.train(lr, n_epochs)```  
目的：訓練 RBF 模型  
learning rate: 0.0001  
n_epochs: 500  
基底函數:
<img src="http://latex.codecogs.com/svg.latex?\phi_j(x)&space;=&space;exp(-&space;\frac{|x-m_j|^2}{2\sigma^{2}_{j}})"> (高斯分布)  
利用 LMS演算法來更新模型參數

- ```RBFNN.predict(X_pred)```  
目的: 輸入要預測的資料，並輸出預測的角度  
X_pred: 被預測的資料(前、右、左三個方向的距離)  
output: 預測的角度(θ)  


使用上方訓練完的 RBF 模型的參數，來預測我們輸入的資料。


## 3. Result
自動車可以順利地到達終點，且不論是在起跑線上的哪一點皆可。

- 右轉  
當偵測到右邊的距離突然變大，則車子會開始進行右轉彎。
<img src="https://i.imgur.com/JFMTsuX.png" width="300px">

- 左轉  
當偵測到左邊的距離突然變大，則車子會開始進行左轉彎。
<img src="https://i.imgur.com/m9BTofv.png" width="300px">

當車子偵測到有某一邊的數值突然變大時，則會開始朝那個方向進行轉彎。全部觀察下來，車子會朝距離大的那個方向進行轉彎，而距離相對小的時候會進行比較小幅度的轉彎，距離大的時候會進行相對較大幅度的轉彎。


## 4. Analysis

下圖是三種距離與角度的對照圖  
<img src="https://i.imgur.com/7cvF7QX.png">  
橫軸分別是前方、右方、左方距離，縱軸是角度(θ)  
紅色點是我的模型跑出來的結果，藍色點是原始的訓練資料  

由圖片可以看出，不論是哪一個方向，相對於助教的資料，我的資料都**沒有辦法產生大角度的輸出** (最多大概就到 25 度)，也就代表訓練出來的自動車沒有辦法做大角度的轉彎 (實測出來也是如此)。
會造成這樣的原因，我的猜想是因為大角度的資料比較少，這樣導致在第一步做 K-means 的時候，和大角度相關的資料中心點個數就會比較少，也因此後續訓練的時候，很容易就被平均掉了，所以如果要避免產生這種現象，應該要**多增加資料量**。

不過即使沒有大角度的資料，我的自動車一樣可以順利地到達終點，而它就會比較早開始轉彎，轉彎的幅度也會比較小。
