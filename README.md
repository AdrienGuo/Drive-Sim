# Drive-Sim
自駕車模擬。

利用深度學習 RBF 模型，讓自駕車根據過往的行駛資料來分析與學習，要在什麼時後進行轉彎，順利走出道路。

---

道路示意圖如下

<img src="https://i.imgur.com/YPbvKNk.png" width="350px">

圓形是車子<br>
左上角顯示是前右左3個距離的方向<br>
紅色區域是終點<br>

### 軌道說明
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

### 訓練資料
檔案有兩種 "data/train4dAll.txt", "data/train4dAll.txt"  
- train4dAll.txt 格式: 前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)
- train6dAll.txt 格式: X 座標、Y 座標、前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)


