# 專題 `topic`


──────────────███████──███████  
──────────████▓▓▓▓▓▓████░░░░░██  
────────██▓▓▓▓▓▓▓▓▓▓▓▓██░░░░░░██  
──────██▓▓▓▓▓▓████████████░░░░██  
────██▓▓▓▓▓▓████████████████░██  
────██▓▓████░░░░░░░░░░░░██████  
──████████░░░░░░██░░██░░██▓▓▓▓██  
──██░░████░░░░░░██░░██░░██▓▓▓▓██  
██░░░░██████░░░░░░░░░░░░░░██▓▓██  
██░░░░░░██░░░░██░░░░░░░░░░██▓▓██  
──██░░░░░░░░░███████░░░░██████  
────████░░░░░░░███████████▓▓██  
──────██████░░░░░░░░░░██▓▓▓▓██  
────██▓▓▓▓██████████████▓▓██  
──██▓▓▓▓▓▓▓▓████░░░░░░████  
████▓▓▓▓▓▓▓▓██░░░░░░░░░░██  
████▓▓▓▓▓▓▓▓██░░░░░░░░░░██  
██████▓▓▓▓▓▓▓▓██░░░░░░████████  
──██████▓▓▓▓▓▓████████████████  
────██████████████████████▓▓▓▓██  
──██▓▓▓▓████████████████▓▓▓▓▓▓██  
████▓▓██████████████████▓▓▓▓▓▓██  
██▓▓▓▓██████████████████▓▓▓▓▓▓██  
██▓▓▓▓██████████──────██▓▓▓▓████  
██▓▓▓▓████──────────────██████  
──████───────────────────────────  



[偵測座標](https://hackmd.io/3NVpFWgkSwy2dmf8QNXrbg)


## python3
### 目前版本 3.6.6

opentest.py
```python=
import random
import webbrowser
import pyautogui
import pydirectinput
import time

def change():
    pyautogui.keyDown("Alt")
    pyautogui.keyDown("Tab")
    pyautogui.keyUp("Tab")
    pyautogui.keyUp("Alt")

mario_url = "https://supermario-game.com"
webbrowser.open(mario_url, new=0, autoraise=True)

time.sleep(5)
# change()
pyautogui.moveTo(968,420)
pyautogui.scroll(-500)
pyautogui.click()

goal_steps = 30
score_requirement = -19
intial_games = 2
keys = ['d w', 'a', 'd']


def model_data_preparation():
        training_data = []
        accepted_scores = []
        for game_index in range(intial_games):
            score = 0
            game_memory = []
            for step_index in range(goal_steps):
                ransom = random.randrange(0, 3)
                if len(keys[ransom]) != 1:
                    move = keys[ransom].split()
                    pydirectinput.keyDown(move[0])
                    pydirectinput.keyDown(move[1])
                    time.sleep(0.35)
                    pydirectinput.keyUp(move[1])
                    pydirectinput.keyUp(move[0])
                    print(move, end='\n')
                    reward = 1
                    score += reward
                else:
                    pydirectinput.keyDown(keys[ransom])
                    print(keys[ransom], end='\n')
                    time.sleep(0.1)
                    pydirectinput.keyUp(keys[ransom])
                    reward = -1
                    score += reward

            if score >= score_requirement:
                accepted_scores.append(score)

        print(accepted_scores)

model_data_preparation()

print("done.")
```


## Main

### [_det.py](https://hackmd.io/uuDoD4hzTIushlW_53UqKw)

### server.cpp
```cpp=
#include <iostream>
#include <WS2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

using namespace std;

void timer(int sec) {
	Sleep(sec * 1);
}
void down(int vk) {
	keybd_event(vk, 0, 0, 0);
}
void up(int vk) {
	keybd_event(vk, 0, KEYEVENTF_KEYUP, 0);
}

void press(char key) {
	switch (key) {
	case 'd':
		//up(65);
		down(68);
		timer(50);
		//timer(2);
		//up(65);
		//up(87);
		break;
	case 'a':
		up(68);
		down(65);
		timer(2);
		//timer(2);
		up(68);
		up(87);
		break;
	case 'w':
		down(87);
		timer(410);
		up(87);
		break;
	case 'r':
		down(39);
		timer(50);
		down(38);
		timer(400);
		up(38);
	}
}



int main() {
	//initialze winsock
	WSADATA wsData;
	WORD ver = MAKEWORD(2, 2);

	int wsOk = WSAStartup(ver, &wsData);
	if (wsOk != 0) {
		cerr << "Can't initialize winsock! Quitting" << endl;
		return 0;
	}

	//create a socket
	SOCKET listening = socket(AF_INET, SOCK_STREAM, 0);
	if (listening == INVALID_SOCKET) {
		cerr << "Can't create a socket! Quitting" << endl;
		return 0;
	}

	//bind the socket to an ip address and port
	sockaddr_in hint;
	hint.sin_family = AF_INET;
	hint.sin_port = htons(54000);
	hint.sin_addr.S_un.S_addr = INADDR_ANY;

	bind(listening, (sockaddr*)&hint, sizeof(hint));

	//tell winsock the socket is for listening
	listen(listening, SOMAXCONN);

	//wait for a connection
	sockaddr_in client;
	int clientSize = sizeof(client);

	SOCKET clientSocket = accept(listening, (sockaddr*)&client, &clientSize);
	char host[NI_MAXHOST];
	char service[NI_MAXHOST];

	ZeroMemory(host, NI_MAXHOST);
	ZeroMemory(service, NI_MAXHOST);

	if (getnameinfo((sockaddr*)&client, sizeof(client), host, NI_MAXHOST, service, NI_MAXHOST, 0) == 0) {
		cout << host << "connected on port " << service << endl;
	}
	else {
		inet_ntop(AF_INET, &client.sin_addr, host, NI_MAXHOST);
		cout << host << "connected on port " << ntohs(client.sin_port) << endl;
	}

	//close listening socket
	closesocket(listening);

	//while loop: accept and echo message back to client
	char buf[4096];
	while (true) {
		ZeroMemory(buf, 4096);
		//wait for client to send data
		int bytesReveived = recv(clientSocket, buf, 4096, 0);
		if (bytesReveived == SOCKET_ERROR) {
			cerr << "Error in recv(). Quitting" << endl;
			break;
		}
		if (bytesReveived == 0) {
			cout << "Client disconnected " << endl;
			break;
		}
		//echo message back to client
		send(clientSocket, buf, bytesReveived + 1, 0);
		cout << "input  :" << buf[0] << endl;
		press(buf[0]);
	}
	//close the socket
	closesocket(clientSocket);

	//shutdown winsock
	WSACleanup();
}
```

### client.py
```python=
import socket
import time

import keyboard
import msvcrt
from msvcrt import getch
# import sys
# import threading
# import tkinter
# from tkinter import ttk
# import time
# import threading
# from threading import Timer
# from xmlrpc.client import SERVER_ERROR

HEADER = 64
PORT = 54000
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
# SERVER = "192.168.1.101"
# SERVER = "192.168.50.128"
# SERVER = "10.100.0.161"
# SERVER = "127.0.0.1"
# SERVER = "172.20.10.9"
SERVER = "192.168.0.104"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    msg = str(msg)
    message = msg.encode(FORMAT)
    # msg_length = len(message)
    # send_length = str(msg_length).encode(FORMAT)
    # send_length += b' ' * (HEADER - len(send_length))
    # client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

# for line in sys.stdin:
#     send(line)

# def createTimer():
#     t = threading.Timer(1,repeat)
#     t.start()
#
# def repeat():
#     createTimer()

# while True:
#     time.sleep(0.1)
#     if keyboard.read_key() == 'w':
#         send('w')
#     elif keyboard.read_key() == 's':
#         send('s')
#     elif keyboard.read_key() == 'a':
#         send('a')
#     elif keyboard.read_key() == 'd':
#         send('d')

while True:
    time.sleep(0.1)
    if keyboard.is_pressed('w'):
        send('w')
    elif keyboard.is_pressed('s'):
        send('s')
    elif keyboard.is_pressed('a'):
        send('a')
    elif keyboard.is_pressed('d'):
        send('d')
    elif keyboard.is_pressed('8'):
        print('Quit!')
        break
```

### move.py
```python=
import random
import pydirectinput
import time

time.sleep(3)

pydirectinput.click()
keys = ['w','a','s','d']
for i in range(50):
    ransom = random.randint(0,3)
    pydirectinput.keyDown(keys[ransom])
    time.sleep(0.25)
    pydirectinput.keyUp(keys[ransom])

print("done.")
```

### opentest.py
```python=
import random
import webbrowser
import pyautogui
import pydirectinput
import time

def change():
    pyautogui.keyDown("Alt")
    pyautogui.keyDown("Tab")
    pyautogui.keyUp("Tab")
    pyautogui.keyUp("Alt")

mario_url = "https://supermario-game.com"
webbrowser.open(mario_url, new=0, autoraise=True)

time.sleep(5)
# change()
pyautogui.moveTo(968,420)
pyautogui.scroll(-500)
pyautogui.click()
keys = ['d w', 'a']

for i in range(100):
    ransom = random.randint(0, len(keys) - 1)
    if len(keys[ransom]) != 1:
        move = keys[ransom].split()
        pydirectinput.keyDown(move[0])
        pydirectinput.keyDown(move[1])
        time.sleep(0.35)
        pydirectinput.keyUp(move[1])
        pydirectinput.keyUp(move[0])
        print(move, end='\n')
    else:
        pydirectinput.keyDown(keys[ransom])
        print(keys[ransom], end='\n')
        time.sleep(0.1)
        pydirectinput.keyUp(keys[ransom])

print("done.")

```


### AI

#### colab (train,test,detect)

`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`

#### anaconda 3/30

- download anaconda
        - 

- open Anaconda Prompt

**`conda create --name yolov5`**

**`activate yolov5`**

**`conda install pytorchvision torchaudio cudatoolkit=11.3 -c pytorch`**


`(base) C:\Users\UserName>activate yolov5`

`(yolov5) C:\Users\UserName>`

`(yolov5) C:\Users\UserName>conda install pytorchvision torchaudio cudatoolkit=11.3 -c pytorch`


download -> [Github](https://github.com/ultralytics/yolov5)

and move to UserName/

`(yolov5) C:\Users\UserName>cd yolov5`

`(yolov5) C:\Users\UserName\yolov5>`

`(yolov5) C:\Users\UserName\yolov5>pip install -r requirements.txt`

`(yolov5) C:\Users\kuras1\yolov5>python detect.py --source 0` 
> 開始載東西 like yolov5.pt 就可以打開webcam

`(yolov5) C:\Users\kuras1\yolov5>python detect.py --source 0 --weights your.pt` 
> `--source 1` 外接 webcam 

`python detect.py --help`
```
optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS [WEIGHTS ...]
                        model path(s)
  --source SOURCE       file/dir/URL/glob, 0 for webcam
  --data DATA           (optional) dataset.yaml path
  --imgsz IMGSZ [IMGSZ ...], --img IMGSZ [IMGSZ ...], --img-size IMGSZ [IMGSZ ...]
                        inference size h,w
  --conf-thres CONF_THRES
                        confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     maximum detections per image
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --view-img            show results
  --save-txt            save results to *.txt
  --save-conf           save confidences in --save-txt labels
  --save-crop           save cropped prediction boxes
  --nosave              do not save images/videos
  --classes CLASSES [CLASSES ...]
                        filter by class: --classes 0, or --classes 0 2 3
  --agnostic-nms        class-agnostic NMS
  --augment             augmented inference
  --visualize           visualize features
  --update              update all models
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
  --exist-ok            existing project/name ok, do not increment
  --line-thickness LINE_THICKNESS
                        bounding box thickness (pixels)
  --hide-labels         hide labels
  --hide-conf           hide confidences
  --half                use FP16 half-precision inference
  --dnn                 use OpenCV DNN for ONNX inference
```


###### import
```
pip install psutil
pip install pydirectinput
pip install keyboard
pip install torch
# pip install cv2 #
pip install opencv-contrib-python
pip install IPthon
# pip install PIL #
pip install Pillow
# pip install torchvision #
# pip install pytorchvision torchaudio cudatoolkit=11.3 #
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
pip install matplotlib
pip install seaborn
pip install tensorflow-estimator==2.1
```
### RL




## 討論進度

### ...
### 3/10
### 3/17


### 3/24



### 3/31
- **準確率提高**
- `frame`框住遊戲大小 (640x480) **(x,y,w,h) in game** 
- 跑 `random` 設定 `reward`
- 計算 `distance`
- 先做絕對再做相對位置

>iPhone XR 搭載了一塊1792 x 828 像素解析度的LCD 螢幕
>Oppo Reno 5Z 螢幕解析度: 2400 x 1080 螢幕類型: AMOLED
### 4/28
- **定義reward很重要**
- 可以先定義一個爛的reward再慢慢優化
- reward暫定:跳過障礙加分(X大於他就加分)
- 往右走加分往左走扣分
- 跳過梯子加分
- 最後加分的reward:最短時間碰到旗子時間總長
- 只有右鍵跑越久reward越高

### 5/6
- 紀錄**state、reward、action**
- 盡可能短時間(幾周)完成

### 5/19
- 往上0往右1往左-1
- m510 爛筆電 確認機器
- 加入更多reward多討論

### 5/25
- 確認input shape偵測到的東西*2
- 瑪里歐相對位置
- out shape=幾個動作
- 碰到旗子done

### 5/29
- 25個input_shape下次秀文件
- 然後要設定視窗外的預設偵測
- 一個在前一個在後  一定一正一負
~~mario 1~~

|標籤|數量|
|-|-|
|mushroom|2|
|turtle|1|
|water pipe|2|
|ladder|2|
|brick|2|
|question|2|
|Vulnerability|2|
|end|1|

input shape= 15
output = 3 (右跳，右，左)
發現yolo座標y越上面越小越左上越小

### 6/2

抓出每一個位置並且print :heavy_check_mark: 
預設每個的位置 :heavy_exclamation_mark: 
**少量data** 避免收集花費太多時間 :-1: 
正負分成兩群都抓最近 (x軸) :question: 
資料OK 傳給 **server** 
action **previously**
reward **now**

###### 資料結構

儲存資料  用list or map **建議用np.array**



## 進度

### 3月初開始製作
### 3/ 完成辨識
### 3/10 完成socket
### 3/24 完成遠端控制
### 4/26 完成辨識率提高
### 4/27 命令行demo
### 5/4 `完成move.py`
### 5/5 `完成opentest.py`
### 5/15 `改寫detect.py`
### 5/19 完成goal 1 !!
### 5/25 優化中
### 7/26 取得狀態and記錄
### 8/9 初步想法:Input_shape=抓到的東西的n(抓到的數量)+起來
### 8/24 上傳 [dataset](https://www.kaggle.com/datasets/imcyj123/smbcyimtopic)
>補充
`gym_super_mario_bros -e <the environment ID to play> -m <'human' or 'random'>`
`gym_super_mario_bros -e SuperMarioBros-v0 -m human`
### 11/9 文件完成
### 11/28 完成專案


## 出現問題

###### 5/26 NotReadableError: Could not start video source
###### 5/25 win11 anaconda 無法 start up yolov5(已解決)
###### 5/24 `ERROR: torch has an invalid wheel, .dist-info directory not found`
解決方法 
> pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

###### 5/16 ![](~~https://i.imgur.com/LrnR1tx.png~~)(已解決)
###### 5/2 train `nes_py` or `online game`
###### 5/1 pyautogui don't work!! (已解決)
解決方法
> `使用 pydirectinput 取代 pyautogui`
`把 pyautogui.keyDown 換成 pydirectinput.keyDown`
###### 4/28 LoomieLive會導致鏡頭無法正常運作!!! (已處理)
###### 4/26 `pip install nes-py` failed building (已解決)
###### 8/24 tensor(-9.5000) 轉整數int (已解決)
解決方法
> `x = tensor(-9.5000)  output = x[7:-1]`
###### dataset 未更新



# goal

1. 可以亂跑,random就好 :heavy_check_mark: 
2. 以偵測取得的狀態去跑，記錄下來 :heavy_check_mark: 
3. 拿記錄檔去訓練 :heavy_check_mark: 
4. 以偵測取得狀態 餵給訓練好的模組 產出action :heavy_check_mark: 
5. 再記錄下來擇優去訓練更新的模組 :heavy_check_mark: 
6. 照這個流程持續優化 :heavy_check_mark: 
7. yolo增強準確率 :heavy_check_mark: 
8. 讓它動起來(training) :heavy_check_mark: 
9. mario 0+(x2-x1)/2 x 座標 :heavy_check_mark: 
10. mario y1 y 座標 :heavy_check_mark: 



```
rd = open('lastest.txt', 'r')
line = rd.readline()
cnt = 0
while line:
    line = line[1:-2]
    print(line)
    lst = [int(x) for x in line.split(", ")]
    action = lst[-1]
    lst = lst[:-2]
    print(lst)
    print(action)
    _lst = np.array(lst)
    _lst = _lst.reshape(-1, len(_lst))
    #print(_lst)
    #print(lst[:-1])  ## input
    line = rd.readline()
    #x = lst[:-1]
    
    if action == 1:
        choose = [0,1]
    elif action == 0:
        choose = [1,0]
    _choose = np.array(choose)
    _choose = _choose.reshape(-1,len(_choose))
    
    #_action = np.array(action)
    x = _lst
    y = _choose
    """
    y = np.empty(28)
    y.fill(action)
    _choose = y.reshape(-1,len(y))
    """
    print(x)
    print(y)

    try:
        model = load_model('right_model.h5')
    except Exception as e:
        print(e)
    try:
        model.fit(x, y, epochs=1)
    except Exception as e:
        print(e)
    try:
        oput = np.argmax(model.predict(x))
        print("oput "+str(oput))
        print("y " + str(y))
    except:
        continue
    try:
      cnt += 1
      if oput == action:
          model.save('right_model.h5')
          print("save done")
          print("第",cnt,"次save成功")
    except Exception as e:
        raise e
rd.close()
```
