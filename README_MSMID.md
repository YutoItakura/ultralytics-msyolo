# YOLOv8 マルチスペクトル ミドル融合 (yolov8-msmid)

[Ultralytics](https://github.com/ultralytics/ultralytics) v8.4.129 のフォークです。マルチスペクトル TIFF 用の
**デュアルバックボーン + ミドル融合 YOLOv8** を追加しています。

```
input ─┬─ MSSplit(A: G,R)  ── YOLOv8 backbone A ── P3a, P4a, P5a ─┐
       └─ MSSplit(B: 残り) ── YOLOv8 backbone B ── P3b, P4b, P5b ─┤
              MSFusion = Concat + 1x1 Conv (P3 / P4 / P5)  ◄──────┘
                         └── YOLOv8 PAN-FPN neck + Detect head
```

| TIFF のバンド順 | `channels` | Stream A | Stream B |
|---|---|---|---|
| (G, R, NIR) | 3 | G, R | NIR |
| (G, R, RE) | 3 | G, R | RE |
| (G, R, NIR, RE) | 4 | G, R | NIR, RE |

## インストール（Google Colab）

```python
!pip install -q git+https://github.com/YutoItakura/ultralytics-msyolo.git
```

## 使い方

```python
from ultralytics import YOLO

model = YOLO("yolov8m-msmid.pt")   # COCO で学習済みの YOLOv8m 重みを両ストリームに移植したもの（自動ダウンロード）
# model = YOLO("yolov8m-msmid.yaml")  # スクラッチから学習（n/s/m/l/x はファイル名で選択）

model.train(data="data.yaml", epochs=100, imgsz=1024)
```

`data.yaml` の例:

```yaml
path: /content/dataset
train: images/train
val: images/val
channels: 4        # 4 = (G,R,NIR,RE), 3 = (G,R,NIR) or (G,R,RE)
names:
  0: mosaic
  1: brown-spot
```

## 追加・変更点

| ファイル | 内容 |
|---|---|
| `ultralytics/nn/modules/multispectral.py` | `MSSplit`（ストリームごとのバンド選択）、`MSFusion`（Concat + 1x1 Conv） |
| `ultralytics/cfg/models/v8/yolov8-msmid.yaml` | デュアルバックボーン + ミドル融合のモデル定義 |
| `ultralytics/utils/multispectral.py` | `yolov8{n,s,m,l,x}.pt` → `yolov8{scale}-msmid.pt` の重み変換 |
| `ultralytics/nn/tasks.py` | `parse_model` に MSSplit/MSFusion を登録。チャネル数が違うときにストリームのステム重みを適応させる処理を追加 |
| `ultralytics/utils/downloads.py` | `yolov8*-msmid.pt` を `weights/` から自動取得。無ければその場で変換する |
| `ultralytics/nn/modules/block.py` | `SPPF(cv1_act=...)` オプションを追加（元の YOLOv8 SPPF と同じ動作にするため） |
| `weights/yolov8m-msmid.pt` | 変換済みの重み（4ch, COCO 80 クラス） |

### 重みの移植方法
- 両方のバックボーンを YOLOv8 のバックボーン（層 0〜9）で初期化します。
- ステム Conv の RGB 重みを、出力フィルタごとの入力チャネル和が保たれるように再配分します。
  - Stream A: `w_G = W_G + W_B/2`, `w_R = W_R + W_B/2`
  - Stream B: `w_k = (W_R + W_G + W_B) / n`
- `MSFusion` の 1x1 Conv は 2 ストリームの平均（`[0.5·I | 0.5·I]`）で初期化します。
- Neck/Head（層 10〜22）はインデックスを +16 ずらしてそのままコピーします。

この結果、グレー画像を入力すると元の YOLOv8m と同じ出力になります（fp16 の丸め誤差の範囲内）。
3ch データで学習する場合は、Stream B のステム（4ch 用は 2ch）を学習開始時に 1ch へ自動で変換します。

### 他のスケールの重みを作る

```python
from ultralytics.utils.multispectral import convert_yolov8_to_msmid
convert_yolov8_to_msmid("yolov8s.pt", "yolov8s-msmid.pt", ch=4)
```

`YOLO("yolov8s-msmid.pt")` のように指定してファイルが無い場合も、同じ変換が自動で実行されます。

### 注意
- Ultralytics は 3ch 画像の並びを常に反転（BGR→RGB）します。`MSSplit` はこれを補正するので、
  3ch TIFF (G,R,X) でも Stream A には (G,R) が入ります。
- 3ch 画像には HSV 色拡張が適用されます。マルチスペクトルでは `hsv_h=0, hsv_s=0, hsv_v=0` を推奨します
  （4ch には元々適用されません）。
- 2ch（G,R のみ）の入力には対応していません（Stream B が空になるため）。
