# YOLOv8 マルチスペクトル 融合モデル (`yolov8-msmid` / `yolov8-mscg` / `yolov8-mscmg`)

[Ultralytics](https://github.com/ultralytics/ultralytics) v8.4.129 のフォークです。マルチスペクトル TIFF 用の
**デュアルバックボーン融合 YOLOv8** を搭載しています。

1. **`yolov8m-msmid.pt` (`MSFusion`)**: P3 / P4 / P5 で `Concat + 1x1 Conv` によるシンプルなミドル融合を行うベースラインモデル
2. **`yolov8m-mscg.pt` (`CGFusion`)**: **`MSFusion` に「局所位置ずれ吸収（3x3 DWConv）」と「動的チャネルゲート（Channel Gate）」のみを追加したモデル（推奨）**
   - **3x3 Depthwise 位置ズレ吸収**: 可視光 `(G, R)` と `NIR / RE` の各ストリームで微小なレジストレーション残差を吸収
   - **全チャネル動的ゲート + 1x1 Conv**: 結合した全 $2C$ チャネルに対して大域コンテキスト（Avg + Max Pool）から動的チャネル重み $2\sigma(\cdot)$ を乗算した上で、1x1 Conv で全チャネル間を混合（空間ゲートや追加 BatchNorm を省き、少数データでの過学習を防止）
3. **`yolov8m-mscmg.pt` (`CMGFusion`)**: 位置ズレ吸収＋動的チャネル/空間ゲート＋スペクトル差分ブランチモデル

```
input ─┬─ MSSplit(A: G,R)  ── YOLOv8 backbone A ── P3a, P4a, P5a ─┐
       └─ MSSplit(B: 残り) ── YOLOv8 backbone B ── P3b, P4b, P5b ─┤
              MSFusion または CMGFusion (P3 / P4 / P5)   ◄────────┘
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

# 局所位置ずれ吸収 + 動的チャネルゲートモデル (推奨)
model = YOLO("yolov8m-mscg.pt")

# 既存モデル
# model = YOLO("yolov8m-msmid.pt")
# model = YOLO("yolov8m-mscmg.pt")

model.train(data="data.yaml", epochs=100, imgsz=1024, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
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
  2: yellow-leaf
```

## 追加・変更点

| ファイル | 内容 |
|---|---|
| `ultralytics/nn/modules/multispectral.py` | `MSSplit`、`MSFusion`（Concat + 1x1 Conv）、`CMGFusion`（差分＋位置ズレ吸収＋動的ゲート融合） |
| `ultralytics/cfg/models/v8/yolov8-msmid.yaml` | `MSFusion` を用いたデュアルバックボーンモデル定義 |
| `ultralytics/cfg/models/v8/yolov8-mscmg.yaml` | `CMGFusion` を用いたデュアルバックボーンモデル定義 |
| `ultralytics/utils/multispectral.py` | `yolov8{n,s,m,l,x}.pt` → `yolov8{scale}-{msmid\|mscmg}.pt` の重み変換 |
| `ultralytics/nn/tasks.py` | `parse_model` に `MSSplit` / `MSFusion` / `CMGFusion` を登録。3ch/4ch 自動ステム適応 |
| `ultralytics/utils/downloads.py` | `yolov8*-msmid.pt` / `yolov8*-mscmg.pt` を自動取得・変換 |
| `weights/yolov8m-msmid.pt` | `MSFusion` 版の変換済み重み（4ch, 3ch 自動適応） |
| `weights/yolov8m-mscmg.pt` | `CMGFusion` 版の変換済み重み（4ch, 3ch 自動適応） |

### 他のスケールの重みを作る

```python
from ultralytics.utils.multispectral import convert_yolov8_to_mscmg, convert_yolov8_to_msmid

convert_yolov8_to_mscmg("yolov8s.pt", "yolov8s-mscmg.pt", ch=4)
convert_yolov8_to_msmid("yolov8s.pt", "yolov8s-msmid.pt", ch=4)
```

`YOLO("yolov8s-mscmg.pt")` のように指定してファイルが無い場合も、自動で変換が実行されます。
