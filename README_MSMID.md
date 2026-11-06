# YOLOv8 マルチスペクトル 融合モデル (`yolov8-msmid` / `yolov8-mscmg`)

[Ultralytics](https://github.com/ultralytics/ultralytics) v8.4.129 のフォークです。マルチスペクトル TIFF 用の
**2種類のデュアルバックボーン融合 YOLOv8** を搭載しています。

1. **`yolov8m-msmid.pt` (`MSFusion`)**: P3 / P4 / P5 で `Concat + 1x1 Conv` によるシンプルなミドル融合を行うベースラインモデル
2. **`yolov8m-mscmg.pt` (`CMGFusion`)**: 植物病害・微小病斑検出に特化した **Cross-Modal Gated & Differential Fusion** モデル
   - **3x3 Depthwise 位置ズレ吸収**: 可視光 `(G, R)` と `NIR / RE` 間の微小なレジストレーション残差・葉の揺れを吸収
   - **スペクトル差分・乗算ブランチ**: `B - A`（NDVI/GNDVI/NDRE に対応する波長間コントラスト）と `SiLU(A) * SiLU(B)`（構造共起）を明示的に抽出
   - **動的チャネル＋空間クロスゲート**: 照明条件や影・病斑の種類に応じて、チャネルごと（SE Gate）および空間位置ごと（7×7 Spatial Gate）に可視光と近赤外の重み $\alpha$ を動的に制御

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

# 新モデル: Cross-Modal Gated & Differential Fusion (推奨)
model = YOLO("yolov8m-mscmg.pt")

# 既存モデル: Concat + 1x1 Conv ミドル融合
# model = YOLO("yolov8m-msmid.pt")

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
