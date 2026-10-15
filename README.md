# OSSSeg：RGB 植物语义分割

基于 [MODA/OSSDet](https://github.com/shuaihao-han/MODA) 的公开特征模块，替换本仓库原 ResNet50-U-Net。网络、模块及损失均在 model/，不依赖 MMRotate、MMDetection、MMCV。这是 OSSDet 衍生的 RGB 分割模型，不是原论文的八波段旋转框检测模型，原论文 mAP 不适用于它。

## 模型

RGB → ResNet-50四级特征 → 通道统一为256 → CFB深层交互 → SSAF自顶向下融合 → 像素监督的植物前景引导 → FEM自底向上精炼 → 新增多尺度分割解码器与RGB细节分支 → 两通道logits。

- model/ossseg.py：总体网络。默认 ResNet-50，新增分割部分用GroupNorm。
- model/cfb.py、ssaf.py、fem.py：原样引入的官方特征模块。
- model/losses.py：交叉熵 + 前景Dice + 0.2×辅助前景BCE。
- model/README.md：原始模块与改造部分的区别、来源与许可证。

512输入时CFB在16×16特征上计算全局相关性。骨干保留RGB输入，不能产生RGB没有的近红外等信息。三个特征交互模块在AMP中以FP32执行以保护相关性/距离计算；骨干及分割头仍可混合精度运行。

## 安装

Python 3.10+。先按GPU/CUDA选择配套的torch与torchvision，再安装其余依赖：

~~~bash
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
~~~

CPU测试工作流使用Python3.11、torch2.5.1、torchvision0.20.1；具体运行结果见GitHub Actions。代码支持CPU/CUDA，默认device=auto。单卡起始batch-size=2，显存不足先减为1。改变width会改变实验模型容量，默认256。

## VOC2012 数据

目录如下（文件名是示例）：

| 路径 | 内容 |
|---|---|
| VOCdevkit/VOC2012/JPEGImages/sample001.jpg | RGB图像，允许jpg/jpeg/png等扩展名 |
| VOCdevkit/VOC2012/SegmentationClass/sample001.png | 二值掩膜 |
| VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt | 训练样本ID |
| VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt | 独立验证样本ID |
| VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt | 独立测试样本ID，可选，但需在训练前准备好才能最终测试 |

每行一个同名主干，不带扩展名。图片与掩膜必须都是512×512；程序不悄悄改变尺寸。--data-path接受VOCdevkit或VOC2012路径。不会自行划分；trainval.txt不能当作独立验证集。如果目前只有trainval，先依据真实划分整理为互不重叠的train/val/test。

**背景为0，支持0/255或0/1二分类、0..K-1多分类。** 使用下方“标签编码与多分类”中的参数明确类别数与编码；这是VOC目录布局，不自动采用VOC的21类或void语义。

~~~bash
python check_data.py --data-path VOCdevkit
~~~

检查独立划分、同名配对、尺寸、标签编码。支持索引色PNG的原始索引；真正彩色掩膜报错。验证集不做随机增强；训练同步翻转与90度旋转。图像统一ImageNet mean/std归一化，训练和预测相同。

## ResNet-50预训练

ResNet-50是骨干结构；加载已有权重后才是预训练骨干。原OSSDet配置也使用ResNet-50并指定本地预训练文件。本工程接受本地 **torchvision RGB ResNet-50 ImageNet state_dict**，严格检查键和尺寸，只忽略fc分类器；不自动联网下载。

将权重放在 model/weights/resnet50.pth（该目录及二进制权重未打包），或传任意绝对路径。原U-Net权重、八波段OSSDet检测权重不能直接用。不会在载入后执行旧模板的全模型weights_init。默认冻结骨干BatchNorm运行统计，卷积等参数仍全量训练；--no-freeze-backbone-bn可启用BN统计更新。

## 训练与恢复

使用本地预训练：

~~~bash
python train_medical.py --data-path VOCdevkit --backbone-weights model/weights/resnet50.pth --batch-size 2 --epochs 100 --output run/train/ossseg
~~~

只想检查流程、暂时没有预训练文件，可显式随机初始化：

~~~bash
python train_medical.py --data-path VOCdevkit --from-scratch --batch-size 2 --epochs 2 --output run/train/smoke
~~~

这种短训练不代表最终效果。真实对比应统一预训练来源、数据和训练预算。

~~~bash
python train_medical.py --data-path VOCdevkit --resume run/train/ossseg/last.pth --batch-size 2 --epochs 100 --output run/train/ossseg
~~~

恢复保留原总epochs及其他训练/模型参数，恢复优化器、调度器、AMP与随机状态。不要把--epochs改为剩余轮数。--output须为原运行目录；新实验不能覆盖非空目录。支持--amp/--no-amp，CPU自动禁用AMP。--num-classes沿用旧命令的“前景类别数”含义，二分类为1（输出2通道），五分类为4（输出5通道）。

输出：
- best.pth：验证集**Foreground IoU**最高的checkpoint。
- last.pth：末轮checkpoint，含完整恢复状态。
- config.json、splits.json、history.json：参数、划分和指标记录。
- loss_curve.png、metrics_curve.png：沿用模板绘图。

跨硬件/版本不保证完全一致。划分记录校验ID，不对图像内容做哈希；不要在训练后替换样本。

## 验证、最终测试、预测

~~~bash
python val.py --data-path VOCdevkit --weights run/train/ossseg/best.pth --split val
python val.py --data-path VOCdevkit --weights run/train/ossseg/best.pth --split test
python predict.py --data-path VOCdevkit/VOC2012/JPEGImages --weights run/train/ossseg/best.pth --output run/predict/ossseg
~~~

评估读取checkpoint自带的结构，检查划分与训练时一致。按全数据集混淆矩阵计算前景IoU、背景IoU、mIoU、Dice、Precision、Recall等，不再逐batch简单平均。无定义的指标输出null；mIoU纳入union>0的类。test不用于选择best。

predict输出原尺寸512×512的 *_mask.png（0/255）、*_prob.npy（植物概率）和 *_overlay.png。阈值固定0.5，平局选背景。输出目录须为空，以免覆盖已有预测。当前不包含大图滑窗和实例计数。

## 以后怎样替换模型

~~~python
from model import OSSSeg
model = OSSSeg(num_classes=2)
logits = model(images)                      # [B,2,512,512]
training_output = model(images, return_aux=True)
# {"logits": ..., "aux_logits": [B,1,128,128]}
~~~

训练脚本只依赖此接口及model_config、本地权重加载。后续网络可在model/中适配；若特殊损失/多输入不同，仍需对应修改。此次移除了旧model/MPC.py、resnet_backbone.py、unet_resnet.py、unet_training.py及跟踪的pycache；旧版本留在Git历史。utils下部分旧通用辅助文件保留但不被新训练入口使用。

## 验证

~~~bash
python -m compileall -q model utils train_medical.py val.py predict.py check_data.py tests
python -m unittest discover -s tests -v
~~~

包含默认256通道模型512前向/反向/重载、辅助分支梯度、标签映射、划分泄漏、指标和合成VOC的训练/续训/测试/预测检查。CPU工作流不会下载模型权重、访问用户数据或使用GPU。未在真实植物数据上训练，未验证真实精度及CUDA/AMP表现。原论文的多光谱检测成绩不能作为本改造版的分割成绩。

## 完整训练日志与最终结果（2026-10）

工程保留OSSSeg原有CFB/SSAF/FEM核心模块；多分类适配输出头、损失和评估流程。
参考 xuanpluspro/vegann-seg-benchmark 增加tqdm训练/验证/测试进度条，
显示batch完成比例、时间/预计剩余时间，训练和验证显示累计平均loss。
每轮打印loss、mIoU、Dice、最佳前景IoU及耗时，并导出history.csv。

项目根目录新增data/，可分别放data/RiceSeg/VOC2012、data/VegAnn/VOC2012等。
数据不会提交Git；--data-path指向选定VOC2012总目录。当前仍要求
JPEGImages、SegmentationClass、ImageSets/Segmentation/{train,val,test}.txt，
图像/掩膜512×512，磁盘标签支持0/255、0/1或多分类索引（索引色读取原索引）。
不会自动混合数据、生成划分或改变尺寸；各划分独立。test可选。

训练结束自动加载best.pth（按验证集前景IoU选择），重新评估val和已登记test：

| 文件 | 内容 |
|---|---|
| metrics.json | 最佳epoch、val/test所有精度及benchmark汇总 |
| results.csv | 验证/测试各一行，Excel可打开；精度为0~1 |
| benchmark.json | Params、部分/完整GFLOPs、FPS、平均/中位数/P95延迟及设备信息 |
| history.json / history.csv | 每轮损失、精度、学习率、时间 |
| loss_curve.png / metrics_curve.png | 损失与指标曲线 |
| best.pth / last.pth | 最佳权重 / 完整续训checkpoint |

精度包括前景/背景IoU、mIoU、Dice(F1)、Precision、Recall、Specificity、
PixAcc、mAcc、FWIoU、Boundary F1、HD95、ASSD及混淆矩阵。
区域指标按整个数据集累计像素；Boundary F1/HD95/ASSD按图像取平均。
距离单位为512评估分辨率下的**像素**，Boundary F1容差固定2像素，非毫米。
双方无前景时距离=0、Boundary F1=1；仅一方无前景时距离=图像对角线、F1=0，
与参考仓库口径一致；大量空图会影响这些均值。HD95/ASSD越低越好。

测速口径：PyTorch eager FP32、TF32关闭、batch=1、512×512，预热50次、
测量200次CUDA event，仅模型前向，不含读图/传输/预处理/后处理。
fvcore一次乘加计一次；不支持算子会注明，counted_gflops是部分计数，
gflops仅在无遗漏时填写。CPU不输出GPU FPS。性能失败仍保留精度结果。
--no-benchmark可跳过测速，仍输出最终精度；无test则明确注明而不冒充测试结果。

更新并安装依赖：
```bash
git pull && pip install -r requirements.txt
```

训练示例（指定已有本地torchvision ResNet-50预训练权重）：
```bash
python train_medical.py --data-path data/RiceSeg/VOC2012 --backbone-weights model/weights/resnet50.pth --epochs 30 --batch-size 32 --workers 4 --device cuda --amp --output run/train/riceseg_pretrained
```

训练已完成的模型可补生成结果，无需重训：
```bash
python summarize.py --data-path data/RiceSeg/VOC2012 --weights run/train/riceseg_pretrained/best.pth --device cuda --batch-size 32 --workers 4
```

或者仅测速：
```bash
python benchmark.py --weights run/train/riceseg_pretrained/best.pth --device cuda
```

独立评估依然可用val.py；原模型checkpoint结构兼容，训练数据/超参数不改变时可续训。
32GB最大可用batch尚未实测，batch=32显存不足时降低。边界距离在CPU计算，会增加验证时间。


### 标签编码与多分类

训练和 `check_data.py` 支持 `--mask-encoding auto|binary255|index`：

| 原始掩码像素值 | 参数 | 模型输出通道 |
|---|---|---|
| 0、255（二分类） | `--num-classes 1 --mask-encoding binary255` | 2 |
| 0、1（二分类） | `--num-classes 1 --mask-encoding index` | 2 |
| 0、1、2、3、4（五分类） | `--num-classes 4 --mask-encoding index` | 5 |

`--num-classes` 延续原接口，表示**不含背景的类别数**。默认 `auto` 在二分类时接受 0/1 或 0/255；多分类保留类别编号，越界报错，不将所有非零值合并。背景始终为0；不支持把255作为忽略标签。P模式PNG按调色板索引读取，彩色RGB标签需先转换为类别索引。图像和掩码仍要求512×512。

多分类主分支使用K通道交叉熵和前景类别Dice，辅助分支学习所有非背景类别的并集；最佳模型按mIoU选取（二分类保持前景IoU）。`metrics.json`包含混淆矩阵、逐类指标，`class_metrics.csv`保存逐类结果；Dice/Precision/Recall取前景类别宏平均，未定义项不参与平均。Foreground IoU、HD95、ASSD、Boundary F1在多分类时衡量前景并集，不能衡量前景类别之间的混淆，应结合逐类IoU和mIoU阅读。

预测多分类输出PNG像素为0..K-1，概率文件为K×H×W；二分类保持默认0/255和H×W概率，可加`--mask-output-encoding index`导出0/1。标签编码和类别数随检查点保存，验证和汇总自动复用。旧二分类检查点仍可评估；恢复旧检查点训练请显式指定`--mask-encoding binary255`。

当前0/1数据集训练示例（输出目录须为空）：

```bash
python train_medical.py --data-path data/VOC201201 --mask-encoding index --num-classes 1 --backbone-weights pretrained/resnet50.pth --batch-size 32 --epochs 30 --output run/train/vegann01_pretrained_bs32_ep30
```
