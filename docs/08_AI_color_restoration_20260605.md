# 2026-06-05 AI 色彩偏灰修正记录

用户反馈：AI 输出仍偏灰、低饱和，后面手动调整变化不明显。针对这个问题，本轮不再只靠 L1 像素损失训练，因为 L1/MAE 在小数据集上容易学成平均颜色，结果就是灰、低饱和、低局部对比。

## 底层原则

- 彩色负片去色罩不是简单反相，需要先处理橙色片基，再做通道曝光、白点黑点、色彩平衡和曲线。
- 小模型直接回归 RGB 时，最容易牺牲色度和局部对比来降低平均误差。
- 训练损失需要同时约束亮度、色度、饱和度、结构和边缘。
- 推理阶段需要保留确定性去色罩算法作为锚点，避免神经网络在小样本下漂到灰雾或奇怪色偏。

## 代码改动

- `src/film_mask_automation/ml/train.py`
  - 新增 SSIM、opponent chroma、saturation、local contrast、gradient 组合损失。
  - 新增 `--progress-json`，每个 epoch 记录训练进度。
  - checkpoint 内记录 `loss_config`。
- `src/film_mask_automation/ml/inference.py`
  - 新增 `enhance_model_output()`，对低饱和模型输出做色偏中和、饱和度恢复和对比恢复。
  - 默认 AI 推理改为 `AI model + smart-rule anchor` 混合输出。
  - 最终混合参数：model weight 0.55，color 1.35，contrast 1.12。
- `src/film_mask_automation/cli.py`
  - 新增 `--no-ai-enhance`。
  - 新增 `--no-ai-hybrid`，可关闭规则锚点混合，查看裸 AI 输出。
- `tools/create_experiment_run.py`
  - 最后一列改名为 `AI hybrid`，避免误解为裸模型输出。

## 训练记录

- 数据集：71 对真实/公开负片与目标图。
- 训练集：57 对。
- 测试集：14 对。
- 模型：`models/film_mask_tiny_mixed_true_negative.pt`
- 参数：10 epochs，80 steps/epoch，batch size 2，crop 160，base channels 32，CPU。
- 新组合损失训练：epoch 1 loss 0.36754，最低 epoch 9 loss 0.30679，epoch 10 loss 0.30858。

## 最新实验

- 实验目录：`experiments/20260605-142322_color_loss_hybrid_sat_contrast_71_pairs`
- 最新副本：`experiments/latest`
- 输出总览：`experiments/latest/03_model_outputs.jpg`
- 指标：`experiments/latest/metrics.json`

14 张测试集平均误差：

- rule neutral：49.35
- fixed warm：51.00
- smart auto：44.06
- AI hybrid：40.48

饱和度统计：

- target：0.1217
- smart auto：0.1056
- AI hybrid：0.1295
- 旧裸 AI/弱增强输出曾低到约 0.0765，确实偏灰；当前版本已明显提高。

## 当前判断

- 色彩偏灰问题已改善，AI hybrid 在测试集上也优于 smart auto。
- 仍存在低动态、灰雾和个别偏绿问题，尤其是曝光很暗、目标图本身不够一致的样片。
- 下一轮应该重点改局部曲线、分区对比和目标风格分层，而不是继续盲目提高全局饱和度。

## 参考来源

- Negatop negative inversion notes: https://negatop.com/
- SSIM paper: Wang et al., Image quality assessment: from error visibility to structural similarity, IEEE TIP 2004, https://doi.org/10.1109/TIP.2003.819861
- BlueNeg dataset: https://huggingface.co/datasets/ttgroup/blueneg-release
- BlueNeg ICCV 2025 paper page: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.html
