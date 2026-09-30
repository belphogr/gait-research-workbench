# 模型评估说明

README 中的准确率来自项目已有的**按受试者汇总交叉验证**，不是训练集准确率，也不是网页上传视频的验证结果。表中数值按来源记录四舍五入到小数点后两位；没有对原始结果上调。

| 场景 | 验证设置 | 准确率均值 | 折间标准差 | Macro-F1 均值 | 原始结果文件 |
| --- | --- | ---: | ---: | ---: | --- |
| W | 固定被试级五折；原始 stride ExtraTrees 经周期概率汇总到人 | 63.99% | 9.64% | 61.75% | `tree_model_w/results/w_xy_plus_calibrated_z/seed42_5fold/summary.csv` |
| OW | 原训练实验五折；CNN1D 按人汇总 | 86.57% | 9.82% | 81.00% | `results/threeway_integration/OW/CNN1D/work123_tuned_ow_3class_2026_08_04-17_20_59_739619/subject_generalize_OW_result_metrics.csv` |
| WT | 原训练实验两折；CNN1D 按人汇总 | 57.77% | 5.39% | 39.38% | `results/threeway_integration/WT/CNN1D/work123_wt_3class_2026_08_04-17_32_34_087658/subject_generalize_WT_result_metrics.csv` |

上述文件属于组内保管的原研究工程，不随公开源码提供。W 数值还记录在原工程 `tree_model_w/固定五折验证结果.md` 中。OW/WT 的表值描述对应训练配置的交叉验证实验，不能视为网页所载单个检查点在新视频上的独立性能验证。

整合数据总账记录 W 3584、OW 503、WT 237 个周期，包含 README 等级构成图中的 L0/L1/L2。早期 OW/WT 候选标签表缺少 L0，不能与整合后数据混为同一阶段。OW/WT 类别不平衡；WT 尤其不适合只看准确率，应同时看 Macro-F1。数据来源与检查点关系见组内 `gait-demo/validation/training-lineage/论文与训练链路核对.md`。

W 现场视频目前使用补写的周期处理链路，尚未完成与原训练 CSV 的逐项一致性验证。这里的交叉验证数字不能用于宣称现场视频分级准确率。
