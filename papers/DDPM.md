# 01 · DDPM：核心公式、训练与采样，以及与 DMD 的联系

## 基本信息

- 论文：**Denoising Diffusion Probabilistic Models**，Jonathan Ho、Ajay Jain、Pieter Abbeel，NeurIPS 2020。
- 阅读与整理日期：2026-11-16。
- 来源：[论文](https://arxiv.org/abs/2006.11239) · [PDF](https://arxiv.org/pdf/2006.11239) · [官方代码](https://github.com/hojonathanho/diffusion) · [项目主页](https://hojonathanho.github.io/diffusion/)。
- 阅读状态：核心公式初读；尚未完成逐行源码阅读和训练复现。
- 本篇范围：解释前向加噪、噪声预测训练、反向采样，补充与 score、DMD 的数学联系。推导和直觉解释是学习整理，不是新增实验结论。

## 一句话理解

DDPM 先规定一个逐步破坏图像的高斯加噪过程，再训练神经网络预测不同噪声等级下的噪声；生成时从高斯噪声出发，通过多次反向转移得到图像。多次网络前向是其推理成本的重要来源，也是后续少步蒸馏关注的问题。

## 1. 先统一符号

| 符号 | 含义 |
| --- | --- |
| $x_0$ | 干净的真实图像；数学上视为向量 |
| $x_t$ | 加噪到时间步 $t$ 的样本 |
| $T$ | 总扩散步数 |
| $\beta_t$ | 第 $t$ 步加入的噪声方差，预先设定 |
| $\alpha_t=1-\beta_t$ | 单步信号保留系数，其平方根作用于样本幅值 |
| $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$ | 累计信号保留系数；规定 $\bar\alpha_0=1$ |
| $\epsilon\sim\mathcal N(0,I)$ | 与样本形状相同的标准高斯噪声 |
| $\epsilon_\theta(x_t,t)$ | 网络对噪声的预测；$\theta$ 是网络参数 |
| $q$、$p_\theta$ | 分别表示规定的前向过程和学习的反向生成过程 |

**记号提醒：**有些论文写 $x_t=a_t x_0+b_t\epsilon$。与本篇对应时，$a_t=\sqrt{\bar\alpha_t}$、$b_t=\sqrt{1-\bar\alpha_t}$，不能把其他论文的幅值系数直接当作这里的 $\alpha_t$。

## 2. 前向过程：逐步加噪

### 2.1 单步加噪

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I).
$$

等价的采样形式是：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t.
$$

每一步的噪声独立。这个过程不需要学习，时间步和噪声调度由我们设定。使用平方根是因为缩放系数的平方决定方差：若输入具有单位方差且与噪声独立，输出方差仍为 $\alpha_t+(1-\alpha_t)=1$。真实数据不一定具有单位方差，这只是解释该参数化的一个例子。

### 2.2 一次跳到任意时间步

训练中最常用的是论文公式 (4)：

$$
\boxed{x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.}
$$

为什么可以合并？展开前两步：

$$
x_2=\sqrt{\alpha_2\alpha_1}x_0+
\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+
\sqrt{1-\alpha_2}\epsilon_2.
$$

后两项是独立高斯变量，其总方差为：

$$
\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_2\alpha_1.
$$

递推后就得到累计系数 $\bar\alpha_t$。这是条件分布上的等价采样；合并后的 $\epsilon$ 代表累计噪声，不是最后一步的增量噪声 $\epsilon_t$。

当 $\bar\alpha_t$ 接近 1，图像信号强；接近 0 时，样本接近标准高斯噪声。因此训练可以随机抽一个 $t$，直接构造 $x_t$，不用执行前面的全部加噪步骤。

## 3. 训练：预测自己加入的噪声

论文公式 (14) 的简化目标是：

$$
\boxed{L_{\mathrm{simple}}(\theta)=
\mathbb E_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].}
$$

其中 $x_0$ 来自数据集，$t$ 通常均匀采样，$\epsilon$ 是本次随机生成的高斯噪声。监督信号来自我们自己的加噪操作，无须人工标注。

这个目标来自变分训练目标的重新加权，并不等于把原始负 ELBO 原封不动地写成无权重 MSE。本篇先掌握训练机制，完整 ELBO 推导留作后续。

### 3.1 为什么不能理解为“网络精确找回每次噪声”

同一个 $x_t$ 可能由不同的 $x_0$ 和噪声组合产生。在无限容量与理想优化条件下，MSE 的最优预测是条件均值：

$$
\epsilon^*(x_t,t)=\mathbb E[\epsilon\mid x_t,t].
$$

因此模型学习的是统计规律，而不是对每个带噪样本都能唯一还原随机噪声。高噪声时，原图的不确定性尤其大。

### 3.2 从噪声预测换算干净图像

对加噪公式移项，得到论文公式 (12) 对应的估计：

$$
\boxed{\hat x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.}
$$

直觉是先减去估计的噪声，再补偿被缩小的图像信号。它也是 DMD 实现中从噪声预测转为干净样本预测的基础。

但 $\hat x_0$ 不等于一次成功采样：理想预测对应 $\mathbb E[x_0\mid x_t,t]$，可能平均多个不同的原图；当 $\bar\alpha_t$ 很小时，噪声预测误差还会被放大。

## 4. 反向过程：为什么这样逐步生成

### 4.1 训练时已知原图，后验可以精确写出

利用高斯分布和贝叶斯公式，可以得到论文公式 (6)、(7)：

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_t I),
$$

$$
\tilde\mu_t(x_t,x_0)=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
$$

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

**这里必须保留条件 $x_0$。** $q(x_{t-1}\mid x_t,x_0)$ 是可解析的高斯；只给定 $x_t$ 的真实反向条件分布一般不严格是单个高斯。

### 4.2 生成时不知道原图，用模型估计

DDPM 建模：

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2 I).
$$

将 $\hat x_0$ 代入上面的后验均值并整理，可得到论文公式 (11)：

$$
\boxed{\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).}
$$

从 $x_T\sim\mathcal N(0,I)$ 开始，按 $t=T,\ldots,1$ 依次执行：

$$
x_{t-1}=\mu_\theta(x_t,t)+\sigma_t z,
\qquad z\sim\mathcal N(0,I).
$$

最后一步通常令 $z=0$。固定方差可以选 $\sigma_t^2=\tilde\beta_t$；原论文也讨论了 $\beta_t$ 的选择，不能把某一种方差方案当成唯一版本。

**为什么去噪还加入噪声？** 因为这是从反向条件分布中采样，需要保留尚未消除的不确定性。简单删除所有随机项并不自动等同于 DDIM，后者有自己的采样推导。

## 5. 训练与采样的代码对照

以下是教学用 PyTorch 风格片段，不是官方代码的逐行复现，也不是完整可运行训练项目。假设已有模型、优化器、数据和噪声调度；模型输出与输入形状相同。这里数组使用从 0 开始的索引，对应公式中的时间步 1 到 $T$。

```python
import torch
import torch.nn.functional as F

# betas: shape [T]，已放在与样本相同的 device 上
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
alpha_bar_prev = torch.cat([torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]])
posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

def extract(values, t, x):
    # [B] -> [B, 1, 1, 1]，也支持其他维数的输入
    return values[t].reshape(t.shape[0], *([1] * (x.ndim - 1)))

def training_step(model, optimizer, x0):
    model.train()
    t = torch.randint(len(betas), (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    a = extract(alpha_bar, t, x0)
    xt = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, noise)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()

@torch.no_grad()
def sample(model, shape):
    model.eval()
    x = torch.randn(shape, device=betas.device)
    for i in reversed(range(len(betas))):
        t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
        eps_hat = model(x, t)
        mean = (x - betas[i] / (1.0 - alpha_bar[i]).sqrt() * eps_hat) \
               / alphas[i].sqrt()
        if i > 0:
            x = mean + posterior_var[i].sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x
```

训练一次只需要对随机时间步进行一次网络预测；标准逐步采样要循环调用模型。片段省略了图像归一化、网络结构、EMA、混合精度、预测裁剪和模型保存等工程配置。上述片段尚未执行，不代表已复现论文。

## 6. 与 score 的联系：从预测噪声走向 DMD

这一节是为后续 DMD 补充的数学桥梁。令 $p_t(x_t)$ 表示干净数据经高斯扰动后的边缘分布，score 定义为：

$$
s_t(x_t)=\nabla_{x_t}\log p_t(x_t).
$$

它是与样本同维度的向量，表示对数概率密度增长最快的局部方向，不是一个标量“质量评分”。

### 6.1 从条件高斯推导

先对已知 $x_0$ 的高斯加噪分布求导：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

但模型需要的是边缘分布的 score。对可能的原图取后验平均（固定 $t$，并假设可交换微分与积分）：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=\frac{\int p_0(x_0)\nabla_{x_t}q(x_t\mid x_0)\,dx_0}{p_t(x_t)}\\
&=\mathbb E_{x_0\mid x_t,t}[\nabla_{x_t}\log q(x_t\mid x_0)]\\
&=-\frac{\mathbb E[\epsilon\mid x_t,t]}{\sqrt{1-\bar\alpha_t}}.
\end{aligned}
$$

结合 MSE 的最优解，可得到学习到的近似 score：

$$
\boxed{s_\theta(x_t,t)\approx
-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.}
$$

这解释了为什么去噪训练能估计分布的 score。更一般的理论背景见 Vincent 的 [Denoising Score Matching 论文](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)。

### 6.2 对 DMD 的意义

同一种去噪训练机制，换一个训练数据来源，就在估计另一个分布的 score：

| 去噪模型 | 学习的数据来源 | 在 DMD 中的作用 |
| --- | --- | --- |
| 预训练教师 | 目标数据分布 | 冻结，提供目标 score 的近似 |
| 辅助 fake 模型 | 当前学生的生成样本 | 持续训练，追踪学生分布的 score |

在相同的 $x_t,t$ 上比较两者，就能构造分布匹配方向。fake 模型是辅助去噪器，不是学生生成器。原版 DMD 还包含回归损失，不能将整个方法概括为只计算 score 差；权重、KL 梯度和停止梯度操作留到 [DMD 原论文](https://arxiv.org/abs/2311.18828) 笔记详细推导。

## 7. 常见误区

| 误区 | 应如何理解 |
| --- | --- |
| 训练必须先逐步加噪到 $T$ | 可以直接采样任意时间步的 $x_t$ |
| 网络预测最后一步的增量噪声 | 简化目标预测的是从 $x_0$ 到 $x_t$ 的累计等价噪声 |
| 网络总能恢复每一次真实噪声 | MSE 的理想最优解是条件期望 |
| 得到 $\hat x_0$ 就完成了高质量一步生成 | 条件均值不等于从目标分布采样，高噪声下尤其如此 |
| 去噪过程绝不能再加入噪声 | DDPM 的反向转移是随机采样 |
| 所有反向条件分布都严格是高斯 | 已知 $x_0$ 的后验是高斯；未知原图时一般不是 |
| score 是模型对图片质量的标量评分 | 它是对数密度对样本的梯度向量 |
| DDPM 已解决视频实时生成 | 本文是图像生成基础，不提供流式视频和低延迟交互方案 |

## 8. 证据、实验状态与适用边界

- **论文依据：**公式 (4) 对应直接加噪，(6)、(7) 对应后验，(11)、(12) 对应噪声参数化与原图估计，(14) 对应简化训练损失；Algorithm 1、2 分别给出训练和采样流程。
- **官方实现：**原作者仓库使用 TensorFlow；本篇的 PyTorch 风格代码是教学整理。官方 README 记载实验使用 Google Cloud TPU v3-8，不能把训练硬件信息直接当作实时推理性能。
- **我的理解：**网络学会在各个噪声等级下提供去噪方向，采样器组织这些局部预测逐步生成样本。减少采样步数需要处理质量损失，不是单纯少运行几次循环。
- **实测结果：**未运行训练、生成或速度基准，未测显存。没有据此声称任何画质、FPS 或加速比。

| 实时性项目 | 本篇记录 |
| --- | --- |
| 视频长度、时序一致性 | 不适用：原论文主要研究图像生成 |
| 首帧延迟、持续视频 FPS、交互响应延迟 | 未做视频测试，不以图像实验替代 |
| 本地硬件、精度、分辨率、采样步数 | 尚未建立实验配置 |
| 计时范围、预热与峰值显存 | 未测试 |

## 9. 自测与下一步

这些问题用于检验自己的理解，不代表已经完成复现：

1. 能否从两步加噪推导出累计方差 $1-\bar\alpha_t$？
2. 为什么噪声预测 MSE 的最优解是条件期望？
3. 为什么能估计 $x_0$，却仍需要多步采样？
4. 为什么 $q(x_{t-1}\mid x_t,x_0)$ 和 $q(x_{t-1}\mid x_t)$ 不能混为一谈？
5. 能否推导 $s_t=-\mathbb E[\epsilon\mid x_t,t]/\sqrt{1-\bar\alpha_t}$？
6. 在 DMD 中，用学生样本训练辅助模型，为什么能估计学生分布的 score？

后续行动：补充负 ELBO 到加权噪声预测的推导；逐行阅读官方去噪与采样代码；运行一个小规模加噪和还原实验；随后学习 DMD 的反向 KL 梯度。

## 参考资料与相关笔记

1. Ho et al. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
2. [DDPM 官方代码](https://github.com/hojonathanho/diffusion)。
3. Vincent. [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf), 2011.
4. Yin et al. [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828), CVPR 2024.
5. [学习路线](../LEARNING-PLAN.md) · [返回首页](../README.md)。
