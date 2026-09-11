using System;

/// <summary>
/// GCC-PHAT 延迟估计器（FFT O(N log N) 实现）
/// 无需第三方库，纯 C# + Unity 可用
///
/// v5 优化：
///   1. 语音频带加权：数字人语音能量集中在 300Hz~4kHz，白化时对带外频点
///      按权重衰减，聚焦有效频段，相关峰更尖锐、抗噪声。
///
/// v3 优化：
///   1. 信号能量门控：任一通道 RMS 过低（静音/噪声）时直接返回 0 + 低置信度，
///      避免 PHAT 白化把噪声放大成虚假相关峰。
///   2. 置信度改用 peak-to-RMS（原为 peak-to-mean，有符号求和会正负抵消
///      导致比值爆炸到数百甚至负数）。
///   3. PHAT 白化加相对 eps 正则化，避免噪声频点被放大到 1。
///
/// v2 优化：
///   1. FFT 前加 Hanning 窗，降低频谱泄漏
///   2. 输出置信度（peak-to-mean ratio），供调用方拒绝劣质估计
///   3. 窗系数预计算复用
/// </summary>
public static class DelayEstimator
{
    // ── v5: 语音频带加权参数 ────────────────────────────────────────────────
    // 数字人语音能量集中在 300Hz~4kHz，此范围外的频点主要是噪声/混响尾。
    private const float SAMPLE_RATE = 16000f;
    private const float SPEECH_LOW_HZ = 300f;     // 带通下限
    private const float SPEECH_HIGH_HZ = 4000f;   // 带通上限
    private const float SPEECH_BAND_MIN_WEIGHT = 0.05f;  // 带外频点保留的权重（非 0，避免完全丢弃）
    private const float SPEECH_TRANSITION_HZ = 500f;    // 带通边缘过渡带宽（平滑衰减）

    // ── 复数结构 ────────────────────────────────────────────────────────────

    private struct Complex
    {
        public float R, I;
        public Complex(float r, float i) { R = r; I = i; }

        public static Complex operator *(Complex a, Complex b) =>
            new Complex(a.R * b.R - a.I * b.I, a.R * b.I + a.I * b.R);

        // 共轭乘法：a × conj(b)
        public static Complex MulConj(Complex a, Complex b) =>
            new Complex(a.R * b.R + a.I * b.I, a.I * b.R - a.R * b.I);

        public float Mag => MathF.Sqrt(R * R + I * I);
    }

    // ── 核心 API ────────────────────────────────────────────────────────────

    /// <summary>
    /// 估计 mic 相对于 lpb 的延迟（单位：samples）
    /// 正值 = mic 滞后 lpb（麦克风听到的声音比 loopback 晚）
    /// </summary>
    /// <param name="mic">近端麦克风信号</param>
    /// <param name="lpb">远端 loopback 参考信号（与 mic 等长）</param>
    /// <param name="maxLagSamples">搜索范围上限（samples），建议设为采样率 × 最大延迟秒数</param>
    /// <param name="confidence">输出置信度（peak-to-mean ratio），&lt;2.0 视为不可靠</param>
    public static int Estimate(float[] mic, float[] lpb, int maxLagSamples, out float confidence)
    {
        int n = mic.Length;

        // ── v3: 信号能量门控 ──
        // 任一通道 RMS 过低（静音/纯噪声）时，GCC-PHAT 会白化放大噪声，
        // 产生虚假相关峰。此时直接返回 0 延迟 + 低置信度，不进入 FFT。
        float micRms = Rms(mic);
        float lpbRms = Rms(lpb);
        const float MIN_RMS = 1e-3f; // 约 -60 dBFS
        if (micRms < MIN_RMS || lpbRms < MIN_RMS)
        {
            confidence = 0f;
            return 0;
        }

        int fftSize = NextPow2(2 * n - 1);

        Complex[] X = ToComplex(mic, fftSize);
        Complex[] Y = ToComplex(lpb, fftSize);

        // v2: Hanning 窗降低频谱泄漏
        ApplyHanningWindow(X, n);
        ApplyHanningWindow(Y, n);

        // 正向 FFT
        FFT(X, false);
        FFT(Y, false);

        // 互功率谱 + PHAT 白化（v3: 相对 eps 正则化，避免噪声放大）
        float refMag = 0f;
        for (int k = 0; k < fftSize; k++)
        {
            float m = MathF.Abs(X[k].R) + MathF.Abs(X[k].I);
            if (m > refMag) refMag = m;
        }
        float eps = refMag * 1e-6f + 1e-12f;

        // v5: 语音频带加权 —— 数字人语音能量集中在 300Hz~4kHz，高频段（>4kHz）
        //     几乎全是噪声，PHAT 白化会把噪声放大成虚假相关峰，干扰延迟估计。
        //     白化时对语音频带外的频点按权重衰减，让 GCC-PHAT 聚焦有效频段。
        for (int k = 0; k < fftSize; k++)
        {
            Complex cross = Complex.MulConj(X[k], Y[k]);
            float mag = cross.Mag;
            float weight = SpeechBandWeight(k, fftSize);
            X[k] = mag > eps ? new Complex(cross.R / mag * weight, cross.I / mag * weight)
                             : new Complex(0, 0);
        }

        // 逆 FFT → GCC-PHAT 相关序列
        FFT(X, true);

        // 在 [-maxLag, +maxLag] 内找峰值
        int clampedLag = Math.Min(maxLagSamples, fftSize / 2 - 1);
        float bestVal = float.MinValue;
        int bestLag = 0;

        // 正延迟段
        for (int i = 0; i <= clampedLag; i++)
        {
            if (X[i].R > bestVal) { bestVal = X[i].R; bestLag = i; }
        }

        // 负延迟段
        for (int i = fftSize - clampedLag; i < fftSize; i++)
        {
            if (X[i].R > bestVal) { bestVal = X[i].R; bestLag = i - fftSize; }
        }

        // v3: 计算置信度 = 峰值 / RMS（排除峰值附近 ±5 bins），避免有符号求和爆炸
        confidence = CalculateConfidence(X, clampedLag, fftSize, bestLag);

        return bestLag;
    }

    /// <summary>
    /// 信号均方根（RMS），用于静音/噪声门控。
    /// </summary>
    private static float Rms(float[] x)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++)
            sumSq += (double)x[i] * x[i];
        return (float)Math.Sqrt(sumSq / x.Length);
    }

    /// <summary>
    /// v5: 语音频带加权 —— 返回频点 k 对应的权重，聚焦数字人语音能量集中的频段。
    ///
    /// 频点 k 对应物理频率 f = k * SAMPLE_RATE / fftSize。
    /// 在 [SPEECH_LOW_HZ, SPEECH_HIGH_HZ] 内权重为 1，带外在过渡带内平滑衰减到
    /// SPEECH_BAND_MIN_WEIGHT。这样 PHAT 白化时带外噪声频点被压低，相关峰更尖锐。
    /// </summary>
    private static float SpeechBandWeight(int k, int fftSize)
    {
        float freq = k * SAMPLE_RATE / fftSize;

        // 高于奈奎斯特频率的频点（k > fftSize/2）在实信号 FFT 里是镜像，权重按正频率对称
        if (freq > SAMPLE_RATE / 2f)
        {
            freq = SAMPLE_RATE - freq;
        }

        // 带内 → 权重 1
        if (freq >= SPEECH_LOW_HZ && freq <= SPEECH_HIGH_HZ)
            return 1f;

        // 低频过渡带（0 → SPEECH_LOW_HZ）
        if (freq < SPEECH_LOW_HZ)
        {
            float dist = SPEECH_LOW_HZ - freq;
            float t = Clamp01(dist / SPEECH_TRANSITION_HZ);
            return Lerp(1f, SPEECH_BAND_MIN_WEIGHT, t);
        }

        // 高频过渡带（SPEECH_HIGH_HZ → 奈奎斯特）
        float distH = freq - SPEECH_HIGH_HZ;
        float tH = Clamp01(distH / SPEECH_TRANSITION_HZ);
        return Lerp(1f, SPEECH_BAND_MIN_WEIGHT, tH);
    }

    // 纯 C# 实现（避免引入 UnityEngine 依赖，保持本类可独立单测）
    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    /// <summary>
    /// 重载：兼容旧调用方（不输出置信度）
    /// </summary>
    public static int Estimate(float[] mic, float[] lpb, int maxLagSamples = 4800)
    {
        return Estimate(mic, lpb, maxLagSamples, out _);
    }

    // ── 置信度计算 ──────────────────────────────────────────────────────────

    /// <summary>
    /// peak-to-RMS ratio：峰值相对于均方根的倍数。
    /// 高置信度（&gt;3.0）表示相关峰尖锐；低置信度（&lt;2.0）表示无明显峰（静音/噪声）
    ///
    /// v3: 改用 RMS（均方根）而非有符号求和。GCC-PHAT 相关序列围绕 0 正负振荡，
    ///     有符号 sum 会正负抵消趋近 0，导致 peak/mean 爆炸到数百甚至负数。
    ///     RMS 恒为正，能稳定反映相关序列的"本底水平"。
    /// </summary>
    private static float CalculateConfidence(Complex[] gcc, int clampedLag, int fftSize, int bestLag)
    {
        int excludeRadius = 5;
        float peak = gcc[bestLag >= 0 ? bestLag : bestLag + fftSize].R;

        if (peak <= 0f) return 0f;

        double sumSq = 0;
        int count = 0;

        // 正延迟段
        for (int i = 0; i <= clampedLag; i++)
        {
            if (Math.Abs(i - bestLag) <= excludeRadius) continue;
            sumSq += (double)gcc[i].R * gcc[i].R;
            count++;
        }

        // 负延迟段
        for (int i = fftSize - clampedLag; i < fftSize; i++)
        {
            int lag = i - fftSize;
            if (Math.Abs(lag - bestLag) <= excludeRadius) continue;
            sumSq += (double)gcc[i].R * gcc[i].R;
            count++;
        }

        if (count == 0) return 0f;

        float rms = (float)Math.Sqrt(sumSq / count);
        return rms > 1e-10f ? peak / rms : 0f;
    }

    // ── Hanning 窗 ───────────────────────────────────────────────────────────

    /// <summary>
    /// 对前 n 个元素施加 Hanning 窗（原地修改）
    /// </summary>
    private static void ApplyHanningWindow(Complex[] buf, int n)
    {
        for (int i = 0; i < n; i++)
        {
            float w = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / (n - 1));
            buf[i] = new Complex(buf[i].R * w, buf[i].I * w);
        }
    }

    // ── Cooley-Tukey 基 2 FFT（迭代，无递归开销）───────────────────────────

    /// <param name="inverse">true = IFFT，自动归一化</param>
    private static void FFT(Complex[] buf, bool inverse)
    {
        int n = buf.Length;

        // 位反转置换
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (buf[i], buf[j]) = (buf[j], buf[i]);
        }

        // 蝶形运算（迭代）
        for (int len = 2; len <= n; len <<= 1)
        {
            float ang = 2 * MathF.PI / len * (inverse ? 1 : -1);
            Complex wLen = new Complex(MathF.Cos(ang), MathF.Sin(ang));

            for (int i = 0; i < n; i += len)
            {
                Complex w = new Complex(1, 0);
                for (int j = 0; j < len / 2; j++)
                {
                    Complex u = buf[i + j];
                    Complex v = buf[i + j + len / 2] * w;
                    buf[i + j] = new Complex(u.R + v.R, u.I + v.I);
                    buf[i + j + len / 2] = new Complex(u.R - v.R, u.I - v.I);
                    w = w * wLen;
                }
            }
        }

        // IFFT 归一化
        if (inverse)
        {
            float scale = 1f / n;
            for (int i = 0; i < n; i++)
                buf[i] = new Complex(buf[i].R * scale, buf[i].I * scale);
        }
    }

    // ── 工具方法 ────────────────────────────────────────────────────────────

    private static Complex[] ToComplex(float[] src, int targetLen)
    {
        var buf = new Complex[targetLen];
        int copyLen = Math.Min(src.Length, targetLen);
        for (int i = 0; i < copyLen; i++)
            buf[i] = new Complex(src[i], 0);
        return buf;
    }

    /// <summary>大于等于 n 的最小 2 的幂</summary>
    private static int NextPow2(int n)
    {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }
}
