using System;

/// <summary>
/// GCC-PHAT 延迟估计器（FFT O(N log N) 实现）
/// 无需第三方库，纯 C# + Unity 可用
/// 
/// v2 优化：
///   1. FFT 前加 Hanning 窗，降低频谱泄漏
///   2. 输出置信度（peak-to-mean ratio），供调用方拒绝劣质估计
///   3. 窗系数预计算复用
/// </summary>
public static class DelayEstimator
{
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
        int fftSize = NextPow2(2 * n - 1);

        Complex[] X = ToComplex(mic, fftSize);
        Complex[] Y = ToComplex(lpb, fftSize);

        // v2: Hanning 窗降低频谱泄漏
        ApplyHanningWindow(X, n);
        ApplyHanningWindow(Y, n);

        // 正向 FFT
        FFT(X, false);
        FFT(Y, false);

        // 互功率谱 + PHAT 白化
        for (int k = 0; k < fftSize; k++)
        {
            Complex cross = Complex.MulConj(X[k], Y[k]);
            float mag = cross.Mag;
            X[k] = mag > 1e-10f ? new Complex(cross.R / mag, cross.I / mag)
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

        // v2: 计算置信度 = 峰值 / 均值（排除峰值附近 ±5 bins）
        confidence = CalculateConfidence(X, clampedLag, fftSize, bestLag);

        return bestLag;
    }

    /// <summary>
    /// 重载：兼容旧调用方（不输出置信度）
    /// </summary>
    public static int Estimate(float[] mic, float[] lpb, int maxLagSamples = 4800)
    {
        return Estimate(mic, lpb, maxLagSamples, out _);
    }

    // ── 置信度计算 ──────────────────────────────────────────────────────────

    /// <summary>
    /// peak-to-mean ratio：峰值相对于均值的倍数。
    /// 高置信度（&gt;3.0）表示相关峰尖锐；低置信度（&lt;2.0）表示无明显峰（静音/噪声）
    /// </summary>
    private static float CalculateConfidence(Complex[] gcc, int clampedLag, int fftSize, int bestLag)
    {
        int excludeRadius = 5;
        float peak = gcc[bestLag >= 0 ? bestLag : bestLag + fftSize].R;

        float sum = 0f;
        int count = 0;

        // 正延迟段
        for (int i = 0; i <= clampedLag; i++)
        {
            if (Math.Abs(i - bestLag) <= excludeRadius) continue;
            sum += gcc[i].R;
            count++;
        }

        // 负延迟段
        for (int i = fftSize - clampedLag; i < fftSize; i++)
        {
            int lag = i - fftSize;
            if (Math.Abs(lag - bestLag) <= excludeRadius) continue;
            sum += gcc[i].R;
            count++;
        }

        if (count == 0 || peak <= 0f) return 0f;

        float mean = sum / count;
        return mean > 1e-10f ? peak / mean : 0f;
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
