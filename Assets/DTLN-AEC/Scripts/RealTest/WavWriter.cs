using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// 轻量 WAV 写入工具，用于验证 AEC 效果
/// 用法：
///   var w = new WavWriter(16000);
///   w.Write(samples);          // 可多次调用
///   w.Save("/path/to/out.wav");
/// </summary>
public class WavWriter
{
    private readonly int _sampleRate;
    private readonly List<float> _samples = new List<float>();

    public WavWriter(int sampleRate = 16000)
    {
        _sampleRate = sampleRate;
    }

    public void Write(float[] samples)
    {
        _samples.AddRange(samples);
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));

        using (var fs = new FileStream(path, FileMode.Create))
        using (var bw = new BinaryWriter(fs))
        {
            int sampleCount = _samples.Count;
            int byteCount = sampleCount * 2;      // 16-bit PCM

            // RIFF header
            bw.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(36 + byteCount);
            bw.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            // fmt chunk
            bw.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16);           // chunk size
            bw.Write((short)1);     // PCM
            bw.Write((short)1);     // mono
            bw.Write(_sampleRate);
            bw.Write(_sampleRate * 2);  // byte rate
            bw.Write((short)2);     // block align
            bw.Write((short)16);    // bits per sample

            // data chunk
            bw.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            bw.Write(byteCount);

            foreach (float s in _samples)
            {
                short pcm = (short)Mathf.Clamp(s * 32767f, -32768f, 32767f);
                bw.Write(pcm);
            }
        }

        Debug.Log($"[WavWriter] 已保存 {_samples.Count} 样本 → {path}");
        _samples.Clear();
    }
}