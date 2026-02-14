using System.Collections.Generic;
using uMicrophoneWebGL;
using UnityEngine;

public class RealTest3 : MonoBehaviour
{
    DtlnaecProcessor2 dtlnaecProcessor;
    public MicrophoneWebGL microphoneWebGL;
    public AudioSource audioSource;
    public AudioPlayer audioPlayer;

    private bool isPlay = false;

    // 用于保存完整音频数据（调试用）
    private List<float> mic = new List<float>();
    private List<float> lpb = new List<float>();
    private List<float> output = new List<float>();

    // Loopback音频队列
    private Queue<float> farQueue = new Queue<float>();

    // 临时缓冲区
    private const int ExpectedFrameSize = 128;

    void Start()
    {
        // 配置音频设置为16kHz
        AudioConfiguration config = AudioSettings.GetConfiguration();
        config.sampleRate = 16000;
        config.speakerMode = AudioSpeakerMode.Mono; // 改为Mono以简化处理
        config.dspBufferSize = 512; // 设置为512以便更好地对齐
        AudioSettings.Reset(config);

        Debug.Log($"Audio Config - SampleRate: {config.sampleRate}, DSP Buffer: {config.dspBufferSize}");

        // 启动AudioSource播放
        audioSource.Play();

        // 初始化DTLN-AEC处理器
        dtlnaecProcessor = new DtlnaecProcessor2();
        bool initialized = dtlnaecProcessor.Initialize(
            Application.streamingAssetsPath + "/dtln_aec_128_1.onnx",
            Application.streamingAssetsPath + "/dtln_aec_128_2.onnx"
        );

        if (!initialized)
        {
            Debug.LogError("Failed to initialize DTLN-AEC processor");
            return;
        }

        // 设置麦克风数据回调
        microphoneWebGL.dataEvent.AddListener(OnData);
        microphoneWebGL.Begin(ExpectedFrameSize);

        isPlay = true;
        Debug.Log("RealTest3 started successfully");
    }

    void Update()
    {
        // 监控队列状态
        if (Time.frameCount % 60 == 0) // 每秒打印一次
        {
            Debug.Log($"Queue size: {farQueue.Count}, Frames processed: {dtlnaecProcessor.FramesProcessed}, Padding phase: {dtlnaecProcessor.IsPaddingPhase}");
        }
    }

    /// <summary>
    /// 麦克风数据回调
    /// </summary>
    void OnData(float[] data)
    {
        if (!isPlay || dtlnaecProcessor == null)
            return;

        // 验证数据长度
        if (data.Length != ExpectedFrameSize)
        {
            Debug.LogWarning($"Unexpected mic frame size: {data.Length}, expected {ExpectedFrameSize}");
            return;
        }

        // 保存麦克风数据
        mic.AddRange(data);

        // 准备loopback数据帧
        float[] lpbFrame = new float[ExpectedFrameSize];

        // 从队列中提取loopback数据
        if (farQueue.Count >= ExpectedFrameSize)
        {
            for (int i = 0; i < ExpectedFrameSize; i++)
            {
                lpbFrame[i] = farQueue.Dequeue();
            }
        }
        else
        {
            // 如果loopback数据不足，使用静音
            // Debug.LogWarning($"Insufficient loopback data: {farQueue.Count}/{ExpectedFrameSize}");
        }

        // 执行回声消除处理
        float[] processedFrame = dtlnaecProcessor.ProcessFrame(data, lpbFrame);

        if (processedFrame != null && processedFrame.Length == ExpectedFrameSize)
        {
            // 播放处理后的音频
            if (audioPlayer != null)
            {
                audioPlayer.AddData(processedFrame);
            }

            // 保存处理后的数据
            output.AddRange(processedFrame);
        }
        else
        {
            Debug.LogWarning("ProcessFrame returned invalid data");
        }
    }

    /// <summary>
    /// Unity音频回调 - 用于捕获播放的音频（loopback）
    /// </summary>
    private void OnAudioFilterRead(float[] data, int channels)
    {
        if (!isPlay)
            return;

        int samplesPerChannel = data.Length / channels;

        // 提取单声道数据并添加到队列
        for (int i = 0; i < samplesPerChannel; i++)
        {
            // 提取左声道（或单声道）
            float sample = (channels == 2) ? data[i * 2] : data[i];

            // 应用音量缩放（防止过载）
            //sample *= 0.25f;

            // 保存loopback数据
            lpb.Add(sample);
            farQueue.Enqueue(sample);
        }

        // 限制队列大小，防止内存溢出
        const int MaxQueueSize = 16000 * 2; // 2秒的缓冲
        while (farQueue.Count > MaxQueueSize)
        {
            farQueue.Dequeue();
            Debug.LogWarning("Loopback queue overflow, dropping old samples");
        }
    }

    private void OnDestroy()
    {
        Debug.Log("RealTest3 shutting down...");

        isPlay = false;

        if (dtlnaecProcessor != null)
        {
            // Flush剩余数据
            float[] endData = dtlnaecProcessor.Flush();
            if (endData != null && endData.Length > 0)
            {
                output.AddRange(endData);
            }

            // 保存音频文件用于分析
            try
            {
                string basePath = Application.dataPath;

                if (output.Count > 0)
                {
                    Util.SaveClip(1, 16000, output.ToArray(), basePath + "/output.wav");
                    Debug.Log($"Saved output.wav ({output.Count} samples)");
                }

                if (mic.Count > 0)
                {
                    Util.SaveClip(1, 16000, mic.ToArray(), basePath + "/mic.wav");
                    Debug.Log($"Saved mic.wav ({mic.Count} samples)");
                }

                if (lpb.Count > 0)
                {
                    Util.SaveClip(1, 16000, lpb.ToArray(), basePath + "/lpb.wav");
                    Debug.Log($"Saved lpb.wav ({lpb.Count} samples)");
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"Failed to save audio files: {ex.Message}");
            }

            // 释放处理器资源
            dtlnaecProcessor.Dispose();
            dtlnaecProcessor = null;
        }

        // 清理麦克风回调
        if (microphoneWebGL != null)
        {
            microphoneWebGL.dataEvent.RemoveListener(OnData);
        }
    }
}