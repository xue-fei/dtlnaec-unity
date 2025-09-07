using NAudio.Wave;
using System;
using System.Collections.Generic;
using UnityEngine;

public class RealTest2 : MonoBehaviour
{
    DtlnaecProcessor2 dtlnaecProcessor;
    WaveInEvent waveSource = null;
    bool isPlay = false;
    List<float> mic = new List<float>();
    List<float> lpb = new List<float>();
    List<float> output = new List<float>();

    // Start is called before the first frame update
    void Start()
    {
        dtlnaecProcessor = new DtlnaecProcessor2();
        dtlnaecProcessor.Initialize(Application.streamingAssetsPath + "/dtln_aec_128_1.onnx",
            Application.streamingAssetsPath + "/dtln_aec_128_2.onnx");

        waveSource = new WaveInEvent();
        waveSource.DeviceNumber = 0; // 选择录音设备，通常0是默认设备
        waveSource.WaveFormat = new WaveFormat(16000, 1); // 设置采样率和通道数，例如44100 Hz, 单声道
        waveSource.DataAvailable += OnData;
        waveSource.BufferMilliseconds = 10;
        waveSource.StartRecording();
        isPlay = true;
    }

    // Update is called once per frame
    void Update()
    {

    }

    Queue<float> nearQueue = new Queue<float>();
    float[] tempFar = new float[128];
    float[] tempNear = new float[128];
    byte[] bs = new byte[320];
    float[] tempFloat;
    void OnData(object sender, WaveInEventArgs e)
    {
        //Debug.Log(e.Buffer.Length);
        Array.Copy(e.Buffer, bs, e.BytesRecorded);
        tempFloat = BytesToFloat(bs);
        for (int i = 0; i < tempFloat.Length; i++)
        {
            nearQueue.Enqueue(tempFloat[i]);
        }
        Debug.Log(tempFloat.Length);
        mic.AddRange(tempFloat);
        if (farQueue.Count >= 128 && nearQueue.Count >= 128)
        {
            for (int i = 0; i < tempFar.Length; i++)
            {
                tempFar[i] = farQueue.Dequeue();
            }
            for (int i = 0; i < tempNear.Length; i++)
            {
                tempNear[i] = nearQueue.Dequeue();
            }
        }
        float[] processedFrame = dtlnaecProcessor.ProcessFrame(tempNear, tempFar);
        output.AddRange(processedFrame);
    }

    Queue<float> farQueue = new Queue<float>();

    private void OnAudioFilterRead(float[] data, int channels)
    {
        if (isPlay)
        {
            //lpb.AddRange(data);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] * 0.25f;
                lpb.Add(data[i]);
                farQueue.Enqueue(data[i]);
            }
        }
    }

    private void OnDestroy()
    {
        isPlay = false;
        waveSource.StopRecording();
        if (dtlnaecProcessor != null)
        {
            float[] end = dtlnaecProcessor.Flush();
            if (end != null)
            {
                output.AddRange(end);
            }
            Util.SaveClip(1, 16000, output.ToArray(), Application.dataPath + "/output.wav");
            Util.SaveClip(1, 16000, mic.ToArray(), Application.dataPath + "/mic.wav");
            Util.SaveClip(1, 16000, lpb.ToArray(), Application.dataPath + "/lpb.wav");
        }
    }

    public float[] BytesToFloat(byte[] byteArray)
    {
        float[] sounddata = new float[byteArray.Length / 2];
        for (int i = 0; i < sounddata.Length; i++)
        {
            sounddata[i] = BytesToFloat(byteArray[i * 2], byteArray[i * 2 + 1]);
        }
        return sounddata;
    }

    private float BytesToFloat(byte firstByte, byte secondByte)
    {
        //小端和大端顺序要调整
        short s;
        if (BitConverter.IsLittleEndian)
            s = (short)((secondByte << 8) | firstByte);
        else
            s = (short)((firstByte << 8) | secondByte);
        // convert to range from -1 to (just below) 1
        return s / 32768.0F;
    }
}