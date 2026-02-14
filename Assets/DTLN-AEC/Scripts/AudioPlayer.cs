using System.Collections.Generic;
using UnityEngine;

public class AudioPlayer : MonoBehaviour
{
    public AudioSource audioSource;
    int SampleRate = 16000;
    AudioClip audioClip;
    /// <summary>
    /// 存储合成过程中回调产生的音频浮点数据（范围[-1,1]）
    /// </summary>
    List<float> audioData = new List<float>();
    /// <summary>
	/// 当前要读取的索引位置
	/// </summary>
	int curAudioClipPos = 0;

    private void Awake()
    {
        //audioSource = GetComponent<AudioSource>();
        audioClip = AudioClip.Create("SynthesizedAudio", SampleRate * 2, 1,
                    SampleRate, true, (float[] data) =>
                    {
                        ExtractAudioData(data);
                    });
        audioSource.clip = audioClip;
        audioSource.loop = true;
        audioSource.Play();
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void AddData(float[] data)
    {
        audioData.AddRange(data);
        //Debug.Log("音频长度增加 " + (float)data.Length / (float)SampleRate + "秒");
    }

    bool ExtractAudioData(float[] data)
    {
        if (data == null || data.Length == 0)
        {
            return false;
        }
        bool hasData = false;//是否真的读取到数据
        int dataIndex = 0;//当前要写入的索引位置
        if (audioData != null && audioData.Count > 0)
        {
            while (curAudioClipPos < audioData.Count && dataIndex < data.Length)
            {
                data[dataIndex] = audioData[curAudioClipPos];
                curAudioClipPos++;
                dataIndex++;
                hasData = true;
            }
        }

        //剩余部分填0
        while (dataIndex < data.Length)
        {
            data[dataIndex] = 0;
            dataIndex++;
        }
        return hasData;
    }
}