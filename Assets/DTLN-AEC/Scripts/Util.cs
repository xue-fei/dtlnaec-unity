using System;
using System.IO;

public class Util
{
    public static float[] ReadWav(string filePath)
    {
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            // 读取WAV文件头
            string riff = new string(reader.ReadChars(4));    // "RIFF"
            int fileSize = reader.ReadInt32();                // 文件总大小-8
            string wave = new string(reader.ReadChars(4));    // "WAVE"
            string fmt = new string(reader.ReadChars(4));     // "fmt "
            int fmtSize = reader.ReadInt32();                 // fmt块大小（至少16）

            // 读取音频格式信息
            short audioFormat = reader.ReadInt16();           // 1=PCM
            short numChannels = reader.ReadInt16();           // 通道数
            int sampleRate = reader.ReadInt32();              // 采样率
            int byteRate = reader.ReadInt32();                // 字节率
            short blockAlign = reader.ReadInt16();            // 块对齐
            short bitsPerSample = reader.ReadInt16();         // 采样深度

            // 验证文件格式
            if (riff != "RIFF" || wave != "WAVE" || fmt != "fmt ")
                throw new Exception("无效的WAV文件头");

            // 跳过fmt块的额外信息（如果有）
            if (fmtSize > 16)
                reader.ReadBytes(fmtSize - 16);

            // 查找数据块
            string dataChunkId;
            do
            {
                dataChunkId = new string(reader.ReadChars(4));
                if (dataChunkId != "data")
                    reader.ReadBytes(reader.ReadInt32()); // 跳过非数据块
            } while (dataChunkId != "data");

            int dataSize = reader.ReadInt32(); // 数据块大小（字节）

            // 验证音频参数
            //if (audioFormat != 1) 
            //    throw new Exception("audioFormat:" + audioFormat+ "仅支持PCM格式");
            if (numChannels != 1)
                throw new Exception("仅支持单声道音频");
            if (sampleRate != 16000)
                throw new Exception("仅支持16kHz采样率");
            //if (bitsPerSample != 16)
            //    throw new Exception("仅支持16位采样深度");

            // 读取PCM数据并转换为float
            int sampleCount = dataSize / 2; // 16位 = 2字节/样本
            float[] floatData = new float[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                // 小端序读取16位样本
                byte lowByte = reader.ReadByte();
                byte highByte = reader.ReadByte();
                short pcmValue = (short)((highByte << 8) | lowByte);

                // 将16位PCM值转换为[-1.0, 1.0]范围的float
                floatData[i] = pcmValue / 32768.0f;
            }

            return floatData;
        }
    }

    public static void SaveClip(int channels, int frequency, float[] data, string filePath, float valume = 1.0f)
    {
        using (FileStream fileStream = new FileStream(filePath, FileMode.Create))
        {
            using (BinaryWriter writer = new BinaryWriter(fileStream))
            {
                // 写入RIFF头部标识
                writer.Write("RIFF".ToCharArray());
                // 写入文件总长度（后续填充）
                writer.Write(0);
                writer.Write("WAVE".ToCharArray());
                // 写入fmt子块
                writer.Write("fmt ".ToCharArray());
                writer.Write(16); // PCM格式块长度
                writer.Write((short)1); // PCM编码类型
                writer.Write((short)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2); // 字节率
                writer.Write((short)(channels * 2)); // 块对齐
                writer.Write((short)16); // 位深度
                                         // 写入data子块
                writer.Write("data".ToCharArray());
                writer.Write(data.Length * 2); // 音频数据字节数
                                               // 写入PCM数据（float转为short）
                foreach (float sample in data)
                {
                    writer.Write((short)(sample * 32767 * valume));
                }
                // 返回填充文件总长度
                fileStream.Position = 4;
                writer.Write((int)(fileStream.Length - 8));
            }
        }
    }

    public static float[] BytesToFloat(byte[] byteArray)
    {
        float[] sounddata = new float[byteArray.Length / 2];
        for (int i = 0; i < sounddata.Length; i++)
        {
            sounddata[i] = BytesToFloat(byteArray[i * 2], byteArray[i * 2 + 1]);
        }
        return sounddata;
    }

    private static float BytesToFloat(byte firstByte, byte secondByte)
    {
        //小端和大端顺序要调整
        short s;
        if (BitConverter.IsLittleEndian)
        {
            s = (short)((secondByte << 8) | firstByte);
        }
        else
        {
            s = (short)((firstByte << 8) | secondByte);
        }
        // convert to range from -1 to (just below) 1
        return s / 32768.0F;
    }

    public static byte[] FloatToByte16(float[] floatArray)
    {
        byte[] byteArray = new byte[floatArray.Length * 2];
        int byteIndex = 0;
        foreach (float sample in floatArray)
        {
            short intValue = (short)(Math.Clamp(sample, -1.0f, 1.0f) * short.MaxValue);
            byteArray[byteIndex++] = (byte)(intValue & 0xFF);
            byteArray[byteIndex++] = (byte)((intValue >> 8) & 0xFF);
        }
        return byteArray;
    }
}