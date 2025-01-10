import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

#本实验使用python编程，完成对44.1KHz语音信号进行采样，使其采样频率降低到8KHz。
# 主要步骤：读取WAV文件，根据信号特征和实验需求选择数字滤波器设计方法和参数，
# 进行抗混叠滤波和抽样，输出8KHz音频文件。
#
#

# 读取WAV文件
def read_wav(filename):
    rate, data = wav.read(filename)
    return rate, data


# 绘制时域波形图
def plot_time_domain(data, rate, title="Time Domain Signal"):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(data)) / rate, data)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


# 绘制频谱图
def plot_spectrum(data, rate, title="Frequency Spectrum"):
    # 计算FFT
    n = len(data)
    freqs = np.fft.fftfreq(n, 1 / rate)
    fft_data = np.fft.fft(data)
    magnitude = np.abs(fft_data[:n // 2])
    freqs = freqs[:n // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()


# 低通抗混叠滤波
def low_pass_filter(data, original_rate, target_rate):
    nyquist = 0.5 * original_rate
    cutoff = target_rate / 2.0  # 截止频率为目标采样率的一半
    # 设计低通滤波器
    b, a = signal.butter(4, cutoff / nyquist, btype='low')
    # 使用lfilter进行单向滤波
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


# 使用resample进行降采样
def downsample(data, original_rate, target_rate):
    # 计算重采样的目标长度
    num_samples = int(len(data) * target_rate / original_rate)
    # 使用resample进行重采样
    downsampled_data = signal.resample(data, num_samples)
    return downsampled_data



# 主函数
def process_audio(input_file, target_rate=8000):
    # 1. 读取原始信号文件
    original_rate, original_data = read_wav(input_file)
    print(f"Original Sampling Rate: {original_rate} Hz")

    # 2. 绘制原始时域波形图和频谱图
    plot_time_domain(original_data, original_rate, "Original Time Domain Signal")
    plot_spectrum(original_data, original_rate, "Original Frequency Spectrum")

    # 3. 抗混叠滤波
    filtered_data = low_pass_filter(original_data, original_rate, target_rate)
    plot_time_domain(filtered_data, original_rate, "Filtered Time Domain Signal")
    plot_spectrum(filtered_data, original_rate, "Filtered Frequency Spectrum")

    # 4. 降采样到目标采样率
    downsampled_data = downsample(filtered_data, original_rate, target_rate)

    # 5. 绘制降采样后时域波形图和频谱图
    plot_time_domain(downsampled_data, target_rate, "Downsampled Time Domain Signal")
    plot_spectrum(downsampled_data, target_rate, "Downsampled Frequency Spectrum")

    # 6. 保存降采样后的音频文件
    wav.write("downsampled_audio.wav", target_rate, downsampled_data.astype(np.int16))
    print("Downsampled audio saved as 'downsampled_audio.wav'")


# 运行
input_filename = 'yuyin1.wav'  # 替换为你的WAV文件路径
process_audio(input_filename)
