import os
import librosa.display
import matplotlib.pyplot as plt
import matplotx
import librosa
import numpy as np


class AudioVisualizer:
    figsize = {
        "waveplot": (15, 5),
        "spectrogram": (12, 9)
    }

    def __init__(self,
                 file_path: str,
                 input_signal: np.ndarray,
                 sampling_rate: int,
                 n_fft: int = 2048,
                 hop_length: int = 512
                 ):
        """
        Initialize AudioVisualizer Class. You must load the audio file with librosa before using this.
        Also sets some plotting options. They're not too important but rather how I prefer it.

        Args:
            file_path: path to audio file
            input_signal: numpy array containing the audio signal data
            sampling_rate: sampling rate for the data
            n_fft: used to calculate num of rows in short-term fourier transform results. More = longer result clip
            hop_length: number of columns in the short-term fourier transform.
        """
        self.file_name = file_path.split(os.sep)[-1]
        self.signal = input_signal
        self.sr = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_length = input_signal.shape[0] / sampling_rate
        self.amplitude = np.abs(librosa.stft(input_signal, n_fft=n_fft, hop_length=hop_length))

        plt.style.use(matplotx.styles.nord)
        plt.rcParams.update({
            'figure.autolayout': True,
            'axes.grid': True,
            'axes.grid.axis': 'x',
            'xtick.minor.visible': True,
            'axes.axisbelow': True,
            'axes.grid.which': 'both'
        })

    def gen_waveplot(self) -> plt.figure:
        """
        Generate a waveplot of the audio.

        Returns:
            plot: a plt object containing the figure of the waveplot
        """
        plot = plt.figure(figsize=self.figsize.get("waveplot"))
        librosa.display.waveshow(y=self.signal, sr=self.sr, alpha=0.9)

        return plot

    def gen_spectrogram(self) -> plt.figure:
        """
        Generate a decibel-scaled spectrogram of the audio.

        Returns:
            plot: a plt object containing the figure of the spectrogram
        """
        plot = plt.figure(figsize=self.figsize.get("spectrogram"))
        decibels_scaled = librosa.amplitude_to_db(self.amplitude, ref=np.max)
        librosa.display.specshow(decibels_scaled,
                                 sr=self.sr,
                                 hop_length=self.hop_length,
                                 x_axis='time',
                                 y_axis='log'
                                 )
        plt.colorbar()
        return plot

    def gen_mel_spectrogram(self) -> plt.figure:
        """
        Generates a mel-scaled spectrogram of the audio

        Returns:
            plot: a plt object c

        """
        plot = plt.figure(figsize=self.figsize.get("spectrogram"))
        mel_signal = librosa.feature.melspectrogram(y=self.signal, sr=self.sr)
        decibels_scaled = librosa.amplitude_to_db(mel_signal, ref=np.max)
        librosa.display.specshow(decibels_scaled,
                                              sr=self.sr,
                                              hop_length=self.hop_length,
                                              x_axis='time',
                                              y_axis='log'
                                              )
        plt.colorbar()
        return plot

    def gen_mfccs(self) -> plt.figure:
        """
        Generate a plot displaying the mel frequency cepstral coefficient (MFCC) of the audio

        Returns:
            plot: a plt object containing the figure of the spectrogram
        """
        pass

    def display_plot(self, plot: plt.figure) -> None:
        """
        Displays the plot passed to it.

        Args:
            plot: plt object containing the plot to display
        """
        plt.title(self.file_name, fontsize=20)
        plot.show()

    def save_plot(self, plot: plt.figure, save_loc: str = None, file_name: str = None) -> None:
        """
        Saves the plot at the given save location.

        Args:
            plot: plt object containing the plot to display
            save_loc: location where to save the plot.
            file_name: name to save the file with. Defaults to the same name as the audio file
        """
        plt.title(self.file_name, fontsize=20)

        plot.savefig(os.path.join(save_loc, file_name),
                     bbox_inches='tight',
                     pad_inches=0,
                     dpi=100
                     )

        plt.close()


# input_path = os.path.join("..", "data", "gtzan", "train", "audio", "blues", "blues.00000.wav")
# visualizer = AudioVisualizer(input_path, *librosa.load(input_path))
#
# mel = visualizer.gen_spectrogram()
# mel2 = visualizer.gen_mel_spectrogram()
#
# visualizer.display_plot(mel)
# visualizer.display_plot(mel2)
#
# save_path = os.path.join("..", "output")
#
# print(save_path, input_path.split(os.sep)[-1].replace("wav", "png"))
#
# visualizer.save_plot(mel, save_path.replace("audio", "image"), save_path.split(os.sep)[-1].replace("wav", "png"))
