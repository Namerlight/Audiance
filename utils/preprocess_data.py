import os
import librosa
from utils import AudioVisualizer
from utils import process_imgs


def preprocess_GTZAN(data_root: str):

    data_splits = ["train", "test", "val"]

    for split in data_splits:

        list_of_genres_paths = [os.path.join(data_root, split, "audio", genre) for genre in os.listdir(os.path.join(data_root, split, "audio"))]
        list_of_images_paths = [path.replace("audio", "image") for path in list_of_genres_paths]

        count = 0

        for image_path in list_of_images_paths:
            if not os.path.exists(image_path):
                os.makedirs(image_path)

        for genre_path in list_of_genres_paths:
            list_of_songs_paths = [os.path.join(genre_path, song) for song in os.listdir(genre_path)]

            total_songs = len(list_of_genres_paths) * len(list_of_songs_paths)

            for song_path in list_of_songs_paths:
                try:
                    visualizer = AudioVisualizer(song_path, *librosa.load(song_path))
                except Exception as e:
                    continue
                visualizer.save_plot(plot=visualizer.gen_spectrogram(),
                                     save_loc=genre_path.replace("audio", "image"),
                                     file_name=song_path.split(os.sep)[-1].replace("wav", "png")
                                     )
                count += 1
                if count % 100 == 0:
                    print(f"{count} songs out of {total_songs} converted.")

        print(f"{count} audio files from {split} processed into images.")

        count = 0
        for genre_path in list_of_images_paths:
            list_of_imgs_paths = [os.path.join(genre_path, image) for image in os.listdir(genre_path)]

            for img in list_of_imgs_paths:
                process_imgs.crop_spectrograms(img_path=img)
                count +=1

        print(f"{count} image files from {split} cropped.")


# path = os.path.join("..", "data", "gtzan")
# preprocess_GTZAN(path)