# Audio sync Wav2Lip Deepfake model 🚀 🚀

## Brief 🚀

As we know, Generative AI , Deepfake engineering, AI is booming 🤯 🤯 nowadays. For e.g. President is speaking something which we can't expect he can say that so behind this, there's a deepfake tech in which their lips are analyzed and some other audio is synced with it. Here's the task that I'm talking about. 

![Screenshot from 2023-09-14 22-05-53](https://github.com/Hg03/lip-sync-wav2lip/assets/69637720/8ae07ace-3b03-4096-af31-e71f752c4c0f)



## Objective 🚀

The objective of this assignment is to demonstrate your skills in creating an AI model that is proficient in lip-syncing 👄 i.e. synchronizing an audio file 🔇 with a video file 📹📹. Your task is to ensure the model is accurately matching the lip movements of the characters in the given video file with the corresponding audio file.

## Approach 🚀

In terms of 🔇 to 📹 syncing , **Wav2Lip** model, **A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild**. It's aim to lip-sync unconstrained videos in the wild to any desired target speech. Current works excel at producing accurate lip movements on a static image or videos of specific people seen during the training phase. However, they fail to accurately morph the lip movements of arbitrary identities in dynamic, unconstrained talking face videos, resulting in significant parts of the video being out-of-sync with the new audio.

How the model works, view the detailed [paper](https://arxiv.org/abs/2008.10010)

## Implementation of Wav2Lip model on custom audio and video 🚀

![Screenshot from 2023-09-14 22-05-35](https://github.com/Hg03/lip-sync-wav2lip/assets/69637720/a4c6c516-5391-4c17-a625-5ddaf6a2ff9f)



For implementation, we are referring [rudrabha's github wav2lip](https://github.com/Rudrabha/Wav2Lip) which involves following steps to use the **Wav2Lip** model :

### 🔮 Importing and Installing Wav2Lip model

#### Downloading model and cloning the repo

```python

# Clone the repository
!git clone https://github.com/zabique/Wav2Lip

# download the pretrained model
!wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O '/content/Wav2Lip/checkpoints/wav2lip_gan.pth'
a = !pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl

# !pip uninstall tensorflow tensorflow-gpu
!cd Wav2Lip && pip install -r requirements.txt

#download pretrained model for face detection
!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "/content/Wav2Lip/face_detection/detection/sfd/s3fd.pth"

!pip install -q youtube-dl # if inferencing some youtube videos
!pip install ffmpeg-python
!pip install librosa==0.9.1
```

#### Importing some libraries to support the model process

```python

# We are using google colab, so to load and display video files and support it to download, we require these libraries

from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg

from IPython.display import clear_output 
```

### 🔮 Upload the audio and video file

- As model is not that supported for larger video and audio files
- Use video file (of extension **.mp4**) and audio file (of extension **.wav**) of maximum **30 seconds**.
- Make sure, **length of audio and video should be same** in terms of time duration.
- Make sure, **video should always have the face concentrated whole time**, other model confuse where to sync the audio.
- Remove audio from the video file also.

```python
%cd sample_data/
from google.colab import files
uploaded = files.upload()
%cd ..
```

### 🔮 It's time to sync your video with audio 

- After installing and downloading models, **wav2lip** folder is created, where we have **inference.py** file which is reponsible to built the resulted video

```python
!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "/content/sample_data/input_video.mp4" --audio "/content/sample_data/input_audio.wav"
```

Here **input_video.mp4** and **input_audio.wav** are two files that we've uploaded. 


https://github.com/Hg03/lip-sync-wav2lip/assets/69637720/e415aa4d-085b-4d20-8787-2617c75ad8c5

Link of the audio file - [https://openinapp.co/o9vuj](https://openinapp.co/o9vuj)

So here's the video in which person is speaking in **telugu** language, we are going to sync the **hindi** audio with it.

### 🔮 Display the video

- If you are using colab, to display the video saved in the folder, you can use the below code

```python
from IPython.display import HTML
from base64 import b64encode
mp4 = open('/content/Wav2Lip/results/result_voice.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f"""
<video width="50%" height="50%" controls>
      <source src="{data_url}" type="video/mp4">
</video>""")
```

### 🔮 After running the model, resulted video I got is 

https://github.com/Hg03/lip-sync-wav2lip/assets/69637720/fd331c0d-72f0-4cd6-ab0f-fc88a952cc2b

[Drive link of output video](https://drive.google.com/file/d/1NDJmtoyfmbLn3u9dKSVRhDLVVtwchtSt/view?usp=sharing)

### 🔮 We can tweak some parameters also 

#### resize_factor - We can use resize factor to reduce the video resolution, we might get better results in terms of audio sync, but resolution reduces.

```python
!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "/content/sample_data/input_video.mp4" --audio "/content/sample_data/input_audio.wav" --resize_factor 2
```

- After using resize_factor , we'll get the output vide like below

https://github.com/Hg03/lip-sync-wav2lip/assets/69637720/99f01cf3-49d9-4660-82e0-074daefa8bca

[Drive link of output video](https://drive.google.com/file/d/1QT4HCwe5ojCzxhwRw-6dMVP0_KR-2BpM/view?usp=sharing)

#### pads - Use more padding to include the chin region

```python
!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "/content/sample_data/input_video.mp4" --audio "/content/sample_data/input_audio.wav" --pads 0 20 0 0
```












