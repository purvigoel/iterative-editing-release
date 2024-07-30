from pytube import YouTube
from lib.utils.demo_utils import download_youtube_clip
import ipdb
from tqdm import tqdm


# video_file = 'https://www.youtube.com/watch?v=JALBN7OEEyM'  # Nadel Practice
# video_file = 'https://www.youtube.com/watch?v=hGczqANGDDY'
# video_file = 'https://www.youtube.com/shorts/vlT4M4A5XjI' # Short, does it work? YES


video_files = [
	# # Pro Women
	# 'https://www.youtube.com/watch?v=chYyI5LYhmA',
	# # Pro Men
	# 'https://www.youtube.com/watch?v=M3eC5egY3wA',
	# 'https://www.youtube.com/watch?v=UEhEvPrTjwc',
	# 'https://www.youtube.com/watch?v=OF6i6n_1SJg',
	# 'https://www.youtube.com/watch?v=iWthqNronaY',
	# 'https://www.youtube.com/watch?v=Roc4Yao6iqE',
	# 'https://www.youtube.com/watch?v=0edfZQPFeDc',
	# 'https://www.youtube.com/watch?v=hCokmem5zoA',
	# # College Men
	# 'https://www.youtube.com/watch?v=hGczqANGDDY',
	# 'https://www.youtube.com/watch?v=LSW7Ey4fcSs'
	# College Women
	'https://www.youtube.com/watch?v=vCPEEMoo-9M'
]
 

for video_file in tqdm(video_files):
	download_youtube_clip(video_file, '/home/groups/syyeung/wangkua1/videos/youtube')

