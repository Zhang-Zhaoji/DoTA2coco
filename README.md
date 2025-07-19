# DoTA2coco

convert [DoTA Dataset(Traffic Anomaly)](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)  into coco-detection like structure.

you may need to create a **Soft Link** in DoTA/train and DoTA/val through `ln -s ..`, due to coco-style dataset trainers like [mmdetection](https://github.com/open-mmlab/mmdetection) might try to access `DoTA/train/Dota/train/<img_name>` and `DoTA/train/<img_name>` at the same time.

Due to download GoogleDisk is to some extent hard for chinese developers, I would offer you a [BaiduNetDisk link for DoTA dataset](https://pan.baidu.com/s/1XF1IIT2eVFiM-Tr72qTHuw?pwd=1919)

Or use the link here: ```https://pan.baidu.com/s/1XF1IIT2eVFiM-Tr72qTHuw?pwd=1919``` code: 1919.

I use `cat` to concatenate seperated folders together, which may caused some error using `unzip`. I would recommend using `7z` on windows/linux to obtain flawless documents. you may find some installation guides [here](https://www.7-zip.org/)

I am sorry for people use MacOS that they may have to find their way to do similar things, but I am not pretty sure if there are some developers do such things through MacOS. Anyway you are free to pull any requests I guess...
