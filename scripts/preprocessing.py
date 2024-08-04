import os
import cv2 as cv
import numpy as np
from collections import defaultdict

def preprocessAllVideosFrames(inputPath, outputPath):
    videosFilesNames = defaultdict(list)
    for fileName in os.listdir(inputPath):
        if not (fileName.endswith('.jpg') or fileName.endswith('.png')):
            continue
        videoNum = fileName.split('_')[0]
        videosFilesNames[videoNum].append(fileName)

    for videoNum, videoFilesNames in videosFilesNames.items():
        preprocessVideoFrames(videoNum, videoFilesNames, inputPath, outputPath)

def preprocessVideoFrames(videoNum, videoFilesNames, inputPath, outputPath):
    print(f"------------------------------")
    print(f"Preprocessing Video {videoNum}")
    print(f"------------------------------")

    print(f"Reading the images ...", end='\r')
    imgsNum = len(videoFilesNames)
    imgs = []
    for idx, fileName in enumerate(videoFilesNames):
        print(f"Reading the images ... ({(idx+1)/imgsNum :.3%})", end='\r')
        imgPath = os.path.join(inputPath, fileName)
        img = cv.imread(imgPath)
        imgs.append(img)
    print(f"Reading the images done.         ")

    background = extractingBackground(imgs)

    print(f"Processing ...", end='\r')
    for idx, (frame, fileName) in enumerate(zip(imgs, videoFilesNames)):
        print(f"Processing ... ({(idx+1)/imgsNum :.3%})", end='\r')
        result = preprocessFrame(background, frame)
        cv.imwrite(os.path.join(outputPath, fileName), result)
    print(f"Processing done.         ")
    print()

def preprocessVideoFile(FileName, inputPath, outputPath):
    print(f"------------------------------")
    print(f"Preprocessing Video {FileName}")
    print(f"------------------------------")

    print(f"Reading the frames ...", end='\r')
    videoPath = os.path.join(inputPath, FileName)
    cap = cv.VideoCapture(videoPath)
    capFps = cap.get(cv.CAP_PROP_FPS)
    capFramesCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    capFrameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    capFrameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    if not cap.isOpened():
        print(f"\nError: Cannot open video file {videoPath}\n")
        return
    frames = []
    for idx in range(capFramesCount):
        print(f"Reading the frames ... ({(idx+1)/capFramesCount :.3%})", end='\r')
        sucess, img = cap.read()
        if not sucess:
            break        
        frames.append(img)
    framesNum = len(frames)
    cap.release()
    print(f"Reading the frames done.         ")

    background = extractingBackground(frames)

    print(f"Processing ...", end='\r')
    out = cv.VideoWriter(os.path.join(outputPath, FileName), cv.VideoWriter_fourcc(*'mp4v'), capFps, (capFrameWidth, capFrameHeight), isColor=False)
    for idx, frame in enumerate(frames):
        print(f"Processing ... ({(idx+1)/framesNum :.3%})", end='\r')
        result = preprocessFrame(background, frame)
        out.write(result)
    out.release()
    print(f"Processing done.         ")
    print()

def extractingBackground(imgs):
    print(f"Extracting the background ...", end='\r')
    framesStack = np.stack(imgs, axis=-1)
    background = np.median(framesStack, axis=-1).astype(np.uint8)
    del framesStack
    print(f"Extracting the background done.")
    return background

def preprocessFrame(background, frame):
    foreground = cv.absdiff(frame, background)
    foreground_gray = cv.cvtColor(foreground, cv.COLOR_BGR2GRAY)
    foreground_bin = cv.threshold(foreground_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1].astype(np.uint8)
    foreground_cln = cv.erode(foreground_bin, np.ones((3, 3), np.uint8), iterations=2)
    foreground_cln = cv.dilate(foreground_cln, np.ones((3, 3), np.uint8), iterations=50)
    foreground_cln_3c = np.array([[p] * 3 for p in foreground_cln.flatten()], dtype=np.uint8).reshape((*foreground_cln.shape, 3))
    result = frame & foreground_cln_3c
    return result



# * To preprocess the whole dataset, run:
# INPUT_PATH = './datasets/org/train'
# OUTPUT_PATH = './datasets/preprocessed/train'
# if not os.path.exists(OUTPUT_PATH):
#     os.makedirs(OUTPUT_PATH)
# preprocessAllVideosFrames(INPUT_PATH, OUTPUT_PATH)

# INPUT_PATH = './datasets/org/test'
# OUTPUT_PATH = './datasets/preprocessed/test'
# if not os.path.exists(OUTPUT_PATH):
#     os.makedirs(OUTPUT_PATH)
# preprocessAllVideosFrames(INPUT_PATH, OUTPUT_PATH)

# INPUT_PATH = '/datasets/org/val'
# OUTPUT_PATH = '/datasets/preprocessed/val'
# if not os.path.exists(OUTPUT_PATH):
#     os.makedirs(OUTPUT_PATH)
# preprocessAllVideosFrames(INPUT_PATH, OUTPUT_PATH)

# * To preprocess a video to predict, run:
# INPUT_PATH = '/path/to/input/folder'
# OUTPUT_PATH = '/path/to/output/folder'
# VIDEO_NAME = 'video_name.mp4'
# if not os.path.exists(OUTPUT_PATH):
#     os.makedirs(OUTPUT_PATH)
# preprocessVideoFile(VIDEO_NAME, INPUT_PATH, OUTPUT_PATH)
