# This is a sample Python script.
from pickletools import uint8

import cv2
import math
import numpy as np
import random
from moviepy import *
import string
import yt_dlp
from PIL import Image, ImageFont, ImageDraw
import time
import torch
import multiprocessing
from itertools import repeat
import cProfile
import os
import shutil
import torch.nn.functional as F
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def Create_letter_images(framewidth, frameheight, textscale):
    '''letters = [' ', '.', "'", '`', '^', '"', ',', ':', ';', 'I', 'l', '!', 'i', '>', '<', '~', '+', '_', '-', '?', ']',
              '[', '}', '{', '1', ')', '(', '|', '\\', '/', 't', 'f', 'j', 'r', 'x', 'n', 'u', 'v', 'c', 'z', 'X', 'Y',
              'U', 'J', 'C', 'L', 'Q', '0', 'O', 'Z', 'm', 'w', 'q', 'p', 'd', 'b', 'k', 'h', 'a', 'o', '*', '#', 'M',
              'W', '&', '8', '%', 'B', '@', '$']'''
    #letters = list("+<>i!lI?/|(){}[]tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")
    #letters = ['z', 'X', 'Y', 'U', 'J', 'L', 'm', 'w', 'q', 'p', 'd','b', 'k', 'h','K','%','$','@','#']
    letters = ["D", "X", "Q", "K", "R", "N", "H", "&", "%", "$", "8", "B", "W", "M", "#", "@"]
    np.set_printoptions(threshold=np.inf)
    framewidth = int(framewidth)
    frameheight = int(frameheight)
    textscale = ClosestDiv(frameheight, textscale)
    fontheight = int(frameheight / textscale)
    fontpoint = math.ceil(fontheight * .75)
    fnt = ImageFont.truetype('Fonts/SpaceMono-Regular.ttf', fontpoint)
    fontwidth = int(fnt.getlength('a'))
    letterimage = []
    for letter in letters:
        letterimg1 = np.zeros((fontheight, fontwidth, 3), np.uint8)
        letterimg1 = Image.fromarray(letterimg1)
        draw1 = ImageDraw.Draw(letterimg1)
        draw1.text((0, 0), letter, fill=tuple((1, 1, 1)), font=fnt)
        letterimage.append(np.array(letterimg1))  # np.divide(np.array(letterimg1), 255)
    return torch.from_numpy(np.array(letterimage,dtype=np.uint8)).cuda()

def gcd(x, y):
    if x == 0:
        return y
    return gcd(y % x, x)


def common_divs(x, y):
    divs = []
    GCD = gcd(x, y)
    for i in range(1, int(math.sqrt(GCD)) + 1):
        if GCD % i == 0:
            divs.append(i)
    return divs


def ClosestDiv(n, m):
    for i in range(int(m / 2)):
        if n % (m + i) == 0:
            return m + i
        elif n % (m - i) == 0:
            return m - i


def GetVideoInfo(video):
    FrameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    FrameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    FrameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = video.get(cv2.CAP_PROP_FPS)
    print('fps:', end='')
    print(FPS)
    return FrameCount, FrameWidth, FrameHeight, FPS


def CreateNewVideo(path, framewidth, frameheight, FPS):
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    size = (framewidth, frameheight)
    print(size)
    return cv2.VideoWriter(path, fourcc, FPS, size)


def MapToAscii(density):
    ''' letters = [' ', '.', "'", '`', '^', '"', ',', ':', ';', 'I', 'l', '!', 'i', '>', '<', '~', '+', '_', '-', '?', ']',
              '[', '}', '{', '1', ')', '(', '|', '\\', '/', 't', 'f', 'j', 'r', 'x', 'n', 'u', 'v', 'c', 'z', 'X', 'Y',
              'U', 'J', 'C', 'L', 'Q', '0', 'O', 'Z', 'm', 'w', 'q', 'p', 'd', 'b', 'k', 'h', 'a', 'o', '*', '#', 'M',
              'W', '&', '8', '%', 'B', '@', '$'] '''
    #letters = list("+<>i!lI?/|(){}[]tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")
    letters = [';', 'i', 'l', '!', '>', '<',
               '~', '+', '_', '/', 't', 'f', 'j', 'r', 'x', 'n', 'u', 'v', 'c',
               'z', 'X', 'Y', 'U', 'J', 'L', 'C', 'O', 'm', 'w', 'q', 'p', 'd',
               'b', 'k', 'h', 'a', 'o']
    a = letters[math.floor((density * (len(letters)-1)) / 255)]
    return a


def GetTextLine(sums, fontsize):
    text = []
    color = []

    for sum in sums:
        color.append(sum.tolist())
        a = MapToAscii(int(sum.max().item()))
        text.append(a[0])
    return text, np.array(color,dtype=np.uint8)


def GetColorlessTextLine(sums):
    text = []
    for sum in sums:
        text.append(MapToAscii(sum.item()))
    return text


def ColorlessAsciiFrame(framesums, font, framesize, fontsize):
    asciiframe = np.zeros((int(framesize[1] / fontsize[1]), fontsize[1], framesize[0], 3))
    frameline = np.zeros((fontsize[1], framesize[0], 3))
    '''
    for i in range(len(framesums)):
        text = GetColorlessTextLine(framesums[i])
        for j in range(len(text)):
            frameline[:, j * fontsize[0]:(j + 1) * fontsize[0]] = letterimage[text[j]]
        frameline[:, framesize[0] - (framesize[0] % fontsize[0]):framesize[0]] = np.zeros(
            (fontsize[1], framesize[0] % fontsize[0], 3))
        asciiframe[i] = frameline
    '''
    asciiframe = torch.from_numpy(np.array(asciiframe))
    asciiframe = torch.flatten(asciiframe, start_dim=0, end_dim=1).numpy()
    return asciiframe


def GetColorlessAsciiFrames(frames, framewidth, frameheight, textscale, indexs, framedic, output_rez):
    AsciiFrames = []
    letterimage = []
    framewidth = int(framewidth)
    frameheight = int(frameheight)
    if output_rez != (0, 0):
        textscale = int(gcd(frameheight, output_rez[1]) / 4)
        outfontheight = int(output_rez[1] / textscale)
        outfontpoint = math.ceil(outfontheight * .75)
        outfnt = ImageFont.truetype('Fonts/SpaceMono-Regular.ttf', outfontpoint)
        outfontwidth = int(outfnt.getlength('a'))
    else:
        textscale = ClosestDiv(frameheight, textscale)
    fontheight = int(frameheight / textscale)
    fontpoint = math.ceil(fontheight * .75)
    fnt = ImageFont.truetype('Fonts/SpaceMono-Regular.ttf', fontpoint)
    fontwidth = int(fnt.getlength('a'))
    if output_rez != (0, 0):
        letterimage = Create_letter_images(output_rez[0], output_rez[1], textscale)
    else:
        letterimage = Create_letter_images(framewidth, frameheight, textscale)
    splitframes = []
    torch.cuda.empty_cache()
    start = time.time()
    for frame in frames:
        frame = torch.from_numpy(frame).split(int(fontheight), dim=0)
        temp_frame = []
        for i in range(len(frame)):
            temp_frame.append(torch.stack(frame[i].split(int(fontwidth), dim=1)[:-1]))
        splitframes.append(torch.from_numpy(np.array(temp_frame)))
    framessums = torch.from_numpy(np.array(splitframes)).to('cuda').div(fontwidth * fontheight * 3).sum(
        dim=(3, 4, 5)).to(
        'cpu')
    if output_rez != (0, 0):
        for framesums in framessums:
            frame = ColorlessAsciiFrame(framesums, outfnt, output_rez, (outfontwidth, outfontheight))
            AsciiFrames.append(frame)
    else:
        for framesums in framessums:
            frame = ColorlessAsciiFrame(framesums, fnt, (framewidth, frameheight), (fontwidth, fontheight))
            AsciiFrames.append(frame)
    AsciiFrames = torch.from_numpy(np.array(AsciiFrames))
    for i in range(len(indexs)):
        framedic[indexs[i]] = AsciiFrames[i]
    return

def AsciiFrame(framesums, framesize, fontsize, letterimage):
    colorframe = framesums.unsqueeze(2).unsqueeze(2).expand(-1,-1,fontsize[1],fontsize[0],-1).permute(0,2,1,3,4)
    #asciiframe = letterimage[torch.floor(torch.mul(torch.max(framesums.to('cuda'), dim=2).values, 50 / 255).long()).to('cpu')]
    luminance = (0.2126 * framesums[..., 0] +
                 0.7152 * framesums[..., 1] +
                 0.0722 * framesums[..., 2])
    gamma = 2.2  # tweak for brighter/darker output
    luminance = (luminance / 255.0) ** (1 / gamma)  # normalize to 0..1
    asciiframe = letterimage[
        torch.floor(
            torch.mul(luminance.to('cuda'), (len(letterimage)-1))
        ).long()
    ]
    asciiframe = asciiframe.permute(0,2,1,3,4)
    asciiframe =  torch.flatten(torch.flatten(asciiframe, start_dim=0, end_dim=1),start_dim=1,end_dim=2)
    colorframe = torch.flatten(torch.flatten(colorframe, start_dim=0, end_dim=1),start_dim=1,end_dim=2)
    widthpad = framesize[0] - asciiframe.shape[1]
    left_pad = widthpad // 2
    right_pad = widthpad - left_pad
    ascii_t = asciiframe.permute(2, 0, 1)
    color_t = colorframe.permute(2, 0, 1)
    ascii_t = F.pad(ascii_t, (left_pad, right_pad, 0, 0))
    color_t = F.pad(color_t, (left_pad, right_pad, 0, 0))
    asciiframe = ascii_t.permute(1, 2, 0)
    colorframe = color_t.permute(1, 2, 0)
    return asciiframe.mul(colorframe)


def GetAsciiFrames(frames, framewidth, frameheight, fontwidth, fontheight, letterimage):
    framessums = torch.from_numpy(frames).cuda().permute(0,3,1,2).unfold(2,fontheight,fontheight).unfold(3,fontwidth,fontwidth).sum(dim=(-2,-1)).permute(0,2,3,1)/(fontwidth * fontheight)
    AsciiFrames = torch.empty((len(framessums), frameheight, framewidth, 3), dtype=torch.uint8, device='cuda')
    for i, framesums in enumerate(framessums):
        AsciiFrames[i] = AsciiFrame(framesums, (framewidth,frameheight), (fontwidth, fontheight), letterimage)
    return AsciiFrames.cpu().numpy()


def AddFramesToVideo(frames, video, frameskip):
    for i in range(len(frames)):
        for j in range(frameskip):
            video.write(frames[i])

def TimedGetAsciiFrames(frames, framewidth, frameheight, textscale, indexs, framedic, output_rez):
    profiler = cProfile.Profile()
    profiler.enable()
    GetAsciiFrames(frames, framewidth, frameheight, textscale, indexs, framedic, output_rez)
    profiler.disable()
    profiler.dump_stats(f"profiles/profile_{os.getpid()}.prof")
    return


def CreateAsciiVideo(path, textscale, resize=(0, 0), output_rez=(0, 0), frameskip=0, FramesPer=50):
    frameskip = frameskip + 1
    vid = cv2.VideoCapture(path)

    FrameCount, FrameWidth, FrameHeight, FPS = GetVideoInfo(vid)
    if resize != (0, 0):
        FrameWidth = resize[0]
        FrameHeight = resize[1]
    print(FrameCount)
    if output_rez != (0, 0):
        OutputVid = CreateNewVideo(path[:path.index('.mp4')] + '_ascii.mp4', output_rez[0], output_rez[1], round(FPS))
    else:
        OutputVid = CreateNewVideo(path[:path.index('.mp4')] + '_ascii.mp4', FrameWidth, FrameHeight, round(FPS))
    suc = True
    Frames = []
    i = 1
    j = 0
    framewidth = int(FrameWidth)
    frameheight = int(FrameHeight)
    if output_rez != (0, 0):
        textscale = int(gcd(frameheight, output_rez[1]) / 4)
        outfontheight = int(output_rez[1] / textscale)
        outfontpoint = math.ceil(outfontheight * .75)
        outfnt = ImageFont.truetype('Fonts/SpaceMono-Regular.ttf', outfontpoint)
        outfontwidth = int(outfnt.getlength('a'))
    else:
        textscale = ClosestDiv(frameheight, textscale)
    fontheight = int(frameheight / textscale)
    fontpoint = math.ceil(fontheight * .75)
    fnt = ImageFont.truetype('Fonts/SpaceMono-Regular.ttf', fontpoint)
    fontwidth = int(fnt.getlength('a'))
    if output_rez != (0, 0):
        letterimage = Create_letter_images(output_rez[0], output_rez[1], textscale)
    else:
        letterimage = Create_letter_images(framewidth, frameheight, textscale)
    while vid.isOpened():
        suc, image = vid.read()
        if suc:
            if resize != (0, 0):
                image = cv2.resize(image, resize)
            if i % frameskip == 0:
                Frames.append(image)
            if len(Frames) == FramesPer:
                start = time.time()
                processFrames = np.array(Frames,dtype=np.uint8)
                AsciiFrames = GetAsciiFrames(processFrames, FrameWidth, FrameHeight,fontwidth, fontheight,letterimage)
                AddFramesToVideo(AsciiFrames, OutputVid, frameskip)
                Frames = []
                end = time.time()
                print('Percent Done: '+str((j*100) // FrameCount)+'%')
                '''print(str(FramesPer) + " Frames time:", end=" ")
                print(end - start)'''
            i = i + 1
            j = j + 1
        else:
            AsciiFrames = {}
            processFrames = np.array(Frames, dtype=np.uint8)
            AsciiFrames = GetAsciiFrames(processFrames, FrameWidth, FrameHeight, fontwidth, fontheight, letterimage)
            ''' `AsciiFrames = manager.dict()
            processFrames = np.array_split(np.array(Frames), cpucount)
            indexs = np.array_split(np.arange(len(Frames)), cpucount)
            processes = []
            for j in range(cpucount):
                p = multiprocessing.Process(target=TimedGetAsciiFrames,
                                            args=(
                                                processFrames[j], FrameWidth, FrameHeight, textscale, indexs[j],
                                                AsciiFrames, output_rez))
                processes.append(p)
                p.start()
            for process in processes:
                process.join()'''
            # AsciiFrames = GetAsciiFrames(Frames, FrameWidth, FrameHeight, textscale)
            AddFramesToVideo(AsciiFrames, OutputVid, frameskip)
            break
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    prof_dir = os.path.join(BASE_DIR, 'profiles')
    if os.path.exists(prof_dir):
        shutil.rmtree(prof_dir)
    os.makedirs(prof_dir)
    profiler = cProfile.Profile()
    profiler.enable()
    print(multiprocessing.cpu_count())
    file_path = os.path.join(BASE_DIR, 'Videos')
    file_name = 'Dare'
    url = 'https://www.youtube.com/watch?v=7G_MqnMBCmI'  # example URL — replace with your own
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(file_path, file_name + '.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    videosound = VideoFileClip('Videos/' + file_name + '.mp4')
    # Get video info
    video_path = os.path.join(file_path, file_name + '.mp4')
    vid = cv2.VideoCapture(video_path)
    FrameCount, FrameWidth, FrameHeight, FPS = GetVideoInfo(vid)
    print(f"Resolution: {FrameWidth}x{FrameHeight}")
    print(f"Total Frames: {FrameCount}")
    print(f"FPS: {FPS}")

    # Prompt user for textscale and FramesPer
    textscale_input = input("Enter textscale (default 120): ")
    FramesPer_input = input("Enter FramesPer batch (default 50): ")

    textscale = int(textscale_input) if textscale_input else 120
    FramesPer = int(FramesPer_input) if FramesPer_input else 50

    start = time.time()
    CreateAsciiVideo(video_path, textscale=textscale, frameskip=0, FramesPer=FramesPer)
    end = time.time()
    print('total time')
    print(end - start)
    video = VideoFileClip("Videos/" + file_name + "_ascii.mp4")
    video = video.with_audio(videosound.audio)
    video.write_videofile("Videos/" + file_name + "_ascii_with_sound.mp4")
    profiler.disable()
    profiler.dump_stats(f"profiles/profile_main.prof")

    # YouTube('https://www.youtube.com/watch?v=iR-K2rUP86M').streams.get_audio_only().download(
    #    output_path=file_path + '\\audio')
    # frames = CreateAsciiVideo(file_path + '\\test.mp4')
    # for frame in frames[48][0]:
    #    print(frame)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/'''
