from pytube import YouTube
# not necessary, just for demo purposes.
from pprint import pprint
import re
import urllib.request
import urllib.error
import sys
import time

url='https://www.youtube.com/playlist?list=PLRwlRpYDDfLCR5KA8cQp8Pf_7lh5FfdAY'

def crawl(url):
    sTUBE = ''
    cPL = ''
    amp = 0
    final_url = []
     
    if 'list=' in url:
        eq = url.rfind('=') + 1
        cPL = url[eq:]
             
    else:
        print('Incorrect Playlist.')
        exit(1)
     
    try:
        yTUBE = urllib.request.urlopen(url).read()
        sTUBE = str(yTUBE)
    except urllib.error.URLError as e:
        print(e.reason)
     
    tmp_mat = re.compile(r'watch\?v=\S+?list=' + cPL)
    mat = re.findall(tmp_mat, sTUBE)
 
    if mat:
           
        for PL in mat:
            yPL = str(PL)
            if '&' in yPL:
                yPL_amp = yPL.index('&')
            final_url.append('http://www.youtube.com/' + yPL[:yPL_amp])
 
        all_url = list(set(final_url))
 
    else:
        print('No videos found.')
        exit(1)
    return all_url

video_list_url=crawl('https://www.youtube.com/playlist?list=PLRwlRpYDDfLCR5KA8cQp8Pf_7lh5FfdAY')

# crawl video from youtube

def crawl_from_youtube(urls, path, resolution, setname=None):
    yt = YouTube(urls)
    print(yt.get_videos())
    
    if setname:
        yt.set_filename(setname)
    
    video = yt.get('mp4', resolution)
    video.download(path)

crawl_from_youtube(urls=video_list_url[0],resolution='720p',path='/home/slave1/桌面/GC')

# crawl them all
for video in video_list_url:
    crawl_from_youtube(urls=video, resolution='720p', path='/home/slave1/桌面/GC')
