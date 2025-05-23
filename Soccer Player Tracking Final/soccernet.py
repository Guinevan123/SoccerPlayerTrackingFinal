from SoccerNet.Downloader import SoccerNetDownloader
import os
from tqdm import tqdm
import urllib.request

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=r"C:\Users\User\Desktop\SoccerNet")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv"], split=["train"])