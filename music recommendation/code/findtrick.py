import json
import pandas as pd
import os
def is_english_char(ch):
    if ord(ch) not in (97,122) and ord(ch) not in (65,90):
        return False
    return True
title=["album_name","album_uri","artist_name","artist_uri","duration_ms","pos","track_name","track_uri","pid","pname","num_followers"]
inputpath="../data/"
def readOneJson(myjsonname,pid):
    with open(myjsonname) as f1:
        temp = json.loads(f1.read())
        list1=[]
        playlist=list(filter(lambda arg: arg["pid"]==pid,temp['playlists']))[0]
        tracks = pd.DataFrame()
        count=0
        for track in playlist["tracks"]:
            tracks[count] = pd.Series(track)
            tracks.loc["pid"] = playlist["pid"]
            tracks.loc["pname"] =playlist["name"]
            tracks.loc["num_followers"]=playlist["num_followers"]
            count = count + 1
            list1.append(track["track_uri"])
        
        return list1,tracks
def f(pid):
    #mpd.slice.1000-1999.json
    s=(int(pid)//1000)*1000
    e=str(s+999)
    myJson= inputpath+"mpd.slice."+str(s)+"-"+e+".json"
    return readOneJson(myJson, pid)
def getFileLists(dflist):
    allneedURL=[]
    Alltracks = pd.DataFrame([], columns=title)
    for j in dflist:
        mylist,tracks=f(j)
        allneedURL.extend(mylist)
        Alltracks = pd.concat([Alltracks, tracks.T])
    allneedURL=list(set(allneedURL))
    pd.DataFrame(allneedURL, columns=["url"]).to_csv('NeedConcern/needConcernTrick.csv',index=False)
    Alltracks.to_csv('NeedConcern/trickWithDetail.csv',index=False)
    print("Done!")
    return 

  #readOneJson(r"mpd.slice.0-999.json",r"mpd.slice.0-999")
df=pd.read_csv("NeedConcern/pidlist.csv")
getFileLists(df["pid"].tolist())
