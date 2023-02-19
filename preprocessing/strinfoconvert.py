# 字符匹配
import re

# 时差
import time as t
import datetime as d


def match_str(strtarget, strsrc):
    is_match = re.search(strtarget, strsrc, re.M | re.I)
    if is_match == None:
        return False
    else:
        return True


def getDateInterval(nowDate):
    matchNowDate = re.search(r'(\d+)/(\d+)/(\d+)', nowDate, re.M | re.I)
    year = int(matchNowDate.group(1))
    month = int(matchNowDate.group(2))
    day = int(matchNowDate.group(3))
    cur = d.datetime(year, month, day)
    pre = d.datetime(2011, 1, 1)
    dayInterval = (cur - pre).days
    return dayInterval


# 匹配时间 循环 r'(\d+:\d+)-(\d+:\d+)‘
def getStopTime(remarkstr):
    strStop = ' stop pump'
    matchStopPump = re.search(strStop, remarkstr, re.M | re.I)

    if matchStopPump == None:
        return 0
    else:
        idxEnd = matchStopPump.regs[0][0]
        sumStopTime = 0
        matchTime = re.search(r'(\d+:\d+)-(\d+:\d+)', remarkstr, re.M | re.I)

        if matchTime == None:
            matchTime = re.search(r'(\d+:\d+)', remarkstr, re.M | re.I)
            if matchTime == None:
                return 0
            timestr = matchTime.group(1)
            return myDate(timestr, '23:59')

        idxStart = matchTime.regs[2][1]
        subStr = remarkstr[idxStart:idxEnd]

        while 1:
            timestr = matchTime.group(1)
            timeend = matchTime.group(2)
            sumStopTime += myDate(timestr, timeend)

            matchTime = re.search(r'(\d+:\d+)-(\d+:\d+)', subStr, re.M | re.I)
            if matchTime == None:
                break
            idxStart = matchTime.regs[2][1]
            subStr = subStr[idxStart:]

    return sumStopTime


# 定义时间差函数
def myDate(date1, date2):
    date1 = t.strptime(date1, "%H:%M")
    matchObjtimEnd = re.search(r'(\d+):(\d+)', date2, re.M | re.I)

    if matchObjtimEnd.group(1) == '24':
        date2 = '23:59'
    date2 = t.strptime(date2, "%H:%M")
    startTime = t.strftime("%H:%M", date1)
    endTime = t.strftime("%H:%M", date2)

    startTime = d.datetime.strptime(startTime, "%H:%M")
    endTime = d.datetime.strptime(endTime, "%H:%M")
    date = endTime - startTime

    # print(date.seconds / 3600.0)
    # print(date)
    return date.seconds / 3600.0


