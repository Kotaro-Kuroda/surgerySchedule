import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

# -*- coding: utf-8 -*-


def create_ganttchart(url):
    mpl.rcParams['font.family'] = "IPAexGothic"
    df = pd.ExcelFile(url)
    shiftLength = 480
    for sheet in df.sheet_names:
        excel = pd.read_excel(
            url,
            sheet_name=sheet, header=0, index_col=0)
        ind = excel.columns
        label_x = []
        plt.figure(figsize=(30, 10), dpi=50)
        plt.plot([shiftLength, shiftLength], [0, len(ind)], "black")
        for i in range(0, 96):
            hour = int((i * 5) / 60 + 9)
            minute = (i * 5) % 60
            time = ""
            if (minute < 10):
                time = str(hour) + ":" + "0" + str(minute)
            else:
                time = str(hour) + ":" + str(minute)
            label_x.append(time)
        k = 0
        l = 0
        m = 0
        for i in range(0, len(excel)):
            data = excel.iloc[i, :]
            btm = 0
            for j in range(0, i):
                btm += np.array(excel.iloc[j, :].values)
            if (excel.index[i] == "空白"):
                plt.barh(
                    ind, data, height=0.8, left=btm, align='center',
                    color="white")
            else:
                if ("準備" in excel.index[i]):
                    if k == 0:
                        plt.barh(ind, data, height=0.8, left=btm, align='center', color="red", label="preparation")
                        k += 1
                    else:
                        plt.barh(ind, data, height=0.8, left=btm, align='center', color="red")
                if("手術" in excel.index[i]):
                    if l == 0:
                        plt.barh(ind, data, height=0.8, left=btm, align='center', color="blue", label="surgery")
                        l += 1
                    else:
                        plt.barh(ind, data, height=0.8, left=btm, align='center', color="blue")
                if ("清掃" in excel.index[i]):
                    if m == 0:
                        plt.barh(ind, data, height=0.8, left=btm, align='center', color="yellow", label="cleaning")
                        m += 1
                    else:
                        plt.barh(ind, data, height=0.8, left=btm, align='center', color="yellow")
        plt.grid(axis="x", alpha=0.8)
        plt.title("手術スケジュール", fontsize=36)
        timeList = []
        time = ""
        for i in range(0, max(480, max(btm)) + 30, 30):
            hour = int(i / 60) + 9
            minute = i % 60
            if minute < 10:
                time = str(hour) + ":" + "0" + str(minute)
            else:
                time = str(hour) + ":" + str(minute)
            timeList.append(time)
        plt.tick_params(labelsize=18)
        plt.xticks(range(0, max(shiftLength, max(btm)) + 30, 30), timeList)
        plt.legend(bbox_to_anchor=(1.06, 0), loc='lower right', borderaxespad=0)
        plt.savefig(sheet + ".png")
    plt.show()


def create_surgeon_gantt_chart(url):
    mpl.rcParams['font.family'] = "IPAexGothic"
    shiftLength = 480
    df = pd.ExcelFile(url)
    for sheet in df.sheet_names:
        excel = pd.read_excel(
            url,
            sheet_name=sheet, header=0, index_col=0)
        ind = excel.columns
        label_x = []
        plt.figure(figsize=(50, 10), dpi=50)
        plt.plot([shiftLength, shiftLength], [0, len(ind)], "black")
        for i in range(0, 96):
            hour = int((i * 5) / 60 + 9)
            minute = (i * 5) % 60
            time = ""
            if (minute < 10):
                time = str(hour) + ":" + "0" + str(minute)
            else:
                time = str(hour) + ":" + str(minute)
            label_x.append(time)
        l = 0
        for i in range(0, len(excel)):
            data = excel.iloc[i, :]
            btm = 0
            for j in range(0, i):
                btm += np.array(excel.iloc[j, :].values)
            if excel.index[i] == "空白":
                plt.barh(
                    ind, data, height=0.8, left=btm, align='center',
                    color="white")
            else:
                if l == 0:
                    plt.barh(ind, data, height=0.8, left=btm, align='center', color="blue", label="surgery")
                    l += 1
                else:
                    plt.barh(ind, data, height=0.8, left=btm, align='center', color="blue")
        plt.grid(axis="x", alpha=0.8)
        plt.title(sheet)
        plt.legend(bbox_to_anchor=(1.06, 0), loc='lower right', borderaxespad=0)
        timeList = []
        time = ""
        for i in range(0, max(480, max(btm)) + 30, 30):
            hour = int(i / 60) + 9
            minute = i % 60
            if minute < 10:
                time = str(hour) + ":" + "0" + str(minute)
            else:
                time = str(hour) + ":" + str(minute)
            timeList.append(time)
        plt.tick_params(labelsize=18)
        plt.xticks(range(0, max(shiftLength, max(btm)) + 30, 30), timeList)
        plt.savefig(sheet + ".png")
    plt.show()
