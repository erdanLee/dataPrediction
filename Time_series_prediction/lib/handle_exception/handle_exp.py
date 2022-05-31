
import pandas as pd
import numpy as np

def remove_Outranges(list, silenceMode=True):
    while True:
        ser1=list_to_ser(list)
        outrange=three_sigma(ser1)
        if not silenceMode:
            print("outrange:",outrange)
        if len(outrange) != 0:
            list.remove(max(outrange))
            print("删除一个异常值：",max(outrange))
        else:
            break
    return list

def replace_Outranges_with_th(list):
    while True:
        ser1=list_to_ser(list)
        outrange=three_sigma(ser1)
        th=ser1.mean()+3*ser1.std()
        print("outrange:",outrange)
        if len(outrange) != 0:
            list = [int(th) if x == max(outrange) else x for x in list]
            print("替换异常值 %s 为 %s:"%(max(outrange),int(th)))
        else:
            break
    return list

def replace_Outranges_with_avg(list):
    while True:
        ser1=list_to_ser(list)
        outrange=three_sigma(ser1)
        avg=ser1.mean()
        print("outrange:",outrange)
        if len(outrange) != 0:
            list = [int(avg) if x == max(outrange) else x for x in list]
            print("替换异常值 %s 为 %s:"%(max(outrange),int(avg)))
        else:
            break
    return list

# 定义3σ法则识别异常值函数
def three_sigma(Ser1) -> pd.DataFrame:
    '''
    Ser1：表示传入DataFrame的某一列。
    '''
    rule = (Ser1.mean()-3*Ser1.std()>Ser1) | (Ser1.mean()+3*Ser1.std()< Ser1)
    index = np.arange(Ser1.shape[0])[rule]
    outrange_list = Ser1.iloc[index].head().tolist()
    return outrange_list


def list_to_ser(list):
    return pd.Series(list,index=np.arange(len(list)))
