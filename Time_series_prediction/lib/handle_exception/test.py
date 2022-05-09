import pandas as pd
import handle_exp

Ser1 = pd.read_excel(r"Monthly_static_w.xlsx")["weekly"]
list=Ser1.tolist()
list1=handle_exp.remove_Outranges(list)
print(list1)
print("---------------------------------")

Ser1 = pd.read_excel(r"Monthly_static_w.xlsx")["weekly"]
list=Ser1.tolist()
list2=handle_exp.replace_Outranges_with_th(list)
print(list2)
print("---------------------------------")

Ser1 = pd.read_excel(r"Monthly_static_w.xlsx")["weekly"]
list=Ser1.tolist()
list3=handle_exp.replace_Outranges_with_avg(list)
print(list3)
print("---------------------------------")
