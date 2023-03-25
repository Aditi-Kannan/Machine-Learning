# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:14:06 2023

@author: Aditi Kannan
"""

#hebb network

x1=[]
x2=[]
y=[]
bias=1;
ip=4
for i in range(ip):
    x11=int(input("enter value for x1: "))
    x1.append(x11)
    x22=int(input("enter value for x2: "))
    x2.append(x22)
    y1=int(input("enter value for y: "))
    y.append(y1)
    
w1=0
w2=0
b=0
print("\t |x1 \t |x2 \t |bias \t |y \t |dw1 \t |dw2 \t |db \t |w1 \t |w2 \t |b \t |")
for i in range(ip):
    dw1=(x1[i]*y[i])
    dw2=(x2[i]*y[i])
    db=y[i]
    w1 = w1 + dw1
    w2 = w2 + dw2
    b=b+db
    print(" \t |",x1[i]," \t |",x2[i]," \t |",bias," \t |",y[i]," \t |",dw1," \t |",dw2," \t |",db,"\t |",w1," \t |",w2," \t |",b,"\t |")

g0=(x1[0]*w1) + (x2[0]*w2) +b
g1=(x1[1]*w1) + (x2[1]*w2) +b
g2=(x1[2]*w1) + (x2[2]*w2) +b
g3=(x1[3]*w1) + (x2[3]*w2) +b

G=[g0,g1,g2,g3]
Y=[]
for i in G:
    if i>0:
        Y.append(1)
    else:
        Y.append(-1)
