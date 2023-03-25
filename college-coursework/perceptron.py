# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:24:31 2023

@author: Aditi Kannan
"""

x1=[]
x2=[]
y=[]
Y=[];
bias=1
ip=4
typeofip=input("type of input: \na.Binary input bipolar target\nb.Bipolar input and target\n")

epochs=int(input("enter no. of epochs: "))
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
alpha=1
l=[['x1', 'x2', 'bias', 't', 'y', 'f(y)', 'dw1', 'dw2', 'db', 'w1', 'w2', 'b']]

while epochs!=0:
    
    for i in range(ip):
        yin=b+(x1[i]*w1)+(x2[i]*w2)
        if typeofip=='a':
            if yin>0.2:
                fyin=1     
            elif yin<0.2 and yin>-0.2:
                fyin=0
            
        elif typeofip=='b':
            if yin>0:
                fyin=1     
            elif yin==0:
                fyin=0
            else:
                fyin=-1
                
        if(fyin!=y[i]):
            dw1=alpha*x1[i]*y[i]
            dw2=alpha*x2[i]*y[i]
            db=alpha*y[i]
            w1 = w1 + dw1
            w2 = w2 + dw2
            b=b+db
            temp=[x1[i],x2[i],bias,y[i],yin,fyin,dw1,dw2,db,w1,w2,b]
            l.append(temp)
            
        else:
            dw1=0
            dw2=0
            db=0
            temp=[x1[i],x2[i],bias,y[i],yin,fyin,dw1,dw2,db,w1,w2,b]
            l.append(temp)
           
    epochs=epochs-1

for row in l:
    print("{: >5} {: >5} {: >5}{: >5} {: >5} {: >5}{: >5} {: >5} {: >5}{: >5} {: >5} {: >5}".format(*row))

g0=(x1[0]*w1) + (x2[0]*w2) +b
g1=(x1[1]*w1) + (x2[1]*w2) +b
g2=(x1[2]*w1) + (x2[2]*w2) +b
g3=(x1[3]*w1) + (x2[3]*w2) +b

G=[g0,g1,g2,g3]
Y=[]

for i in range(len(G)):
    if typeofip=='a':
        if G[i]>0.2:
            Y.append(1)     
        elif G[i]<0.2 and G[i]>-0.2:
            Y.append(0)
            
    elif typeofip=='b':
        if G[i]>0:
            Y.append(1)     
        elif G[i]==0:
            Y.append(0)
        else:
            Y.append(-1)
