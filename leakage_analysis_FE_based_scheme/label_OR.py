# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:48:34 2020

@author: NaveenHK
"""


words =['org', 'mail', 'list', 'mailman', 'wrote', 'code', 'listinfo', 'help', 'comment', 'perl'] 
#['post', 'unsubscrib', 'stat', 'reproduc', 'minim', 'guid', 'ethz', 'html', 'math', 'contain']

print(len(words))
mails = open('./final_out/email_length.txt','r')
category = open('./categoryIndex','w')
mail = mails.readlines()

count=0
for i in mail:
    val =False
    item = i.strip('\n').split()
    #print((item))
    m= item[1].split('/')
    nam = m[-1]
    file = open('./emails/'+nam,'r')
    mail = file.read().split()
    #print(mail)
    for j in range(len(words)):
        if words[j] in mail:
            category.write('present;'+nam+'\n')
            count+=1
            val=True
            break
    if val==False:
        category.write('notpresent;'+nam+'\n')
        count+=1
        
    print(count)        
    file.close()
    #print(count)
category.close()
print(count)
    