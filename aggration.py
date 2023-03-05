import numpy as np
output_path='C:/Users/Ye Yue/Desktop/CPF-master/outputs/'
pro_list=['ant-1.7','camel-1.2','camel-1.6','jedit-3.2','log4j-1.0','lucene-2.0','lucene-2.2','lucene-2.4','poi-1.5',
          'poi-2.0','poi-2.5','poi-3.0','synapse-1.0','synapse-1.1','synapse-1.2','velocity-1.4','velocity-1.6','xalan-2.5',
          'xalan-2.6','xerces-1.2','xerces-1.3']
teacher_list=['APPNP','GAT','GCN','GCNII','GraphSAGE','MoNet']
for pro in pro_list:
    path1 = output_path + pro + '/'+teacher_list[0] + '/'+'cascade_random_0_30/' + 'output.txt'
    path2=output_path + pro + '/'+teacher_list[1] + '/'+'cascade_random_0_30/' + 'output.txt'
    path3=output_path + pro + '/'+teacher_list[2] + '/'+'cascade_random_0_30/' + 'output.txt'
    path4=output_path + pro + '/'+teacher_list[3] + '/'+'cascade_random_0_30/' + 'output.txt'
    path5=output_path + pro + '/'+teacher_list[4] + '/'+'cascade_random_0_30/' + 'output.txt'
    path6=output_path + pro + '/'+teacher_list[5] + '/'+'cascade_random_0_30/' + 'output.txt'

    softlable1=np.genfromtxt(path1)
    softlable2=np.genfromtxt(path2)
    softlable3=np.genfromtxt(path3)
    softlable4=np.genfromtxt(path4)
    softlable5=np.genfromtxt(path5)
    softlable6=np.genfromtxt(path6)

    softlable_list=[softlable1,softlable2,softlable3,softlable4,softlable5,softlable6]

    aggrate_softlabel1=[]
    aggrate_softlabel2=[]
    aggrate_softlabel3=[]
    aggrate_softlabel4=[]
    aggrate_softlabel5=[]
#拼接
    for i in range(len(softlable1)):
        random_selct=np.random.randint(0,6)#随机选择教师模型
        aggrate_softlabel1.append(softlable_list[random_selct][i])#拼接
    aggrate_softlabel1=np.array(aggrate_softlabel1)
    np.savetxt('C:/Users/Ye Yue/Desktop/softlable/Proposed/'+pro+'_'+str(1)+'.txt',aggrate_softlabel1)
#聚合
    for i in range(len(softlable1)):
        #按平均值聚合
        a=(softlable1[i][0]+softlable2[i][0]+softlable3[i][0]+softlable4[i][0]+softlable5[i][0]+softlable6[i][0])/6
        b=(softlable1[i][1]+softlable2[i][1]+softlable3[i][1]+softlable4[i][1]+softlable5[i][1]+softlable6[i][1])/6
        aggrate_softlabel4.append(np.array([a,b]))#拼接
    aggrate_softlabel4=np.array(aggrate_softlabel4)
    np.savetxt('C:/Users/Ye Yue/Desktop/softlable/Traditional/' + pro + '.txt', aggrate_softlabel4)

    for i in range(len(softlable1)):
        random_selct=np.random.randint(0,6)#随机选择教师模型
        aggrate_softlabel2.append(softlable_list[random_selct][i])#拼接
    aggrate_softlabel2=np.array(aggrate_softlabel1)
    for i in range(len(softlable1)):
        a=(aggrate_softlabel1[i][0]+aggrate_softlabel2[i][0])/2
        b=(aggrate_softlabel1[i][1]+aggrate_softlabel2[i][1])/2
        aggrate_softlabel3.append(np.array([a,b]))
    aggrate_softlabel3=np.array(softlable3)
    np.savetxt('C:/Users/Ye Yue/Desktop/softlable/Proposed/' + pro + '_' + str(3) + '.txt', aggrate_softlabel1)





