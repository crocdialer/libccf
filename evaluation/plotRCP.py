import matplotlib.pyplot as plt
import numpy as np

graphs = []

def rcp(results, graphNames):
  rsltSize = results[0].size / 3 
  prec = range(0,rsltSize)
  precN = range(0,rsltSize)
  recall = range(0, rsltSize)

  for r in results:
    for x in range(0,rsltSize):
      numPos = float(r[0,x][0])
      if numPos == 0:
        prec[x] = 1
      else:
        prec[x] = r[0,x][2] / numPos

    for x in range(0,rsltSize):
      recall[x] = r[0,x][2] / float(r[0,x][1])

    for x in range(0,rsltSize):
      precN[x] = 1- prec[x]

    graph = plt.plot(precN,recall)
    graphs.append(graph)

  # plot settings
  plt.ylabel('Recall')
  plt.xlabel('1 - Precision')
  plt.axis([0, 1.0, 0, 1.0])
  plt.grid(True)

  # legend for our graphs
  plt.figlegend( (graphs), graphNames,'upper left')

# plots f-measure
def fMeasure(results, graphNames):
  rsltSize = results[0].size / 3 
  prec = range(0,rsltSize)
  recall = range(0, rsltSize)
  thrsh = arange(0, 1, 0.01)
  fMeasure = range(0, rsltSize)

  for r in results:
    for x in range(0,rsltSize):
      numPos = float(r[0,x][0])
      if numPos == 0:
        prec[x] = 1
      else:
        prec[x] = r[0,x][2] / numPos

    for x in range(0,rsltSize):
      recall[x] = r[0,x][2] / float(r[0,x][1])
    
    maxF = 0
    maxIndex = 0
    
    for x in range(0,rsltSize):
      fMeasure[x] = (2*recall[x]*prec[x]) / float(recall[x]+prec[x])
      if fMeasure[x] > maxF:
        maxF=fMeasure[x]
        maxIndex = x;     

    graph = plt.plot(thrsh, fMeasure)
    graphs.append(graph)

  # plot settings
  plt.ylabel('F-Measure')
  plt.xlabel('Threshold')
  plt.axis([0, 1.0, 0, 1.0])
  plt.grid(True)

  # legend for our graphs
  plt.figlegend( (graphs), graphNames,'upper left')

  print "Max F-Measure: " + repr(maxF) +" at " + repr(maxIndex) +"\n\n"
  print "Precision: " + repr(prec[maxIndex]) +"\nRecall: " + repr(recall[maxIndex]) +"\n"