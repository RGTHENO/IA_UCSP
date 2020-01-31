import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

# elimina valores repetidos de 'a' que se encuentran en 'b'
def deleteBfromA(a,b):
    index = np.array([], dtype = np.int16)
    for number in range(len(a)):
        if a[number] in b:
            index = np.append(index, number)

    return np.delete(a, index)
        
class tsp_genetic(object):
    #ini
    def __init__(self, xyMax, numberOfStops, maxmen, mutationRate, verbose = False, mutateBest = True):
        
        self.numberOfStops = numberOfStops #num. de puntos a conecter
        self.mutateBest = mutateBest #probabilidad de mutar
        self.maxmen = maxmen #numero max de ruta
        self.xyMax = xyMax #tamano del mapa
        self.mutationRate = mutationRate # taza de mutacion(0.1)
        
        #puntos(x,y) aleatorios(1-1000)
        self.targets = np.random.randint(xyMax, size=(numberOfStops, 2))
        # print("targets: ",self.targets)
        #routa aleatoria
        self.men = np.empty((maxmen, numberOfStops), dtype = np.int32)
        for number in range(maxmen):
            tempman = np.arange(numberOfStops, dtype = np.int32)
            np.random.shuffle(tempman)
            self.men[number] = tempman
        print("men: ",self.men)
        
        # mejor ruta para la primera poblacion
        self.best = np.array(self.bestSalesmen())[...,0][0]
    
    #retorna la mejor ruta
    def bestSalesmen(self):
        #orden temp
        tempOrder = np.empty([len(self.men), 2], dtype = np.int32)
        # guardamos la ruta antes de cambiarla
        for number in range(len(self.men)):
            tempOrder[number] = [number, 0,]
        # tamano del recorrido para todas las rutas
        for number in range(len(self.men)):
            tempLength = 0
            #tamano del recorrido
            for target in range(len(self.targets) - 1):
                diffx = abs(self.targets[self.men[number][target]][0] - self.targets[self.men[number][target + 1]][0])
                diffy = abs(self.targets[self.men[number][target]][1] - self.targets[self.men[number][target + 1]][1])
                diff = diffy + diffx
                tempLength = tempLength + diff
            #agregar le tamano al final
            diffx = abs(self.targets[self.men[number][0]][0] - self.targets[self.men[number][-1]][0])
            diffy = abs(self.targets[self.men[number][0]][1] - self.targets[self.men[number][-1]][1])
            diff = diffy + diffx
            tempLength = tempLength + diff
            #agregar tamano al orden temp
            tempOrder[number][1] = tempLength
        #ordenar ruta por el tamano del recorrido
        tempOrder = sorted(tempOrder, key=lambda x: -x[1])
        # regresa la mejor mitad de la ruta mas uno
        return tempOrder[int(len(tempOrder)/2):]
    
    #llenamos la ruta con una nueva generacion
    def breedNewGeneration(self):
        # indices de la mejor ruta
        best = np.array(self.bestSalesmen())[...,0]
        # reemplazar la ruta por la nueva generacion si es que no estan entre los mejores
        for i in range(len(self.men)):
            if i not in best:
                self.men[i] = self.men[random.choice(best)].copy() # clonamos uno de los mejores
                
    #mutacion
    def mutate(self):
        for i in range(len(self.men)): # por cada ruta
            if self.mutateBest == True or i != self.best: # mutar si el mejor=True
                for j in range(len(self.men[i]) - 1): # por cada individuo
                    if random.random() < self.mutationRate: # si random es < mutationRate
                            #switch con otro gene
                            rand = random.randint(0, self.numberOfStops - 1)
                            temp = self.men[i][j]
                            self.men[i][j] = self.men[i][rand]
                            self.men[i][rand] = temp
        
    #calcular
    def calculate(self, iterations):
        #calcula el mejor y el mejor tamano
        best = np.array(self.bestSalesmen())[...,0][-1]
        bestlength = np.array(self.bestSalesmen())[...,1][-1]
        #imprime los dos valores
        print(self.men[best])
        print('best length: ', bestlength)
        #self.draw(best)
        #repetir por cada generacion
        for number in range(iterations):
            #crear nueva generacion
            self.breedNewGeneration()
            #el mejor nuevo se pone como parametro para la siguiente gen
            self.best = np.array(self.bestSalesmen())[...,0][-1]
            #mutar
            self.mutate()
        #imprimir y dibujar el resultado final
        best = np.array(self.bestSalesmen())[...,0][-1]
        bestlength = np.array(self.bestSalesmen())[...,1][-1]
        print(self.men[best])
        print('best length: ', bestlength)
        self.draw(best)
    
    #graficos
    def draw(self, index):
        plt.scatter(self.targets[[...,0]], self.targets[[...,1]], s=20)
        plt.show()
        plt.scatter(self.targets[[...,0]], self.targets[[...,1]], s=20)
        linearray = self.targets[self.men[index]]
        linearray = np.append(linearray, [linearray[0]], axis = 0)
        plt.plot(linearray[[...,0]], linearray[[...,1]])
        plt.show()

#ini
man = tsp_genetic(1000, 10, 10, 0.1, mutateBest = False)

man.calculate(10000)