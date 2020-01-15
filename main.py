import tkinter as tk
from random import randint
import math
from time import time


# blind SD
class Graph:
    Nodes = []      # [[x,y],[x,y]...]
    Edges = []      # [[...],[...]...]

    # Blind
    SD = []

    # A*
    A_asterisco = []
    Edges_measure = []

    def __init__(self, num_nodes=200, num_edges=5):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.p_begin = 0
        self.p_last = 0

    def create_temp(self):
        self.Nodes = [0, 1, 2, 3, 4, 5, 6, 7]
        self.Edges = [[1, 2], [0, 2, 3], [0, 1, 4], [1, 4, 5], [2, 3, 6], [3, 7], [4], [5]]

        self.p_begin = 0
        self.p_last = 7

    def create_graph(self):
        # Add Elements
        for i in range(self.num_nodes):
            x = randint(0, 1000)
            y = randint(0, 1000)
            self.Nodes.append([x, y])

        # Sort List[List]
        # self.Nodes.sort()
        """
        # Add Edges Force
        for i in range(self.num_nodes):
            self.Edges.append([])
        distance = []
        for i in range(len(self.Nodes)):
            for j in range(len(self.Nodes)):
                distance.append([math.sqrt(pow(self.Nodes[i][0] - self.Nodes[j][0], 2) +
                                           pow(self.Nodes[i][1] - self.Nodes[i][1], 2)), j])
            distance.sort()

            while len(self.Edges[i]) < self.num_edges:
                if len(self.Edges[distance[0][1]]) < 5:
                    self.Edges[i].append(distance[0][1])
                    distance.pop(0)
                else:
                    distance.pop(0)
            distance.clear()

        print(self.Edges)
        """
        # Add Edges
        for i in range(self.num_nodes):
            self.Edges.append([])
        for node in range(self.num_nodes):
            edge = len(self.Edges[node])
            while edge < self.num_edges:
                ite = randint(0, self.num_nodes - 1)
                if len(self.Edges[ite]) >= self.num_edges:
                    edge = edge - 1
                else:
                    self.Edges[node].append(ite)
                    self.Edges[ite].append(node)
                edge += 1

        self.p_begin = randint(0, self.num_nodes - 1)
        self.p_last = randint(0, self.num_nodes - 1)

        for i in self.Nodes:
            temp = math.sqrt(pow(i[0] - self.Nodes[self.p_last][0], 2) +
                             pow(i[1] - self.Nodes[self.p_last][1], 2))
            self.A_asterisco.append(temp)

        for i in self.Edges:
            self.Edges_measure.append([])
            for j in i:
                temp = math.sqrt(pow(self.Nodes[j][0] - i[0], 2) + pow(self.Nodes[j][0] - i[1], 2))
                self.Edges_measure[-1].append(temp)

    def blind_sd(self):
        self.SD.append([self.p_begin, -1])

        while self.SD[0][0] != self.p_last:
            temp = []

            for x in self.SD[0]:
                temp.append(x)
            temp.pop(0)
            temp.pop(0)

            # self.SD.insert(0, temp)

            if len(self.Edges[self.SD[0][0]]) != 0:
                for x in range(len(self.Edges[self.SD[0][0]])):
                    no_puede_ser = []
                    for n in self.SD:
                        no_puede_ser.append(n[1])
                    add = True
                    for n in no_puede_ser:
                        if self.Edges[self.SD[0][0]][x] == n:
                            add = False
                    if add is True:
                        temp.insert(0, self.SD[0][0])
                        temp.insert(0, self.Edges[self.SD[0][0]][x])
            self.SD.insert(0, temp)

            #print(self.SD)
            #input("presione")

    def a(self):
        self.SD.append([self.p_begin, -1])

        while self.SD[0][0] != self.p_last:
            temp = []

            for x in self.SD[0]:
                temp.append(x)
            temp.pop(0)
            temp.pop(0)

            # self.SD.insert(0, temp)

            neighbor = []

            if len(self.Edges[self.SD[0][0]]) != 0:
                for x in range(len(self.Edges[self.SD[0][0]])):
                    no_puede_ser = []
                    for n in self.SD:
                        no_puede_ser.append(n[1])
                    add = True
                    for n in no_puede_ser:
                        if self.Edges[self.SD[0][0]][x] == n:
                            add = False
                    if add is True:
                        # temp.insert(0, self.SD[0][0])
                        # temp.insert(0, self.Edges[self.SD[0][0]][x])
                        neighbor.insert(0, [])
                        neighbor[0].insert(0, self.SD[0][0])
                        neighbor[0].insert(0, self.Edges[self.SD[0][0]][x])
                        neighbor[0].insert(0, self.A_asterisco[self.SD[0][0]] + self.Edges_measure[self.SD[0][0]][x])

                neighbor.sort()
                for ite in range(len(neighbor) - 1, -1, -1):
                    temp.insert(0, neighbor[ite][2])
                    temp.insert(0, neighbor[ite][1])

            self.SD.insert(0, temp)

    def print_ruta_and_weight(self):
        route = [self.SD[0][0]]

        for i in range(len(self.SD)):
            if i != (len(self.SD) - 1):
                while self.SD[i][1] != self.SD[i + 1][0]:
                    i += 1
                route.append(self.SD[i + 1][0])

        end_measure = 0
        for i in range(0, len(route) - 1):
            for j in range(len(self.Edges[route[i]])):
                if self.Edges[route[i]][j] == route[i + 1]:
                    end_measure += self.Edges_measure[route[i]][j]

        # print(self.SD)
        print("End Measure is: " + str(end_measure))
        print(route)


class Aplication:
    A = Graph(200, 5)

    def __init__(self):
        self.A.create_graph()
        print(self.A.p_begin)
        print(self.A.p_last)
        self.A.blind_sd()
        print(self.A.SD)

        for x in range(len(self.A.Nodes)):
            self.A.Nodes[x][0] *= 0.65
            self.A.Nodes[x][0] += 25
            self.A.Nodes[x][1] *= 0.65
            self.A.Nodes[x][1] += 25

        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=700, height=700, background="black")
        self.canvas.grid(column=0, row=0)

        # Draw all Edges
        for i in range(len(self.A.Edges)):
            for j in range(len(self.A.Edges[i])):
                x_start = self.A.Nodes[i][0]
                y_start = self.A.Nodes[i][1]
                x_end = self.A.Nodes[self.A.Edges[i][j]][0]
                y_end = self.A.Nodes[self.A.Edges[i][j]][1]
                self.canvas.create_line(x_start, y_start, x_end, y_end, fill="white")

        # Draw all Points
        size = 4
        for point in self.A.Nodes:
            self.canvas.create_oval(point[0] - size, point[1] - size, point[0] + size, point[1] + size, fill="red")

        # Draw Targets
        size_t = 8
        x_p_begin = self.A.Nodes[self.A.p_begin][0]
        y_p_begin = self.A.Nodes[self.A.p_begin][1]
        x_p_last = self.A.Nodes[self.A.p_last][0]
        y_p_last = self.A.Nodes[self.A.p_last][1]
        self.canvas.create_oval(x_p_begin - size_t, y_p_begin - size_t, x_p_begin + size_t, y_p_begin + size_t, fill="blue")
        self.canvas.create_oval(x_p_last - size_t, y_p_last - size_t, x_p_last + size_t, y_p_last + size_t, fill="blue")

        # Draw all Edges
        for i in range(len(self.A.SD)):
            if i != (len(self.A.SD) - 1):
                x_start = self.A.Nodes[self.A.SD[i][0]][0]
                y_start = self.A.Nodes[self.A.SD[i][0]][1]

                while self.A.SD[i][1] != self.A.SD[i + 1][0]:
                    i += 1
                x_end = self.A.Nodes[self.A.SD[i + 1][0]][0]
                y_end = self.A.Nodes[self.A.SD[i + 1][0]][1]
                self.canvas.create_line(x_start, y_start, x_end, y_end, fill="blue")

        '''
        self.canvas.create_line(0, 0, 100, 50, fill="white")
        self.canvas.create_rectangle(150, 10, 250, 110, fill="white")
        self.canvas.create_oval(10, 30, 500, 50, fill="red")
        self.canvas.create_arc(420, 10, 550, 110, fill="yellow", start=180, extent=90)
        self.canvas.create_rectangle(150, 210, 250, 310, outline="white")
        self.canvas.create_oval(300, 210, 400, 350, outline="red")
        self.canvas.create_arc(420, 210, 550, 310, outline="yellow", start=180, extent=90)
        '''
        self.window.mainloop()

# A*
class Aplication_A_asterisco:
    A = Graph()

    def __init__(self):
        self.A.create_graph()
        print(self.A.p_begin)
        print(self.A.p_last)
        self.A.a()
        print(self.A.SD)

        for x in range(len(self.A.Nodes)):
            self.A.Nodes[x][0] *= 0.65
            self.A.Nodes[x][0] += 25
            self.A.Nodes[x][1] *= 0.65
            self.A.Nodes[x][1] += 25

        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=700, height=700, background="black")
        self.canvas.grid(column=0, row=0)

        # Draw all Edges
        for i in range(len(self.A.Edges)):
            for j in range(len(self.A.Edges[i])):
                x_start = self.A.Nodes[i][0]
                y_start = self.A.Nodes[i][1]
                x_end = self.A.Nodes[self.A.Edges[i][j]][0]
                y_end = self.A.Nodes[self.A.Edges[i][j]][1]
                self.canvas.create_line(x_start, y_start, x_end, y_end, fill="white")

        # Draw all Points
        size = 4
        for point in self.A.Nodes:
            self.canvas.create_oval(point[0] - size, point[1] - size, point[0] + size, point[1] + size, fill="red")

        # Draw Targets
        size_t = 8
        x_p_begin = self.A.Nodes[self.A.p_begin][0]
        y_p_begin = self.A.Nodes[self.A.p_begin][1]
        x_p_last = self.A.Nodes[self.A.p_last][0]
        y_p_last = self.A.Nodes[self.A.p_last][1]
        self.canvas.create_oval(x_p_begin - size_t, y_p_begin - size_t, x_p_begin + size_t, y_p_begin + size_t, fill="blue")
        self.canvas.create_oval(x_p_last - size_t, y_p_last - size_t, x_p_last + size_t, y_p_last + size_t, fill="blue")

        # Draw all Edges
        for i in range(len(self.A.SD)):
            if i != (len(self.A.SD) - 1):
                x_start = self.A.Nodes[self.A.SD[i][0]][0]
                y_start = self.A.Nodes[self.A.SD[i][0]][1]

                while self.A.SD[i][1] != self.A.SD[i + 1][0]:
                    i += 1
                x_end = self.A.Nodes[self.A.SD[i + 1][0]][0]
                y_end = self.A.Nodes[self.A.SD[i + 1][0]][1]
                self.canvas.create_line(x_start, y_start, x_end, y_end, fill="blue")

        '''
        self.canvas.create_line(0, 0, 100, 50, fill="white")
        self.canvas.create_rectangle(150, 10, 250, 110, fill="white")
        self.canvas.create_oval(10, 30, 500, 50, fill="red")
        self.canvas.create_arc(420, 10, 550, 110, fill="yellow", start=180, extent=90)
        self.canvas.create_rectangle(150, 210, 250, 310, outline="white")
        self.canvas.create_oval(300, 210, 400, 350, outline="red")
        self.canvas.create_arc(420, 210, 550, 310, outline="yellow", start=180, extent=90)
        '''
        self.window.mainloop()


if __name__ == "__main__":
    # blind SD
    # aplication1 = Aplication_A_asterisco()

    aplication2 = Aplication()

    # A = Graph()
    # A.create_graph()
    # A.a()

    # A.print_ruta_and_weight()
