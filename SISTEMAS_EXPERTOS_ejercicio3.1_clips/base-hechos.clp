; Practica Numero 3 - Ejercicio 1
; Entrada de datos desde archivos
(defrule importar-data
=>
	(open "nombres.txt" data "r")
	(bind ?data (readline data))
	(while (neq ?data EOF)
		(printout t "Se agrego a "?data" a la base de hechos." crlf)
		(str-assert (str-cat "(persona " ?data ")"))
		(bind ?data (readline data))) 
	(close data))