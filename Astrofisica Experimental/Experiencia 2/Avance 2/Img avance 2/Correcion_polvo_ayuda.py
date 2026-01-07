import numpy as np
import extinction

#https://iopscience.iop.org/article/10.1088/0004-637X/763/2/145/pdf

ObservedHa_Hb = 3.6050844688826675
 #Razon medida de Halpha sobre Hbeta. Notar que son Luminosidades. 

#Esto compara la razón de luminosidad con la razón teorica de 2.86 y lo compara con la curva de extinción de ccm89 para las longitudes de onda de Ha y Hb
EB_V = (2.5/(extinction.calzetti00(np.array([4861.0]), 1, 1)-extinction.calzetti00(np.array([6563.0]), 1, 1)))*np.log10(ObservedHa_Hb/2.86)

#Esta función calcula los factores de correción para una longitud de onda dada usando el EB_V calculado arriba. 
# Esta función devuelve un factor, digamos 1.2x. Entonces el flujo corregido es el flujo observado multiplicado por 1.2. 
extinction.remove(extinction.calzetti00(np.array([6563.0]), EB_V, 1), [1])