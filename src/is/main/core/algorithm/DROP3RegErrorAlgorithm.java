/*
 * This file is part of Instance Selection Library.
 * 
 * Instance Selection Library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Instance Selection Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Instance Selection Library.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * DROP3RegErrorAlgorithm.java
 * Copyright (C) 2015 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable; 

import main.core.exception.NotEnoughInstancesException;
import main.core.util.InstanceIS;

import weka.core.Instances;

/**
 * <b>Descripción</b><br>
 * Algoritmo de selección de instancias en regresión basado en el Drecremental Reduction Optimization Procedure 2,
 * con ordenación inicial de instancias.
 * <p>
 * <b>Detalles</b><br>
 * Elimina las instancias en función de los conjuntos de vecinos más cercanos y asociados. Por cada instancia,
 * recorre sus asociados y evalúa el impacto de esa instancia en el error de clasificación del asociado, si el
 * error con la instancia es mayor que sin ella se elimina. 
 * Posibilita seguir al algoritmo paso a paso.  
 * </p>
 * <p>
 * <b>Pseudocódigo del DROP1</b><br>
 * <span style="font-weight: bold;">Require:</span> Training set&nbsp;<span style="font-style: italic;">
 * </span><span style="font-style: italic;">X</span> = {(x<sub>1</sub>, y<sub>1</sub>),...,(x<sub>n</sub>, 
 * y<sub>n</sub>)}, a selector <span style="font-style: italic;">S = </span><span style="font-style: italic;" 
 * class="mw-headline" id=".E2.88.85.7B.7D">&#8709;<br></span><span style="font-weight: bold;">Ensure: </span>
 * The set of selected instances <span style="font-style: italic;">S&nbsp;&#8834; X</span><br><span 
 * style="font-style: italic;"></span><br>&nbsp; 1:&nbsp;<span style="font-style: italic;">S = X</span><br>
 * &nbsp; 2:&nbsp;<span style="font-weight: bold;">foreach</span> instance <span style="font-style: italic;">
 * P &#8712; S</span> <span style="font-weight: bold;">do</span><span style="font-style: italic;"></span><br>
 * &nbsp; 3:&nbsp;&nbsp;&nbsp;&nbsp;Find <span style="font-style: italic;">P.N</span><sub>1..k+1</sub>, the 
 * k + 1 nearest neighbors of <span style="font-style: italic;">P</span> in <span style="font-style: italic;">
 * S</span><br><span style="font-style: italic;"></span><span style="font-weight: bold;"></span>&nbsp; 4: 
 * &nbsp;&nbsp; Add <span style="font-style: italic;">P</span> to each of its neighbor's list of associates
 * <span style="font-style: italic;"></span><br>&nbsp; 5:&nbsp;<span style="font-weight: bold;">end for</span>
 * <br>&nbsp; 6: <span style="font-weight: bold;">foreach</span> instance <span style="font-style: italic;">P 
 * &#8712; S</span> <span style="font-weight: bold;">do<br></span>&nbsp; 7:&nbsp;&nbsp;&nbsp; &nbsp; Let 
 * <span style="font-style: italic;">errorWith</span> = &Sigma; &epsilon; of associates of <span style="font-style: italic;">P
 * </span> with <span style="font-style: italic;">P</span> as a neighbour<br>&nbsp; 8:
 * &nbsp;&nbsp;&nbsp; &nbsp;Let <span style="font-style: italic;">errorWithout</span> = &Sigma; &epsilon; of associates of 
 * <span style="font-style: italic;">P</span> without <span style="font-style: italic;">P as a neighbour
 * </span><br>&nbsp; 9:&nbsp;&nbsp;&nbsp; &nbsp;<span style="font-weight: bold;"> if</span> 
 * <span style="font-style: italic;">errorWith</span>&nbsp;&#8805; <span style="font-style: italic;">errorWithout</span>
 * <span style="font-weight: bold;">then</span><br>10:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp; Remove 
 * <span style="font-style: italic;">P</span> from <span style="font-style: italic;">S</span><br>11:&nbsp;
 * &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<span style="font-weight: bold;">foreach</span> Associate 
 * <span style="font-style: italic;">A</span> of <span style="font-style: italic;">P</span> 
 * <span style="font-weight: bold;">do</span><br>12:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 * &nbsp; Remove <span style="font-style: italic;">P</span> from <span style="font-style: italic;">A</span>'s
 * list of nearest neighbors<br>13:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;Find a new
 * nearest neighbor for <span style="font-style: italic;">A</span><br>14:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
 * &nbsp;&nbsp;&nbsp; &nbsp;Add <span style="font-style: italic;">A</span> to its new neighbor's list of 
 * associated<br>15:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<span style="font-weight: bold;">end for
 * </span><br>16:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<span style="font-weight: bold;">foreach</span> 
 *  Neighbor <span style="font-style: italic;">N</span> of <span style="font-style: italic;">P</span> 
 * <span style="font-weight: bold;">do<br></span>17:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
 * &nbsp; Remove <span style="font-style: italic;">P</span> from <span style="font-style: italic;">N</span>'s
 * lists of associates<br>18:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;<span style="font-weight: bold;">
 * end for</span><br><span style="font-weight: bold;"></span>19:&nbsp;&nbsp;&nbsp; 
 * <span style="font-weight: bold;">end if</span><br>20: <span style="font-weight: bold;">end for</span>
 * <br>21: <span style="font-weight: bold;"></span><span style="font-weight: bold;">return</span> 
 * <span style="font-style: italic;">S</span>
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa una modificación del algoritmo DROP2 para regresión sin ordenación inicial.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.7
 */
public class DROP3RegErrorAlgorithm extends DROPRegErrorAlgorithmOrder implements Serializable {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = -3104283987324202298L;
	
	/**
	 * Indica si se ha filtrado ya el conjunto de entrenamiento.
	 */
	private boolean mFilter;
	
	/**
	 * Constructor por defecto del algoritmo DROP1.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.
	 */
	public DROP3RegErrorAlgorithm () {
		super();
	} // DROPRegErrorAlgorithm
	
	/**
	 * Constructor del algoritmo DROP1 al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm();<br>
	 * reset(train);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public DROP3RegErrorAlgorithm (Instances train) throws NotEnoughInstancesException {
		super(train);
	} // DROPRegErrorAlgorithm
	
	/**
	 * Constructor del algoritmo DROP1 al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para
	 * establecer el número de vecinos cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * DROP1Algorithm(train);<br>
	 * reset(inputDatasetIndex);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de í­ndices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public DROP3RegErrorAlgorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);
	} // DROPRegErrorAlgorithm
	
	/**
	 * Devuelve el valor de alfa.
	 * 
	 * @return Valor de alfa.
	 */
	public double getBetha () {
		
		return mBeta;
	} // getBeta
	
	/**
	 * Establece el valor de beta.
	 * Beta debe estar en el intervalo [0, 100] y ponderla la sensitividad o la especificidad del
	 * algoritmo en la fase de ordenado y filtrado.
	 * 
	 * @param beta Valor de alfa.
	 */
	public void setBeta (double beta) {
		if (beta < 0 || beta > 100)
			throw new IllegalArgumentException("El valor de beta debe estar en el intervalo [0, 100]");
		
		mBeta = beta;
	} // setBeta
	
	/**
	 * Ejecuta un paso del algoritmo.
	 * En el primer paso elimina las instancias ruidosas mediante RegENN.
	 * En el segundo paso ordena las instancias de mayor a menor distancia al enemigo más próximo y calcula
	 * los conjuntos vecindario y asociados.<br>
	 * En los siguientes pasos, comprueba si without>=with en caso afirmativo la instancia es eliminada en
	 * caso contrario la instancia permanece en el conjunto solución.<br>
	 * Cabe destacar que en este algoritmo se guarda una copia del conjunto de entrada (tras haber eliminado
	 * las instancias duplicadas) en <code>mTempSet</code>, y en cada paso se avanza sobre éste, por ello
	 * <code>mCurrInstancePos</code> se refiere siempre a la posición sobre el conjunto de instancias temporal
	 * <code>mTempSet</code> y no sobre <code>mSolutionSet</code>. Esto es necesario ya que en el DROP2 se
	 * debe considerar el efecto de borrar una instancia sobre todo el conjunto original y no solo sobre el
	 * conjunto solución obtenido hasta ese momento (como en el DROP1).
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contratio.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public boolean step () throws Exception {
		if (!mFilter) {
			// Filtrar por RegENN.
			return filterInstances();
		} else if (!mOrdered) {
			// Borrar las instancias duplicadas.
			InstanceIS.removeDuplicateInstances(mSolutionSet, mOutputDatasetIndex);
			
			// Si tras el filtrado queda una instancia o menos devolver falso.
			if (mSolutionSet.numInstances() <= 1)
				return false;
			
			// Inicializar el algoritmo de vecinos cercanos.
			mNearestNeighbourSearch.setInstances(mSolutionSet);
			
			// Calcular los conjuntos vecindario y asociados.
			calcNeighbourAssociateSets(mSolutionSet);

			// Ordenar en función a la distancia al enemigo más próximo (de mayor a menor).
			orderInstances(false);
			
			// Inicializar las variables para iniciar DROP3.
			mCurrInstancePos = 0;
			mCurrentInstance = mTempSet.firstInstance();
			mCalcNeighbourAssociate = true;
		}
		
		return super.step();
	} // step

	/**
	 * Filtra el conjunto inicial de instancias por RegENN.
	 * 
	 * @return Verdadero si tras el filtrado hay más de una instancia, falso en caso contrario.
	 * @throws Exception Excepción producida durante el filtrado.
	 */
	private boolean filterInstances () throws Exception {
		ENNRegAlgorithm wea = new ENNRegAlgorithm(mTrainSet, vectorToArray(mInputDatasetIndex));
		
		// Asignar el alfa.
		wea.setAlpha(mBeta);
		
		// Establecer el número de vecinos cercanos a utilizar.
		wea.setNumOfNearestNeighbour(mNumOfNearestNeighbour);
		
		// Ejecutar el filtrado.
		wea.allSteps();
		
		// Inicializar el algoritmo con los datos devueltos por el filtrado RegENN.
		mOutputDatasetIndex = wea.getOutputDatasetIndex();
		mSolutionSet = wea.getSolutionSet();
		mFilter = true;
		
		// Si tras el filtrado solo queda una instancia o menos devolver falso.
		if (mSolutionSet.numInstances() <= 1)
			return false;
		
		return true;
	} // filterInstances
	
	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 *  
	 * @param train Conjunto de entrenamiento.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train) throws NotEnoughInstancesException {
		super.reset(train);
	} // reset
	
	/**
	 * Reinicia el algoritmo con un conjunto de entrenamiento nuevo.
	 * Inicializa las variables de trabajo del algoritmo.
	 *  
	 * @param train Conjunto de entrenamiento.
	 * @param inputDatasetIndex Array de Ã­ndices a las instancias del dataset. 
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train, int[] inputDatasetIndex) throws NotEnoughInstancesException {
		super.reset(train, inputDatasetIndex);

		mFilter = false;
	} // reset
	
} // DROP3RegErrorAlgorithm
