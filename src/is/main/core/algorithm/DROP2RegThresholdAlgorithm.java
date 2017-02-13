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
 * DROP2RegThresholdAlgorithm.java
 * Copyright (C) 2014 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable;
import java.util.Vector;

import main.core.algorithm.sort.SortByDistance;
import main.core.exception.NotEnoughInstancesException;
import main.core.util.InstanceIS;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * <b>Descripción</b><br>
 * Algoritmo de selección de instancias Drecremental Reduction Optimization Procedure 3 adaptado a regresión 
 * por medio del umbral.
 * <p>
 * <b>Detalles</b><br>
 * Ordena las instancias en función de la distancia a su enemigo más cercano (de mayor a menor).
 * Elimina las instancias en función de los conjuntos de vecinos más cercanos y asociados.
 * Posibilita seguir al algoritmo paso a paso.  
 * </p>
 * <p>
 * <b>Pseudocódigo del DROP2</b><br>
 * <span style="font-weight: bold;">Require:</span>Training set&nbsp;<span style="font-style: italic;"></span>
 * <span style="font-style: italic;">X</span> = {(x<sub>1</sub>,y<sub>1</sub>),...,(x<sub>n</sub>, y<sub>n
 * </sub>)},a selector <span style="font-style: italic;">S = </span><span style="font-style: italic;" 
 * class="mw-headline" id=".E2.88.85.7B.7D">&#8709;<br></span><span style="font-weight: bold;">Ensure: </span>
 * Theset of selected instances <span style="font-style: italic;">S&nbsp;&#8834;X</span><br>
 * <span style="font-style: italic;"></span><br>&nbsp; 1:&nbsp;<span style="font-style: italic;">S= X<br>
 * </span>&nbsp; 2: Sort instances in <span style="font-style: italic;">S</span> bydistance&nbsp;to their 
 * nearest enemy<br><span style="font-weight: bold;"></span>&nbsp;3:&nbsp;<span style="font-weight: bold;">
 * foreach</span>instance <span style="font-style: italic;">P &#8712; S</span><span style="font-weight: bold;">
 * do</span><span style="font-style: italic;"></span><br>&nbsp; 4:&nbsp;&nbsp;&nbsp;&nbsp;Find 
 * <span style="font-style: italic;">P.N</span><sub>1..k+1</sub>,the k + 1 nearest neighbors of 
 * <span style="font-style: italic;">P</span>in <span style="font-style: italic;">S</span><br>
 * <span style="font-style: italic;"></span><span style="font-weight: bold;"></span>&nbsp; 5:&nbsp; &nbsp; Add 
 * <span style="font-style: italic;">P </span> to each of its neighbor's list of associates
 * <span style="font-style: italic;"></span><br>&nbsp; 6:&nbsp;<span style="font-weight: bold;">endfor</span>
 * <br>&nbsp; 7: <span style="font-weight: bold;">foreach</span>instance <span style="font-style: italic;">P 
 * &#8712; S</span><span style="font-weight: bold;">do<br></span>&nbsp; 8:&nbsp;&nbsp;&nbsp; &nbsp;Let 
 * <span style="font-style: italic;">with</span> = # ofassociates of <span style="font-style: italic;">P
 * </span>classified correctly with <span style="font-style: italic;">P</span>as a neighbor<br>&nbsp; 9:&nbsp;
 * &nbsp;&nbsp;&nbsp;Let <span style="font-style: italic;">without</span> = # ofassociates of 
 * <span style="font-style: italic;">P</span>classified correctly without <span style="font-style: italic;">P
 * </span><br>10:&nbsp;&nbsp;&nbsp; &nbsp;<span style="font-weight: bold;">if</span> 
 * <span style="font-style: italic;">without</span>&nbsp;&#8805; <span style="font-style: italic;">with</span> 
 * <span style="font-weight: bold;">then</span><br>11:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; Remove 
 * <span style="font-style: italic;">P</span> from <span style="font-style: italic;">S</span><br>12:&nbsp;
 * &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;<span style="font-weight: bold;">foreach</span>Associate 
 * <span style="font-style: italic;">A</span>of <span style="font-style: italic;">P</span> 
 * <span style="font-weight: bold;">do</span><br>13:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 * &nbsp; Remove <span style="font-style: italic;">P</span> from <span style="font-style: italic;">A</span>'s 
 * list of nearestneighbors<br>14:&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;Find a new 
 * nearest neighborfor <span style="font-style: italic;">A</span><br>15:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
 * &nbsp;&nbsp;&nbsp; &nbsp;Add <span style="font-style: italic;">A</span> to its newneighbor's list of 
 * associated<br>16:&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;<span style="font-weight: bold;">end for</span>
 * <br>17:&nbsp;&nbsp;&nbsp; <span style="font-weight: bold;">endif</span><br>18: 
 * <span style="font-weight: bold;">end for</span><br>19: <span style="font-weight: bold;"></span>
 * <span style="font-weight: bold;">return</span> <span style="font-style: italic;">S</span>
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Implementa el algoritmo DROP3 para regresión por medio del umbral.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.6
 */
public class DROP2RegThresholdAlgorithm extends DROPRegThresholdAlgorithm implements Serializable {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = -7825386052332193726L;

	/**
	 * Indica si se ha ordenado el conjunto de entrenamiento.
	 */
	protected boolean mOrdered;

	/**
	 * Valor de beta: sensitividad/especificidad.
	 */
	protected double mBeta;
	
	/**
	 * Constructor por defecto del algoritmo DROP2.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.
	 */
	public DROP2RegThresholdAlgorithm () {
		super();
	} // DROP2RegThresholdAlgorithm
	
	/**
	 * Constructor del algoritmo DROP2 al que se le pasa el nuevo conjunto de instancias a tratar.
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
	public DROP2RegThresholdAlgorithm (Instances train) throws NotEnoughInstancesException {
		super(train);
	} // DROP2RegThresholdAlgorithm
	
	/**
	 * Constructor del algoritmo DROP2 al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * Algorithm();<br>
	 * reset(train, inputDatasetIndex);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto inicial
	 * de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public DROP2RegThresholdAlgorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);
	} // DROP2RegThresholdAlgorithm
	
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
	 * En el primer paso ordena las instancias de mayor a menor distancia al enemigo más próximo y 
	 * calcula los conjuntos vecindario y asociados.<br>
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
		if (!mOrdered){
			// Inicializar la solución: S = T.
			mSolutionSet = new Instances(mTrainSet);

			// Inicializar el vector de índices de salida.
			for (Integer index : mInputDatasetIndex)
				mOutputDatasetIndex.add(new Integer(index));

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
			
			// Inicializar las variables para iniciar DROP2.
			mCurrInstancePos = 0;
			mCurrentInstance = mTempSet.firstInstance();
			mCalcNeighbourAssociate = true;
		} else {
			// Si without >= with eliminar la instancia actual.
			if (calcWithout(mTempSet) >= calcWith(mTempSet))
				// Eliminar la instancia actual porque without >= with.
				removeCurrentInstance();

			// Si hemos recorrido todas las instancias del dataset de entrada finalizar el algoritmo.
			if (mCurrInstancePos == mTempSet.numInstances() - 1)
				return false;
			
			// Aumentar la instancia actual.
			mCurrInstancePos++;
			
			// Asignar la siguiente instancia a analizar.
			mCurrentInstance = mTempSet.instance(mCurrInstancePos);
		}
		
		return true;
	} // step
	
	/**
	 * Ordenar el conjunto solución en función a la distancia al enemigo más próximo de mayor a menor.
	 * 
	 * @param orderBy Si es veradero las instancias serán ordenadas de menor a mayor distancia a su enemigo,
	 * en caso contrario las ordenará de mayor a menor distancia.
	 * @return Verdadero si existe más de una instancia tras la eliminación de duplicadas, falso en caso
	 * contrario.
	 */
	protected void orderInstances (boolean orderBy) {
		SortByDistance sbd;
		Vector<Vector<Instance>> tmpAssociates;
		Vector<Vector<Instance>> tmpNeighbours;
		Vector<Integer> tmpIndexes;
		double indexOfInstances[] = SortByDistance.getIndexArray(mSolutionSet.numInstances());
		
		// Ordenar las instancias en función a la distancia a su enemigo más próximo.
		sbd = new SortByDistance(mSolutionSet, mOutputDatasetIndex);
		
		// Ordenar SolutionSet en función de la distancia al enemigo más próximo.
		sbd.orderByNearestEnemyReg(mNeighbours, orderBy, mBeta);

		// Ordenación por QuickSort.
		NearestNeighbourSearch.quickSort(sbd.getDistancesToNearEnemy(), indexOfInstances, 0, mSolutionSet.numInstances() - 1);
		
		// Creamos un conjunto de instancias vacío.
		mTempSet = new Instances(mSolutionSet, mSolutionSet.numInstances());
		
		// Inicializar los vectores temporales para la ordenación.
		tmpAssociates = new Vector<Vector<Instance>>(mSolutionSet.numInstances());
		tmpNeighbours = new Vector<Vector<Instance>>(mSolutionSet.numInstances());
		tmpIndexes = new Vector<Integer>(mSolutionSet.numInstances());

		// Almacenar en tmpInstances las instancias en el orden devuelto por quicksort.
		// Mantener al mismo tiempo el vector de índices de salida.
		if (orderBy)
			for (int i = 0; i < indexOfInstances.length ; i++) {
				mTempSet.add(mSolutionSet.instance((int)indexOfInstances[i]));
				tmpIndexes.add(mInputDatasetIndex.get((int)indexOfInstances[i]));
				tmpAssociates.add(mAssociates.get((int)indexOfInstances[i]));
				tmpNeighbours.add(mNeighbours.get((int)indexOfInstances[i]));
			}
		else
			for (int i = indexOfInstances.length - 1; i >= 0; i--) {
				mTempSet.add(mSolutionSet.instance((int)indexOfInstances[i]));
				tmpIndexes.add(mInputDatasetIndex.get((int)indexOfInstances[i]));
				tmpAssociates.add(mAssociates.get((int)indexOfInstances[i]));
				tmpNeighbours.add(mNeighbours.get((int)indexOfInstances[i]));
			}
		
		// Inicializar el algoritmo con las instancias ordenadas.
		mSolutionSet = new Instances(mTempSet);
		mOutputDatasetIndex = tmpIndexes;
		mAssociates = tmpAssociates;
		mNeighbours = tmpNeighbours;
		mOrdered = true;
	} // orderInstances
	
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
	 * @param inputDatasetIndex Array de índices a las instancias del dataset. 
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public void reset (Instances train, int[] inputDatasetIndex) throws NotEnoughInstancesException {
		super.reset(train, inputDatasetIndex);
		
		mOrdered = false;
	} // reset
	
} // DROP2RegThresholdAlgorithm
