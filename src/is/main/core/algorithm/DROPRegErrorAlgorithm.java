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
 * DROPRegErrorAlgorithm.java
 * Copyright (C) 2014 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable; 

import main.core.exception.NotEnoughInstancesException;
import main.core.util.InstanceIS;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <b>Descripción</b><br>
 * Algoritmo de selección de instancias en regresión basado en el Drecremental Reduction Optimization Procedure 2,
 * pero sin ordenación inicial de instancias.
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
 * @version 1.6
 */
public class DROPRegErrorAlgorithm extends DROPRegAlgorithm implements Serializable {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = -3104283987324202298L;
	
	/**
	 * Constructor por defecto del algoritmo DROP1.
	 * Antes de comenzar la ejecuciÃ³n del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.
	 */
	public DROPRegErrorAlgorithm () {
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
	public DROPRegErrorAlgorithm (Instances train) throws NotEnoughInstancesException {
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
	public DROPRegErrorAlgorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);
	} // DROPRegErrorAlgorithm
	
	/**
	 * Ejecuta un paso del algoritmo.
	 * En el primer paso calcula los conjuntos vecindario y asociados.
	 * A partir del segundo, en cada paso comprueba si without>=with en caso afirmativo la instancia es 
	 * eliminada en caso contrario la instancia permanece en el conjunto solución.
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contrario.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public boolean step () throws Exception {
		if (!mCalcNeighbourAssociate) {
			// Inicializar la solución: S = T.
			mSolutionSet = new Instances(mTrainSet);

			// Inicializar el vector de índices de salida.
			for (Integer index : mInputDatasetIndex)
				mOutputDatasetIndex.add(new Integer(index));
			
			// Borrar las instancias duplicadas.
			InstanceIS.removeDuplicateInstances(mSolutionSet, mOutputDatasetIndex);
			
			// Si solo queda una instancia tras eliminar las instancias duplicadas finaliza el algoritmo.
			if (mSolutionSet.numInstances() == 1)
				return false;
			
			// Inicializar el algoritmo de vecinos cercanos.
			mNearestNeighbourSearch.setInstances(mSolutionSet);
			
			// Calcular los conjuntos vecindario y asociados.
			calcNeighbourAssociateSets(mSolutionSet);
			
			// Inicializar el conjunto temporal de instancias.
			mTempSet = new Instances(mSolutionSet);
			
			// Inicializar las variables para iniciar DROP1.
			mCurrInstancePos = 0;
			mCurrentInstance = mTempSet.firstInstance();
			mCalcNeighbourAssociate = true;
		} else {
			// Si el error sin la instancia es menor o igual al error con ella -> eliminar.
			if (!isUseful(mTempSet))
				// Eliminar la instancia actual
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
	 * Calcula si la instancia actual debe ser eliminada o no en función de su utilidad.
	 * Calcula para cada asociado el error al clasificar dicho asociado en función de sus vecinos 
	 * más cercanos: with (teniendo en cuenta a la instancia actual) y without (sin tener dicha
	 * instancia en cuenta), si el error de without es menor al de with + alpha es que la 
	 * instancia actual es prescindible y no aporta nada al conjunto.
	 * 
	 * @param set Conjunto de instancias con el que se esta trabajando.
	 * @return Verdadero si no debe eliminarse la instancia actual y falso en caso contrario.
	 * @throws Exception Si no puede realizar el cálculo del error.
	 */
	protected boolean isUseful (Instances set) throws Exception {
		Classifier classifier = new IBk(mNumOfNearestNeighbour);
		Instances test, toTrain;
		Evaluation evalWith, evalWithout;
		double errorWith  = 0.0, errorWithout = 0.0;
		int assocPos;
		
		// 20141201 -> Probar con la opción de que tenga en cuenta la distancia de los vecinos para asignar la clase.
//		String[] options = new String[1];
//		options[0] = "-I";
//		((IBk)classifier).setOptions(options);
		// 20141203 -> No funciona mejor
		
		// Recorrer cada asociado.
		for (Instance assoc : mAssociates.elementAt(mCurrInstancePos)) {
			assocPos = InstanceIS.getPosOfInstance(set, assoc);
			test = new Instances(set, 1);
			test.add(assoc);
			toTrain = new Instances(set, mNeighbours.elementAt(assocPos).size());

			// Generar el conjunto entrenamiento sin la instancia actual.
			for (Instance neighbour : mNeighbours.elementAt(assocPos))
				if (!InstanceIS.equals(neighbour, mCurrentInstance))
					toTrain.add(neighbour);
			
			// Evaluar el conjunto sin la instancia actual.
			evalWithout = new Evaluation(toTrain);
			classifier.buildClassifier(toTrain);
			evalWithout.evaluateModel(classifier, test);
			errorWithout += evalWithout.errorRate();
			
			// Añadir la instancia actual y evaluar.
			toTrain.add(mCurrentInstance);
			evalWith = new Evaluation(toTrain);
			classifier.buildClassifier(toTrain);
			evalWith.evaluateModel(classifier, test);
			errorWith += evalWith.errorRate();
		}
		
		// Si el error sin la instancia es menor al error con ella -> eliminar. 
		if (errorWithout <= (errorWith + mAlpha))
			return false;
		
		return true;
	} // isUseful

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
	} // reset
	
} // DROPRegErrorAlgorithm
