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
 * DROPRegAlgorithm.java
 * Copyright (C) 2014 Universidad de Burgos
 */

package main.core.algorithm;

import java.io.Serializable;
import java.util.Vector;

import main.core.algorithm.sort.SortByDistance;
import main.core.exception.NotEnoughInstancesException;
import main.core.util.InstanceIS;
import main.core.util.LinearISNNSearch;

import weka.core.Instance;
import weka.core.Instances;

/**
 * <b>Descripción</b><br>
 * Superclase para los algoritmos de selección de instancias de la familia Drecremental Reduction
 * Optimization Procedure adaptados para regresión.
 * <p>
 * <b>Detalles</b><br>
 * Presenta el concepto de conjuntos de vecinos más cercanos y asociados y, en función de estos, elimina o
 * mantiene las instancias.
 * Posibilita seguir al algoritmo paso a paso.  
 * </p>
 * <p>
 * <b>Funcionalidad</b><br>
 * Establece los métodos comunes para los algoritmos de la familia DROP2 (el DROP1 no contempla el impacto
 * de la eliminación sobre todo el conjunto de entrenamiento.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.6
 */
public abstract class DROPRegAlgorithm extends AlgorithmReg implements Serializable {
	
	/**
	 * Para la serialización.
	 */
	private static final long serialVersionUID = -4682926144827586156L;

	/**
	 * Conjunto de instancias temporal donde se almacenan las instancias resultantes de la eliminación de 
	 * duplicadas y la ordenación. Este conjunto es necesario porque el DROP2 siempre tiene en cuenta todas
	 * las instancias, incluso las ya eliminadas. 
	 */
	protected Instances mTempSet;
	
	/**
	 * Valor de alfa: sensitividad/especificidad.
	 */
	protected double mAlpha;
	
	/**
	 * Indica si se ha calculado ya el vecindario y los asociados de las instancias.
	 */
	protected boolean mCalcNeighbourAssociate;
	
	/**
	 * Conjunto de los k vecinos más cercanos ordenados.
	 */
	protected Vector<Vector<Instance>> mNeighbours;
	
	/**
	 * Conjunto de los asociados ordenados.
	 */
	protected Vector<Vector<Instance>> mAssociates;
	
	/**
	 * Número de vecinos cercanos a buscar.
	 */
	protected int mNumOfNearestNeighbour;
	
	/**
	 * Número de pasos que el algoritmo lleva ejecutados. 
	 */
	protected int mNumOfIterations;
	
	/**
	 * Constructor por defecto del algoritmo DROP.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.
	 */
	public DROPRegAlgorithm () {
		super();

		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;

		// Por defecto el alfa utilizado es 0.05.
		mAlpha = 0.05;
	} // DROPAlgorithm
	
	/**
	 * Constructor del algoritmo DROP al que se le pasa el nuevo conjunto de instancias a tratar.
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
	public DROPRegAlgorithm (Instances train) throws NotEnoughInstancesException {
		super(train);
		
		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;

		// Por defecto el alfa utilizado es 0.05.
		mAlpha = 0.05;
	} // DROPAlgorithm
	
	/**
	 * Constructor del algoritmo DROP1 al que se le pasa el nuevo conjunto de instancias a tratar.
	 * Antes de comenzar la ejecución del algoritmo debe llamarse a setNumOfNearestNeighbour para establecer
	 * el número de vecinos cercanos deseado, por defecto es 1.<br>
	 * Es equivalente a:<br>
	 * <code>
	 * DROPAlgorithm();<br>
	 * reset(train, inputDatasetIndex);
	 * </code>
	 * 
	 * @param train Conjunto de instancias a seleccionar.
	 * @param inputDatasetIndex Array de índices para identificar cada instancia dentro del conjunto
	 * inicial de instancias.
	 * @throws NotEnoughInstancesException Si el dataset no tiene instancias.
	 */
	public DROPRegAlgorithm (Instances train, int [] inputDatasetIndex) throws NotEnoughInstancesException {
		super(train, inputDatasetIndex);
		
		// Por defecto el número de vecinos cercanos es 1.
		mNumOfNearestNeighbour = 1;

		// Por defecto el alfa utilizado es 0.05.
		mAlpha = 0.05;
	} // DROPAlgorithm
	
	/**
	 * Devuelve el número de vecinos a utilizar.
	 * 
	 * @return Número de vecinos cercanos a utilizar.
	 */
	public int getNumOfNearestNeighbour () {
		
		return mNumOfNearestNeighbour;
	} // getNumOfNearestNeighbour
	
	/**
	 * Devuelve el valor de alfa.
	 * 
	 * @return Valor de alfa.
	 */
	public double getAlpha () {
		
		return mAlpha;
	} // getAlpha
	
	/**
	 * Establece el número de vecinos a utilizar.
	 * 
	 * @param nn Número de vecinos cercanos a utilizar.
	 * @throws IllegalArgumentException Es lanzada si el número de vecinos es menor que 1 o número par.
	 */
	public void setNumOfNearestNeighbour (int nn) {
		if (nn < 1)
			throw new IllegalArgumentException("El número de vecinos cercanos debe ser mayor de 0.");
		
		if (nn%2 == 0)
			throw new IllegalArgumentException("El número de vecinos debe ser impar.");
		
		mNumOfNearestNeighbour = nn;
	} // setNumOfNearestNeighbour

	/**
	 * Establece el valor de alfa.
	 * Alfa debe estar en el intervalo [0, 100] y ponderla la sensitividad o la especificidad del
	 * algoritmo.
	 * 
	 * @param alpha Valor de alfa.
	 */
	public void setAlpha (double alpha) {
		if (alpha < 0 || alpha > 100)
			throw new IllegalArgumentException("El valor de alfa debe estar en el intervalo [0, 100]");
		
		mAlpha = alpha;
	} // setAlpha
	
	/**
	 * Ejecuta un paso del algoritmo.
	 * Cada algoritmo deberá implementar este método.
	 * 
	 * @return Verdadero si quedan pasos que ejecutar, falso en caso contratio.
	 * @throws Exception Excepción producida durante el paso del algoritmo.
	 */
	public abstract boolean step () throws Exception;
	
	/**
	 * Calcula los conjuntos vecindario y asociados.
	 *  
	 * @param set Conjunto de instancias a utilizar para calcular el vecindario y asociados.
	 * @throws Exception Excepción producida en el cálculo de los vecinos cercanos. 
	 */
	protected void calcNeighbourAssociateSets (Instances set) throws Exception {
		// Inicializar los conjuntos vecindario y asociados.
		initNeighbourAssociateSets(set.numInstances());

		// Calcular el conjunto vecindario.
		calcNeighbourSet(set);
		
		// Calcular el conjunto de asociados.
		calcAssociateSet(set);
	} // calcNeighbourAssociateSets
	
	/**
	 * Inicializa los conjuntos vecindario y asociados.
	 * Crea los vectores mNeighbours y mAssociates, la capacidad de cada uno de los vectores de vecinos será 
	 * la que se indica por parámetro y la de asociados el doble.
	 * 
	 * @param capacity Capacidad de los conjuntos vecindario y asociados.
	 */
	protected void initNeighbourAssociateSets (int capacity) {
		Vector<Instance> v;
		int realNumOfNeighbours = mNumOfNearestNeighbour + 1;

		// Inicializar los vectores de vectores.
		mNeighbours = new Vector<Vector<Instance>>(capacity);
		mAssociates = new Vector<Vector<Instance>>(capacity);
		
		for (int i = 0; i < capacity; i++) {
			v = new Vector<Instance>(realNumOfNeighbours);
			mNeighbours.add(v);
			v = new Vector<Instance>(realNumOfNeighbours*2);
			mAssociates.add(v);
		}
	} // initNeighbourAssociateSets
	
	/**
	 * Calcula el conjunto vecindario para todas las instancias del algoritmo.
	 * Selecciona los "k" vecinos más cercanos de cada instancia y los ordena de menor a mayor distancia.
	 * 
	 * @param instances Conjunto de instancias a utilizar para el cálculo del vecindario.
	 * @throws Exception Excepción producida en el cálculo de los vecinos cercanos. 
	 */
	protected void calcNeighbourSet (Instances instances) throws Exception {
		// Calcular el conjunto vecindario.
		for (int i = 0; i < instances.numInstances(); i++)
			mNeighbours.set(i, getNeighbours(instances.instance(i)));
	} // calcNeighbourSet
	
	/**
	 * Devuelve el conjunto vecindario para una instancia dada.
	 * Selecciona los vecinos mas cercanos a cada instancia y los ordena de menor a mayor distancia.
	 * 
	 * @param instance Instancia para la cual se va a calcular el vecindario. 
	 * @return Vector de vecinos ordenados de menor a mayor distancia.
	 * @throws Exception Excepción producida en el cálculo de los vecinos cercanos.
	 */
	protected Vector<Instance> getNeighbours (Instance instance) throws Exception {
		Instances nearNeighbours, reducedNearNeighbours;
		double distances[], reducedDistances[];
		int realNumOfNeighbours = mNumOfNearestNeighbour + 1;
		
		// Obtener los vecinos mas cercanos de instance.
		nearNeighbours = mNearestNeighbourSearch.kNearestNeighbours(instance, realNumOfNeighbours);
		
		// Obtener las distancias a los vecinos.
		distances = mNearestNeighbourSearch.getDistances();
		
		// Si hay mas vecinos cercanos que los establecidos en el algoritmo.
		if (nearNeighbours.numInstances() > realNumOfNeighbours ) {
			reducedNearNeighbours = new Instances(nearNeighbours, realNumOfNeighbours);
			reducedDistances = new double[realNumOfNeighbours];
			
			for (int i = 0; i < realNumOfNeighbours; i++) {
				reducedNearNeighbours.add(nearNeighbours.instance(i));
				reducedDistances[i] = distances[i];
			}
			
			nearNeighbours = reducedNearNeighbours;
			distances = reducedDistances;
		}
		
		// Ordenar los vecinos más cercanos en función de la distancia. 
		return SortByDistance.getSortVectorOfInstances(nearNeighbours, distances, true);
	} // getNeighbours
	
	/**
	 * Devuelve los nuevos vecinos de una instancia dada.
	 * A partir de los vecinos antiguos, calcula el nuevo y devuelve un vector con los nuevos vecinos
	 * ordenados.
	 * 
	 * @param instance Instancia a calcular sus vecinos.
	 * @param oldNeighbours Antiguos vecinos de instance.
	 * @param set Conjunto de instancias con el que se esta trabajando.
	 * @return Vector ordenado de los vecinos cercanos.
	 * @throws Exception Excepción producida en el cálculo de los vecinos cercanos.
	 */
	protected Vector<Instance> getNewNeighbours (Instance instance, Vector<Instance> oldNeighbours, 
	                                             Instances set) throws Exception {
		Vector<Instance> nearNeighbours;
		
		// Obtener los vecinos mas cercanos de instance.
		nearNeighbours = getNeighbours(instance);
		
		// Buscar el nuevo vecino de instance.
		for (int newNeighbourPos, i = 0; i < nearNeighbours.size() && i < mNumOfNearestNeighbour + 1; i++) {
			boolean found = false;
			
			// Buscar si el nuevo vecino ya era vecino antes.
			for (int j = 0; j < oldNeighbours.size() && !found; j++)
				if (InstanceIS.equals(oldNeighbours.elementAt(j), nearNeighbours.elementAt(i)))
					found = true;
			
			// Si no era antes vecino quiere decir que es nuevo y no hay que seguir buscando
			if (!found) {
				// Añadir el nuevo vecino.
				oldNeighbours.add(nearNeighbours.elementAt(i));
				
				// Obtener la posición del nuevo vecino.
                newNeighbourPos = InstanceIS.getPosOfInstance(set, nearNeighbours.elementAt(i));
                
                // Añadir como asociado del nuevo vecino la instancia A.
                mAssociates.elementAt(newNeighbourPos).add(instance);
				break;
			}
		}
		
		// No es necesario ordenar las instancias ya que getNeighbours(instance) ya lo devuelve ordenado. 
		return oldNeighbours;
	} // getNeighbours
	
	/**
	 * Calcula el conjunto de asociados para todas las instancias del algoritmo.
	 * Selecciona los asociados de cada instancia y los ordena de menor a mayor distancia.
	 *
	 * @param instances Conjunto de instancias a utilizar para el cálculo de los asociados.
	 */
	protected void calcAssociateSet (Instances instances) {
		// Calcular la lista de asociados.
		for (int i = 0; i < instances.numInstances(); i++)
			// Recorrer todos los vecindarios.
			for (int j = 0; j < mNeighbours.size(); j++)
				// No recorrer sus propios vecinos.
				if (i != j)
					for (int k = 0; k < mNeighbours.elementAt(j).size(); k++)
						// Si esta en el vecindario asignarlo a su lista de asociados.
						if (InstanceIS.equals(mNeighbours.elementAt(j).elementAt(k), instances.instance(i)))
							mAssociates.elementAt(i).add(instances.instance(j));
		
		// Ordenar la lista de asociados de cada instancia en función de la distancia.
		for (int i = 0; i < mAssociates.size(); i++)
			mAssociates.set(i, getSortVectorByDistance(instances.instance(i), mAssociates.elementAt(i)));
	} // calcAssociateSet
	
	/**
	 * Devuelve el vector ordenado en función de la distancia a la instancia dada.
	 * Ordena el vector dado en función de la distancia a la instancia dada.
	 * 
	 * @param instance Instancia con la que calcular las distancias.
	 * @param vectorToSort Vector de instancias a ordenar por distancias.
	 * @return Vector ordenado de instancias en función de la distancia.
	 */
	protected Vector<Instance> getSortVectorByDistance (Instance instance, Vector<Instance> vectorToSort) {
		double distances[] = new double[vectorToSort.size()];
		
		// Calcular la distancia con todos los elementos del vector.
		for (int i = 0; i < vectorToSort.size(); i++)
			distances[i] = mNearestNeighbourSearch.getDistanceFunction().distance(instance,
			                 vectorToSort.elementAt(i));
		
		return SortByDistance.getSortVectorOfInstances(vectorToSort, distances, true);
	} // getSortVectorByDistance
	
	/**
	 * Calcula el valor de with.
	 * Recorre la lista de asociados de la instancia actual contando cuantos asociados se clasifican
	 * correctamente teniendo a la instancia actual como vecino.<br>
	 * Cabe destacar que solo se tienen en cuenta los "n" vecinos próximos (no los n + 1 almacenados en
	 * mNeighbours), de este modo el conjunto with y el without son comparables.
	 * 
	 * @param set Conjunto de instancias con el que se esta trabajando.
	 * @return Valor de with.
	 * @throws Exception Excepción en el cálculo de vecinos cercanos.
	 */
	protected int calcWith (Instances set) throws Exception {
		Instances neighbours;
		int pos, with = 0;
		double theta;

		// Recorrer los asociados de la instancia actual.
		for (Instance assoc : mAssociates.elementAt(mCurrInstancePos)) {
			pos = InstanceIS.getPosOfInstance(set, assoc);
			
			// Copiar la lista de vecinos eliminado al último de la lista.
			neighbours = new Instances(mSolutionSet, mAssociates.elementAt(mCurrInstancePos).size() - 1);
			
			for (int i = 0; i < mNeighbours.elementAt(pos).size() - 1; i++)
				neighbours.add(mNeighbours.elementAt(pos).elementAt(i));
			
//			theta = getTheta(neighbours, mAlpha, assoc.classIndex());
			theta = getTheta(mNeighbours.elementAt(pos), mAlpha, assoc.classIndex());
			
			// Incrementar with si el asociado se clasifica correctamente teniendo a la instancia actual como
			// vecino.
			if (!isMisclassified(assoc, neighbours, theta))
				with++;
		}
		
		return with;
	} // calcWith
	
	/**
	 * Calcula el valor de without.
	 * Recorre la lista de asociados de la instancia actual contando cuantos asociados se clasifican
	 * correctamente sin tener la instancia actual como vecino.
	 * 
	 * @param set Conjunto de instancias con el que se esta trabajando.
	 * @return Valor de without.
	 * @throws Exception Excepción en el cálculo de vecinos cercanos.
	 */
	protected int calcWithout (Instances set) throws Exception {
		Instances neighbours;
		int pos, without = 0;
		double theta;
		
		// Recorrer los asociados de la instancia actual.
		for (Instance assoc : mAssociates.elementAt(mCurrInstancePos)) {
			pos = InstanceIS.getPosOfInstance(set, assoc);
			
			// Copiar la lista de vecinos sin incluir a la instancia actual.
			neighbours = new Instances(mSolutionSet, mAssociates.elementAt(mCurrInstancePos).size() - 1);
			
			for (Instance tmpInstance : mNeighbours.elementAt(pos))
				if (!InstanceIS.equals(tmpInstance, mCurrentInstance))
					neighbours.add(tmpInstance);
			
//			theta = getTheta(neighbours, mAlpha, assoc.classIndex());
			theta = getTheta(mNeighbours.elementAt(pos), mAlpha, assoc.classIndex());
			
			// Incrementar without si el asociado se clasifica correctamente sin tener a la instancia actual
			// como vecino.
			if (!isMisclassified(assoc, neighbours, theta))
				without++;
		}
		
		return without;
	} // calcWithout

	/**
	 * Elimina la instancia actual.
	 * Será llamado en caso de que el conjunto with sea mayor o igual que without.
	 * Elimina la instancia actual y recorre la lista de sus asociados eliminándose a si misma como vecino y
	 * buscándoles su nuevo vecino.
	 * 
	 * @throws Exception Excepción producida durante la eliminación de la instancia actual.
	 */
	protected void removeCurrentInstance () throws Exception {
		Vector<Instance> newNeighbours;
		int solutionSetPosition, associatePos;
		
		// Obtener la posición de la instancia actual en el conjunto solución.
		solutionSetPosition = InstanceIS.getPosOfInstance(mSolutionSet, mCurrentInstance);
		
		// Eliminar la instancia del conjunto solución.
		mSolutionSet.delete(solutionSetPosition);
		
		// Eliminar el índice de la instancia borrada.
		mOutputDatasetIndex.remove(solutionSetPosition);
		
		// Reiniciar el algoritmo de cálculo de distancias.
		mNearestNeighbourSearch.setInstances(mSolutionSet);
		
		// Recorrer todos los asociados (A) de P.
		for (Instance assoc : mAssociates.elementAt(mCurrInstancePos)) {
			// Posición del asociado A.
			associatePos = InstanceIS.getPosOfInstance(mTempSet, assoc);
			
			// Eliminar la instancia actual de los vecinos.
			InstanceIS.removeInstanceFromVector(mCurrentInstance, mNeighbours.elementAt(associatePos));
			
			// Calcular el vector ordenado de los nuevos vecinos de A.
			newNeighbours = getNewNeighbours(assoc, mNeighbours.elementAt(associatePos), mTempSet);
			
			// Asignar los nuevos vecinos de A.
			mNeighbours.set(associatePos, newNeighbours);
		}
	} // removeCurrentInstance

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
		
		// Inicializar el número de iteraciones y el flag de cálculo de conjuntos vecindario y asociados.
		mNumOfIterations = 0;
		mCalcNeighbourAssociate = false;
		
		// Crear el algoritmo de cálculo de distancias.
		mNearestNeighbourSearch = new LinearISNNSearch();
	} // reset
	
} // DROPRegAlgorithm
