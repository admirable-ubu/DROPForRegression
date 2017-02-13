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
 * AllISFilter.java
 * Copyright (C) 2016 Universidad de Burgos
 */
package weka.filters;

import weka.core.Instances;
import weka.filters.supervised.instance.InstanceSelectionFilterIF;

/**
 * <b>Descripción</b><br>
 * Filtro que permite utilizar ISFilterClassifier sin utilizar ningún filtro.
 * <p>
 * <b>Detalles</b><br>
 * Deja pasar todas las instancias.
 * <p>
 * <b>Funcionalidad</b><br>
 * Utiliza la biblioteca de algoritmos de selección de instancias realizada para el proyecto de final
 * de carrera en la Universidad de Burgos. Tutelado por: César García Osorio y Juan José Rodríguez Díez.
 * </p>
 * 
 * @author Álvar Arnaiz González
 * @version 1.2
 */
public class AllISFilter extends AllFilter implements InstanceSelectionFilterIF {
	/**
	 * For serialization
	 */
	private static final long serialVersionUID = 8854454316908803297L;

	@Override
	public Instances getSolutionSet() {
		
		return new Instances(getInputFormat());
	}

	@Override
	public long getFilterCPUTime() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long getFilterUserTime() {
		// TODO Auto-generated method stub
		return 0;
	}
}
