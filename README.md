# DROPReg: Drecremental Reduction Optimization Procedure algorithms for Regression

This is an open-source filter for Weka that implements DROP algorithms for regression.


###Cite this software as:
 **Á. Arnaiz-González, J-F. Díez Pastor, Juan J. Rodríguez, C. García Osorio.** _Instance selection for regression: Adapting DROP._ Neurocomputing, 201, 66-81. [doi: 10.1016/j.neucom.2016.04.003](doi: 10.1016/j.neucom.2016.04.003)

```
@article{ArnaizGonzalez2016,   
  title = "Instance selection for regression: Adapting {DROP} ",   
  journal = "Neurocomputing ",   
  volume = "201",   
  number = "",   
  pages = "66 - 81",   
  year = "2016",   
  issn = "0925-2312",   
  doi = "10.1016/j.neucom.2016.04.003",   
  author = "\'Alvar Arnaiz-Gonz\'alez and Jos\'e F. D\'iez-Pastor and Juan J. Rodr\'iguez and C\'esar Garc\'ia-Osorio"   
}
```


#How to use

##Download and build with ant
- Download source code: It is host on GitHub. To get the sources and compile them we will need git instructions. The specifically command is:
```git clone https://github.com/alvarag/DROPReg.git ```
- Build jar file: 
```ant dist_all ```
It generates the jar file under /dist/weka



##How to run

Include the file instanceselection.jar into the path. Example: 

```java -cp instanceselection.jar:weka.jar weka.gui.GUIChooser```

The new filter can be found in: weka/filters/supervised/instance.

