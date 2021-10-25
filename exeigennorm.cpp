#include "exeigennorm.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

// Primera funcion: Lectura de ficheros csv
// Vector de vectores String
// La idea es leer linea por linea y almacenar en un vector de vectores tipo string

std::vector<std::vector<std::string>> ExEigenNorm::leerCSV(){
    // Se abre el archicho solamente para lectura
    std::ifstream Archivo(setDatos);
    // Vector de vectores del tipo string que tendra los datos del dataSet
    std::vector<std::vector<std::string>> datoString;
    // Se itera a traves de cada linea del dataset, y se divide el contenido
    // dado por el delimitador provisto por el constructor
    std::string linea="";

    while(getline(Archivo,linea)){
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
        datoString.push_back(vectorFila);
    }

    // Se cierra el fichero
    Archivo.close();
    // Se retorna el vector de vectores de tipo string
    return datoString;

}

/*
 * Se crea la segunda función para guardar el vector de vectores de tipo string
 * a una matrix Eigen. Similar a Pandas(Python) para presentar un dataframe.
*/

Eigen::MatrixXd ExEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int col){
    /*
     * Si tiene cabecera la removemos.
     */

    /*
     * Se itera sobre filas y columnas para almacenar en la matrix vacia(Tamaño+filas+columnas),
     * que basicamente almacenará String en un vector: luego lo pasaremos a float para ser manipulados.
     */
        Eigen::MatrixXd dfMatriz(col,filas);
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < col; j++) {
                dfMatriz(j,i) = atof(setDatos[i][j].c_str());
            }
        }
        /*
         * Se transpone la matriz para tener filas por columnas.
         */
        return dfMatriz.transpose();
    }
/*A continuacion se van a implementar las funciones para la normalizacion.*/

    /* En c++, la palabra clave auto especifica que el tipo de variable que se empieza a declarar
     * de deducira automaticamente de su inicializador y para las funciones,
     * si su tipo de retorno es auto, se evaluara mediante la expresion
     * del tipo de retorno en tiempo de ejecucion */

 auto ExEigenNorm::Promedio(Eigen::MatrixXd datos)->decltype(datos.colwise().mean()){
     //std::cout<<"Promedio: \n"<<datos.colwise().mean()<<std::endl;
     return datos.colwise().mean();
}

 /* Para implementar la funcion de desviacion estandar
      * datos = x_1 - promedio(x) */

 auto ExEigenNorm::Desviacion(Eigen::MatrixXd datos)->decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt()){
     //std::cout<<"Desviacion: \n"<<((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt()<<std::endl;
     return ((datos.array().square().colwise().sum()) / (datos.rows())).sqrt();
}

 Eigen::MatrixXd ExEigenNorm::Normalizador(Eigen::MatrixXd datos) {
     Eigen::MatrixXd datos_escalados = datos.rowwise() - Promedio(datos);
     Eigen::MatrixXd matrixNorm = datos_escalados.array().rowwise()/Desviacion(datos_escalados);

     return matrixNorm;
 }

 /* A continuacion se hara una funcion para dividir los datos en conjunto de datos
  * de entrenamiento y conjunto de datos de prueba
  */

 std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> ExEigenNorm::TrainTestSplit(Eigen::MatrixXd datos,float sizeTrain){
     int filas = datos.rows();
     int filasTrain = round(sizeTrain*filas);
     int filasTest = filas-filasTrain;

     /* Con Eigen se puede especificar un bloque de una matriz, por ejemplo
      * se pueden seleccionar las filas superiores para el conjunto de
      * entrenamiento indicando cuantas filas se desean, se selecciona desde 0
      * (fila 0) hasta el numero de filas indicado
      */

     Eigen::MatrixXd entrenamiento = datos.topRows(filasTrain);

     /* Seleccionadas las filas superiores para entrenamiento, se
      * seleccionan las 11 primeras columnas (colunmas a la izquierda)
      * que representan las variables independientes FEATURES
      */

     Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);

     /* Se selecciona la variable dependiente que corresponde a la
      * ultima columna
      */

     Eigen::MatrixXd y_train = entrenamiento.rightCols(1);

     // Se realiza lo mismo para el conjunto de pruebas

     Eigen::MatrixXd test = datos.bottomRows(filasTest);

     Eigen::MatrixXd X_test = test.leftCols(datos.cols()-1);

     Eigen::MatrixXd y_test = test.rightCols(1);

     /* Finalmente se retorna la tuple dada por el conjunto
      * de datos, prueba y entrenamiento
      */

     return std::make_tuple(X_train,y_train,X_test,y_test);
}
     /*Se implementan  2funciones para exportar a ficheros desde vector y desde origen*/

void ExEigenNorm::VectorToFile(std::vector<float> vector, std::string nombre){
    std::ofstream fichero(nombre);
    std::ostream_iterator<float> iterador(fichero, "\n");
    std::copy(vector.begin(), vector.end(),iterador);
}

void ExEigenNorm::EigenToFile(Eigen::MatrixXd datos, std::string nombre){
    std::ofstream fichero(nombre);
    if(fichero.is_open()){
        fichero<<datos<<"\n";
    }
}





