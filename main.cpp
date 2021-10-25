#include "exeigennorm.h"
#include "linealregression.h"

#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

/* En primer lugar, se creara una clase llamadda "ExEigenNomr", la cual nos permitira
 * leer un database, extraer los datos, montar sobre la estructura Eigen para normalizar los datos
 */


int main(int argc, char *argv[]){



    // Se crea una biblioteca de tipo ExEigenNorm
    // Se incluyen los tres argumentos del contructor
    // Nombre del dataset, delimitador, flag

    ExEigenNorm extraccion(argv[1],argv[2],argv[3]);
    linealregression LR;

    // Se leen los datos del archivo, por la funcion LeerCSV()

    std::vector<std::vector<std::string>> dataFrame = extraccion.leerCSV();

    int filas = dataFrame.size();
    int columnas = dataFrame[0].size();
    Eigen::MatrixXd matrizEigen = extraccion.CSVtoEigen(dataFrame,filas,columnas);
    Eigen::MatrixXd normMatriz = extraccion.Normalizador(matrizEigen);

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> divDatos = extraccion.TrainTestSplit(normMatriz,0.8);


    /* Se desempaca la tupla, se usa std::tie
     * https://en.cppreference.com/w/cpp/utility/tuple/tie
     */

    Eigen::MatrixXd X_Train, y_Train, X_Test, y_Test;

    std::tie(X_Train, y_Train, X_Test, y_Test) = divDatos;

    /* Inspeccion visual de la division de los datos para entrenamiento
     * y prueba
     */

    /*
    std::cout << "**** VARIABLES INDEPENDIENTES ****" <<std::endl;
    std::cout << "Tamaño original ->               " << normMatriz.rows() <<std::endl;
    std::cout << "Tamaño entrenamiento (filas) ->  " << X_Train.rows() <<std::endl;
    std::cout << "Tamaño entrenamiento (cols) ->   " << X_Train.cols() <<std::endl;
    std::cout << "Tamaño prueba (filas) ->         " << X_Test.rows() <<std::endl;
    std::cout << "Tamaño prueba (cols) ->          " << X_Test.cols() <<std::endl;
    std::cout << "\n**** VARIABLES DEPENDIENTES ****" <<std::endl;
    std::cout << "Tamaño original ->               " << normMatriz.rows() <<std::endl;
    std::cout << "Tamaño entrenamiento (filas) ->  " << y_Train.rows() <<std::endl;
    std::cout << "Tamaño entrenamiento (cols) ->   " << y_Train.cols() <<std::endl;
    std::cout << "Tamaño prueba (filas) ->         " << y_Test.rows() <<std::endl;
    std::cout << "Tamaño prueba (cols) ->          " << y_Test.cols() <<std::endl;
*/

    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_Train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_Train.rows());
    /*Redimension de las matrices para ubicacion en los vectores e ONES (similar a reshape
     * con Numpy)*/

    X_Train.conservativeResize(X_Train.rows(), X_Train.cols() + 1);
    X_Train.col(X_Train.cols()-1) = vectorTrain;

    X_Test.conservativeResize(X_Train.rows(), X_Train.cols() + 1);
    X_Test.col(X_Train.cols()-1) = vectorTest;

    /* Se define el vector theta que pasara al algoritmo de gradiente descendiente basicamente
     * un vectoe de ZEROS del mismo tamaño del vector de entrenamiento. Adicional se pasara alpha
     * y el numero de iteraciones
    */

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_Train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    /*Se define las variables de salida que representan los coeficientes el vector de costo*/

    Eigen::VectorXd thetaOut;
    std::vector<float> costo;
    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteD = LR.GradienteDescendiente(X_Train, y_Train, theta, alpha, iteraciones);
    std::tie(thetaOut, costo) = gradienteD;

    /*Se imprimen los valores de los coeficientes theta para cada FEAURES*/
    std::cout<<"\nTheta\n"<<thetaOut<<std::endl;
    std::cout<<"\nCosto\n"<<std::endl;
    for(auto valor:costo){
        std::cout<<valor<<std::endl;
    }

    /*Exportamos a ficheros,  csto y thetaOut*/

    extraccion.VectorToFile(costo, "Costo.txt");
    extraccion.EigenToFile(thetaOut, "ThetaOut.txt");

    return EXIT_SUCCESS;
}
