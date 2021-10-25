#include "linealregression.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

float linealregression::funcionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X * theta - y).array(),2);
    return (diferencia.sum()/(2*X.rows()));

}

/*
 *  Se implementa la funcion para dar al algoritmo lo svalores de thra iniciaes,
 *  que cambiaran iterativamente hasta que converga al valor minimo de la funcion
 *  de costo. Basicamente describira el gradiente descendiente: El cual es dado por
 *  la derivada parcial de la funcion. L funcion tiene un alpha que representa el
 *  salto del gradiente y el numero de iteraciones ueq se necesitam para actualizar theta
 *  hasta que la funcio nconverga al minimo esperado
 */

std::tuple<Eigen::VectorXd, std::vector<float>> linealregression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones){
    /*Almacenamiteno temporal para los valores de theta*/
    Eigen::MatrixXd temporal = theta;
    /*Variable con la cantidad de parametors m (FEATURES)*/
    int parametros = theta.rows();
    /*Ubicar el costo inicial que se actalizara iterativamente con los pesos*/
    std::vector<float> costo;
    costo.push_back(funcionCosto(X,y,theta));
    /*Por cada iteracion se calcula la funcion de error */
    for(int i = 0 ; i<iteraciones ; ++i){
        Eigen::MatrixXd error = X*theta - y;
        for(int j = 0 ; j<parametros ; ++j){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = theta(j,0)-((alpha/X.rows())*termino.sum());
        }
        theta = temporal;
        costo.push_back(funcionCosto(X,y,theta));
    }
    return std::make_tuple(theta,costo);
}






