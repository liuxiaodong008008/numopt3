cmake_minimum_required(VERSION 3.12)
project(numopt3)

set(CMAKE_CXX_STANDARD 17)

add_executable(numopt3 main.cpp matrix.h headers.h matrix_operation.h hyperdual.h problem.h optimizer.h)