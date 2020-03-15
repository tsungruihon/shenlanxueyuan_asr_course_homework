cmake_minimum_required(VERSION 3.15)
project(07_LM)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(07_LM
        lab3_lm.C
        lab3_lm.H
        lang_model.C
        lang_model.H
        main.C
        util.C
        util.H)
