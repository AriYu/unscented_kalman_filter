cmake_minimum_required(VERSION 2.8)
project( unscented_kalman_filter )
find_package( OpenCV REQUIRED )


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
ADD_DEFINITIONS(
  -g
  -O0
  -Wall
  -std=c++11
  )

add_executable( 
  test test.cpp 
  unscented_kalman_filter.cpp
  )
target_link_libraries( test ${OpenCV_LIBS} )
