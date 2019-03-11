PROJECT(ls CXX)
SET(CMAKE_CXX_COMPILER)
SET(CMAKE_CXX_FLAGS "-O2")
SET (LSSRC
  domain.cc
  random.cc
  srandom.cc
  problemcec2014.cc
  cec2014_func.cpp
  problem.cc
  simplex.cc
  cmaeshan.cc
  origcmaes.cc
  main.cpp
)

ADD_LIBRARY(localsearch SHARED ${LSSRC})
TARGET_LINK_LIBRARIES(localsearch)

ADD_EXECUTABLE(main main.cpp)
TARGET_LINK_LIBRARIES(main localsearch)
