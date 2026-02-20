cmake_minimum_required(VERSION 3.15)
project(edge_predictors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/lib")
file(MAKE_DIRECTORY ${OUTPUT_DIR})

add_library(lgbm_predictor SHARED lgbm_predictor.cpp)
add_library(lr SHARED lr.cpp)

if(MSVC)
  target_compile_options(lgbm_predictor PRIVATE /O2 /W3)
  target_compile_options(lr PRIVATE /O2 /W3)
else()
  target_compile_options(lgbm_predictor PRIVATE -O3 -Wall -Wextra)
  target_compile_options(lr PRIVATE -O3 -Wall -Wextra)
  if(UNIX AND NOT APPLE)
    target_link_libraries(lgbm_predictor PRIVATE dl)
  endif()
endif()

set_target_properties(lgbm_predictor PROPERTIES
  OUTPUT_NAME "lgbm_predictor"
  POSITION_INDEPENDENT_CODE ON
  RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

set_target_properties(lr PROPERTIES
  OUTPUT_NAME "lr"
  POSITION_INDEPENDENT_CODE ON
  RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
  set_target_properties(lgbm_predictor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${OUTPUT_DIR}"
  )
  set_target_properties(lr PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${OUTPUT_DIR}"
  )
endforeach()
