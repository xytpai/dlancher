cmake_minimum_required(VERSION 3.0)
project(dlancher)
aux_source_directory(${CMAKE_SOURCE_DIR}/examples DIR_EXAMPLES)
message("dlancher example files: ${DIR_EXAMPLES}")

find_package(CUDA)
if(CUDA_FOUND)
    message("cuda include path: ${CUDA_INCLUDE_DIRS}")
    aux_source_directory(${CMAKE_SOURCE_DIR}/cuda DIR_SRCS)
    message("dlancher source files: ${DIR_SRCS}")
    cuda_add_library(dlancher SHARED ${DIR_SRCS})
    # example
    foreach(EXAMPLE ${DIR_EXAMPLES})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_EXAMPLE ${EXAMPLE})
        cuda_add_executable(test_${FILE_EXAMPLE} ${EXAMPLE} ${DIR_SRCS})
    endforeach()
endif(CUDA_FOUND)

install(TARGETS dlancher
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)