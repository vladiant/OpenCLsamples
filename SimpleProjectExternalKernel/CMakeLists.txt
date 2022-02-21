cmake_minimum_required(VERSION 3.10)

project(SimpleOpenCL LANGUAGES C)

find_package(OpenCL REQUIRED)

configure_file(example.cl . COPYONLY)

add_executable(
    ${PROJECT_NAME}
    main.c
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        OpenCL::OpenCL
)
