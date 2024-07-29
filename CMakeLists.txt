# cmake_minimum_required(VERSION 3.23)
# project(Fry02Opencv)


# # 使用 file(GLOB ...) 来查找所有的 .c* 和 .h* 源文件
# file(GLOB SOURCE_CODE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)

# # message(CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR})
# # message(SOURCE_CODE: ${SOURCE_CODE})
# # ================== 设置输出目录相关 ==================
# # 设置输出目录的变量
# set(OUTPUT_IMAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_images")
# #set(INPUT_IMAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/input_images")
# # 创建输出目录
# file(MAKE_DIRECTORY ${OUTPUT_IMAGE_DIR})
# # 在CMakeLists.txt文件中，直接将路径添加为宏定义：
# add_definitions(-DOUTPUT_IMAGE_DIR="${OUTPUT_IMAGE_DIR}")
# # 代码中使用： std::string saveDir = std::string(OUTPUT_IMAGE_DIR);
# # ================== END 设置输出目录相关 ==================


# # =================== Opencv ===================
# # 寻找OpenCV库
# # 设置 OpenCV_DIR 为后面的find_package指定了opencv的查找路径
# set(OpenCV_DIR D:/opencv/opencv/build/x64/vc15/lib)
# # 用find_package命令指定CMake需要寻找OpenCV，
# # 并且必须包含指定的组件
# # （calib3d, core, features2d, flann, gapi, highgui, imgcodecs, imgproc, world）。
# # 如果这些组件中有任何一个未找到，CMake将会报错。
# find_package(OpenCV REQUIRED COMPONENTS calib3d core features2d flann gapi highgui imgcodecs imgproc world)
# # 用get_target_property命令获取opencv_world目标的调试版本DLL的位置，
# # 并将其存储在变量__dll_dbg中。
# get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
# # 获取opencv_world目标的发布版本DLL的位置，并将其存储在变量__dll_release中。
# get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

# # include_directories 就是用来包含头文件路径
# # 用include_directories命令为项目添加包含目录
# # ${PROJECT_NAME}是当前项目的名字，
# # ${CMAKE_BINARY_DIR}是当前项目的二进制目录（通常是构建目录）
# # ${OpenCV_INCLUDE_DIRS}是OpenCV头文件的目录。
# # PRIVATE指定了这些包含目录的范围，这里只对${PROJECT_NAME}目标有效。
# include_directories(${PROJECT_NAME} PRIVATE
#     ${CMAKE_BINARY_DIR}
# 	${OpenCV_INCLUDE_DIRS} 
# 	include
# )
# # =================== Opencv ===================

# #include_directories(include)

# add_executable(Fry02Opencv ${SOURCE_CODE})


# # target_link_libraries 可以链接动态库和静态库
# # 链接OpenCV库
# target_link_libraries(
# 	Fry02Opencv
# 	${OpenCV_LIBS}
#     ${OpenCV_LIBRARIES}
# )


cmake_minimum_required(VERSION 3.25)
project(CNN)


# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2")
# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR D:\\rain_programing\\vscodeProject\\buildOpenCV\\buildOpenCV\\install)
# set(OpenCV_DIR_64 D:/opencv/opencv/build/x64/vc15)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include
)
message(STATUS 
*****OpenCV_LIBS:****** ${OpenCV_LIBS} \n 
*****OpenCV_DIR:****** ${OpenCV_DIR} \n
*****OpenCV_INCLUDE_DIRS:******${OpenCV_INCLUDE_DIRS}\n)

# set_target_properties(opencv PROPERTIES
#     IMPORTED_LOCATION_DEBUG ${OpenCV_LIB_DIR}/opencv_world454d.lib
#     IMPORTED_LOCATION_RELEASE ${OpenCV_LIB_DIR}/opencv_world454.lib
# )

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(IMAGE_DIR ${PROJECT_SOURCE_DIR}/imagesDir)
file(MAKE_DIRECTORY ${IMAGE_DIR})
# 在CMakeLists.txt文件中，直接将路径添加为宏定义：
add_definitions(-DIMAGE_DIR="${IMAGE_DIR}")

#message(Hello World)
#message(STATUS ${SRC_LIST})

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})