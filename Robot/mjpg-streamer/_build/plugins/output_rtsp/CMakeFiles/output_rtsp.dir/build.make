# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/Desktop/newCapstone/middleware/camera

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/Desktop/newCapstone/middleware/camera/_build

# Include any dependencies generated for this target.
include plugins/output_rtsp/CMakeFiles/output_rtsp.dir/depend.make

# Include the progress variables for this target.
include plugins/output_rtsp/CMakeFiles/output_rtsp.dir/progress.make

# Include the compile flags for this target's objects.
include plugins/output_rtsp/CMakeFiles/output_rtsp.dir/flags.make

plugins/output_rtsp/CMakeFiles/output_rtsp.dir/output_rtsp.c.o: plugins/output_rtsp/CMakeFiles/output_rtsp.dir/flags.make
plugins/output_rtsp/CMakeFiles/output_rtsp.dir/output_rtsp.c.o: ../plugins/output_rtsp/output_rtsp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Desktop/newCapstone/middleware/camera/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object plugins/output_rtsp/CMakeFiles/output_rtsp.dir/output_rtsp.c.o"
	cd /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/output_rtsp.dir/output_rtsp.c.o   -c /home/pi/Desktop/newCapstone/middleware/camera/plugins/output_rtsp/output_rtsp.c

plugins/output_rtsp/CMakeFiles/output_rtsp.dir/output_rtsp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/output_rtsp.dir/output_rtsp.c.i"
	cd /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pi/Desktop/newCapstone/middleware/camera/plugins/output_rtsp/output_rtsp.c > CMakeFiles/output_rtsp.dir/output_rtsp.c.i

plugins/output_rtsp/CMakeFiles/output_rtsp.dir/output_rtsp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/output_rtsp.dir/output_rtsp.c.s"
	cd /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pi/Desktop/newCapstone/middleware/camera/plugins/output_rtsp/output_rtsp.c -o CMakeFiles/output_rtsp.dir/output_rtsp.c.s

# Object files for target output_rtsp
output_rtsp_OBJECTS = \
"CMakeFiles/output_rtsp.dir/output_rtsp.c.o"

# External object files for target output_rtsp
output_rtsp_EXTERNAL_OBJECTS =

plugins/output_rtsp/output_rtsp.so: plugins/output_rtsp/CMakeFiles/output_rtsp.dir/output_rtsp.c.o
plugins/output_rtsp/output_rtsp.so: plugins/output_rtsp/CMakeFiles/output_rtsp.dir/build.make
plugins/output_rtsp/output_rtsp.so: plugins/output_rtsp/CMakeFiles/output_rtsp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/Desktop/newCapstone/middleware/camera/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library output_rtsp.so"
	cd /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/output_rtsp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
plugins/output_rtsp/CMakeFiles/output_rtsp.dir/build: plugins/output_rtsp/output_rtsp.so

.PHONY : plugins/output_rtsp/CMakeFiles/output_rtsp.dir/build

plugins/output_rtsp/CMakeFiles/output_rtsp.dir/clean:
	cd /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp && $(CMAKE_COMMAND) -P CMakeFiles/output_rtsp.dir/cmake_clean.cmake
.PHONY : plugins/output_rtsp/CMakeFiles/output_rtsp.dir/clean

plugins/output_rtsp/CMakeFiles/output_rtsp.dir/depend:
	cd /home/pi/Desktop/newCapstone/middleware/camera/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/Desktop/newCapstone/middleware/camera /home/pi/Desktop/newCapstone/middleware/camera/plugins/output_rtsp /home/pi/Desktop/newCapstone/middleware/camera/_build /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp /home/pi/Desktop/newCapstone/middleware/camera/_build/plugins/output_rtsp/CMakeFiles/output_rtsp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : plugins/output_rtsp/CMakeFiles/output_rtsp.dir/depend

