# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/petertseng/Desktop/Programming/kaldi_test/07-LM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/07_LM.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/07_LM.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/07_LM.dir/flags.make

CMakeFiles/07_LM.dir/lab3_lm.C.o: CMakeFiles/07_LM.dir/flags.make
CMakeFiles/07_LM.dir/lab3_lm.C.o: ../lab3_lm.C
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/07_LM.dir/lab3_lm.C.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/07_LM.dir/lab3_lm.C.o -c /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/lab3_lm.C

CMakeFiles/07_LM.dir/lab3_lm.C.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/07_LM.dir/lab3_lm.C.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/lab3_lm.C > CMakeFiles/07_LM.dir/lab3_lm.C.i

CMakeFiles/07_LM.dir/lab3_lm.C.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/07_LM.dir/lab3_lm.C.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/lab3_lm.C -o CMakeFiles/07_LM.dir/lab3_lm.C.s

CMakeFiles/07_LM.dir/lang_model.C.o: CMakeFiles/07_LM.dir/flags.make
CMakeFiles/07_LM.dir/lang_model.C.o: ../lang_model.C
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/07_LM.dir/lang_model.C.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/07_LM.dir/lang_model.C.o -c /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/lang_model.C

CMakeFiles/07_LM.dir/lang_model.C.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/07_LM.dir/lang_model.C.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/lang_model.C > CMakeFiles/07_LM.dir/lang_model.C.i

CMakeFiles/07_LM.dir/lang_model.C.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/07_LM.dir/lang_model.C.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/lang_model.C -o CMakeFiles/07_LM.dir/lang_model.C.s

CMakeFiles/07_LM.dir/main.C.o: CMakeFiles/07_LM.dir/flags.make
CMakeFiles/07_LM.dir/main.C.o: ../main.C
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/07_LM.dir/main.C.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/07_LM.dir/main.C.o -c /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/main.C

CMakeFiles/07_LM.dir/main.C.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/07_LM.dir/main.C.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/main.C > CMakeFiles/07_LM.dir/main.C.i

CMakeFiles/07_LM.dir/main.C.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/07_LM.dir/main.C.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/main.C -o CMakeFiles/07_LM.dir/main.C.s

CMakeFiles/07_LM.dir/util.C.o: CMakeFiles/07_LM.dir/flags.make
CMakeFiles/07_LM.dir/util.C.o: ../util.C
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/07_LM.dir/util.C.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/07_LM.dir/util.C.o -c /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/util.C

CMakeFiles/07_LM.dir/util.C.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/07_LM.dir/util.C.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/util.C > CMakeFiles/07_LM.dir/util.C.i

CMakeFiles/07_LM.dir/util.C.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/07_LM.dir/util.C.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/util.C -o CMakeFiles/07_LM.dir/util.C.s

# Object files for target 07_LM
07_LM_OBJECTS = \
"CMakeFiles/07_LM.dir/lab3_lm.C.o" \
"CMakeFiles/07_LM.dir/lang_model.C.o" \
"CMakeFiles/07_LM.dir/main.C.o" \
"CMakeFiles/07_LM.dir/util.C.o"

# External object files for target 07_LM
07_LM_EXTERNAL_OBJECTS =

07_LM: CMakeFiles/07_LM.dir/lab3_lm.C.o
07_LM: CMakeFiles/07_LM.dir/lang_model.C.o
07_LM: CMakeFiles/07_LM.dir/main.C.o
07_LM: CMakeFiles/07_LM.dir/util.C.o
07_LM: CMakeFiles/07_LM.dir/build.make
07_LM: CMakeFiles/07_LM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable 07_LM"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/07_LM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/07_LM.dir/build: 07_LM

.PHONY : CMakeFiles/07_LM.dir/build

CMakeFiles/07_LM.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/07_LM.dir/cmake_clean.cmake
.PHONY : CMakeFiles/07_LM.dir/clean

CMakeFiles/07_LM.dir/depend:
	cd /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/petertseng/Desktop/Programming/kaldi_test/07-LM /Users/petertseng/Desktop/Programming/kaldi_test/07-LM /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug /Users/petertseng/Desktop/Programming/kaldi_test/07-LM/cmake-build-debug/CMakeFiles/07_LM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/07_LM.dir/depend
