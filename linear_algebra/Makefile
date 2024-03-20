CXX = g++
OPT = 0
CXXFLAGS = -std=c++11 -Wall -O$(OPT)

INCLUDES = -I/opt/homebrew/opt/google-benchmark/include/
LDFLAGS = -L/opt/homebrew/opt/google-benchmark/lib/
LDLIBS = -lbenchmark -pthread

TARGET_EXEC := run

BUILD_DIR := ./build
SRC_DIRS := ./src

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
CPPFLAGS := $(INC_FLAGS) -MMD -MP

.PHONY: classic simd

# Define object files for each target
CLASSIC_OBJS := $(BUILD_DIR)/classic.o
SIMD_OBJS := $(BUILD_DIR)/simd.o

# Executable targets
classic: $(BUILD_DIR)/classic
simd: $(BUILD_DIR)/simd

# Rule to link the classic executable
$(BUILD_DIR)/classic: $(CLASSIC_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Rule to link the simd executable
$(BUILD_DIR)/simd: $(SIMD_OBJS)
	$(CXX) -target arm64-apple-macosx -march=armv8-a+simd -mfpu=neon $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Rule to compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIRS)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Include dependency files if they exist
-include $(CLASSIC_OBJS:.o=.d)
-include $(SIMD_OBJS:.o=.d)

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)