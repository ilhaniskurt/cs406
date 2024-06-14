TARGET_EXEC := run

BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_DIRS += /opt/homebrew/opt/google-benchmark/include/

INC_FLAGS := $(addprefix -I,$(INC_DIRS))

LDFLAGS = -L/opt/homebrew/opt/google-benchmark/lib/
LDLIBS = -lbenchmark -pthread

CXX = g++
OPT = 0
CXXFLAGS = -std=c++11 -Wall -O$(OPT)

CPPFLAGS := $(INC_FLAGS) -MMD -MP

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)


# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)